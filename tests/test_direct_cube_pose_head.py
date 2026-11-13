import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from live_recognition.direct_cube_pose import (  # noqa: E402
    CHECKPOINT_FORMAT, DirectCubePoseModel, DirectPoseHead, SymmetryPoseLoss,
    load_direct_pose_checkpoint, make_detector, project_points, rotation_from_6d,
)
from live_recognition.direct_cube_pose_data import (  # noqa: E402
    collate_direct_pose, landmark_template, split_entries, target_tensors,
)
from scripts import export_direct_cube_pose_dataset as exporter  # noqa: E402

RECORDS = json.loads((Path(__file__).parent / "fixtures/direct_cube_pose_metadata.json").read_text())
MANIFEST = exporter.symmetry_manifest()


def payload(record):
    target = exporter.geometry.build_target(copy.deepcopy(record))
    target = exporter.add_center_targets(target, record, set(), MANIFEST)
    target["pose_supervision"] = {"enabled": True}
    return target


def batched(target):
    return {key: value[None] for key, value in target_tensors(target).items()}


def prediction(target):
    value = batched(target)
    result = {name: value[field].clone() for name, field in (
        ("rotation_matrix", "rotation"), ("angles", "angles"), ("center", "center"),
        ("scale", "scale"), ("perspective", "perspective"), ("principal", "principal"),
        ("center_keypoints", "centers"),
    )}
    result["quality_logit"] = torch.zeros(1)
    result["quality"] = torch.full((1,), .5)
    centers = torch.tensor(exporter.geometry.FACE_AXES, dtype=torch.float32)[None] * 1.5
    result["projected_centers"] = project_points(centers, result)
    return result


@pytest.mark.parametrize("index", range(len(RECORDS)))
def test_all_real_modes_and_24_remapped_poses_have_zero_joint_geometric_loss(index):
    reference = payload(RECORDS[index])
    rest, owners = landmark_template()
    criterion = SymmetryPoseLoss(MANIFEST, rest, owners)
    target = batched(reference)
    for item in MANIFEST["candidates"]:
        matrix = np.asarray(item["candidate_to_reference"])
        alternative = exporter.symmetry.transform_target(reference, matrix)
        losses, metrics, selected = criterion(prediction(alternative), target)
        assert selected.item() == item["id"]
        for name in ("rotation", "projection", "angles", "centers", "geometry"):
            assert losses[name].item() < 1e-9, (index, item["id"], name)
        assert metrics["center_rmse_px"] < .001
        assert metrics["landmark_rmse_px"] < .001


def test_rotation_and_turns_cannot_choose_different_symmetries():
    reference = payload(RECORDS[0])
    item = next(item for item in MANIFEST["candidates"] if item["reference_face_to_candidate_face"]["F"] == "R")
    alternative = exporter.symmetry.transform_target(reference, np.asarray(item["candidate_to_reference"]))
    mixed = prediction(alternative)
    mixed["angles"] = batched(reference)["angles"]
    criterion = SymmetryPoseLoss(MANIFEST, *landmark_template())
    losses, _, _ = criterion(mixed, batched(reference))
    assert sum(losses[name].item() for name in ("rotation", "angles", "geometry")) > .01


def test_disallowed_symmetry_does_not_win():
    reference = payload(RECORDS[0])
    target = batched(reference)
    target["allowed"][:] = False
    target["allowed"][:, 0] = True
    item = MANIFEST["candidates"][1]
    alternative = exporter.symmetry.transform_target(reference, np.asarray(item["candidate_to_reference"]))
    losses, _, selected = SymmetryPoseLoss(MANIFEST, *landmark_template())(prediction(alternative), target)
    assert selected.item() == 0
    assert losses["rotation"].item() > .1


def test_projector_matches_independent_homogeneous_camera_projection():
    target = payload(RECORDS[0])
    pred = prediction(target)
    points = torch.tensor([[[-1.5, 1.5, .5], [.5, -.5, -1.5], [0, 0, 0]]])
    cam = points @ pred["rotation_matrix"].transpose(-1, -2) + torch.tensor(target["reference_camera"]["translation_in_cubie_units"])
    homogeneous = cam @ torch.tensor(target["reference_camera"]["intrinsics"]).T
    expected = homogeneous[..., :2] / homogeneous[..., 2:] / target["image_size"][0]
    torch.testing.assert_close(project_points(points, pred), expected, atol=1e-6, rtol=1e-6)


def test_all_masked_batch_has_zero_geometry_gradients_but_trains_quality():
    target = batched(payload(RECORDS[0]))
    target["enabled"][:] = False
    pred = prediction(payload(RECORDS[0]))
    pred = {name: value.detach().requires_grad_(True) for name, value in pred.items()}
    losses, _, _ = SymmetryPoseLoss(MANIFEST, *landmark_template())(pred, target)
    losses["total"].backward()
    assert losses["rotation"] == 0 and losses["geometry"] == 0
    assert pred["angles"].grad.abs().max() == 0
    assert pred["quality_logit"].grad.abs().min() > 0


def test_quality_measures_articulated_projection_accuracy_and_target_is_detached():
    reference = payload(RECORDS[0])
    criterion = SymmetryPoseLoss(MANIFEST, *landmark_template())
    exact = prediction(reference)
    _, metrics, _ = criterion(exact, batched(reference))
    assert metrics["quality_target"] > .999
    wrong = prediction(reference)
    wrong["center"] = wrong["center"] + .3
    wrong["projected_centers"] = wrong["projected_centers"] + .3
    _, metrics, _ = criterion(wrong, batched(reference))
    assert metrics["quality_target"] < .01
    observation_mismatch = batched(reference)
    observation_mismatch["observed_landmarks"] += .3
    _, metrics, _ = criterion(exact, observation_mismatch)
    assert metrics["quality_target"] < .01
    assert metrics["observed_landmark_rmse_px"] > 20


def test_rotation_decoder_handles_zero_and_collinear_inputs_with_finite_gradients():
    value = torch.tensor([[0., 0, 0, 0, 0, 0], [1., 0, 0, 2, 0, 0], [1., 2, 3, 2, 3, 1]], requires_grad=True)
    matrix = rotation_from_6d(value)
    torch.testing.assert_close(matrix.transpose(-1, -2) @ matrix, torch.eye(3).expand(3, -1, -1), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.linalg.det(matrix), torch.ones(3))
    matrix.sum().backward()
    assert torch.isfinite(value.grad).all()


def test_head_uses_both_features_and_predicted_class_scored_points():
    torch.manual_seed(17)
    head = DirectPoseHead([16, 32, 64], width=8, topk=4)
    features = [torch.randn(2, channels, size, size, requires_grad=True)
                for channels, size in ((16, 8), (32, 4), (64, 2))]
    xy = torch.rand(2, 84, 2, requires_grad=True)
    scores = torch.rand(2, 9, 84, requires_grad=True)
    output = head(features, xy, scores)
    output["center"].sum().backward()
    assert all(feature.grad.abs().sum() > 0 for feature in features)
    assert xy.grad.abs().sum() > 0 and scores.grad.abs().sum() > 0
    assert output["center_keypoints"].shape == (2, 6, 2)
    assert output["angles"].shape == (2, 6)


def test_actual_yolo26_wrapper_ignores_presence_and_uses_class_scores():
    model = DirectCubePoseModel(make_detector("yolo26n-pose.yaml"), width=8, topk=4).eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        reference = model(image)["pose"]

        def alter_presence(_module, _inputs, output):
            raw = output[1]
            for branch in (raw["one2many"], raw["one2one"]):
                branch["kpts"] = branch["kpts"].clone()
                branch["kpts"][:, 2::3] += 100
            return output

        handle = model.detector.model[-1].register_forward_hook(alter_presence)
        changed_presence = model(image)["pose"]
        handle.remove()
        torch.testing.assert_close(changed_presence["center"], reference["center"], atol=0, rtol=0)
        torch.testing.assert_close(changed_presence["quality"], reference["quality"], atol=0, rtol=0)

        def alter_scores(_module, _inputs, output):
            output[1]["one2one"]["scores"] = output[1]["one2one"]["scores"] + 20
            return output

        handle = model.detector.model[-1].register_forward_hook(alter_scores)
        changed_scores = model(image)["pose"]
        handle.remove()
        assert (changed_scores["center"] - reference["center"]).abs().max() > 1e-7

    # Direct loss alone must reach the shared backbone; the independent YOLO
    # loss is deliberately absent from this backward pass.
    model.train()
    model.freeze_detector(False)
    model(torch.rand(2, 3, 64, 64))["pose"]["center"].sum().backward()
    assert next(model.detector.model[0].parameters()).grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model.freeze_detector(True)
    model(torch.rand(2, 3, 64, 64))["pose"]["center"].sum().backward()
    assert all(parameter.grad is None for parameter in model.detector.parameters())
    assert any(parameter.grad is not None for parameter in model.pose_head.parameters())


def test_empty_independent_labels_collate_to_valid_yolo_shapes():
    target = target_tensors(payload(RECORDS[0]))
    target["enabled"] = torch.tensor(False)
    batch, direct, _ = collate_direct_pose([
        (torch.zeros(3, 64, 64), target, torch.empty(0, 8), "empty"),
    ])
    assert batch["cls"].shape == (0, 1)
    assert batch["keypoints"].shape == (0, 1, 3)
    assert not direct["enabled"].any()


def test_actual_yolo_loss_and_direct_head_train_on_an_all_empty_batch():
    model = DirectCubePoseModel(make_detector("yolo26n-pose.yaml"), width=8, topk=4).train()
    target = target_tensors(payload(RECORDS[0]))
    target["enabled"] = torch.tensor(False)
    batch, direct, _ = collate_direct_pose([
        (torch.rand(3, 64, 64), target, torch.empty(0, 8), f"empty{index}") for index in range(2)
    ])
    output = model(batch["img"])
    independent_loss, _ = model.detector.init_criterion()(output["independent"], batch)
    direct_loss, _, _ = SymmetryPoseLoss(MANIFEST, *landmark_template())(output["pose"], direct)
    total = independent_loss.sum() / 2 + direct_loss["total"]
    assert torch.isfinite(total)
    total.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    assert direct_loss["geometry"] == 0


def test_stratified_split_is_reproducible_disjoint_and_includes_all_images():
    entries = [{"key": str(index), "source_dataset": str(index % 2), "mode": str(index % 3), "symmetry_count": 4}
               for index in range(60)]
    first = split_entries(entries, .2, 17)
    assert first == split_entries(entries, .2, 17)
    train, val = ({entry["key"] for entry in split} for split in first)
    assert not train & val
    assert train | val == {entry["key"] for entry in entries}


def test_zero_and_nonfinite_allowed_targets_are_rejected():
    value = payload(RECORDS[0])
    value["symmetry"]["allowed_ids"] = []
    with pytest.raises(ValueError, match="symmetry"):
        target_tensors(value)
    value = payload(RECORDS[0])
    value["projection"]["center_xy_normalized"][0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        target_tensors(value)


@pytest.mark.parametrize("end2end", [False, True])
def test_checkpoint_restores_runtime_detection_mode_and_exact_predictions(tmp_path, end2end):
    detector = make_detector("yolo26n-pose.yaml")
    with torch.no_grad():
        detector.model[-1].one2one_cv4_kpts[0].bias.add_(4)
    detector.end2end = end2end
    model = DirectCubePoseModel(detector, width=8, topk=4).eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        expected = model(image)["pose"]
    path = tmp_path / "checkpoint.pt"
    torch.save({"format": CHECKPOINT_FORMAT, "detector_yaml": detector.yaml,
                "detector_end2end": end2end, "head_config": model.head_config,
                "model_state": model.state_dict(), "train_args": {"epochs": 1}}, path)
    restored, _ = load_direct_pose_checkpoint(path)
    assert restored.detector.end2end == end2end
    with torch.no_grad():
        actual = restored(image)["pose"]
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
