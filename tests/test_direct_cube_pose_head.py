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
    CHECKPOINT_FORMAT, DirectCubePoseModel, DirectPoseHead, SymmetryPoseLoss, balanced_face_angle_loss,
    independent_pose_criterion, load_direct_pose_checkpoint, make_detector, project_points, rotation_from_6d,
    visible_reprojection_loss,
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
    # Test exact symmetry algebra with consistent visible observations. Real
    # corner-cut observations can differ from the dominant-angle approximation.
    target["observed_landmarks"] = target["landmarks"].clone()
    target["observed_centers"] = target["centers"].clone()
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


def test_angle_loss_balances_moving_and_stationary_groups_without_diluting_one_moving_face():
    expected = torch.tensor([[20., 0, 0, 0, 0, 0], [20., -20, 0, 0, 0, 0]])
    errors = torch.tensor([[2., 1, 1, 1, 1, 1], [2., 2, 1, 1, 1, 1]], requires_grad=True)
    result = balanced_face_angle_loss(errors, expected)
    torch.testing.assert_close(result, torch.tensor([1.5, 1.5]))
    result.sum().backward()
    torch.testing.assert_close(errors.grad[0], torch.tensor([.5, .1, .1, .1, .1, .1]))


@pytest.mark.parametrize("count", [1, 9, 10, 20, 62])
def test_visible_geometry_keeps_every_good_match_regardless_of_count(count):
    points = torch.linspace(.001, .01, count).reshape(1, 1, count, 1).expand(-1, -1, -1, 2).clone().requires_grad_()
    loss, _ = visible_reprojection_loss(points, torch.zeros_like(points), torch.ones(1, 1, count, dtype=torch.bool))
    loss.sum().backward()
    assert points.grad.min() > 0


def test_visible_geometry_bounds_outlier_influence_and_excludes_only_hidden_points():
    points = torch.full((1, 1, 11, 2), .01)
    points[..., 9, :] = .8
    points[..., 10, :] = 5.
    points.requires_grad_()
    observed = torch.zeros_like(points)
    mask = torch.ones(1, 1, 11, dtype=torch.bool)
    mask[..., 10] = False
    loss, rmse = visible_reprojection_loss(points, observed, mask)
    torch.testing.assert_close(loss, torch.tensor([[(9 * .005 + 2 ** .5 * .8 - .01) / 10]]))
    torch.testing.assert_close(rmse, torch.tensor([[((9 * 2 * .01 ** 2 + 2 * .8 ** 2) / 10) ** .5]]))
    loss.sum().backward()
    assert points.grad[..., :9, :].min() > 0
    torch.testing.assert_close(points.grad[..., 9, :].norm(dim=-1), torch.tensor([[.1]]))
    assert points.grad[..., 10, :].abs().max() == 0


def test_visible_geometry_keeps_gradients_when_every_match_has_large_error():
    points = torch.ones(1, 1, 20, 2, requires_grad=True)
    loss, _ = visible_reprojection_loss(points, torch.zeros_like(points), torch.ones(1, 1, 20, dtype=torch.bool))
    loss.sum().backward()
    assert loss > 0 and torch.isfinite(points.grad).all() and points.grad.min() > 0


def test_visible_metric_and_quality_include_large_errors_without_percentile_trimming():
    reference = payload(RECORDS[0])
    target = batched(reference)
    target["allowed"][:] = False
    target["allowed"][:, 0] = True
    target["observed_center_mask"][:] = False
    target["observed_mask"][:] = False
    target["observed_mask"][:, :10] = True
    target["observed_landmarks"] = target["landmarks"].clone()
    target["observed_landmarks"][:, 0] += .2
    _, metrics, _ = SymmetryPoseLoss(MANIFEST, *landmark_template())(prediction(reference), target, 128)
    torch.testing.assert_close(metrics["visible_keypoint_rmse_px"], torch.tensor((.08 / 10) ** .5 * 128), atol=1e-4, rtol=0)
    assert metrics["quality_target"] < .001


def test_no_visible_points_produce_zero_geometry_and_quality_without_gt_fallback():
    reference = payload(RECORDS[0])
    target = batched(reference)
    target["observed_mask"][:] = False
    target["observed_center_mask"][:] = False
    pred = prediction(reference)
    pred["angles"] = pred["angles"].clone().requires_grad_()
    losses, metrics, _ = SymmetryPoseLoss(MANIFEST, *landmark_template())(pred, target)
    assert losses["geometry"] == 0
    assert metrics["visible_keypoint_rmse_px"] == 0 and metrics["quality_target"] == 0
    losses["geometry"].backward()
    assert torch.isfinite(pred["angles"].grad).all() and pred["angles"].grad.abs().max() == 0


def test_geometry_uses_visible_positions_and_ignores_ideal_gt_landmarks():
    reference = payload(RECORDS[0])
    target = batched(reference)
    target["allowed"][:] = False
    target["allowed"][:, 0] = True
    target["observed_landmarks"] = target["landmarks"].clone() + .04
    pred = prediction(reference)
    criterion = SymmetryPoseLoss(MANIFEST, *landmark_template())
    loss, _, _ = criterion(pred, target)
    changed = {k: v.clone() for k, v in target.items()}
    changed["landmarks"] += 10
    changed["observed_landmarks"][~changed["observed_mask"]] += 100
    other, _, _ = criterion(pred, changed)
    assert loss["geometry"] > 0
    torch.testing.assert_close(loss["geometry"], other["geometry"])


def test_visible_center_sticker_can_supervise_geometry_without_landmarks():
    reference = payload(RECORDS[0])
    target = batched(reference)
    target["observed_mask"][:] = False
    target["observed_center_mask"][:] = False
    target["observed_center_mask"][:, 0] = True
    target["observed_centers"] = target["centers"].clone() + .05
    pred = prediction(reference)
    pred["projected_centers"] = pred["projected_centers"].clone().requires_grad_()
    loss, _, _ = SymmetryPoseLoss(MANIFEST, *landmark_template())(pred, target)
    assert loss["geometry"] > 0
    loss["geometry"].backward()
    assert pred["projected_centers"].grad[:, 0].abs().min() > 0
    assert pred["projected_centers"].grad[:, 1:].abs().max() == 0


def test_center_observation_masks_preserve_retained_visibility_one_and_two():
    reference = payload(RECORDS[0])
    anchors = reference["virtual_core_anchors"]
    for index, visibility in enumerate((1, 2, 0)):
        anchors[index]["source_label_visibility"] = visibility
        anchors[index]["source_observed_pixel"] = [20., 30.]
    target = target_tensors(reference)
    assert target["observed_center_mask"].tolist() == [True, True, False, False, False, False]
    torch.testing.assert_close(target["observed_centers"][0], torch.tensor([20., 30.]) / reference["image_size"][0])


def test_visible_geometry_optimization_follows_observations_instead_of_ideal_target():
    reference = payload(RECORDS[0])
    target = batched(reference)
    target["allowed"][:] = False
    target["allowed"][:, 0] = True
    target["observed_center_mask"][:] = False
    pred = prediction(reference)
    desired_center = pred["center"] + torch.tensor([[.06, -.04]])
    desired = {**pred, "center": desired_center}
    from live_recognition.direct_cube_pose import articulated_landmarks
    criterion = SymmetryPoseLoss(MANIFEST, *landmark_template(), rotation=0, projection=0,
                                 angles=0, centers=0, quality=0)
    points = articulated_landmarks(criterion.rest_points, criterion.owners, pred["angles"])
    target["observed_landmarks"] = project_points(points, desired).detach()
    center = pred["center"].clone().requires_grad_()
    optimizer = torch.optim.Adam([center], lr=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        losses, _, _ = criterion({**pred, "center": center}, target, 128)
        losses["total"].backward()
        optimizer.step()
        scheduler.step()
    _, metrics, _ = criterion({**pred, "center": center}, target, 128)
    torch.testing.assert_close(center, desired_center, atol=1e-4, rtol=0)
    assert metrics["visible_keypoint_rmse_px"] < .02
    assert metrics["landmark_rmse_px"] > 5


@pytest.mark.parametrize("target_value", [0., 20.])
def test_angle_loss_supervises_the_only_present_group(target_value):
    expected = torch.full((2, 24, 6), target_value)
    errors = torch.rand_like(expected, requires_grad=True)
    result = balanced_face_angle_loss(errors, expected)
    torch.testing.assert_close(result, errors.mean(-1))
    result.sum().backward()
    assert torch.isfinite(errors.grad).all() and errors.grad.min() > 0


def test_articulation_geometry_and_angle_gradients_match_finite_differences():
    reference = payload(RECORDS[0])
    target = {k: v.double() if v.is_floating_point() else v for k, v in batched(reference).items()}
    target["allowed"][:] = False
    target["allowed"][:, 0] = True
    criterion = SymmetryPoseLoss(MANIFEST, *landmark_template()).double()
    pred = {k: v.double() for k, v in prediction(reference).items()}
    angles = (target["angles"] + .7).requires_grad_(True)

    def value(angles, component):
        return criterion({**pred, "angles": angles}, target)[0][component]

    for component in ("angles", "geometry"):
        gradient = torch.autograd.grad(value(angles, component), angles)[0]
        numerical = torch.empty_like(angles)
        for face in range(6):
            offset = torch.zeros_like(angles)
            offset[:, face] = 1e-5
            numerical[:, face] = (value(angles + offset, component) - value(angles - offset, component)) / 2e-5
        torch.testing.assert_close(gradient, numerical, atol=1e-7, rtol=1e-4)


def test_moving_and_stationary_metrics_use_their_respective_faces():
    reference = payload(RECORDS[0])
    target = batched(reference)
    target["allowed"][:] = False
    target["allowed"][:, 0] = True
    target["angles"][:] = torch.tensor([20., -20., 0, 0, 0, 0])
    pred = prediction(reference)
    pred["angles"] = torch.tensor([[10., -15., 1., -2., 3., -4.]])
    _, metrics, _ = SymmetryPoseLoss(MANIFEST, *landmark_template())(pred, target)
    assert metrics["moving_angle_mae_deg"] == 7.5
    assert metrics["stationary_angle_mae_deg"] == 2.5
    torch.testing.assert_close(metrics["angle_mae_deg"], torch.tensor(25 / 6))


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
            raw = output[1]
            branch = raw.get("one2one" if model.detector.end2end else "one2many", raw)
            branch["scores"] = branch["scores"] + 20
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
    independent_loss, _ = independent_pose_criterion(model.detector)(output["independent"], batch)
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


def test_non_end2end_pose26_stock_loss_uses_the_same_xy_formula_as_inference():
    detector = make_detector("yolo26n-pose.yaml")
    detector.end2end = False
    head = detector.model[-1]
    criterion = independent_pose_criterion(detector)
    anchors = torch.tensor([[.5, .5], [1.5, .5]])
    raw = torch.tensor([[[[.2, .3, 1.]], [[.4, .6, 1.]]]])
    head.anchors, head.strides = anchors.T, torch.ones(1, 2)
    inference = head.kpts_decode(raw[:, :, 0].transpose(1, 2))[:, :2].transpose(1, 2)
    training = criterion.kpts_decode(anchors, raw)[:, :, 0, :2]
    torch.testing.assert_close(training, inference)


@pytest.mark.parametrize("end2end", [False, True])
def test_observation_tokens_follow_the_configured_inference_branch(end2end):
    detector = make_detector("yolo26n-pose.yaml")
    detector.end2end = end2end
    model = DirectCubePoseModel(detector, width=8, topk=4).eval()
    # Make the two point heads observably different; comparing a checkpoint
    # against its own reload alone cannot detect consistently wrong selection.
    with torch.no_grad():
        for layer in detector.model[-1].one2one_cv4_kpts:
            layer.bias.add_(4)
        image = torch.rand(2, 3, 64, 64)
        output, _, points, scores = model.encode_observations(image)
        decoded, raw = output
        if end2end:
            # Every postprocessed object must have an identical raw point and
            # class score among the tokens used by the custom pose branch.
            for rows, xy, probability in zip(decoded, points, scores):
                classes = rows[:, 5].long()
                distance = (rows[:, None, 6:8] - xy[None] * 64).abs().amax(-1)
                score_error = (rows[:, 4, None] - probability[classes]).abs()
                assert ((distance < 1e-5) & (score_error < 1e-6)).any(-1).all()
        else:
            # Non-E2E inference retains every anchor, so compare all values.
            torch.testing.assert_close(points * 64, decoded[:, 13:15].transpose(1, 2))
            torch.testing.assert_close(scores, decoded[:, 4:13])


@pytest.mark.parametrize("initial_mode,actual_mode", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("frozen", [False, True])
def test_independent_loss_uses_real_output_layout_during_warmup_joint_and_validation(initial_mode, actual_mode, frozen):
    from ultralytics.utils.loss import E2ELoss, PoseLoss26

    detector = make_detector("yolo26n-pose.yaml")
    detector.end2end = initial_mode
    criterion = independent_pose_criterion(detector)
    # Reproduce the reported failure: the forward output can differ from the
    # mode seen when the criterion was constructed.
    detector.end2end = actual_mode
    model = DirectCubePoseModel(detector, width=8, topk=4)
    model.freeze_detector(frozen)
    model.train()
    target = target_tensors(payload(RECORDS[0]))
    labels = torch.tensor([[0., .5, .5, .4, .4, .5, .5, 2.], [3., .2, .3, .1, .1, .2, .3, 1.]])
    batch, _, _ = collate_direct_pose([
        (torch.rand(3, 64, 64), target, labels, f"sample{index}") for index in range(2)
    ])
    output = model(batch["img"])["independent"]
    loss, items = criterion(output, batch)
    raw = output[1] if isinstance(output, tuple) else output
    # Old Ultralytics factories choose the YOLO11 decoder for non-E2E Pose26;
    # use its actual Pose26 loss and the dual/flat structure emitted at runtime.
    stock = E2ELoss(detector, PoseLoss26) if "one2many" in raw and "one2one" in raw else PoseLoss26(detector)
    expected_loss, expected_items = stock(output, batch)
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(items, expected_items)
    assert torch.isfinite(loss).all()
    if not frozen:
        loss.sum().backward()
        assert all(torch.isfinite(parameter.grad).all() for parameter in detector.parameters() if parameter.grad is not None)
    model.eval()
    with torch.no_grad():
        validation_loss, _ = criterion(model(batch["img"])["independent"], batch)
    assert torch.isfinite(validation_loss).all()
    criterion.update()
    assert criterion.end_to_end.updates == 1


def test_incomplete_pose_output_reports_its_structure_instead_of_keyerror():
    criterion = independent_pose_criterion(make_detector("yolo26n-pose.yaml"))
    with pytest.raises(ValueError, match="no complete pose branch"):
        criterion({"one2many": {"boxes": torch.zeros(1)}}, {})
