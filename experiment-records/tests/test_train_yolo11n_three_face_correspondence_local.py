from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_yolo11n_three_face_correspondence_local.py"
MANIFEST = ROOT / "data/independent_keypoints_dataset_v3_three_face/correspondence_manifest.json"


def load_training_module():
    spec = importlib.util.spec_from_file_location("three_face_correspondence_training", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training script: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_three_face_symmetry_manifest_derives_three_bijections():
    training = load_training_module()
    permutations_by_mask = training.load_symmetry_permutations(MANIFEST)

    assert len(permutations_by_mask) == 8
    assert len(permutations_by_mask[0]) == 15
    assert len(permutations_by_mask[1]) == 4
    assert len(permutations_by_mask[2]) == 4
    assert len(permutations_by_mask[3]) == 2
    assert len(permutations_by_mask[7]) == 3
    for permutations in permutations_by_mask.values():
        assert permutations[0] == tuple(range(37))
        assert len({tuple(permutation) for permutation in permutations}) == len(permutations)
        assert all(set(permutation) == set(range(37)) for permutation in permutations)


def test_shared_UF_seam_does_not_make_F_look_like_two_visible_faces():
    training = load_training_module()
    _permutations, _face_masks, exclusive_face_masks = training.load_symmetry_definition(MANIFEST)
    target = torch.zeros((1, 37, 3), dtype=torch.float32)
    target[0, 12:28, 2] = 2.0
    face_mask = training._visible_face_masks(
        target,
        torch.as_tensor(exclusive_face_masks),
    )

    assert int(face_mask.item()) == training.FACE_BITS["F"]


def test_symmetry_loss_selects_matching_permutation():
    training = load_training_module()
    permutations = torch.as_tensor(training.load_symmetry_permutations(MANIFEST)[7])
    target = torch.zeros((1, 37, 3), dtype=torch.float32)
    target[0, :, 0] = torch.arange(37, dtype=torch.float32) / 37.0
    target[0, :, 1] = 1.0 - target[0, :, 0]
    target[0, :, 2] = 2.0
    target[0, 0, 2] = 0.0

    chosen = 1
    candidate = target[:, permutations[chosen], :]
    prediction = candidate.clone()
    prediction[:, :, 2] = torch.where(
        candidate[:, :, 2] == 0,
        torch.tensor(-20.0),
        torch.tensor(20.0),
    )
    prediction.requires_grad_()
    location_loss, visibility_loss, winner = training._symmetry_candidate_losses(
        prediction,
        target,
        permutations,
        pose_weight=1.0,
        kobj_weight=1.0,
    )

    assert int(winner.item()) == chosen
    assert float(location_loss.detach()) < 1e-6
    assert float(visibility_loss.detach()) < 1e-6
    (location_loss + visibility_loss).backward()
    assert prediction.grad is not None


def test_prepared_data_yaml_declares_37_point_head(tmp_path):
    training = load_training_module()
    empty_negatives = tmp_path / "empty-negatives"
    empty_negatives.mkdir()
    dataset_dir = tmp_path / "prepared"
    data_yaml = training.write_prepared_dataset(
        source_dir=ROOT / "data/independent_keypoints_dataset_v3_three_face",
        dataset_dir=dataset_dir,
        val_fraction=0.01,
        seed=17,
        extra_negative_dir=empty_negatives,
        force=False,
    )

    data_text = data_yaml.read_text(encoding="utf-8")
    assert "kpt_shape: [37, 3]" in data_text
    assert "kpt_symmetry_file: three_face_symmetries.json" in data_text
    symmetry_payload = json.loads((dataset_dir / "three_face_symmetries.json").read_text())
    assert len(symmetry_payload["permutations_by_face_mask"]["1"]) == 4
    assert len(symmetry_payload["permutations_by_face_mask"]["3"]) == 2
    assert len(symmetry_payload["permutations_by_face_mask"]["7"]) == 3
