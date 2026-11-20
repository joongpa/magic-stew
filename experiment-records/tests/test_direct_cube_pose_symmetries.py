import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("direct_pose_symmetries", ROOT / "scripts/verify_direct_cube_pose_symmetries.py")
symmetries = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = symmetries
SPEC.loader.exec_module(symmetries)
RECORDS = json.loads((Path(__file__).parent / "fixtures/direct_cube_pose_metadata.json").read_text())


def test_unconstrained_group_contains_exactly_24_unique_proper_rotations():
    matrices = symmetries.cube_rotations()
    assert len(matrices) == len({tuple(matrix.ravel()) for matrix in matrices}) == 24
    for matrix in matrices:
        np.testing.assert_array_equal(matrix.T @ matrix, np.eye(3))
        assert np.linalg.det(matrix) == pytest.approx(1)


@pytest.mark.parametrize("face", symmetries.geometry.FACE_ORDER)
def test_one_center_leaves_four_rotations_that_fix_its_axis(face):
    matrices = symmetries.cube_rotations([face])
    assert len({tuple(matrix.ravel()) for matrix in matrices}) == 4
    axis = symmetries.geometry.FACE_AXES[symmetries.geometry.FACE_ORDER.index(face)]
    for matrix in matrices:
        np.testing.assert_array_equal(matrix @ axis, axis)


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: Path(record["fixture_source"]).stem)
def test_all_24_candidates_preserve_pose_turns_and_keypoint_projection(record):
    reference = symmetries.geometry.build_target(record)
    snapshot = copy.deepcopy(reference)
    for matrix in symmetries.cube_rotations():
        candidate = symmetries.transform_target(reference, matrix)
        metrics = symmetries.verify_candidate(reference, candidate, matrix)
        assert max(metrics.values()) < 1e-9
        # Clockwise signs are retained when the face's outward normal is renamed.
        for source, renamed in zip(reference["layer_operations"], candidate["layer_operations"]):
            assert renamed["angle_degrees"] == source["angle_degrees"]
        np.testing.assert_allclose(
            [point["projected_pixel"] for point in candidate["matched_observations"]],
            [point["projected_pixel"] for point in reference["matched_observations"]], atol=1e-9,
        )
    assert reference == snapshot


def test_rotation_without_turn_remapping_is_detected():
    reference = symmetries.geometry.build_target(RECORDS[0])
    matrix = next(matrix for matrix in symmetries.cube_rotations() if symmetries.face_permutation(matrix)["F"] != "F")
    candidate = symmetries.transform_target(reference, matrix)
    candidate["layer_operations"] = copy.deepcopy(reference["layer_operations"])
    candidate["layer_angles_degrees"] = reference["layer_angles_degrees"][:]
    assert symmetries.verify_candidate(reference, candidate, matrix)["max_projection_change_px"] > 1


def test_retained_visibility_one_and_two_both_count_as_center_evidence():
    record = {"sticker_keypoints": {"keypoints": [
        {"source_object": "cubie.017", "visibility": 1, "visible_area_ratio": .2},
        {"source_object": "cubie.012", "visibility": 2, "visible_area_ratio": .8},
        {"source_object": "cubie.003", "visibility": 0, "visible_area_ratio": 0},
        {"source_object": "StickerlessMaterialRadialPart.cubie.001.Red", "visibility": 2},
    ]}}
    retained = symmetries.retained_center_labels(record, {"cubie.017", "cubie.012", "cubie.003"})
    assert [point["visibility"] for point in retained] == [1, 2]
