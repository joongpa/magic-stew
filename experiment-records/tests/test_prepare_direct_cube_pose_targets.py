import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "direct_cube_pose_targets", ROOT / "scripts/prepare_direct_cube_pose_targets.py",
)
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)
RECORDS = json.loads((Path(__file__).parent / "fixtures/direct_cube_pose_metadata.json").read_text())


@pytest.mark.parametrize("record", RECORDS, ids=lambda value: Path(value["fixture_source"]).stem)
def test_real_metadata_reconstructs_camera_and_articulated_landmarks(record):
    target = converter.build_target(record)
    assert target["verification"]["source_camera_error_px"]["max"] < .001
    assert target["verification"]["full_source_projection_error_px"]["max"] < .001
    assert target["verification"]["full_source_geometry_error_cubie_units"]["max"] < 1e-4
    if record["layer_state"]["mode"] != "corner_cut":
        assert target["verification"]["target_projection_error_px"]["max"] < .001
        assert target["verification"]["articulated_geometry_error_cubie_units"]["max"] < 1e-4
    assert len(target["landmark_anchors"]) == 56
    assert len(target["virtual_core_anchors"]) == 14
    matrix = np.asarray(target["rotation_matrix"])
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-10)
    assert np.linalg.det(matrix) == pytest.approx(1)


def test_scene_face_is_remapped_after_whole_cube_scramble_and_both_sign_fields_are_used():
    # Real generator record: scene F, negative angle AND negative direction.
    first = converter.layer_angles(RECORDS[0])
    assert first[converter.FACE_ORDER.index("F")] == pytest.approx(-19.336059003007506)
    assert np.count_nonzero(first) == 1
    # The second record's scene D points toward canonical/color B, not color D.
    second = converter.layer_angles(RECORDS[1])
    assert second[converter.FACE_ORDER.index("B")] == pytest.approx(22.184835224418745)
    assert second[converter.FACE_ORDER.index("D")] == 0
    opposite = converter.layer_angles(RECORDS[3])
    assert opposite[converter.FACE_ORDER.index("R")] == pytest.approx(-24.22155563190313)
    assert opposite[converter.FACE_ORDER.index("L")] == pytest.approx(16.71786847428418)


def test_exact_projector_preserves_principal_shift_and_center_in_perspective_numerator():
    target = converter.build_target(RECORDS[0])
    points = np.asarray([[1.5, -.5, .5], [-.5, -1.5, 1.5], [0, 0, 0]])
    matrix = np.asarray(target["rotation_matrix"])
    translation = np.asarray(target["reference_camera"]["translation_in_cubie_units"])
    intrinsics = np.asarray(target["reference_camera"]["intrinsics"])
    camera = points @ matrix.T + translation
    # Independent homogeneous pinhole projection, compared with compact outputs.
    expected = (intrinsics @ camera.T).T
    expected = expected[:, :2] / expected[:, 2:]
    np.testing.assert_allclose(converter.project_target(points, target), expected, atol=1e-10)
    assert np.linalg.norm(converter.project_target(points, target, compact=True) - expected) > .01


def test_wrong_sensor_width_and_turn_sign_fail_the_observation_check():
    target = converter.build_target(RECORDS[0], sensor_width_mm=32)
    assert target["verification"]["source_camera_error_px"]["max"] > 1
    record = copy.deepcopy(RECORDS[0])
    record["layer_state"]["operations"][0]["direction"] *= -1
    target = converter.build_target(record)
    assert target["verification"]["target_projection_error_px"]["max"] > 1


@pytest.mark.parametrize("mode", ["wide_turn", "invalid"])
def test_unsupported_articulation_is_not_silently_treated_as_aligned(mode):
    record = copy.deepcopy(RECORDS[0])
    record["layer_state"]["mode"] = mode
    with pytest.raises(ValueError, match="unsupported articulation"):
        converter.build_target(record)


@pytest.mark.parametrize("record", RECORDS[5:], ids=lambda value: value["layer_state"]["operations"][0]["move"])
def test_slice_rebases_color_frame_and_uses_only_two_opposite_face_offsets(record):
    target = converter.build_target(record)
    angles = np.asarray(target["layer_angles_degrees"])
    selected = np.flatnonzero(angles)
    assert len(selected) == 2
    assert angles.sum() == pytest.approx(0)
    np.testing.assert_array_equal(converter.FACE_AXES[selected[0]], -converter.FACE_AXES[selected[1]])
    assert not np.allclose(target["color_frame_rebase_rotation"], np.eye(3))
    assert len(target["layer_operations"]) == 2
    assert target["verification"]["target_projection_error_px"]["max"] < .001


def test_corner_cut_keeps_only_larger_angle_and_measures_smaller_turn_residual():
    record = RECORDS[4]
    target = converter.build_target(record)
    angles = np.asarray(target["layer_angles_degrees"])
    expected = max(abs(op["angle_degrees"] * op["direction"]) for op in record["layer_state"]["operations"])
    assert np.count_nonzero(angles) == 1
    assert abs(angles).max() == pytest.approx(expected)
    assert len(target["layer_operations"]) == 1
    assert len(target["reference_layer_operations"]) == 2
    assert target["corner_cut_policy"] == "dominant_angle_only"
    assert target["verification"]["full_source_projection_error_px"]["max"] < .001
    assert target["verification"]["target_projection_error_px"]["max"] > .1


def test_heavily_occluded_records_can_verify_the_few_available_landmarks():
    record = copy.deepcopy(RECORDS[5])
    record["sticker_keypoints"]["landmark_keypoints"] = record["sticker_keypoints"]["landmark_keypoints"][:2]
    target = converter.build_target(record)
    assert target["verification"]["target_projection_error_px"]["count"] == 2
    assert target["verification"]["target_projection_error_px"]["max"] < .001


def test_adjacent_or_duplicate_simultaneous_layers_are_rejected():
    for move in ["F", "U"]:
        record = copy.deepcopy(RECORDS[0])
        record["layer_state"]["mode"] = "simultaneous_turn"
        operation = copy.deepcopy(record["layer_state"]["operations"][0])
        operation["move"] = move
        record["layer_state"]["operations"].append(operation)
        with pytest.raises(ValueError, match="opposite faces"):
            converter.layer_angles(record)


def test_discovery_uses_per_image_metadata_and_stratifies_selection(tmp_path):
    source = tmp_path / "source"
    for name in ["images", "pose", "metadata"]:
        (source / name).mkdir(parents=True)
    for index, record in enumerate(RECORDS):
        stem = f"{index:06d}"
        (source / "images" / f"{stem}.png").write_bytes(b"test")
        (source / "pose" / f"{stem}.txt").write_text("existing label\n")
        (source / "metadata" / f"{stem}.json").write_text(json.dumps(record))
    # A legacy aggregate file must not be read, even when it is invalid JSON.
    (source / "metadata.json").write_text("not a usable metadata source")
    (source / "images" / "missing_metadata.png").write_bytes(b"test")
    samples, excluded, inventory = converter.discover_samples([source])
    assert len(samples) == len(RECORDS)
    assert excluded == {}
    assert inventory[0]["images_without_per_image_metadata"] == 1
    selected = converter.select_samples(samples, len(RECORDS), seed=17)
    assert {sample.mode for sample in selected} == converter.SUPPORTED_MODES
    assert len({sample.key for sample in selected}) == len(RECORDS)
