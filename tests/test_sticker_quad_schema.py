import numpy as np
import pytest

from live_recognition.sticker_quad_schema import (
    CLASS_NAMES,
    class_id,
    surface_template,
    transformed_cubie_face,
    yolo_row,
)
from scripts.relabel_sticker_quads import corner_targets, relabel_target


def test_negative_image_without_pose_is_retained_with_no_invented_objects():
    objects, counts = relabel_target({"sticker_observations": [], "rotation_matrix": None})
    assert objects == []
    assert counts == {"retained_stickers": 0, "unmatched_stickers": 0}


def test_all_physical_faces_have_correct_semantic_corner_roles():
    quads, owners, normals, roles, kinds = surface_template()
    assert quads.shape == (54, 4, 3)
    assert len(CLASS_NAMES) == 18
    for q, owner, normal, r, kind in zip(quads, owners, normals, roles, kinds):
        matrix = np.eye(4)
        matrix[:3, 3] = owner
        actual, actual_roles, actual_kind = transformed_cubie_face(matrix, owner, np.eye(3), normal)
        np.testing.assert_allclose(actual, q)
        np.testing.assert_array_equal(actual_roles, r)
        assert actual_kind == kind
        assert sorted(r) == sorted(
            {"corner": (0, 1, 2, 1), "edge": (1, 1, 2, 2), "center": (2, 2, 2, 2)}[kind]
        )


def test_grid_points_follow_exact_cubie_transform_and_keep_roles():
    angle = 0.6
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [3, 4, 5]
    q, roles, kind = transformed_cubie_face(matrix, (1, 1, 1), np.eye(3), rotation @ [0, 0, 1])
    q0, roles0, kind0 = transformed_cubie_face(np.eye(4), (1, 1, 1), np.eye(3), [0, 0, 1])
    np.testing.assert_allclose(q, q0 @ rotation.T + [3, 4, 5])
    np.testing.assert_array_equal(roles, roles0)
    assert kind == kind0 == "corner"


def test_yolo_quad_clips_box_and_zeros_unlabeled_points():
    row = np.asarray(
        yolo_row(
            class_id("blue", "edge"),
            [[-5, 5], [20, 5], [20, 30], [-5, 30]],
            [0, 2, 1, 0],
            [100, 100],
        ).split(),
        float,
    )
    assert len(row) == 17
    np.testing.assert_allclose(row[1:5], [0.1, 0.175, 0.2, 0.25])
    np.testing.assert_allclose(
        row[5:].reshape(4, 3), [[0, 0, 0], [0.2, 0.05, 2], [0.2, 0.3, 1], [0, 0, 0]]
    )


def test_relabel_uses_per_file_exact_operations_not_dominant_corner_cut():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "data/direct_cube_pose_dataset/targets"
    path = root / "keypoints_only_dataset__000087.json"
    if not path.is_file():
        pytest.skip("Local verification dataset is not installed")
    target = json.loads(path.read_text())
    assert target["layer_state_mode"] == "corner_cut"
    assert len(target["reference_layer_operations"]) == 2
    objects, _ = relabel_target(target)
    assert objects
    for obj in objects:
        for c in obj["corners"]:
            if c["matched_landmark"] is not None:
                assert c["matched_landmark"]["visibility"] == c["visibility"]
                assert (
                    np.linalg.norm(
                        np.asarray(c["pixel_point"]) - c["matched_landmark"]["observed_pixel"]
                    )
                    < 2
                )
    missing = {**target, "matched_observations": []}
    objects, _ = relabel_target(missing)
    assert objects
    for o in objects:
        for c in o["corners"]:
            in_frame = np.all(np.asarray(c["pixel_point"]) >= 0) and np.all(
                np.asarray(c["pixel_point"]) < target["image_size"]
            )
            assert c["visibility"] == int(in_frame)


def test_recovered_seam_corners_are_supervised_while_saved_visibility_is_preserved():
    pixels = np.asarray([[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]])
    observations = [
        {"observed_pixel": [10.0, 10.0], "class_id": 0, "visibility": 2},
        {"observed_pixel": [30.0, 10.0], "class_id": 1, "visibility": 1},
    ]
    corners = corner_targets(pixels, [0, 1, 2, 1], observations, [100, 100], 0.1)
    assert [c["visibility"] for c in corners] == [2, 1, 1, 1]
    assert [c["visibility_evidence"] for c in corners] == [
        "saved_landmark",
        "saved_landmark",
        "amodal_reconstruction",
        "amodal_reconstruction",
    ]
    row = np.asarray(
        yolo_row(0, pixels, [c["visibility"] for c in corners], [100, 100]).split(), float
    )
    np.testing.assert_allclose(row[5:].reshape(4, 3)[:, :2], pixels / 100)


def test_out_of_frame_and_explicitly_ignored_saved_corner_stay_ignored():
    pixels = np.asarray([[-1.0, 10.0], [100.0, 10.0], [30.0, 30.0], [10.0, 30.0]])
    observations = [{"observed_pixel": [30.0, 30.0], "class_id": 2, "visibility": 0}]
    corners = corner_targets(pixels, [0, 1, 2, 1], observations, [100, 100], 0.1)
    assert [c["visibility"] for c in corners] == [0, 0, 0, 1]


def test_no_previously_excluded_stickers_are_added():
    # An angle/coverage-filtered object has visibility 0. Its geometry must
    # never qualify it for conversion on its own.
    objects, counts = relabel_target(
        {
            "sticker_observations": [
                {"visibility": 0, "visible_area_ratio": 0.05, "angle_filtered": True}
            ],
            "rotation_matrix": None,
        }
    )
    assert objects == [] and counts["retained_stickers"] == 0


def test_slice_turn_recovers_seam_points_without_old_landmark_support():
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "data/direct_cube_pose_dataset/targets/keypoints_only_dataset_part3__003897.json"
    )
    if not path.is_file():
        pytest.skip("Local verification dataset is not installed")
    target = json.loads(path.read_text())
    objects, _ = relabel_target(target)
    recovered = [
        c
        for o in objects
        for c in o["corners"]
        if c["visibility_evidence"] == "amodal_reconstruction"
    ]
    assert recovered
    assert all(c["visibility"] == 1 for c in recovered)
    # These are new coordinates, not aliases of a saved point relabeled
    # into another keypoint slot.
    old = np.asarray([o["observed_pixel"] for o in target["matched_observations"]])
    distance = np.linalg.norm(
        np.asarray([c["pixel_point"] for c in recovered])[:, None] - old, axis=-1
    )
    assert (distance.min(-1) > 1).any()
