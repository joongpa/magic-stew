import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "review_turn_candidates.py"
SPEC = importlib.util.spec_from_file_location("review_turn_candidates", SCRIPT_PATH)
review_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = review_script
SPEC.loader.exec_module(review_script)


def test_draw_candidate_handles_null_scores_and_missing_heatmaps():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    record = {
        "search_pose": {
            "camera_matrix": [[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]],
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0],
            "rvec": [0.0, 0.0, 0.0],
            "tvec": [0.0, 0.0, 300.0],
            "assignment_image_points": [[20.0, 30.0]],
            "sticker_mask_centers": None,
        },
        "candidates": [
            {
                "label": "R+3",
                "layer_rotations": [
                    {"axis": 2, "side": 1, "angle_degrees": 3.0, "layer_count": 1}
                ],
                "global_mean_error": None,
                "global_robust_error": None,
            }
        ],
    }

    output = review_script.draw_candidate(frame, record, 0)

    assert output.shape == frame.shape
    assert output.dtype == frame.dtype
    assert np.any(output != frame)


def test_review_candidates_marks_static_pose_when_turn_search_was_not_tested():
    candidates = review_script.review_candidates(
        {
            "has_pose": True,
            "search_pose": {"rvec": [0.0, 0.0, 0.0]},
            "static_fit_scores": [0.1, 0.0, 0.1],
            "turn_fit_color_score": 0.7,
            "candidates": [],
        }
    )

    assert len(candidates) == 1
    assert candidates[0]["label"] == "static pose (turn search not tested)"
    assert candidates[0]["global_mean_error"] == 0.1
    assert candidates[0]["color_error"] == 0.7


def test_color_likelihood_payload_exposes_each_identity_and_observation():
    payload = review_script.color_likelihood_payload(
        {
            "turn_color_identity_labels": ["C01 U"],
            "turn_color_observation_centers": [[12.0, 34.0]],
            "turn_color_likelihoods": [[0.875]],
        }
    )

    assert payload == {
        "color_identity_labels": ["C01 U"],
        "color_observation_centers": [[12.0, 34.0]],
        "color_likelihoods": [[0.875]],
    }


def test_rotation_matrix_matches_opencv_rodrigues_convention():
    matrix, pivot = review_script.rotation_matrix_for_layer(
        {"axis": 1, "side": -1, "angle_degrees": 12.0, "layer_count": 1}
    )
    expected_vector = np.asarray([[0.0], [-np.deg2rad(12.0)], [0.0]])
    expected, _ = cv2.Rodrigues(expected_vector)

    np.testing.assert_allclose(matrix, expected)
    np.testing.assert_allclose(pivot, [0.0, -review_script.CUBE_HALF_SIZE_MM, 0.0])


def test_face_points_winding_matches_each_outward_side():
    center = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        for side in (-1, 1):
            points = review_script.face_points(center, axis, side)
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            expected = np.zeros(3, dtype=np.float64)
            expected[axis] = side
            assert float(np.dot(normal, expected)) > 0.0


def test_static_front_view_does_not_project_opposite_cubie_faces():
    pose = {
        "camera_matrix": [[200.0, 0.0, 80.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]],
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0],
        "rvec": [0.0, 0.0, 0.0],
        "tvec": [0.0, 0.0, 300.0],
    }

    assert len(review_script.projected_visible_sticker_faces(pose, [])) == 9


def test_static_layer_removal_candidate_still_marks_selected_layer_red():
    pose = {
        "camera_matrix": [[200.0, 0.0, 80.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]],
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0],
        "rvec": [0.0, 0.0, 0.0],
        "tvec": [0.0, 0.0, 300.0],
    }

    faces = review_script.projected_visible_sticker_faces(
        pose,
        [],
        {"axis": 2, "side": -1, "angle_degrees": 0.0, "layer_count": 1},
    )

    assert len(faces) == 9
    assert all(moving for _polygon, moving in faces)


def test_renderer_selection_is_found_even_when_not_top_ranked():
    turn = {"axis": 0, "side": 1, "angle_degrees": 12.0, "layer_count": 1}
    record = {
        "selected_layer_rotations": [turn],
        "candidates": [
            {"layer_rotations": []},
            {"layer_rotations": [turn]},
        ],
    }

    assert review_script.selected_candidate_index(record) == 1
    record["selected_layer_rotations"] = []
    assert review_script.selected_candidate_index(record) == 0


def test_renderer_selection_distinguishes_color_offsets_with_identical_geometry():
    record = dict(selected_layer_rotations=[], selected_committed_moves=["R"],
                  selected_color_offset_degrees=-90.0, selected_trial_id="0:-90",
                  candidates=[dict(trial_id="0:+0", layer_rotations=[], color_offset=0.0, state_moves=[]),
                              dict(trial_id="0:-90", layer_rotations=[], color_offset=-90.0, state_moves=["R"])])
    assert review_script.selected_candidate_index(record) == 1
    # Older offset logs have no explicit ID, but still carry enough evidence.
    del record["selected_trial_id"]
    assert review_script.selected_candidate_index(record) == 1
    record["selected_trial_id"] = "not-recorded"
    assert review_script.selected_candidate_index(record) is None


def test_search_debug_payload_preserves_forecasts_and_handles_legacy_logs():
    search = dict(status="searched", previous_move="R", selected_path=dict(total_score=None))
    assert review_script.search_debug_payload({"search": search}) == search
    assert review_script.search_debug_payload({}) is None
    assert review_script.search_debug_payload(None) is None


def test_projected_click_regions_identify_face_and_adjacent_layer_edges():
    pose = {
        "camera_matrix": [[200.0, 0.0, 80.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]],
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0],
        "rvec": [0.0, 0.0, 0.0],
        "tvec": [0.0, 0.0, 300.0],
    }
    regions = review_script.projected_face_regions(pose, [])

    assert len(regions) == 9
    assert all((region["axis"], region["side"]) == (2, -1) for region in regions)
    assert sorted(len(region["edges"]) for region in regions) == [0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert {edge["axis"] for region in regions for edge in region["edges"]} == {0, 1}


def test_frame_navigation_decodes_by_index_without_random_seeking(monkeypatch, tmp_path):
    captures = []

    class FakeCapture:
        def __init__(self, _path):
            self.index = 0
            captures.append(self)

        def isOpened(self):
            return True

        def read(self):
            if self.index >= 6:
                return False, None
            frame = np.full((2, 2, 3), self.index, dtype=np.uint8)
            self.index += 1
            return True, frame

        def set(self, *_args):
            raise AssertionError("random frame seeking may not follow decoded frame indices")

        def release(self):
            pass

    monkeypatch.setattr(review_script.cv2, "VideoCapture", FakeCapture)
    state = review_script.ReviewState(tmp_path / "video.mov", [], tmp_path / "labels.json")

    for index in (0, 1, 4, 4, 2, 5):
        assert np.all(state.frame(index) == index)
    assert len(captures) == 1
