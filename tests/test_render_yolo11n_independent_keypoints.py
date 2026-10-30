import importlib.util
import json
from dataclasses import dataclass, replace
from itertools import permutations
from pathlib import Path
from queue import SimpleQueue
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "render_yolo11n_independent_keypoints.py"
)
SPEC = importlib.util.spec_from_file_location(
    "render_yolo11n_independent_keypoints",
    SCRIPT_PATH,
)
render_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = render_script
SPEC.loader.exec_module(render_script)


def test_final_video_parser_option_is_not_color_specific():
    args = render_script.build_parser().parse_args(
        [
            "--source",
            "input.mp4",
            "--save-final-video",
            "runs/final.mp4",
        ]
    )

    assert args.save_final_video == Path("runs/final.mp4")


def test_legacy_colored_video_parser_option_remains_an_alias():
    args = render_script.build_parser().parse_args(
        [
            "--source",
            "input.mp4",
            "--save-colored-video",
            "runs/final.mp4",
        ]
    )

    assert args.save_final_video == Path("runs/final.mp4")


def test_layer_removal_is_the_default_move_recovery_mode():
    args = render_script.build_parser().parse_args(["--source", "input.mp4"])

    assert args.pnp_turn_recovery_mode == "layer-removal"


def test_initial_scramble_parser_accepts_standard_notation_and_alias():
    args = render_script.build_parser().parse_args(
        ["--source", "input.mp4", "--known-scramble", "R U2 F' Rw2 r'"]
    )

    assert args.initial_scramble == ("R", "U2", "F'", "Rw2", "r'")


def test_scramble_recognition_can_be_disabled():
    args = render_script.build_parser().parse_args(
        ["--source", "input.mp4", "--no-recognize-scramble"]
    )

    assert args.recognize_scramble is False

    enabled = render_script.build_parser().parse_args(
        ["--source", "input.mp4", "--recognize-scramble"]
    )
    assert enabled.recognize_scramble is True


def test_classified_colors_are_reported_as_a_solved_to_state_scramble():
    state = render_script.CubeState.from_scramble(("R", "U2", "F'"))
    trace = render_script.StickerColorTrace.create()
    color_names = {
        identity.label: state.color_at(
            (identity.cubie_index, identity.axis, identity.side)
        )
        for identity in trace.identities
    }
    seen_facelets = []

    def fake_solve(facelets: str) -> str:
        seen_facelets.append(facelets)
        return "R U2 F'"

    assert render_script.scramble_from_classified_sticker_colors(
        trace,
        color_names,
        solve=fake_solve,
    ) == "F U2 R'"
    assert len(seen_facelets) == 1


def test_classified_colors_do_not_report_a_scramble_for_an_incomplete_state():
    trace = render_script.StickerColorTrace.create()
    color_names = {trace.identities[0].label: "white"}

    assert render_script.scramble_from_classified_sticker_colors(trace, color_names) is None


def test_single_layer_removal_includes_its_complete_outer_face():
    points = render_script.CUBE_LATTICE_OBJECT_POINTS_MM

    removed = render_script.layer_removal_mask_for_assigned_points(
        points,
        axis=2,
        side=-1,
        layer_count=1,
    )

    assert int(removed.sum()) == 16
    for point_index, coord in enumerate(render_script.CUBE_LATTICE_COORDS):
        assert bool(removed[point_index]) == (coord[2] == 0)


def test_double_layer_removal_includes_the_inner_seam_plane():
    points = render_script.CUBE_LATTICE_OBJECT_POINTS_MM

    single = render_script.layer_removal_mask_for_assigned_points(
        points,
        axis=2,
        side=1,
        layer_count=1,
    )
    double = render_script.layer_removal_mask_for_assigned_points(
        points,
        axis=2,
        side=1,
        layer_count=2,
    )

    assert int(single.sum()) == 16
    assert int(double.sum()) == 28
    assert np.all(double[single])
    assert np.count_nonzero(double & ~single) == 12
    assert not np.any(single[
        [coord[2] == 2 for coord in render_script.CUBE_LATTICE_COORDS]
    ])


def test_layer_removal_selection_colors_selected_layer_without_rotation():
    selected = render_script.CubeLayerRotation(
        axis=2,
        side=1,
        angle_degrees=0.0,
        layer_count=2,
    )
    candidate = SimpleNamespace(
        layer_rotations=(),
        turn_selected_layer=selected,
    )

    assert render_script.projected_cube_layer_color(candidate, True, (1, 2, 3)) == (
        0,
        0,
        255,
    )
    assert render_script.projected_cube_layer_color(candidate, False, (1, 2, 3)) == (
        0,
        0,
        0,
    )


def test_turn_candidate_log_appends_one_complete_record_per_frame(tmp_path):
    path = tmp_path / "turn_candidate_scores.jsonl"
    candidate = SimpleNamespace(
        turn_candidate_scores=(("R+3", 0.12, 7, 0.08, 0.04),),
        turn_candidate_diagnostics=(
            {
                "label": "R+3",
                "global_mean_error": 0.12,
                "global_robust_error": 0.16,
                "layer_rotations": [],
            },
        ),
        turn_fit_evaluated=True,
        turn_fit_unexplained_keypoints=2,
        turn_fit_sticker_score=0.03,
        turn_fit_static_scores=(0.04, 0.02, 0.06),
        turn_fit_articulated_scores=(0.03, 0.01, 0.04),
        turn_fit_search_rvec=np.zeros((3, 1), dtype=np.float64),
        turn_fit_search_tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
        camera_matrix=np.eye(3, dtype=np.float64),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
        assignment_image_points=np.asarray(((1.0, 2.0),), dtype=np.float32),
        assignment_observed_class_ids=np.asarray((2,), dtype=np.int32),
        assignment_fixed_mask=np.asarray((True,)),
        sticker_mask_centers=(np.asarray((3.0, 4.0)),),
        sticker_mask_center_uncertainties=(1.5,),
        layer_rotations=(),
    )

    render_script.append_turn_candidate_log(path, candidate, 12)
    render_script.append_turn_candidate_log(path, None, 13)

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["frame_index"] for record in records] == [12, 13]
    assert records[0]["candidate_count"] == 1
    assert records[0]["candidates"][0]["global_robust_error"] == 0.16
    assert records[0]["search_pose"]["tvec"] == [[0.0], [0.0], [200.0]]
    assert records[0]["search_pose"]["assignment_image_points"] == [[1.0, 2.0]]
    assert records[1]["has_pose"] is False
    assert records[1]["candidate_count"] == 0


def test_turn_center_scoring_rewards_coverage_and_discounts_uncertain_false_detection(monkeypatch):
    @dataclass
    class CenterCandidate:
        sticker_mask_centers: tuple
        sticker_mask_center_uncertainties: tuple
        layer_rotations: tuple = ()

    monkeypatch.setattr(render_script, "projected_cube_diameter", lambda *_args: 100.0)
    monkeypatch.setattr(
        render_script,
        "projected_sticker_faces",
        lambda *_args: [(None, 0, np.tile([0.0, 0.0], (4, 1)))],
    )
    pose = np.zeros((3, 1))

    def score(centers, uncertainties):
        candidate = CenterCandidate(tuple(centers), tuple(uncertainties))
        return render_script.sticker_mask_center_turn_fit_metrics(candidate, pose, pose, (), 5.0)

    close, _, close_count = score(((0.0, 0.0),), (1.0,))
    within_tolerance, _, _ = score(((2.0, 0.0),), (1.0,))
    reliable_false, _, reliable_count = score(((0.0, 0.0), (30.0, 0.0)), (1.0, 1.0))
    uncertain_false, _, uncertain_count = score(((0.0, 0.0), (30.0, 0.0)), (1.0, 4.0))
    assert close == within_tolerance == 0.0
    assert close_count == reliable_count == uncertain_count == 1
    assert 0.0 < uncertain_false < reliable_false
    assert reliable_false == pytest.approx(0.025)


def test_turn_center_scoring_uses_optimal_one_to_one_assignment(monkeypatch):
    @dataclass
    class CenterCandidate:
        sticker_mask_centers: tuple
        sticker_mask_center_uncertainties: tuple
        layer_rotations: tuple = ()

    monkeypatch.setattr(render_script, "projected_cube_diameter", lambda *_args: 100.0)
    monkeypatch.setattr(
        render_script,
        "projected_sticker_faces",
        lambda *_args: [
            (None, 0, np.tile([0.0, 0.0], (4, 1))),
            (None, 0, np.tile([6.0, 0.0], (4, 1))),
        ],
    )
    candidate = CenterCandidate(((2.0, 0.0), (0.0, 0.0)), (1.0, 1.0))
    pose = np.zeros((3, 1))

    ranking, _, match_count = render_script.sticker_mask_center_turn_fit_metrics(
        candidate, pose, pose, (), 5.0
    )

    assert match_count == 2
    assert ranking < 0.025


def test_layer_removal_center_score_keeps_only_centers_outside_removed_layer(monkeypatch):
    top_identity = render_script.StickerIdentity(
        label="U-center",
        category="center",
        cubie_label="U",
        face="U",
        cubie_index=(1, 1, 2),
        axis=2,
        side=1,
    )
    bottom_identity = render_script.StickerIdentity(
        label="D-center",
        category="center",
        cubie_label="D",
        face="D",
        cubie_index=(1, 1, 0),
        axis=2,
        side=-1,
    )

    def polygon(center):
        x, y = center
        return np.asarray(
            ((x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1)),
            dtype=np.float64,
        )

    candidate = SimpleNamespace(
        sticker_mask_centers=(
            np.asarray((0.0, 0.0)),
            np.asarray((20.0, 0.0)),
        ),
        sticker_mask_center_uncertainties=(1.0, 1.0),
        layer_rotations=(),
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
        camera_matrix=np.eye(3, dtype=np.float64),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
    )
    monkeypatch.setattr(render_script, "projected_cube_diameter", lambda *_args: 100.0)
    monkeypatch.setattr(
        render_script,
        "projected_sticker_faces",
        lambda *_args: [
            (top_identity, 1.0, polygon((0.0, 0.0))),
            (bottom_identity, 1.0, polygon((20.0, 0.0))),
        ],
    )

    mean, robust, match_count = render_script.sticker_mask_center_remainder_fit_metrics(
        candidate,
        axis=2,
        side=1,
        layer_count=1,
        max_error=5.0,
    )

    assert mean == pytest.approx(0.0)
    assert robust == pytest.approx(0.0)
    assert match_count == 1


def test_quicktime_frame_counter_adds_timecode_track_without_reencoding(
    tmp_path,
    monkeypatch,
):
    output_path = tmp_path / "render.mp4"
    output_path.write_bytes(b"opencv movie")
    commands = []

    monkeypatch.setattr(render_script.shutil, "which", lambda name: "/opt/bin/ffmpeg")

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        Path(command[-1]).write_bytes(b"movie with tmcd track")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(render_script.subprocess, "run", fake_run)

    assert render_script.add_quicktime_frame_counter(output_path)
    assert output_path.read_bytes() == b"movie with tmcd track"
    assert len(commands) == 1
    command, kwargs = commands[0]
    assert command[command.index("-c") + 1] == "copy"
    assert command[command.index("-timecode") + 1] == "00:00:00:00"
    assert kwargs["check"] is False


def test_quicktime_frame_counter_skips_non_quicktime_video_extensions(
    tmp_path,
    monkeypatch,
):
    output_path = tmp_path / "render.avi"
    output_path.write_bytes(b"opencv movie")
    monkeypatch.setattr(
        render_script.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("ffmpeg should not run for AVI"),
    )

    assert not render_script.add_quicktime_frame_counter(output_path)
    assert output_path.read_bytes() == b"opencv movie"


def test_turn_search_defaults_to_all_six_layer_sides():
    args = render_script.build_parser().parse_args(["--source", "input.mp4"])

    assert args.pnp_turn_face_candidates == 6
    assert render_script.PNP_LAYER_MISALIGNMENT_MIN_DEGREES == -45.0
    assert render_script.PNP_LAYER_MISALIGNMENT_MAX_DEGREES == 45.0


def test_turn_overlay_colors_single_and_double_layers_by_role():
    candidate = SimpleNamespace(
        layer_rotations=(
            render_script.CubeLayerRotation(axis=2, side=1, angle_degrees=30.0),
        )
    )
    assert render_script.projected_cube_layer_color(candidate, False, (1, 2, 3)) == (
        0,
        0,
        255,
    )
    assert render_script.projected_cube_layer_color(candidate, True, (1, 2, 3)) == (
        0,
        0,
        0,
    )

    candidate.layer_rotations = (
        render_script.CubeLayerRotation(
            axis=2,
            side=1,
            angle_degrees=30.0,
            layer_count=2,
        ),
    )
    assert render_script.projected_cube_layer_color(candidate, False, (1, 2, 3)) == (
        0,
        0,
        0,
    )
    assert render_script.projected_cube_layer_color(candidate, True, (1, 2, 3)) == (
        0,
        0,
        255,
    )


def test_combined_headless_progress_keeps_neural_and_pose_counts_together(capsys):
    progress = render_script.CombinedHeadlessProgress(("neural pass", "pose pass"))

    progress.update("neural pass", 3, 10)
    progress.update("pose pass", 2, 10)
    progress.finish()

    output = capsys.readouterr().out
    assert "neural pass: 3/10" in output
    assert "pose pass: 2/10" in output


def test_async_pose_consumer_only_selects_candidates(monkeypatch):
    records = SimpleQueue()
    record = SimpleNamespace(frame="frame", frame_index=4)
    records.put(record)
    records.put(None)
    calls = []

    monkeypatch.setattr(
        render_script,
        "make_pose_motion_filter",
        lambda _args: None,
    )

    def fake_render_frame(*_args, **kwargs):
        calls.append(kwargs)
        return None, "cached", 0

    monkeypatch.setattr(render_script, "render_frame", fake_render_frame)
    render_script.consume_neural_cache_for_pose(
        records,
        None,
        None,
        None,
        SimpleNamespace(),
        None,
        None,
        None,
    )

    assert len(calls) == 1
    assert calls[0]["neural_data"] is record
    assert calls[0]["candidate_only"] is True


def test_cached_final_replay_draws_all_observed_keypoints(monkeypatch):
    observed = {}

    def fake_draw_pose(image, frame_points, *args, **kwargs):
        observed["frame_points"] = frame_points.copy()
        return image, len(frame_points)

    monkeypatch.setattr(render_script, "draw_pose", fake_draw_pose)
    monkeypatch.setattr(render_script, "draw_context", lambda image, *args: image)

    neural_data = render_script.NeuralFrameData(
        frame_index=0,
        frame=np.zeros((32, 32, 3), dtype=np.uint8),
        roi_xyxy=(4, 8, 20, 24),
        roi_source="tracker",
        sticker_masks=None,
        local_points=np.asarray([[0.0, 0.0], [8.0, 12.0]], dtype=np.float32),
        point_confidences=np.asarray([0.9, 0.8], dtype=np.float32),
        class_ids=np.asarray([0, 1], dtype=np.int32),
        box_confidences=np.asarray([0.9, 0.8], dtype=np.float32),
        pose_ready=True,
    )
    args = SimpleNamespace(
        view_size=16,
        show_sticker_mask=False,
        show_status=False,
        show_pnp=False,
        keypoint_conf=0.25,
        pose_class_conf={},
        show_labels=False,
        show_confidences=False,
    )

    render_script.render_frame(
        neural_data.frame,
        None,
        None,
        None,
        args,
        neural_data=neural_data,
        cached_final_only=True,
    )

    np.testing.assert_allclose(
        observed["frame_points"],
        np.asarray([[4.0, 8.0], [12.0, 20.0]], dtype=np.float32),
    )


def test_draw_sticker_boundaries_uses_retained_segmentation_contours():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    candidate = SimpleNamespace(
        sticker_contours=(
            np.asarray(
                [[4.0, 4.0], [20.0, 4.0], [20.0, 20.0], [4.0, 20.0]],
                dtype=np.float32,
            ),
        )
    )

    rendered = render_script.draw_sticker_boundaries(image, candidate)

    assert np.count_nonzero(rendered) > 0


def test_reprojection_error_is_drawn_on_frame_without_console_output():
    image = np.zeros((80, 240, 3), dtype=np.uint8)
    candidate = SimpleNamespace(
        overall_reprojection_error=7.25,
        reprojection_error=1.5,
    )

    rendered = render_script.draw_reprojection_error_overlay(image, candidate)

    assert np.count_nonzero(rendered) > 0


def test_pose_reprojection_fit_metric_uses_normalized_median_and_p90(monkeypatch):
    monkeypatch.setattr(
        render_script,
        "projected_cube_diameter",
        lambda *_args, **_kwargs: 100.0,
    )
    candidate = SimpleNamespace(
        camera_matrix=np.eye(3),
        assignment_image_points=np.asarray(
            [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
        ),
        assignment_projected_points=np.zeros((4, 2), dtype=np.float64),
    )

    metric = render_script.pose_reprojection_fit_metric(candidate)

    expected = (2.5 + 0.5 * np.percentile([1.0, 2.0, 3.0, 4.0], 90)) / 100.0
    assert metric == pytest.approx(expected)


def test_sticker_mask_centroids_are_rendered_independently_of_mask_fill():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    sticker_mask = np.zeros((32, 32), dtype=np.uint8)
    sticker_mask[4:28, 4:28] = 1
    predicted = (
        [sticker_mask],
        None,
        np.asarray([0], dtype=np.int32),
        None,
    )
    rendered = render_script.draw_sticker_mask_centroids(
        image,
        (0, 0, 32, 32),
        predicted,
    )

    assert np.all(
        rendered[16, 16] == render_script.STICKER_MASK_CENTROID_COLOR_BGR
    )

    without_ids = render_script.draw_sticker_mask_centroids(
        image,
        (0, 0, 32, 32),
        predicted,
        show_sticker_identities=False,
    )
    with_ids = render_script.draw_sticker_mask_centroids(
        image,
        (0, 0, 32, 32),
        predicted,
        show_sticker_identities=True,
    )
    assert np.count_nonzero(with_ids[6:14, 22:32]) > np.count_nonzero(
        without_ids[6:14, 22:32]
    )


def test_sticker_mask_heatmap_is_transparent_and_redder_at_mask_peak():
    image = np.full((32, 32, 3), 10, dtype=np.uint8)
    sticker_mask = np.zeros((32, 32), dtype=bool)
    sticker_mask[8:24, 8:24] = True
    predicted = (
        [sticker_mask],
        None,
        np.asarray([0], dtype=np.int32),
        None,
    )

    rendered = render_script.draw_sticker_mask_heatmap(
        image,
        (0, 0, 32, 32),
        predicted,
        show_sticker_identities=False,
    )

    np.testing.assert_array_equal(rendered[0, 0], image[0, 0])
    assert rendered[16, 16, 2] > rendered[16, 8, 2] > image[16, 8, 2]


def test_sticker_mask_heatmap_can_hide_centroid_markers():
    image = np.full((32, 32, 3), 10, dtype=np.uint8)
    sticker_mask = np.zeros((32, 32), dtype=bool)
    sticker_mask[8:24, 8:24] = True
    predicted = (
        [sticker_mask],
        None,
        np.asarray([0], dtype=np.int32),
        None,
    )

    with_centroid = render_script.draw_sticker_mask_heatmap(
        image,
        (0, 0, 32, 32),
        predicted,
    )
    without_centroid = render_script.draw_sticker_mask_heatmap(
        image,
        (0, 0, 32, 32),
        predicted,
        show_sticker_centroids=False,
    )

    assert np.any(with_centroid != without_centroid)


def test_sticker_mask_containment_preserves_neighbors_and_first_duplicate():
    large = np.zeros((30, 40), dtype=bool)
    large[2:12, 2:14] = True
    contained = np.zeros_like(large)
    contained[3:8, 3:8] = True
    neighbor = np.zeros_like(large)
    neighbor[2:12, 13:23] = True
    duplicate = large.copy()
    empty = np.zeros_like(large)

    keep = render_script.retained_sticker_mask_indices(
        np.asarray([contained, neighbor, large, duplicate, empty])
    )

    assert keep == [1, 2]


def test_sticker_mask_containment_uses_fraction_of_smaller_mask():
    large = np.zeros((20, 30), dtype=bool)
    large[0:10, 0:12] = True
    mostly_inside = np.zeros_like(large)
    mostly_inside[0:10, 3:13] = True  # 90 of 100 pixels overlap.
    partly_inside = np.zeros_like(large)
    partly_inside[0:10, 4:14] = True  # 80 of 100 pixels overlap.

    assert render_script.retained_sticker_mask_indices(np.asarray([mostly_inside, large])) == [1]
    assert render_script.retained_sticker_mask_indices(np.asarray([partly_inside, large])) == [0, 1]


def test_sticker_mask_containment_uses_union_of_other_masks():
    target = np.zeros((30, 30), dtype=bool)
    target[10:20, 10:20] = True

    first_cover = np.zeros_like(target)
    first_cover[10:20, 10:15] = True  # 50% of the target.
    first_cover[0:10, 0:10] = True  # Make this a larger retained mask.

    second_cover = np.zeros_like(target)
    second_cover[10:20, 16:20] = True  # 40% of the target.
    second_cover[0:10, 20:30] = True  # Make this a larger retained mask.

    # Neither cover reaches the threshold alone, but their union covers 90%
    # of the target, so the target is filtered out.
    assert render_script.retained_sticker_mask_indices(
        np.asarray([target, first_cover, second_cover])
    ) == [1, 2]


def test_sticker_mask_filter_preserves_class_ids_and_separate_cube_mask():
    small = np.zeros((16, 16), dtype=np.float32)
    small[4:7, 4:7] = 1
    large = np.zeros_like(small)
    large[2:10, 2:10] = 1
    cube = np.ones_like(small)
    result = SimpleNamespace(
        masks=SimpleNamespace(data=np.asarray([small, cube, large])),
        boxes=SimpleNamespace(cls=np.asarray([0, 2, 1])),
        names={0: "red", 1: "blue", 2: "cube"},
    )

    masks, _names, classes, cube_mask = render_script.sticker_masks_from_result(result)

    assert len(masks) == 1
    np.testing.assert_array_equal(masks[0], large.astype(bool))
    np.testing.assert_array_equal(classes, [1])
    np.testing.assert_array_equal(cube_mask, cube.astype(bool))


def test_move_search_runs_only_after_static_pose_selection(monkeypatch):
    candidate = render_script.StickerPnPCandidate(
        sticker_type="corner",
        sticker_average_confidence=0.9,
        image_points=np.zeros((4, 2), dtype=np.float32),
        object_points=np.zeros((4, 3), dtype=np.float32),
        projected_sticker_points=np.zeros((4, 2), dtype=np.float32),
        camera_matrix=np.eye(3, dtype=np.float64),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
        reprojection_error=0.0,
        assignment_image_points=np.zeros((4, 2), dtype=np.float32),
        assignment_projected_points=np.zeros((4, 2), dtype=np.float32),
        assignment_object_points=np.zeros((4, 3), dtype=np.float32),
    )
    record = render_script.NeuralFrameData(
        frame_index=0,
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        roi_xyxy=None,
        roi_source="none",
        sticker_masks=None,
        local_points=np.empty((0, 2), dtype=np.float32),
        point_confidences=np.empty((0,), dtype=np.float32),
        class_ids=np.empty((0,), dtype=np.int32),
        box_confidences=np.empty((0,), dtype=np.float32),
        pose_candidate=candidate,
        pose_candidates=(candidate,),
        pose_ready=True,
    )
    calls = []

    def fake_refine(value, **kwargs):
        calls.append(kwargs["recover_layer_turns"])
        return value

    monkeypatch.setattr(
        render_script,
        "solve_pnp_from_low_error_assignments",
        fake_refine,
    )

    def fail_color_work(*_args, **_kwargs):
        raise AssertionError("geometry-only pose recovery must not classify colors")

    monkeypatch.setattr(
        render_script,
        "build_global_color_profiles",
        fail_color_work,
    )
    monkeypatch.setattr(
        render_script,
        "_score_symmetry_observations",
        fail_color_work,
    )

    render_script.recover_cached_pose_symmetries(
        [record],
        render_script.StickerColorTrace.create(),
        refine_pose=True,
        recover_orientation=False,
        recognize_moves=True,
    )

    assert calls == [False, True]


def test_sticker_type_patterns_and_three_point_fallbacks():
    classify = render_script.sticker_type_from_keypoint_classes

    assert classify([0, 1, 1, 2]) == "corner"
    assert classify([1, 1, 2, 2]) == "edge"
    assert classify([2, 2, 2, 2]) == "center"
    assert classify([0, 1, 2]) == "corner"
    assert classify([1, 1, 2]) == "edge"
    assert classify([1, 2, 2]) == "edge"
    assert classify([2, 2, 2]) == "center"
    assert classify([0, 2, 2]) == "unknown"


def test_cube_symmetries_are_the_24_proper_rotations():
    matrices = render_script.CUBE_SYMMETRY_MATRICES

    assert len(matrices) == 24
    assert len({tuple(matrix.reshape(-1)) for matrix in matrices}) == 24
    assert all(np.isclose(np.linalg.det(matrix), 1.0) for matrix in matrices)
    np.testing.assert_array_equal(matrices[0], np.eye(3))


def test_symmetry_transformation_preserves_projected_cube_geometry():
    camera_matrix = render_script.default_camera_matrix((256, 256), 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    symmetry = render_script.CUBE_SYMMETRY_MATRICES[7]
    points = render_script.CUBE_LATTICE_OBJECT_POINTS_MM
    transformed_points = render_script.transform_object_points_by_symmetry(
        points,
        symmetry,
    )
    transformed_rvec = render_script.transform_rvec_by_symmetry(rvec, symmetry)
    original_projection, _ = cv2.projectPoints(
        points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    transformed_projection, _ = cv2.projectPoints(
        transformed_points,
        transformed_rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    np.testing.assert_allclose(
        transformed_projection,
        original_projection,
        atol=1e-4,
    )


def test_symmetry_tracker_selects_equivalent_pose_near_previous_frame():
    object_points = render_script.canonical_sticker_object_points("corner")
    camera_matrix = render_script.default_camera_matrix((256, 256), 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2, 3),
        sticker_type="corner",
        average_confidence=0.95,
    )
    candidate = render_script.solve_sticker_pnp(
        image_points.reshape(-1, 2),
        np.asarray([0, 1, 2, 1], dtype=np.int32),
        match,
        (256, 256),
        1.2,
    )
    assert candidate is not None
    target_rvec = render_script.transform_rvec_by_symmetry(
        candidate.rvec,
        render_script.CUBE_SYMMETRY_MATRICES[9],
    )
    tracker = render_script.CubeSymmetryTracker()
    tracker.observe(target_rvec)

    canonical = render_script.canonicalize_pose_symmetry(candidate, tracker)

    assert render_script.rotation_symmetry_distance(canonical.rvec, target_rvec) < 1e-6


def test_static_refinement_skips_layer_turn_recovery(monkeypatch):
    object_points = render_script.canonical_sticker_object_points("corner")
    camera_matrix = render_script.default_camera_matrix((256, 256), 1.2)
    image_points, _ = cv2.projectPoints(
        object_points,
        np.zeros((3, 1), dtype=np.float64),
        np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
    )
    match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2, 3),
        sticker_type="corner",
        average_confidence=0.95,
    )
    candidate = render_script.solve_sticker_pnp(
        image_points.reshape(-1, 2),
        np.asarray([0, 1, 2, 1], dtype=np.int32),
        match,
        (256, 256),
        1.2,
    )
    assert candidate is not None

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("static refinement must not recover layer turns")

    monkeypatch.setattr(
        render_script,
        "recover_misaligned_cube_layers",
        fail_if_called,
    )
    refined = render_script.solve_pnp_from_low_error_assignments(candidate)

    assert refined.layer_rotations == ()
    assert refined.turn_candidate_scores == ()
    assert refined.final_solve_image_points is not None
    assert len(refined.final_solve_image_points) >= 3


def test_project_cube_orientation_axes_follow_camera_projection():
    candidate = SimpleNamespace(
        camera_matrix=render_script.default_camera_matrix((256, 256), 1.2),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
    )
    projected = render_script.project_cube_orientation_axes(
        candidate,
        np.zeros((3, 1), dtype=np.float64),
        np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),
        axis_length=24.0,
    )

    assert projected is not None
    assert projected.shape == (4, 2)
    assert projected[1, 0] > projected[0, 0]  # +X points right
    assert projected[2, 1] > projected[0, 1]  # +Y points down in the image
    np.testing.assert_allclose(projected[3], projected[0], atol=1e-5)


def test_canonical_sticker_identities_cover_54_stickers_in_intuitive_groups():
    identities = render_script.canonical_sticker_identities()

    assert len(identities) == 54
    assert len({identity.label for identity in identities}) == 54
    assert {identity.category for identity in identities} == {"center", "edge", "corner"}
    assert sum(identity.category == "center" for identity in identities) == 6
    assert sum(identity.category == "edge" for identity in identities) == 24
    assert sum(identity.category == "corner" for identity in identities) == 24
    assert [identity.label for identity in identities[:6]] == [
        "C01 U", "C02 D", "C03 F", "C04 B", "C05 R", "C06 L",
    ]


def test_center_color_classifier_chooses_a_legal_standard_scheme():
    trace = render_script.StickerColorTrace.create()
    expected = {
        "U": "white",
        "D": "yellow",
        "F": "green",
        "B": "blue",
        "R": "red",
        "L": "orange",
    }
    labels_by_face = {
        identity.face: identity.label
        for identity in trace.identities
        if identity.category == "center"
    }
    observations = {
        labels_by_face[face]: np.asarray(render_script.PURE_COLOR_RGB[color], dtype=np.uint8)
        for face, color in expected.items()
    }
    trace.observe(0, observations)
    trace.observe(1, observations)

    classified = render_script.classify_center_color_names(trace)

    assert {
        next(identity.face for identity in trace.identities if identity.label == label): color
        for label, color in classified.items()
    } == expected
    assert len(render_script.legal_center_color_assignments()) == 24


def test_oklab_conversion_round_trips_rgb_values():
    rgb = np.asarray(tuple(render_script.PURE_COLOR_RGB.values()), dtype=np.float64)

    np.testing.assert_allclose(
        render_script.oklab_to_rgb(render_script.rgb_to_oklab(rgb)),
        rgb,
        atol=0.01,
    )


def test_piecewise_profiles_compare_trajectories_in_oklab():
    trace = render_script.StickerColorTrace.create()
    center_labels = {
        identity.face: identity.label
        for identity in trace.identities
        if identity.category == "center"
    }
    red_values = np.asarray(
        [[64, 8, 8], [128, 16, 16], [220, 40, 40], [255, 110, 110]],
        dtype=np.uint8,
    )
    blue_values = np.asarray(
        [[8, 8, 64], [16, 16, 128], [40, 40, 220], [110, 110, 255]],
        dtype=np.uint8,
    )
    for frame_index, (red, blue) in enumerate(zip(red_values, blue_values, strict=True)):
        trace.observe(
            frame_index,
            {
                center_labels["R"]: red,
                center_labels["B"]: blue,
            },
        )

    red_profile = render_script.fit_sticker_color_profile(trace, center_labels["R"])
    blue_profile = render_script.fit_sticker_color_profile(trace, center_labels["B"])

    assert red_profile is not None
    assert blue_profile is not None
    assert red_profile.polyline_rgb.shape == (3, 3)
    assert blue_profile.polyline_rgb.shape == (3, 3)
    assert np.isclose(np.max(red_profile.polyline_rgb[1]), 255.0)
    assert np.isclose(np.max(blue_profile.polyline_rgb[1]), 255.0)
    assert render_script.sticker_color_profile_distance(red_profile, red_profile) < 1e-8
    assert render_script.sticker_color_profile_distance(red_profile, blue_profile) > 0.02


def test_color_observation_likelihoods_are_soft_and_normalized():
    profiles = {
        color_name: render_script.GlobalColorProfile(
            color_name=color_name,
            prior=render_script.pure_rgb_color_profile(color_name),
            fitted=None,
            sample_count=0,
        )
        for color_name in render_script.PURE_COLOR_RGB
    }

    likelihoods = render_script.rgb_observation_color_likelihoods(
        np.asarray((128, 20, 20), dtype=np.uint8),
        profiles,
    )

    assert np.isclose(sum(likelihoods.values()), 1.0)
    assert likelihoods["red"] > likelihoods["orange"]
    assert likelihoods["orange"] > likelihoods["blue"]


def test_global_profiles_use_soft_responsibilities_for_ambiguous_red_orange_samples():
    trace = render_script.StickerColorTrace.create()
    labels = [identity.label for identity in trace.identities]
    red_values = np.asarray(
        [[40, 2, 2], [90, 5, 4], [150, 15, 10], [220, 30, 20], [255, 60, 40]],
        dtype=np.uint8,
    )
    orange_values = np.asarray(
        [[50, 25, 2], [110, 55, 4], [170, 90, 8], [220, 130, 30], [255, 165, 0]],
        dtype=np.uint8,
    )
    ambiguous_orange = np.asarray((255, 100, 0), dtype=np.uint8)
    for frame_index in range(20):
        trace.observe(
            frame_index,
            {
                labels[0]: red_values[frame_index % len(red_values)],
                labels[1]: orange_values[frame_index % len(orange_values)],
                # This sample is closer to the red pure prior, but is still
                # allowed to contribute to the orange profile softly.
                labels[2]: ambiguous_orange,
            },
        )

    profiles, _grouped = render_script.build_global_color_profiles(trace)
    red_pivot = profiles["red"].fitted.polyline_rgb[1]
    orange_pivot = profiles["orange"].fitted.polyline_rgb[1]

    assert orange_pivot[1] > red_pivot[1] + 40.0
    likelihoods = render_script.rgb_observation_color_likelihoods(
        np.asarray((255, 165, 0), dtype=np.uint8),
        profiles,
    )
    assert likelihoods["orange"] > likelihoods["red"]

    prior_distances = np.asarray(
        [
            render_script._piecewise_profile_distances(
                render_script.rgb_to_oklab(ambiguous_orange.reshape(1, 3)),
                render_script.pure_rgb_color_profile(color_name).polyline_oklab,
            )[0]
            for color_name in render_script.GLOBAL_COLOR_NAMES
        ]
    )
    responsibilities = render_script._global_profile_soft_responsibilities(
        prior_distances.reshape(1, -1)
    )[0]
    red_index = render_script.GLOBAL_COLOR_NAMES.index("red")
    orange_index = render_script.GLOBAL_COLOR_NAMES.index("orange")
    assert responsibilities[red_index] > responsibilities[orange_index]
    assert responsibilities[orange_index] > 0.01


def test_global_profiles_can_use_pose_free_mask_samples():
    samples = []
    for frame_index in range(8):
        samples.extend(
            (
                ("M0", frame_index, np.asarray((220, 25, 15), dtype=np.uint8)),
                ("M1", frame_index, np.asarray((240, 125, 20), dtype=np.uint8)),
            )
        )

    profiles, grouped = render_script.build_global_color_profiles_from_samples(samples)

    assert sum(len(entries) for entries in grouped.values()) == len(samples)
    assert profiles["red"].fitted is not None
    assert profiles["orange"].fitted is not None


def test_sticker_observation_likelihood_uses_previous_observations_as_prior():
    trace = render_script.StickerColorTrace.create()
    label = trace.identities[0].label
    trace.observe(0, {label: np.asarray((128, 20, 20), dtype=np.uint8)})
    trace.observe(1, {label: np.asarray((10, 10, 180), dtype=np.uint8)})
    profiles = {
        color_name: render_script.GlobalColorProfile(
            color_name=color_name,
            prior=render_script.pure_rgb_color_profile(color_name),
            fitted=None,
            sample_count=0,
        )
        for color_name in render_script.PURE_COLOR_RGB
    }

    likelihoods = render_script.build_sticker_observation_likelihoods(trace, profiles)

    assert set(likelihoods) == {0, 1}
    first = likelihoods[0][label]
    second = likelihoods[1][label]
    assert np.isclose(sum(first.values()), 1.0)
    assert np.isclose(sum(second.values()), 1.0)
    assert first["red"] > first["orange"]
    assert second["red"] < first["red"]


def test_debug_global_profile_likelihoods_are_independent_per_frame():
    trace = render_script.StickerColorTrace.create()
    label = trace.identities[0].label
    trace.observe(0, {label: np.asarray((128, 20, 20), dtype=np.uint8)})
    trace.observe(1, {label: np.asarray((10, 10, 180), dtype=np.uint8)})
    profiles = {
        color_name: render_script.GlobalColorProfile(
            color_name=color_name,
            prior=render_script.pure_rgb_color_profile(color_name),
            fitted=None,
            sample_count=0,
        )
        for color_name in render_script.PURE_COLOR_RGB
    }

    likelihoods = render_script.build_sticker_global_profile_likelihoods(
        trace,
        profiles,
    )

    assert set(likelihoods) == {0, 1}
    for frame_index, rgb in (
        (0, np.asarray((128, 20, 20), dtype=np.uint8)),
        (1, np.asarray((10, 10, 180), dtype=np.uint8)),
    ):
        expected = render_script.rgb_observation_color_likelihoods(rgb, profiles)
        actual = likelihoods[frame_index][label]
        assert np.isclose(sum(actual.values()), 1.0)
        assert actual == expected


def test_low_reliability_observation_has_tempered_color_effect():
    profiles = {
        color_name: render_script.GlobalColorProfile(
            color_name=color_name,
            prior=render_script.pure_rgb_color_profile(color_name),
            fitted=None,
            sample_count=0,
        )
        for color_name in render_script.PURE_COLOR_RGB
    }
    label = render_script.StickerColorTrace.create().identities[0].label
    trace = render_script.StickerColorTrace.create()
    trace.observe(
        0,
        {label: np.asarray((180, 10, 10), dtype=np.uint8)},
        observation_weights={label: 0.0},
    )

    posterior = render_script.build_sticker_observation_likelihoods(trace, profiles)[0][label]

    assert all(np.isclose(value, 1.0 / len(render_script.PURE_COLOR_RGB)) for value in posterior.values())


def test_final_posterior_color_assignment_is_independent_per_sticker():
    label = render_script.StickerColorTrace.create().identities[0].label
    another_label = render_script.StickerColorTrace.create().identities[1].label

    assigned = render_script.classify_sticker_colors_by_final_posterior(
        {
            0: {
                label: {"red": 0.40, "orange": 0.60},
                another_label: {"blue": 0.75, "green": 0.25},
            },
            1: {
                label: {"red": 0.80, "orange": 0.20},
            },
        }
    )

    assert assigned == {label: "red", another_label: "blue"}


def test_cube_rules_do_not_assign_weakly_observed_sticker():
    trace = render_script.StickerColorTrace.create()
    label = trace.identities[0].label
    trace.observe(0, {label: np.asarray((255, 0, 0), dtype=np.uint8)})

    assigned = render_script.classify_sticker_colors_by_cube_rules(
        trace,
        {label: {"red": 0.95, "orange": 0.05}},
    )

    assert label not in assigned


def test_cube_rules_infer_the_last_corner_color_from_two_known_stickers():
    trace = render_script.StickerColorTrace.create()
    face_colors = render_script.STANDARD_FACE_COLOR_NAMES
    center_identities = [
        identity for identity in trace.identities if identity.category == "center"
    ]
    corner_identities = [
        identity
        for identity in trace.identities
        if identity.category == "corner" and identity.cubie_label == "UBL"
    ]
    corner_identities.sort(key=lambda identity: identity.face)
    missing_identity = corner_identities[2]
    all_corner_identities = [
        identity for identity in trace.identities if identity.category == "corner"
    ]
    observed = {
        identity.label: face_colors[identity.face]
        for identity in center_identities + [
            identity
            for identity in all_corner_identities
            if identity.label != missing_identity.label
        ]
    }
    for frame_index in range(3):
        trace.observe(
            frame_index,
            {
                label: np.asarray(render_script.PURE_COLOR_RGB[color], dtype=np.uint8)
                for label, color in observed.items()
            },
        )

    posterior_by_label = {
        label: {color_name: (0.98 if color_name == color else 0.02 / 5.0)
                for color_name in render_script.PURE_COLOR_RGB}
        for label, color in observed.items()
    }
    assigned = render_script.classify_sticker_colors_by_cube_rules(
        trace,
        posterior_by_label,
    )

    assert assigned[missing_identity.label] == face_colors[missing_identity.face]


def test_pairwise_trajectory_support_reinforces_a_shared_color():
    trace = render_script.StickerColorTrace.create()
    orange_label = trace.identities[0].label
    red_label = trace.identities[1].label
    another_orange_label = trace.identities[2].label
    white_label = trace.identities[3].label
    orange_rgb = np.asarray((255, 165, 0), dtype=np.uint8)
    red_rgb = np.asarray((255, 0, 0), dtype=np.uint8)
    white_rgb = np.asarray((255, 148, 149), dtype=np.uint8)
    for frame_index in range(4):
        trace.observe(
            frame_index,
            {
                orange_label: orange_rgb,
                red_label: red_rgb,
                another_orange_label: orange_rgb,
                white_label: white_rgb,
            },
        )

    assigned = render_script.classify_sticker_colors_by_pairwise_trajectory(
        trace,
        {
            3: {
                orange_label: {"red": 0.80, "orange": 0.20},
                red_label: {"red": 0.95, "orange": 0.05},
                another_orange_label: {"red": 0.10, "orange": 0.90},
                white_label: {"white": 0.99, "orange": 0.01},
            }
        },
    )

    assert assigned[orange_label] == "orange"
    assert assigned[another_orange_label] == "orange"
    assert assigned[red_label] == "red"
    assert assigned[white_label] == "white"


def test_frame_posterior_color_assignment_can_change_by_frame():
    label = render_script.StickerColorTrace.create().identities[0].label

    assigned = render_script.classify_sticker_colors_by_frame_posterior(
        {
            0: {label: {"red": 0.70, "orange": 0.30}},
            1: {label: {"red": 0.35, "orange": 0.65}},
        }
    )

    assert assigned[0][label] == "red"
    assert assigned[1][label] == "orange"


def test_frame_global_profile_color_assignment_is_independent_per_sample():
    trace = render_script.StickerColorTrace.create()
    label = trace.identities[0].label
    trace.observe(0, {label: np.asarray((128, 20, 20), dtype=np.uint8)})
    trace.observe(1, {label: np.asarray((10, 10, 180), dtype=np.uint8)})
    profiles = {
        color_name: render_script.GlobalColorProfile(
            color_name=color_name,
            prior=render_script.pure_rgb_color_profile(color_name),
            fitted=None,
            sample_count=0,
        )
        for color_name in render_script.GLOBAL_COLOR_NAMES
    }

    assigned = render_script.classify_sticker_colors_by_frame_global_profile(
        trace,
        profiles,
    )

    assert assigned == {
        0: {label: "red"},
        1: {label: "blue"},
    }


def test_frame_posteriors_carry_forward_when_a_sticker_is_not_observed():
    label = render_script.StickerColorTrace.create().identities[0].label
    other_label = render_script.StickerColorTrace.create().identities[1].label
    posteriors = render_script.carry_forward_sticker_posteriors(
        {
            0: {label: {"red": 0.70, "orange": 0.30}},
            2: {other_label: {"blue": 0.80, "green": 0.20}},
        },
        (0, 1, 2),
    )

    assert posteriors[1][label] == {"red": 0.70, "orange": 0.30}
    assert posteriors[2][label] == {"red": 0.70, "orange": 0.30}
    assert posteriors[1].get(other_label) is None
    assert posteriors[2][other_label] == {"blue": 0.80, "green": 0.20}


def test_face_on_sticker_views_receive_lower_reliability_weight():
    identity = next(
        item
        for item in render_script.canonical_sticker_identities()
        if item.face == "U" and item.category == "center"
    )
    face_on_candidate = SimpleNamespace(
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),
        layer_rotations=(),
    )
    angled_candidate = SimpleNamespace(
        rvec=np.asarray([[np.deg2rad(70.0)], [0.0], [0.0]], dtype=np.float64),
        tvec=np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),
        layer_rotations=(),
    )

    face_on_weight = render_script.sticker_view_reliability_weight(
        face_on_candidate,
        identity,
    )
    angled_weight = render_script.sticker_view_reliability_weight(
        angled_candidate,
        identity,
    )

    assert np.isclose(face_on_weight, 1.0)
    assert render_script.STICKER_VIEW_WEIGHT_FLOOR == 0.0
    assert 0.0 < angled_weight < face_on_weight


def test_cached_pose_symmetry_recovery_uses_color_identity_history(capsys):
    image_size = (256, 256)
    camera_matrix = render_script.default_camera_matrix(image_size, 1.2)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    object_points = render_script.canonical_sticker_object_points("corner")
    image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
    )
    match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2, 3),
        sticker_type="corner",
        average_confidence=0.95,
    )
    candidate = render_script.solve_sticker_pnp(
        image_points.reshape(-1, 2),
        np.asarray([0, 1, 2, 1], dtype=np.int32),
        match,
        image_size,
        1.2,
    )
    assert candidate is not None

    trace = render_script.StickerColorTrace.create()
    color_by_label = {
        identity.label: np.asarray(
            render_script.PURE_COLOR_RGB[
                render_script.STANDARD_FACE_COLOR_NAMES[identity.face]
            ],
            dtype=np.uint8,
        )
        for identity in trace.identities
    }
    symmetry = render_script.CUBE_SYMMETRY_MATRICES[7]
    identities_by_geometry = {
        (identity.cubie_index, identity.axis, identity.side): identity
        for identity in trace.identities
    }
    permutation = {
        identity.label: render_script.sticker_identity_label_under_symmetry(
            identity,
            symmetry,
            identities_by_geometry,
        )
        for identity in trace.identities
    }
    frame_one = {
        source_label: color_by_label[target_label]
        for target_label, source_label in permutation.items()
    }
    trace.observe(0, color_by_label)
    trace.observe(1, frame_one)

    records = [
        SimpleNamespace(frame_index=0, pose_candidate=candidate),
        SimpleNamespace(frame_index=1, pose_candidate=candidate),
    ]
    corrected_trace = render_script.recover_cached_pose_symmetries(
        records,
        trace,
        debug=True,
    )
    debug_output = capsys.readouterr().out
    assert "pose symmetry frame 1" in debug_output
    assert "tested orientations" in debug_output
    assert "symmetry=07" in debug_output

    recovered = records[1].pose_candidate
    expected_rvec = render_script.transform_rvec_by_symmetry(candidate.rvec, symmetry)
    assert render_script.rotation_symmetry_distance(recovered.rvec, expected_rvec) < 1e-6
    assert corrected_trace.observations[trace.identities[0].label][1][1].tolist() == list(
        color_by_label[trace.identities[0].label]
    )


def test_sticker_assignment_rejects_opposite_colors_on_one_edge():
    trace = render_script.StickerColorTrace.create()
    center_labels = {
        identity.face: identity.label
        for identity in trace.identities
        if identity.category == "center"
    }
    centers = {
        face: np.asarray(render_script.PURE_COLOR_RGB[color], dtype=np.uint8)
        for face, color in {
            "U": "white",
            "D": "yellow",
            "F": "green",
            "B": "blue",
            "R": "red",
            "L": "orange",
        }.items()
    }
    edge_identities = [
        identity
        for identity in trace.identities
        if identity.category == "edge" and identity.cubie_label == "UF"
    ]
    observations = {
        center_labels[face]: value for face, value in centers.items()
    }
    observations[edge_identities[0].label] = np.asarray(
        render_script.PURE_COLOR_RGB["red"], dtype=np.uint8
    )
    observations[edge_identities[1].label] = np.asarray(
        render_script.PURE_COLOR_RGB["orange"], dtype=np.uint8
    )
    trace.observe(0, observations)

    center_colors = render_script.classify_center_color_names(trace)
    classified = render_script.classify_sticker_color_names(trace, center_colors)
    assigned_edge_colors = tuple(classified[label] for label in observations if label.startswith("E"))

    assert len(assigned_edge_colors) == 2
    assert render_script._legal_cubie_color_tuple(assigned_edge_colors)
    assert frozenset(assigned_edge_colors) not in render_script.OPPOSITE_COLOR_PAIRS


def test_sticker_assignment_does_not_reuse_the_same_edge_piece():
    trace = render_script.StickerColorTrace.create()
    center_labels = {
        identity.face: identity.label
        for identity in trace.identities
        if identity.category == "center"
    }
    center_colors = {
        "U": "white",
        "D": "yellow",
        "F": "green",
        "B": "blue",
        "R": "red",
        "L": "orange",
    }
    observations = {
        center_labels[face]: np.asarray(render_script.PURE_COLOR_RGB[color], dtype=np.uint8)
        for face, color in center_colors.items()
    }
    edge_groups = {}
    for identity in trace.identities:
        if identity.category == "edge":
            edge_groups.setdefault(identity.cubie_label, []).append(identity)
    for group_name in ("UF", "UR"):
        for identity in edge_groups[group_name]:
            observations[identity.label] = np.asarray(
                render_script.PURE_COLOR_RGB["red" if identity.face == "U" else "orange"],
                dtype=np.uint8,
            )
    trace.observe(0, observations)

    classified = render_script.classify_sticker_color_names(
        trace,
        render_script.classify_center_color_names(trace),
    )
    assigned_pairs = []
    for group_name in ("UF", "UR"):
        assigned_pairs.append(
            frozenset(classified[identity.label] for identity in edge_groups[group_name])
        )

    assert assigned_pairs[0] != assigned_pairs[1]
    assert len(render_script.legal_edge_color_combinations()) == 12
    assert len(render_script.legal_corner_color_combinations()) == 8


def test_corner_assignment_enforces_proper_color_chirality():
    identities = [
        identity
        for identity in render_script.canonical_sticker_identities()
        if identity.category == "corner" and identity.cubie_label == "UBL"
    ]
    identities.sort(key=lambda identity: {"U": 0, "L": 1, "B": 2}[identity.face])

    # On the UBL corner, white-orange-blue is the clockwise standard order.
    assert render_script.corner_color_assignment_is_chiral(
        identities,
        ("white", "orange", "blue"),
    )
    assert not render_script.corner_color_assignment_is_chiral(
        identities,
        ("white", "blue", "orange"),
    )

    # Exactly half of the permutations of a legal color triple are proper
    # rotations; the other half are reflected/impossible corner assignments.
    assignments = [
        render_script.corner_color_assignment_is_chiral(
            identities,
            permutation,
        )
        for permutation in permutations(("white", "orange", "blue"))
    ]
    assert sum(assignments) == 3


def test_corner_chirality_uses_rotated_center_color_scheme():
    identities = [
        identity
        for identity in render_script.canonical_sticker_identities()
        if identity.category == "corner" and identity.cubie_label == "UBL"
    ]
    identities.sort(key=lambda identity: {"U": 0, "L": 1, "B": 2}[identity.face])

    # This is a valid cube rotation.  Chirality must follow the centers rather
    # than assuming that a particular object face is always a fixed color.
    rotated_face_colors = render_script.legal_center_color_assignments()[1]
    rotated_color_face = {
        color: face
        for face, color in rotated_face_colors.items()
    }
    ordered_colors = tuple(rotated_face_colors[identity.face] for identity in identities)
    assert render_script.corner_color_assignment_is_chiral(
        identities,
        ordered_colors,
        rotated_color_face,
    )


def test_projected_center_color_overlay_only_draws_visible_classified_centers():
    candidate = SimpleNamespace(
        layer_rotations=(),
        camera_matrix=render_script.default_camera_matrix((256, 256), 1.2),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
    )
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    drawn = render_script.draw_projected_center_colors(
        image,
        candidate,
        {"C02 D": "yellow", "C01 U": "white"},
        np.zeros((3, 1), dtype=np.float64),
        np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),
    )

    assert drawn == 1
    assert np.any(image[:, :, 1] > 0)


def test_classified_cube_diagnostic_panel_keeps_cube_stationary():
    common = {
        "layer_rotations": (),
        "rvec": np.zeros((3, 1), dtype=np.float64),
        "camera_matrix": render_script.default_camera_matrix((256, 256), 1.2),
        "dist_coeffs": np.zeros((4, 1), dtype=np.float64),
    }
    first_candidate = SimpleNamespace(
        **common,
        tvec=np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),
    )
    second_candidate = SimpleNamespace(
        **common,
        tvec=np.asarray([[40.0], [-25.0], [320.0]], dtype=np.float64),
    )
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    colors = {"C01 U": "white", "C02 D": "yellow"}

    first = render_script.draw_classified_cube_diagnostic_panel(
        image,
        first_candidate,
        colors,
    )
    second = render_script.draw_classified_cube_diagnostic_panel(
        image,
        second_candidate,
        colors,
    )

    assert first.shape == (256, 512, 3)
    np.testing.assert_array_equal(first[:, 256:], second[:, 256:])


def test_classified_cube_diagnostic_panel_can_hide_sticker_identities(monkeypatch):
    candidate = SimpleNamespace(
        layer_rotations=(),
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),
        camera_matrix=render_script.default_camera_matrix((256, 256), 1.2),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
    )
    calls = []
    monkeypatch.setattr(
        render_script,
        "draw_projected_sticker_identity_labels",
        lambda *args, **kwargs: calls.append((args, kwargs)) or 0,
    )

    render_script.draw_classified_cube_diagnostic_panel(
        np.zeros((256, 256, 3), dtype=np.uint8),
        candidate,
        {"C01 U": "white"},
        show_sticker_identities=False,
    )

    assert calls == []


def test_projected_sticker_faces_match_cube_pose_geometry_after_symmetry():
    symmetry = render_script.CUBE_SYMMETRY_MATRICES[7]
    candidate = SimpleNamespace(
        layer_rotations=(),
        object_symmetry=symmetry,
        camera_matrix=render_script.default_camera_matrix((256, 256), 1.2),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
    )
    rvec = render_script.transform_rvec_by_symmetry(
        np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64),
        symmetry,
    )
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)

    sticker_faces = render_script.projected_sticker_faces(candidate, rvec, tvec)
    cube_faces = render_script.projected_cubelet_faces(candidate, rvec, tvec)

    assert len(sticker_faces) == len(cube_faces)
    for (_identity, sticker_depth, sticker_polygon), (
        cube_depth,
        cube_polygon,
        _anchor,
    ) in zip(sticker_faces, cube_faces, strict=True):
        np.testing.assert_allclose(sticker_depth, cube_depth)
        np.testing.assert_allclose(sticker_polygon, cube_polygon)


def test_sticker_identity_overlay_draws_only_visible_stickers():
    candidate = SimpleNamespace(
        layer_rotations=(),
        camera_matrix=render_script.default_camera_matrix((256, 256), 1.2),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
    )
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    drawn = render_script.draw_projected_sticker_identity_labels(
        image,
        candidate,
        np.zeros((3, 1), dtype=np.float64),
        np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),
    )

    assert drawn == 9
    assert np.any(image)


def test_sticker_likelihood_overlay_is_compact_and_draws_second_color_swatch(monkeypatch):
    identity = render_script.canonical_sticker_identities()[0]
    candidate = SimpleNamespace()
    polygon = np.asarray(
        ((20.0, 20.0), (44.0, 20.0), (44.0, 44.0), (20.0, 44.0)),
        dtype=np.float32,
    )
    monkeypatch.setattr(
        render_script,
        "projected_sticker_faces",
        lambda *_args, **_kwargs: [(identity, 1.0, polygon)],
    )
    drawn_text: list[str] = []
    rectangles: list[tuple[int, int, int]] = []
    real_rectangle = cv2.rectangle

    def record_text(image, text, *args, **kwargs):
        drawn_text.append(text)
        return image

    def record_rectangle(image, _first, _second, color, *args, **kwargs):
        rectangles.append(tuple(int(value) for value in color))
        return real_rectangle(image, _first, _second, color, *args, **kwargs)

    monkeypatch.setattr(render_script.cv2, "putText", record_text)
    monkeypatch.setattr(render_script.cv2, "rectangle", record_rectangle)

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    drawn = render_script.draw_projected_sticker_likelihoods(
        image,
        candidate,
        {identity.label: "red"},
        {
            identity.label: {
                "red": 0.87,
                "orange": 0.10,
                "yellow": 0.01,
                "white": 0.01,
                "blue": 0.005,
                "green": 0.005,
            }
        },
        np.zeros((3, 1), dtype=np.float64),
        np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),
        projected_faces=[(identity, 1.0, polygon)],
    )

    assert drawn == 1
    assert drawn_text == ["0.87, 0.10"]
    assert (0, 165, 255) in rectangles


def test_sticker_color_sampling_uses_projected_mask_and_cube_sticker_intersection():
    image = np.full((256, 256, 3), (60, 80, 100), dtype=np.uint8)
    camera_matrix = render_script.default_camera_matrix((256, 256), 1.2)
    candidate = SimpleNamespace(
        layer_rotations=(),
        camera_matrix=camera_matrix,
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),
    )
    sticker_mask = np.ones((256, 256), dtype=bool)
    cube_mask = np.ones((256, 256), dtype=bool)
    predicted = ([sticker_mask], {0: "sticker"}, np.asarray([0], dtype=np.int32), cube_mask)

    observations = render_script.sample_projected_sticker_colors(
        image,
        (0, 0, 256, 256),
        predicted,
        candidate,
        max_samples_per_sticker=32,
        min_samples_per_sticker=4,
    )

    assert observations
    assert set(observations) == {
        "C02 D",
        "E09 DF/D",
        "E11 DR/D",
        "E13 DB/D",
        "E15 DL/D",
        "K13 DFR/D",
        "K16 DFL/D",
        "K19 DBL/D",
        "K22 DBR/D",
    }
    assert all(np.array_equal(value, np.asarray((100, 80, 60), dtype=np.uint8)) for value in observations.values())

    empty_cube = np.zeros_like(cube_mask)
    empty_predicted = (predicted[0], predicted[1], predicted[2], empty_cube)
    assert render_script.sample_projected_sticker_colors(
        image,
        (0, 0, 256, 256),
        empty_predicted,
        candidate,
    ) == {}


def test_sticker_rgb_html_plot_has_selectors_for_all_stickers(tmp_path):
    pytest.importorskip("plotly.graph_objects")
    trace = render_script.StickerColorTrace.create()
    trace.observe(
        7,
        {trace.identities[0].label: np.asarray((10, 20, 30), dtype=np.uint8)},
    )
    output_path = tmp_path / "sticker_rgb.html"

    render_script.save_sticker_rgb_html_plot(
        trace,
        output_path,
        title="Sticker RGB",
    )

    html = output_path.read_text(encoding="utf-8")
    assert "All stickers" in html
    assert "All colors" in html
    assert "unknown observations" not in html
    assert "global profile" in html
    assert "Centers" in html
    assert "Edges" in html
    assert "Corners" in html
    assert "fitted regression" in html
    assert '"mode":"lines"' in html
    assert '"groupclick":"toggleitem"' in html
    assert all(
        identity.label.replace("/", "\\u002f") in html
        for identity in trace.identities
    )


def test_sticker_keypoint_matches_returns_a_consistent_circular_order():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    points = np.asarray(
        [
            (8, 8),
            (23, 8),
            (23, 23),
            (8, 23),
        ],
        dtype=np.float32,
    )
    point_confidences = np.ones(4, dtype=np.float32)
    class_ids = np.asarray([0, 1, 2, 1], dtype=np.int32)
    box_confidences = np.ones(4, dtype=np.float32)

    indices, sticker_type = render_script.sticker_keypoint_matches(
        mask,
        points,
        point_confidences,
        class_ids,
        box_confidences,
        point_threshold=0.05,
        class_confidences=dict(render_script.DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES),
    )

    assert indices == [0, 1, 2, 3]
    assert sticker_type == "corner"


def test_sticker_selection_prefers_parallelogram_like_geometry():
    points = np.asarray(
        [
            (8.0, 8.0),
            (24.0, 9.0),
            (23.0, 24.0),
            (7.0, 23.0),
        ],
        dtype=np.float32,
    )
    distorted = points.copy()
    distorted[2] += np.asarray((8.0, 7.0))

    good_error = render_script.sticker_shape_error(points, [0, 1, 2, 3])
    bad_error = render_script.sticker_shape_error(distorted, [0, 1, 2, 3])
    good_match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2, 3),
        sticker_type="corner",
        average_confidence=0.60,
        shape_error=good_error,
    )
    bad_match = render_script.StickerKeypointMatch(
        mask_index=1,
        keypoint_indices=(0, 1, 2, 3),
        sticker_type="corner",
        average_confidence=0.99,
        shape_error=bad_error,
    )

    assert good_error < bad_error
    assert render_script.sticker_match_priority_key(good_match) < render_script.sticker_match_priority_key(bad_match)


def test_render_sticker_masks_uses_neutral_class_independent_color():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    local_mask = np.zeros((32, 32), dtype=bool)
    local_mask[8:24, 8:24] = True
    predicted = ([local_mask], {0: "sticker"}, np.asarray([0], dtype=np.int32))
    pose = (
        np.asarray([(8, 8), (23, 8), (23, 23), (8, 23)], dtype=np.float32),
        np.ones(4, dtype=np.float32),
        np.asarray([0, 1, 2, 1], dtype=np.int32),
        np.ones(4, dtype=np.float32),
    )

    output = render_script.render_sticker_masks(
        image,
        (0, 0, 32, 32),
        predicted,
        opacity=1.0,
        border_thickness=0,
        pose_predictions=pose,
        point_threshold=0.05,
        class_confidences=dict(render_script.DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES),
    )

    assert np.all(output[12, 12] == render_script.STICKER_MASK_NEUTRAL_COLOR_BGR)


def test_debug_color_masks_classifies_mask_pixels_without_sticker_identity():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :] = (255, 0, 0)  # BGR blue
    local_mask = np.zeros((32, 32), dtype=bool)
    local_mask[8:24, 8:24] = True
    predicted = (
        [local_mask],
        {0: "sticker"},
        np.asarray([0], dtype=np.int32),
        None,
    )
    profiles = {
        color_name: render_script.GlobalColorProfile(
            color_name=color_name,
            prior=render_script.pure_rgb_color_profile(color_name),
            fitted=None,
            sample_count=0,
        )
        for color_name in render_script.GLOBAL_COLOR_NAMES
    }

    output = render_script.draw_classified_sticker_masks(
        image,
        (0, 0, 32, 32),
        predicted,
        global_profiles=profiles,
        opacity=1.0,
        show_sticker_identities=False,
    )

    assert np.all(output[12, 12] == (255, 0, 0))
    assert np.all(output[0, 0] == image[0, 0])


def test_sticker_mask_erosion_parser_and_roi_masks():
    assert render_script.build_parser().parse_args(["--source", "frame.png"]).sticker_mask_erosion == 0
    args = render_script.build_parser().parse_args(
        ["--source", "frame.png", "--sticker-mask-erosion", "1"]
    )
    assert args.sticker_mask_erosion == 1
    debug_args = render_script.build_parser().parse_args(
        ["--source", "frame.png", "--debug-color-masks"]
    )
    assert debug_args.debug_color_masks is True
    hidden_args = render_script.build_parser().parse_args(
        [
            "--source",
            "frame.png",
            "--no-show-keypoints",
        ]
    )
    assert hidden_args.show_keypoints is False

    local_mask = np.zeros((9, 9), dtype=bool)
    local_mask[2:7, 2:7] = True
    predicted = (
        [local_mask],
        {0: "sticker"},
        np.asarray([0], dtype=np.int32),
        np.ones((9, 9), dtype=bool),
    )
    masks = render_script.sticker_segmentation_masks_in_roi(
        predicted,
        (9, 9),
        sticker_mask_erosion=1,
    )

    assert len(masks) == 1
    assert int(masks[0].sum()) == 9


def test_sticker_mask_parser_preserves_the_model_cube_mask():
    cube_mask = np.zeros((16, 16), dtype=np.float32)
    cube_mask[2:14, 2:14] = 1.0
    sticker_mask = np.zeros((16, 16), dtype=np.float32)
    sticker_mask[5:11, 5:11] = 1.0
    result = SimpleNamespace(
        masks=SimpleNamespace(data=np.asarray([cube_mask, sticker_mask])),
        boxes=SimpleNamespace(cls=np.asarray([0, 1], dtype=np.int32)),
        names={0: "cube", 1: "sticker"},
    )

    parsed = render_script.sticker_masks_from_result(result)

    assert parsed is not None
    sticker_masks, _names, sticker_classes, parsed_cube_mask = parsed
    assert len(sticker_masks) == 1
    np.testing.assert_array_equal(sticker_classes, np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(parsed_cube_mask, cube_mask.astype(bool))


def test_solve_sticker_pnp_uses_canonical_corner_correspondence():
    object_points = render_script.canonical_sticker_object_points("corner")
    camera_matrix = render_script.default_camera_matrix((256, 256), 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2, 3),
        sticker_type="corner",
        average_confidence=0.95,
    )
    candidate = render_script.solve_sticker_pnp(
        image_points.reshape(-1, 2),
        np.asarray([0, 1, 2, 1], dtype=np.int32),
        match,
        (256, 256),
        1.2,
    )

    assert candidate is not None
    assert candidate.reprojection_error < 1e-3
    assert candidate.tvec[2, 0] > 0.0
    assert candidate.assignment_image_points is not None
    assert candidate.assignment_projected_points is not None
    assert len(candidate.assignment_image_points) == 4
    assert len(candidate.assignment_projected_points) == 4


def test_final_assignment_refinement_is_keypoint_only(monkeypatch):
    object_points = render_script.canonical_sticker_object_points("corner")
    camera_matrix = render_script.default_camera_matrix((256, 256), 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2, 3),
        sticker_type="corner",
        average_confidence=0.95,
    )
    candidate = render_script.solve_sticker_pnp(
        image_points.reshape(-1, 2),
        np.asarray([0, 1, 2, 1], dtype=np.int32),
        match,
        (256, 256),
        1.2,
    )
    assert candidate is not None
    candidate = replace(
        candidate,
        mask_fit_error=12.0,
        mask_boundary_sample_count=7,
    )
    mask_refinement_calls = []
    monkeypatch.setattr(
        render_script,
        "refine_pose_with_sticker_boundaries",
        lambda value: mask_refinement_calls.append(value) or value,
    )

    refined = render_script.solve_pnp_from_low_error_assignments(
        candidate,
    )

    assert mask_refinement_calls == []
    assert not np.isfinite(refined.mask_fit_error)
    assert refined.mask_boundary_sample_count == 0

    assert len(mask_refinement_calls) == 0


def test_initial_sticker_hypotheses_defer_full_refinement(monkeypatch):
    object_points = render_script.canonical_sticker_object_points("corner")
    camera_matrix = render_script.default_camera_matrix((256, 256), 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    base_match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2, 3),
        sticker_type="corner",
        average_confidence=0.95,
    )
    base_candidate = render_script.solve_sticker_pnp(
        image_points.reshape(-1, 2),
        np.asarray([0, 1, 2, 1], dtype=np.int32),
        base_match,
        (256, 256),
        1.2,
    )
    assert base_candidate is not None
    matches = tuple(
        render_script.StickerKeypointMatch(
            index,
            (0, 1, 2, 3),
            "corner",
            0.9,
        )
        for index in range(3)
    )
    refinement_flags = []

    def fake_sticker_solve(*_args, **_kwargs):
        match = _args[2]
        return replace(
            base_candidate,
            sticker_type=match.sticker_type,
            sticker_average_confidence=match.average_confidence,
            mask_fit_error=float(match.mask_index),
        )

    def fake_assignment_solve(candidate, **kwargs):
        refinement_flags.append(kwargs)
        return candidate

    monkeypatch.setattr(render_script, "solve_sticker_pnp", fake_sticker_solve)
    monkeypatch.setattr(
        render_script,
        "solve_pnp_from_low_error_assignments",
        fake_assignment_solve,
    )

    candidates = render_script.best_sticker_pnp_candidates(
        np.zeros((12, 2), dtype=np.float32),
        np.zeros((12,), dtype=np.int32),
        matches,
        (128, 128),
        1.2,
    )

    assert len(candidates) == 3
    assert refinement_flags == []
    assert [candidate.mask_fit_error for candidate in candidates] == [0.0, 1.0, 2.0]


def test_projected_cube_diameter_uses_the_initial_pose_scale():
    image_size = (256, 256)
    camera_matrix = render_script.default_camera_matrix(image_size, 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    candidate = SimpleNamespace(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvec=rvec,
        tvec=tvec,
    )

    projected_cube, _ = cv2.projectPoints(
        render_script.CUBE_CORNER_POINTS_MM,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    _center, radius = cv2.minEnclosingCircle(
        projected_cube.reshape(-1, 2).astype(np.float32)
    )

    assert render_script.projected_cube_diameter(candidate) == pytest.approx(
        2.0 * radius
    )


def test_cube_layer_rotation_changes_only_the_selected_face_layer():
    points = render_script.CUBE_CORNER_POINTS_MM
    rotation = render_script.CubeLayerRotation(axis=2, side=1, angle_degrees=90.0)
    rotated = render_script.apply_cube_layer_rotations(points, (rotation,))
    mask = render_script.cube_layer_mask(points, axis=2, side=1)

    np.testing.assert_allclose(rotated[~mask], points[~mask])
    assert not np.allclose(rotated[mask], points[mask])
    np.testing.assert_allclose(
        rotated[mask][:, 2],
        np.full(int(mask.sum()), render_script.CUBE_HALF_SIZE_MM),
    )


def test_single_layer_mask_leaves_the_shared_seam_stationary():
    points = render_script.CUBE_LATTICE_OBJECT_POINTS_MM
    seam_coordinate = render_script.CUBE_HALF_SIZE_MM - render_script.CUBE_LAYER_SIZE_MM

    positive_mask = render_script.cube_layer_mask(points, axis=2, side=1)
    negative_mask = render_script.cube_layer_mask(points, axis=2, side=-1)

    assert not np.any(positive_mask[np.isclose(points[:, 2], seam_coordinate)])
    assert not np.any(negative_mask[np.isclose(points[:, 2], -seam_coordinate)])


def test_orientation_anchor_uses_stationary_complement_of_turn_layer():
    candidate = SimpleNamespace(
        layer_rotations=(
            render_script.CubeLayerRotation(axis=2, side=1, angle_degrees=30.0),
        )
    )

    anchor = render_script.orientation_anchor_for_candidate(candidate)

    assert anchor == render_script.OrientationAnchor(axis=2, layers=(0, 1))


def test_orientation_anchor_is_whole_cube_without_a_turn():
    candidate = SimpleNamespace(layer_rotations=())

    assert render_script.orientation_anchor_for_candidate(candidate) == (
        render_script.all_cube_orientation_anchor()
    )


def test_projected_cube_faces_mark_only_anchor_layers():
    image_size = (256, 256)
    candidate = SimpleNamespace(
        layer_rotations=(),
        orientation_anchor=render_script.OrientationAnchor(axis=2, layers=(0, 1)),
        camera_matrix=render_script.default_camera_matrix(image_size, 1.2),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
    )
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)

    faces = render_script.projected_cubelet_faces(candidate, rvec, tvec)

    assert faces
    assert any(not is_anchor for _depth, _face, is_anchor in faces)
    assert any(is_anchor for _depth, _face, is_anchor in faces)


def test_temporal_pose_filter_refits_from_stationary_layers():
    image_size = (256, 256)
    camera_matrix = render_script.default_camera_matrix(image_size, 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    rotation = render_script.CubeLayerRotation(axis=2, side=1, angle_degrees=30.0)
    assignment_object_points = render_script.apply_cube_layer_rotations(
        render_script.CUBE_LATTICE_OBJECT_POINTS_MM,
        (rotation,),
    )
    assignment_image_points, _ = cv2.projectPoints(
        assignment_object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    sticker_object_points = render_script.canonical_sticker_object_points("corner")
    sticker_image_points, _ = cv2.projectPoints(
        sticker_object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    candidate = render_script.StickerPnPCandidate(
        sticker_type="corner",
        sticker_average_confidence=0.9,
        image_points=sticker_image_points.reshape(-1, 2),
        object_points=sticker_object_points,
        projected_sticker_points=sticker_image_points.reshape(-1, 2),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvec=rvec,
        tvec=tvec,
        reprojection_error=0.0,
        assignment_image_points=assignment_image_points.reshape(-1, 2),
        assignment_projected_points=assignment_image_points.reshape(-1, 2),
        assignment_object_points=assignment_object_points,
        layer_rotations=(rotation,),
    )

    filtered = render_script.apply_temporal_pose_filter(
        candidate,
        render_script.PoseMotionFilter(),
    )

    assert filtered.orientation_anchor == render_script.OrientationAnchor(
        axis=2,
        layers=(0, 1),
    )
    assert filtered.motion_diagnostics is not None
    assert filtered.raw_rvec is not None
    assert filtered.assignment_projected_points is not None


def test_rank_misaligned_layer_faces_prefers_fewest_explained_observations():
    points = render_script.CUBE_LATTICE_OBJECT_POINTS_MM
    distances = np.zeros(len(points), dtype=np.float32)
    positive_z = render_script.cube_layer_mask(points, axis=2, side=1)
    distances[positive_z] = 12.0

    ranked = render_script.rank_misaligned_layer_faces(
        points,
        points,
        distances,
        fixed_mask=np.zeros(len(points), dtype=bool),
        assignment_error_limit=4.0,
        max_face_candidates=2,
    )

    assert ranked[0][:2] == (2, 1)
    assert ranked[0][2] == 0.0
    assert len(ranked) == 2


def test_transformed_visibility_rejects_occluded_lattice_points():
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64)
    back_center_index = render_script.CUBE_LATTICE_COORDS.index((1, 1, 3))

    unturned_points = render_script.CUBE_LATTICE_OBJECT_POINTS_MM
    unturned_hidden = render_script.transformed_surface_occlusion_mask(
        unturned_points,
        rvec,
        tvec,
        (),
    )
    assert unturned_hidden[back_center_index]

    rvec = np.asarray([[0.5], [0.3], [0.0]], dtype=np.float64)
    rotation = render_script.CubeLayerRotation(axis=0, side=-1, angle_degrees=30.0)
    turned_points = render_script.apply_cube_layer_rotations(
        unturned_points,
        (rotation,),
    )
    turned_hidden = render_script.transformed_surface_occlusion_mask(
        turned_points,
        rvec,
        tvec,
        (rotation,),
    )
    visible_points, _class_ids = render_script.visible_transformed_cube_lattice_points(
        rvec,
        tvec,
        (rotation,),
    )

    transformed_normals = render_script.CUBE_LATTICE_BOUNDARY_NORMALS.copy()
    layer_mask = render_script.cube_layer_mask(
        unturned_points,
        rotation.axis,
        rotation.side,
        rotation.layer_count,
    )
    layer_rvec = np.zeros((3, 1), dtype=np.float64)
    layer_rvec[rotation.axis, 0] = rotation.side * np.deg2rad(
        rotation.angle_degrees
    )
    layer_matrix, _ = cv2.Rodrigues(layer_rvec)
    transformed_normals[layer_mask] = np.einsum(
        "ij,nfj->nfi",
        layer_matrix,
        transformed_normals[layer_mask],
    )
    camera_matrix, _ = cv2.Rodrigues(rvec)
    front_facing = np.any(
        render_script.CUBE_LATTICE_BOUNDARY_FACE_MASK
        & (
            np.einsum(
                "ij,nfj->nfi",
                camera_matrix,
                transformed_normals,
            )[..., 2]
            < 0.0
        ),
        axis=1,
    )

    assert np.any(turned_hidden & front_facing)
    assert len(visible_points) < len(turned_points)


def test_recover_misaligned_layer_updates_assignment_geometry(monkeypatch):
    image_size = (256, 256)
    camera_matrix = render_script.default_camera_matrix(image_size, 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    sticker_object_points = render_script.canonical_sticker_object_points("corner")
    sticker_image_points, _ = cv2.projectPoints(
        sticker_object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    sticker_image_points = sticker_image_points.reshape(-1, 2)

    visible_objects, visible_classes = render_script.visible_cube_lattice_points(
        rvec,
        tvec,
    )
    layer_choice = None
    for axis in range(3):
        for side in (-1, 1):
            mask = render_script.cube_layer_mask(visible_objects, axis, side)
            if int(mask.sum()) >= 3 and not np.any(
                np.all(
                    np.isclose(
                        visible_objects[mask][:, None, :],
                        sticker_object_points[None, :, :],
                        atol=1e-4,
                    ),
                    axis=2,
                )
            ):
                layer_choice = (axis, side, mask)
                break
        if layer_choice is not None:
            break
    assert layer_choice is not None
    axis, side, layer_mask = layer_choice
    rotation = render_script.CubeLayerRotation(axis, side, 45.0)
    rotated_objects = render_script.apply_cube_layer_rotations(
        visible_objects,
        (rotation,),
    )
    layer_indices = np.flatnonzero(layer_mask)
    source_index = int(
        layer_indices[
            np.argmax(
                np.linalg.norm(
                    rotated_objects[layer_indices] - visible_objects[layer_indices],
                    axis=1,
                )
            )
        ]
    )
    extra_point, _ = cv2.projectPoints(
        rotated_objects[source_index : source_index + 1],
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    all_points = np.concatenate((sticker_image_points, extra_point.reshape(-1, 2)))
    all_classes = np.concatenate(
        (
            np.asarray([0, 1, 2, 1], dtype=np.int32),
            np.asarray([visible_classes[source_index]], dtype=np.int32),
        )
    )
    match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2, 3),
        sticker_type="corner",
        average_confidence=0.95,
    )
    candidate = render_script.solve_sticker_pnp(
        all_points,
        all_classes,
        match,
        image_size,
        1.2,
        scoring_points=all_points,
    )
    assert candidate is not None
    candidate = replace(
        candidate,
        sticker_contours=(np.zeros((4, 2), dtype=np.float32),),
    )
    def fail_sticker_score(*_args, **_kwargs):
        raise AssertionError("sticker scoring must not run during pose-only turn fitting")

    monkeypatch.setattr(
        render_script,
        "sticker_mask_wireframe_score",
        fail_sticker_score,
    )
    monkeypatch.setattr(
        render_script,
        "sticker_mask_layer_wireframe_score",
        fail_sticker_score,
    )
    limit = (
        render_script.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION
        * render_script.projected_cube_diameter(candidate)
    )

    recovered = render_script.recover_misaligned_cube_layers(candidate, limit)
    exhaustive = render_script.recover_misaligned_cube_layers(
        candidate,
        limit,
        max_face_candidates=6,
    )
    shortlisted = render_script.recover_misaligned_cube_layers(
        candidate,
        limit,
        max_face_candidates=2,
    )
    logged = render_script.recover_misaligned_cube_layers(
        candidate,
        limit,
        max_face_candidates=6,
        record_all_turn_candidates=True,
    )

    assert recovered.layer_rotations
    assert recovered.assignment_projected_points is not None
    recovered_distances = np.linalg.norm(
        recovered.assignment_image_points
        - recovered.assignment_projected_points,
        axis=1,
    )
    assert np.min(recovered_distances) <= limit
    assert shortlisted.layer_rotations == exhaustive.layer_rotations
    np.testing.assert_allclose(
        shortlisted.assignment_projected_points,
        exhaustive.assignment_projected_points,
    )
    searched_rotations = [
        rotation
        for debug_candidate in logged.turn_candidate_debug
        for rotation in debug_candidate.layer_rotations
    ]
    assert len({
        (rotation.axis, rotation.side)
        for rotation in searched_rotations
    }) == 6
    assert {rotation.layer_count for rotation in searched_rotations} == {1, 2}
    assert min(rotation.angle_degrees for rotation in searched_rotations) == -45.0
    assert max(rotation.angle_degrees for rotation in searched_rotations) == 45.0
    assert all(
        rotation.angle_degrees % 3.0 == 0.0
        for rotation in searched_rotations
    )
    assert len(logged.turn_candidate_scores) > render_script.PNP_TURN_DEBUG_CANDIDATE_COUNT
    assert len(logged.turn_candidate_debug) == len(logged.turn_candidate_scores)


def test_three_point_edge_uses_image_gap_to_preserve_cyclic_correspondence():
    object_points = render_script.canonical_sticker_object_points("edge")
    # p0, p1, p2 are observed; p3 is the missing corner.  The p2 -> p0
    # segment is therefore the long cyclic gap in the ordered image points.
    image_points = np.asarray(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
        dtype=np.float32,
    )

    sequences = render_script.sticker_correspondence_object_indices(
        "edge",
        [1, 1, 2],
        image_points,
    )

    assert sequences == ((0, 1, 2),)
    np.testing.assert_allclose(object_points[list(sequences[0])], object_points[:3])


def test_three_point_pnp_defers_segmentation_until_after_pose_refinement():
    object_points = render_script.canonical_sticker_object_points("corner")
    camera_matrix = render_script.default_camera_matrix((256, 256), 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    full_image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    full_image_points = full_image_points.reshape(-1, 2).astype(np.float32)

    match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2),
        sticker_type="corner",
        average_confidence=0.95,
    )
    candidate = render_script.solve_sticker_pnp(
        full_image_points[[2, 3, 0]],
        np.asarray([2, 1, 0], dtype=np.int32),
        match,
        (256, 256),
        1.2,
        sticker_contour=full_image_points,
    )
    candidate_without_mask = render_script.solve_sticker_pnp(
        full_image_points[[2, 3, 0]],
        np.asarray([2, 1, 0], dtype=np.int32),
        match,
        (256, 256),
        1.2,
    )

    assert candidate is not None
    assert candidate_without_mask is not None
    np.testing.assert_allclose(candidate.tvec, candidate_without_mask.tvec)
    np.testing.assert_allclose(candidate.rvec, candidate_without_mask.rvec)
    assert not np.isfinite(candidate.mask_fit_error)


def test_sticker_mask_center_uncertainty_increases_for_asymmetric_mask():
    clean = np.zeros((64, 64), dtype=np.uint8)
    clean[20:40, 20:40] = 1
    asymmetric = clean.copy()
    asymmetric[20:40, 40:54] = 1
    barely_visible = np.zeros((64, 64), dtype=np.uint8)
    barely_visible[30:34, 30:45] = 1

    centers, uncertainties = (
        render_script.sticker_mask_center_observations_to_frame(
            [clean, asymmetric, barely_visible],
            (0, 0, 64, 64),
        )
    )

    assert len(centers) == 3
    assert uncertainties[1] > uncertainties[0]
    assert uncertainties[2] > uncertainties[0]


def test_sticker_mask_center_hypothesis_score_uses_uncertainty(monkeypatch):
    monkeypatch.setattr(
        render_script,
        "projected_cube_diameter",
        lambda _candidate: 100.0,
    )

    def fake_matches(candidate, _rvec, _tvec, _max_error):
        uncertainty = candidate.sticker_mask_center_uncertainties[0]
        return (
            np.zeros((1, 2), dtype=np.float64),
            np.zeros((1, 3), dtype=np.float64),
            np.asarray([8.0], dtype=np.float64),
            np.asarray([uncertainty], dtype=np.float64),
        )

    monkeypatch.setattr(
        render_script,
        "matched_sticker_mask_center_correspondences",
        fake_matches,
    )
    common = dict(
        overall_reprojection_error=1.0,
        reprojection_error=1.0,
        sticker_mask_centers=(np.asarray((20.0, 20.0)),),
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )
    clean = SimpleNamespace(
        **common,
        sticker_mask_center_uncertainties=(1.0,),
    )
    uncertain = SimpleNamespace(
        **common,
        sticker_mask_center_uncertainties=(8.0,),
    )

    assert (
        render_script.sticker_mask_center_hypothesis_score(uncertain)
        < render_script.sticker_mask_center_hypothesis_score(clean)
    )


def test_sticker_wireframe_score_prefers_the_pose_matching_mask_boundaries():
    object_points = render_script.canonical_sticker_object_points("corner")
    image_size = (256, 256)
    camera_matrix = render_script.default_camera_matrix(image_size, 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2, 3),
        sticker_type="corner",
        average_confidence=0.95,
    )
    candidate = render_script.solve_sticker_pnp(
        image_points.reshape(-1, 2),
        np.asarray([0, 1, 2, 1], dtype=np.int32),
        match,
        image_size,
        1.2,
    )
    assert candidate is not None
    contours = tuple(
        polygon
        for _identity, _depth, polygon in render_script.projected_sticker_faces(
            candidate,
            rvec,
            tvec,
        )
    )
    candidate = replace(candidate, sticker_contours=contours)

    matching_score, matching_count = render_script.sticker_mask_wireframe_score(
        candidate,
        rvec,
        tvec,
    )
    shifted_score, shifted_count = render_script.sticker_mask_wireframe_score(
        candidate,
        rvec + np.asarray([[0.02], [0.0], [0.0]]),
        tvec,
    )

    assert matching_count > 0
    assert shifted_count > 0
    assert matching_score < shifted_score


def test_sticker_wireframe_score_tolerates_a_merged_mask_boundary():
    object_points = render_script.canonical_sticker_object_points("corner")
    image_size = (256, 256)
    camera_matrix = render_script.default_camera_matrix(image_size, 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    match = render_script.StickerKeypointMatch(0, (0, 1, 2, 3), "corner", 0.95)
    candidate = render_script.solve_sticker_pnp(
        image_points.reshape(-1, 2),
        np.asarray([0, 1, 2, 1], dtype=np.int32),
        match,
        image_size,
        1.2,
    )
    assert candidate is not None
    polygons = [
        polygon
        for _identity, _depth, polygon in render_script.projected_sticker_faces(
            candidate,
            rvec,
            tvec,
        )
    ]
    merged = np.concatenate(polygons[:2], axis=0).astype(np.float32)
    merged_hull = cv2.convexHull(merged.reshape(-1, 1, 2)).reshape(-1, 2)
    candidate = replace(candidate, sticker_contours=(merged_hull,))

    score, sample_count = render_script.sticker_mask_wireframe_score(
        candidate,
        rvec,
        tvec,
    )

    assert np.isfinite(score)
    assert sample_count > 0


def test_best_sticker_prefers_four_points_over_higher_confidence_three_points(monkeypatch):
    matches = (
        render_script.StickerKeypointMatch(0, (0, 1, 2), "corner", 0.999),
        render_script.StickerKeypointMatch(1, (0, 1, 2, 3), "edge", 0.900),
    )
    selected = []

    def fake_solve(_points, _classes, match, _image_size, _focal_scale, *_args):
        selected.append(match)
        return SimpleNamespace(
            match=match,
            overall_reprojection_error=1.0,
            reprojection_error=0.0,
            sticker_average_confidence=match.average_confidence,
            assignment_image_points=None,
            assignment_projected_points=None,
            assignment_object_points=None,
        )

    monkeypatch.setattr(render_script, "solve_sticker_pnp", fake_solve)
    result = render_script.best_sticker_pnp_candidate(
        np.zeros((4, 2), dtype=np.float32),
        np.zeros((4,), dtype=np.int32),
        matches,
        (128, 128),
        1.2,
    )

    assert result.match is matches[1]
    assert selected == [matches[1]]


def test_best_sticker_tries_all_four_point_stickers_and_chooses_lowest_overall_error(
    monkeypatch,
):
    matches = (
        render_script.StickerKeypointMatch(0, (0, 1, 2, 3), "corner", 0.99),
        render_script.StickerKeypointMatch(1, (4, 5, 6, 7), "edge", 0.90),
        render_script.StickerKeypointMatch(2, (8, 9, 10, 11), "center", 0.80),
        render_script.StickerKeypointMatch(3, (12, 13, 14, 15), "corner", 0.70),
        render_script.StickerKeypointMatch(4, (8, 9, 10), "center", 1.00),
    )
    selected = []

    def fake_solve(_points, _classes, match, _image_size, _focal_scale, *_args):
        selected.append(match.mask_index)
        return SimpleNamespace(
            match=match,
            overall_reprojection_error={0: 12.0, 1: 4.0, 2: 8.0}[match.mask_index],
            reprojection_error=0.0,
            sticker_average_confidence=match.average_confidence,
            assignment_image_points=None,
            assignment_projected_points=None,
            assignment_object_points=None,
        )

    monkeypatch.setattr(render_script, "solve_sticker_pnp", fake_solve)
    result = render_script.best_sticker_pnp_candidate(
        np.zeros((16, 2), dtype=np.float32),
        np.zeros((16,), dtype=np.int32),
        matches,
        (128, 128),
        1.2,
    )

    assert result.match is matches[1]
    assert selected == [0, 1, 2]


def test_best_sticker_candidates_retains_ranked_alternatives(monkeypatch):
    matches = tuple(
        render_script.StickerKeypointMatch(
            index,
            (index * 4, index * 4 + 1, index * 4 + 2, index * 4 + 3),
            "corner",
            0.9,
        )
        for index in range(3)
    )

    def fake_solve(_points, _classes, match, _image_size, _focal_scale, *_args):
        return SimpleNamespace(
            match=match,
            overall_reprojection_error=float(3 - match.mask_index),
            reprojection_error=0.0,
            sticker_average_confidence=match.average_confidence,
        )

    monkeypatch.setattr(render_script, "solve_sticker_pnp", fake_solve)
    candidates = render_script.best_sticker_pnp_candidates(
        np.zeros((12, 2), dtype=np.float32),
        np.zeros((12,), dtype=np.int32),
        matches,
        (128, 128),
        1.2,
    )

    assert [candidate.match.mask_index for candidate in candidates] == [2, 1, 0]


def test_pose_hypothesis_buffer_keeps_current_set_separate_from_history():
    buffer = render_script.PoseHypothesisBuffer(max_frames=2)
    first = SimpleNamespace(rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)))
    second = SimpleNamespace(rvec=np.ones((3, 1)), tvec=np.ones((3, 1)))
    third = SimpleNamespace(rvec=np.full((3, 1), 2.0), tvec=np.full((3, 1), 2.0))

    buffer.append((first,), 10)
    buffer.append((second,), 11)
    assert buffer.previous_candidates() == (first,)
    buffer.append((third,), 12)

    assert [frame.frame_index for frame in buffer.frames] == [11, 12]
    assert buffer.previous_candidates() == (second,)


def test_pose_hypothesis_buffer_exposes_only_best_candidate_for_fast_graph():
    buffer = render_script.PoseHypothesisBuffer(max_frames=2)
    first = SimpleNamespace()
    alternate = SimpleNamespace()
    second = SimpleNamespace()
    buffer.append((first, alternate), 10)
    buffer.append((second,), 11)

    fast_frames = buffer.best_candidate_frames()

    assert [frame.candidates for frame in fast_frames] == [(first,), (second,)]


def test_hypothesis_selection_prefers_continuity_only_for_close_visual_scores():
    def candidate(error, angle):
        return SimpleNamespace(
            overall_reprojection_error=error,
            reprojection_error=0.0,
            rvec=np.asarray([[0.0], [0.0], [angle]], dtype=np.float64),
            tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
        )

    near = candidate(10.0, 0.01)
    far = candidate(10.5, 1.0)
    selected = render_script.select_pose_hypothesis(
        (far, near),
        (near.rvec, near.tvec),
    )

    assert selected is near


def test_hypothesis_selection_keeps_much_better_visual_candidate_on_turn():
    good_turn = SimpleNamespace(
        overall_reprojection_error=2.0,
        reprojection_error=0.0,
        rvec=np.asarray([[0.0], [0.0], [1.0]], dtype=np.float64),
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )
    stale = SimpleNamespace(
        overall_reprojection_error=8.0,
        reprojection_error=0.0,
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )

    selected = render_script.select_pose_hypothesis(
        (good_turn, stale),
        (stale.rvec, stale.tvec),
    )

    assert selected is good_turn


def test_pose_graph_prefers_path_with_bounded_derivatives():
    def candidate(position, error=0.0):
        return SimpleNamespace(
            overall_reprojection_error=error,
            reprojection_error=0.0,
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.asarray([[position], [0.0], [200.0]], dtype=np.float64),
        )

    frames = (
        render_script.PoseHypothesisFrame(0, (candidate(0.0),)),
        render_script.PoseHypothesisFrame(
            1,
            (candidate(1.0), candidate(50.0)),
        ),
        render_script.PoseHypothesisFrame(2, (candidate(2.0),)),
    )
    selection = render_script.select_pose_graph_path(
        frames,
        render_script.PoseMotionFilterConfig(
            dt_seconds=1.0,
            max_acceleration_units_s2=5.0,
            max_jerk_units_s3=5.0,
            max_angular_acceleration_rad_s2=5.0,
            max_angular_jerk_rad_s3=5.0,
        ),
    )

    assert selection is not None
    assert selection.path[1].tvec[0, 0] == 1.0
    assert selection.violates_limits is False


def test_pose_graph_does_not_use_jerk_as_a_constraint():
    def candidate(position):
        return SimpleNamespace(
            overall_reprojection_error=0.0,
            reprojection_error=0.0,
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.asarray([[position], [0.0], [200.0]], dtype=np.float64),
        )

    frames = tuple(
        render_script.PoseHypothesisFrame(index, (candidate(position),))
        for index, position in enumerate((0.0, 0.0, 0.0, 1.0))
    )
    selection = render_script.select_pose_graph_path(
        frames,
        render_script.PoseMotionFilterConfig(
            dt_seconds=1.0,
            max_acceleration_units_s2=2.0,
            max_jerk_units_s3=0.001,
            max_angular_acceleration_rad_s2=5.0,
            max_angular_jerk_rad_s3=0.001,
        ),
    )

    assert selection is not None
    assert selection.violates_limits is False


def test_pose_assignment_keeps_selected_sticker_correspondences_fixed():
    image_size = (256, 256)
    camera_matrix = render_script.default_camera_matrix(image_size, 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [240.0]], dtype=np.float64)
    object_points, class_ids = render_script.visible_cube_lattice_points(rvec, tvec)
    observed, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    observed = observed.reshape(-1, 2)

    _score, rows, columns, _projected, visible_objects, _visible_classes = (
        render_script.evaluate_pose_hypothesis(
            rvec,
            tvec,
            observed,
            class_ids,
            image_size,
            camera_matrix,
            dist_coeffs,
            fixed_observed_indices=np.asarray([0, 1, 2, 3], dtype=np.int32),
            fixed_object_points=object_points[:4],
        )
    )

    assert len(rows) == len(observed)
    np.testing.assert_array_equal(rows[:4], np.asarray([0, 1, 2, 3]))
    np.testing.assert_allclose(visible_objects[columns[:4]], object_points[:4])


def test_naive_assignment_allows_multiple_observations_to_choose_one_target():
    observed = np.asarray([(10.0, 10.0), (10.5, 10.25), (80.0, 80.0)])
    projected = np.asarray([(10.0, 10.0), (40.0, 40.0), (80.0, 80.0)])

    score, rows, columns = render_script.observed_keypoint_reprojection_nearest_assignment(
        observed,
        projected,
        image_size=(100, 100),
    )

    assert np.isfinite(score)
    np.testing.assert_array_equal(rows, np.asarray([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(columns, np.asarray([0, 0, 2], dtype=np.int32))


def test_class_aware_assignment_rejects_known_class_mismatches():
    observed = np.asarray([(10.0, 10.0)])
    projected = np.asarray([(10.0, 10.0)])

    score, rows, columns = (
        render_script.observed_keypoint_reprojection_nearest_assignment(
            observed,
            projected,
            image_size=(100, 100),
            observed_class_ids=np.asarray([0], dtype=np.int32),
            projected_class_ids=np.asarray([1], dtype=np.int32),
        )
    )

    assert np.isinf(score)
    assert len(rows) == 0
    assert len(columns) == 0


def test_pivot_consistency_penalizes_a_swapped_free_assignment():
    image_size = (640, 480)
    camera_matrix = render_script.default_camera_matrix(image_size, 1.2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    true_rvec = np.asarray([[0.22], [-0.16], [0.10]], dtype=np.float64)
    true_tvec = np.asarray([[5.0], [-4.0], [250.0]], dtype=np.float64)
    object_points, class_ids = render_script.visible_cube_lattice_points(
        true_rvec,
        true_tvec,
    )
    observed, _ = cv2.projectPoints(
        object_points,
        true_rvec,
        true_tvec,
        camera_matrix,
        dist_coeffs,
    )
    observed = observed.reshape(-1, 2)
    candidate_rvec = true_rvec + np.asarray([[0.03], [-0.02], [0.025]])
    candidate_tvec = true_tvec + np.asarray([[2.0], [-1.5], [1.0]])
    projected, _ = cv2.projectPoints(
        object_points,
        candidate_rvec,
        candidate_tvec,
        camera_matrix,
        dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    rows = np.arange(len(observed), dtype=np.int32)
    correct_columns = rows.copy()
    swapped_columns = correct_columns.copy()
    swapped_columns[6], swapped_columns[7] = swapped_columns[7], swapped_columns[6]
    fixed_rows = np.arange(4, dtype=np.int32)

    correct_score = render_script.pivot_residual_consistency_score(
        observed,
        projected,
        object_points,
        rows,
        correct_columns,
        fixed_rows,
        camera_matrix,
        dist_coeffs,
        candidate_rvec,
        candidate_tvec,
    )
    swapped_score = render_script.pivot_residual_consistency_score(
        observed,
        projected,
        object_points,
        rows,
        swapped_columns,
        fixed_rows,
        camera_matrix,
        dist_coeffs,
        candidate_rvec,
        candidate_tvec,
    )
    class_aware_swapped_score = render_script.pivot_residual_consistency_score(
        observed,
        projected,
        object_points,
        rows,
        swapped_columns,
        fixed_rows,
        camera_matrix,
        dist_coeffs,
        candidate_rvec,
        candidate_tvec,
        observed_class_ids=class_ids,
        projected_class_ids=class_ids,
        use_class_aware_assignment=True,
    )

    assert swapped_score > correct_score
    assert np.isinf(class_aware_swapped_score)


def test_assignment_residual_direction_statistics_detects_common_direction():
    observed = np.asarray([(4.0, 1.0), (12.0, 2.0), (8.0, 8.0)])
    projected = observed - np.asarray((3.0, 2.0))

    consistency, mean_residual = (
        render_script.assignment_residual_direction_statistics(observed, projected)
    )

    assert consistency > 0.99
    np.testing.assert_allclose(mean_residual, np.asarray((3.0, 2.0)))


