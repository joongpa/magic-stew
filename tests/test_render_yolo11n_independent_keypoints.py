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


STANDARD_FACE_COLOR_NAMES = {
    "U": "white", "D": "yellow", "F": "green", "B": "blue", "R": "red", "L": "orange",
}


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


def _solve_sticker_pnp(*args, **kwargs):
    candidates = render_script.solve_sticker_pnp_candidates(*args, **kwargs)
    return candidates[0] if candidates else None


def _best_sticker_pnp_candidate(*args, **kwargs):
    candidates = render_script.best_sticker_pnp_candidates(*args, **kwargs)
    return candidates[0] if candidates else None


def test_final_video_parser_option_is_not_color_specific():
    args = render_script.build_parser().parse_args(
        ["--initial-scramble", "",
            "--source",
            "input.mp4",
            "--save-final-video",
            "runs/final.mp4",
        ]
    )

    assert args.save_final_video == Path("runs/final.mp4")


def test_legacy_colored_video_parser_option_remains_an_alias():
    args = render_script.build_parser().parse_args(
        ["--initial-scramble", "",
            "--source",
            "input.mp4",
            "--save-colored-video",
            "runs/final.mp4",
        ]
    )

    assert args.save_final_video == Path("runs/final.mp4")


def test_move_accumulator_does_not_reemit_same_state_after_neutral_frame():
    rotation = render_script.CubeLayerRotation(0, 1, 90.0, 1)
    candidate = SimpleNamespace(
        layer_rotations=(rotation,),
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )
    accumulator = render_script.MoveAccumulator()

    accumulator.observe(candidate)
    first_move_count = len(accumulator.moves)
    accumulator.observe(None)
    accumulator.observe(candidate)

    assert first_move_count == 1
    assert len(accumulator.moves) == 1
    assert accumulator.displacement_mm == 0.0


def test_move_accumulator_displays_core_displacement_between_good_poses():
    rotation = render_script.CubeLayerRotation(0, 1, 90.0, 1)
    first = SimpleNamespace(
        layer_rotations=(rotation,),
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )
    second = SimpleNamespace(
        layer_rotations=(rotation,),
        tvec=np.asarray([[12.0], [0.0], [200.0]], dtype=np.float64),
    )
    accumulator = render_script.MoveAccumulator()

    accumulator.observe(first)
    accumulator.observe(second)

    assert len(accumulator.moves) == 1
    assert accumulator.displacement_mm == 12.0


def test_move_accumulator_displays_rejected_pose_displacement_without_accepting_it():
    rotation = render_script.CubeLayerRotation(0, 1, 90.0, 1)
    first = SimpleNamespace(
        layer_rotations=(rotation,),
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )
    rejected = SimpleNamespace(
        layer_rotations=(rotation,),
        tvec=np.asarray([[35.0], [0.0], [200.0]], dtype=np.float64),
    )
    accumulator = render_script.MoveAccumulator()

    accumulator.observe(first)
    accumulator.observe(None, displacement_candidate=rejected)

    assert accumulator.current_fit == "rejected"
    assert accumulator.displacement_mm == 35.0
    assert len(accumulator.moves) == 1


def test_move_accumulator_ignores_whole_cube_rotations_as_state_changes():
    face_rotation = render_script.CubeLayerRotation(0, 1, 90.0, 1)
    whole_cube_rotation = render_script.CubeLayerRotation(0, 1, 90.0, 3)
    face_candidate = SimpleNamespace(
        layer_rotations=(face_rotation,),
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )
    whole_cube_candidate = SimpleNamespace(
        layer_rotations=(whole_cube_rotation,),
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )
    accumulator = render_script.MoveAccumulator()

    accumulator.observe(face_candidate)
    accumulator.observe(whole_cube_candidate)
    accumulator.observe(face_candidate)

    assert len(accumulator.moves) == 1


def test_move_accumulator_snaps_layer_turn_past_45_degrees():
    partial_rotation = render_script.CubeLayerRotation(0, 1, 84.0, 1)
    partial_candidate = SimpleNamespace(
        layer_rotations=(partial_rotation,),
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )
    accumulator = render_script.MoveAccumulator()

    accumulator.observe(partial_candidate)

    assert accumulator.moves == ["R'"]


def test_move_accumulator_keeps_turn_at_or_below_45_degrees_uncommitted():
    partial_rotation = render_script.CubeLayerRotation(0, 1, 45.0, 1)
    partial_candidate = SimpleNamespace(
        layer_rotations=(partial_rotation,),
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )
    accumulator = render_script.MoveAccumulator()

    accumulator.observe(partial_candidate)

    assert accumulator.moves == []


def test_move_accumulator_applies_completed_moves_to_displayed_state():
    initial_state = render_script.CubeState.solved()
    identities = render_script.canonical_sticker_identities()
    initial_colors = {
        identity.label: initial_state.color_at(
            (identity.cubie_index, identity.axis, identity.side)
        )
        for identity in identities
    }
    rotation = render_script.CubeLayerRotation(0, 1, 90.0, 1)
    candidate = SimpleNamespace(
        layer_rotations=(rotation,),
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )
    accumulator = render_script.MoveAccumulator()

    accumulator.observe(candidate, sticker_color_names=initial_colors)

    # Colors are keyed by persistent physical identity, so applying a move
    # changes the identity's projected location, not its color.
    assert accumulator.current_color_names() == initial_colors


def test_move_accumulator_display_colors_are_read_before_current_turn_commit():
    identities = render_script.canonical_sticker_identities()
    initial_colors = {
        identity.label: identity.label
        for identity in identities
    }
    rotation = render_script.CubeLayerRotation(0, 1, 90.0, 1)
    candidate = SimpleNamespace(
        layer_rotations=(rotation,),
        cube_geometry_state=render_script.CubeGeometryState.identity(),
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )
    accumulator = render_script.MoveAccumulator()

    colors_for_current_geometry = accumulator.current_color_names(initial_colors)
    accumulator.observe(candidate, sticker_color_names=initial_colors)

    assert colors_for_current_geometry == initial_colors
    assert (
        accumulator._geometry_state.signature()
        == render_script.CubeGeometryState.identity()
        .apply_move("R'")
        .signature()
    )


def test_stateful_move_accumulator_commits_repeated_moves_from_matching_models():
    rotation = render_script.CubeLayerRotation(0, 1, 90.0, 1)
    first_state = render_script.CubeGeometryState.identity()
    second_state = first_state.apply_move("R'")
    first = SimpleNamespace(
        layer_rotations=(rotation,),
        cube_geometry_state=first_state,
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )
    second = SimpleNamespace(
        layer_rotations=(rotation,),
        cube_geometry_state=second_state,
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )
    accumulator = render_script.MoveAccumulator()

    accumulator.observe(first)
    accumulator.observe(second)

    assert accumulator.moves == ["R'", "R'"]
    assert accumulator._geometry_state.signature() == second_state.apply_move("R'").signature()


def test_stateful_move_accumulator_rejects_a_stale_model_update():
    rotation = render_script.CubeLayerRotation(0, 1, 90.0, 1)
    identity_state = render_script.CubeGeometryState.identity()
    candidate = SimpleNamespace(
        layer_rotations=(rotation,),
        cube_geometry_state=identity_state,
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )
    accumulator = render_script.MoveAccumulator()

    accumulator.observe(candidate)
    moves_before = list(accumulator.moves)
    state_before = accumulator._geometry_state.signature()
    accumulator.observe(candidate)

    assert accumulator.current_fit == "state-mismatch"
    assert accumulator.moves == moves_before
    assert accumulator._geometry_state.signature() == state_before


def test_initial_scramble_parser_accepts_standard_notation_and_alias():
    args = render_script.build_parser().parse_args(
        ["--source", "input.mp4", "--known-scramble", "R U2 F' Rw2 r'"]
    )

    assert args.initial_scramble == ("R", "U2", "F'", "Rw2", "r'")


def test_scramble_recognition_can_be_disabled():
    args = render_script.build_parser().parse_args(
        ["--initial-scramble", "", "--source", "input.mp4", "--no-recognize-scramble"]
    )

    assert args.recognize_scramble is False

    enabled = render_script.build_parser().parse_args(
        ["--initial-scramble", "", "--source", "input.mp4", "--recognize-scramble"]
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


def test_turn_candidate_log_appends_one_complete_record_per_frame(tmp_path):
    path = tmp_path / "turn_candidate_scores.jsonl"
    color_likelihoods = np.zeros((54, 1), dtype=np.float64)
    color_likelihoods[:2, 0] = (0.8, 0.2)
    candidate = SimpleNamespace(
        turn_candidate_scores=(("R+3", 0.12, 7, 0.08, 0.04),),
        turn_candidate_diagnostics=(
            {
                "label": "R+3",
                "global_mean_error": 0.12,
                "global_robust_error": 0.16,
                "color_error": 0.04,
                "color_match_count": 1,
                "layer_rotations": [],
            },
        ),
        turn_fit_evaluated=True,
        turn_fit_unexplained_keypoints=2,
        turn_fit_sticker_score=0.03,
        turn_fit_color_score=0.05,
        turn_color_likelihoods=color_likelihoods,
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
    assert records[0]["candidates"][0]["color_error"] == 0.04
    assert len(records[0]["turn_color_identity_labels"]) == 54
    assert records[0]["turn_color_likelihoods"][:2] == [[0.8], [0.2]]
    assert records[0]["search_pose"]["tvec"] == [[0.0], [0.0], [200.0]]
    assert records[0]["search_pose"]["assignment_image_points"] == [[1.0, 2.0]]
    assert records[1]["has_pose"] is False
    assert records[1]["candidate_count"] == 0


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


def test_turn_search_defaults_to_all_six_layer_sides_and_half_period():
    assert render_script.PNP_LAYER_MISALIGNMENT_MIN_DEGREES == -90.0
    assert render_script.PNP_LAYER_MISALIGNMENT_MAX_DEGREES == 90.0
    assert render_script.PNP_LAYER_MISALIGNMENT_STEP_DEGREES == 6.0
    angles = render_script.turn_search_angles()
    assert len(angles) == 31
    assert angles[0] == -90.0
    assert angles[-1] == 90.0


def test_turn_color_weight_matches_combined_geometry_weight():
    assert render_script.PNP_TURN_COLOR_WEIGHT == pytest.approx(
        1.0 + render_script.PNP_TURN_MASK_CENTER_WEIGHT
    )


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


def test_turn_overlay_renders_selected_single_layer_in_red():
    image_size = (256, 256)
    rotation = render_script.CubeLayerRotation(
        axis=2,
        side=1,
        angle_degrees=30.0,
        layer_count=1,
    )
    candidate = SimpleNamespace(
        layer_rotations=(rotation,),
        orientation_anchor=render_script.orientation_anchor_for_candidate(
            SimpleNamespace(layer_rotations=(rotation,))
        ),
        camera_matrix=render_script.default_camera_matrix(image_size, 1.2),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
    )
    image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    assert render_script.draw_projected_cube_pose(
        image,
        candidate,
        np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64),
        np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64),
        (0, 0, 0),
        2,
    )
    assert np.any(np.all(image == np.asarray((0, 0, 255), dtype=np.uint8), axis=2))


def test_gpu_sticker_center_scoring_uses_nearest_faces_without_hungarian(monkeypatch):
    torch = pytest.importorskip("torch")
    candidate = SimpleNamespace(
        sticker_mask_centers=(
            np.asarray((0.0, 0.0), dtype=np.float64),
            np.asarray((1.0, 0.0), dtype=np.float64),
        ),
        sticker_mask_center_uncertainties=(1.0, 1.0),
    )
    projected_face_centers = torch.as_tensor(
        [[[0.0, 0.0]]],
        dtype=torch.float64,
    )
    visible_faces = torch.ones((1, 1), dtype=torch.bool)

    monkeypatch.setattr(
        render_script,
        "linear_sum_assignment",
        lambda *_args, **_kwargs: pytest.fail("GPU center scoring should not use Hungarian matching"),
    )
    means, robust, match_counts = render_script._gpu_sticker_center_fit_metrics(
        candidate,
        torch,
        projected_face_centers,
        visible_faces,
        scale=100.0,
        max_error=10.0,
    )

    assert match_counts.tolist() == [2]
    assert means.shape == (1,)
    assert robust.shape == (1,)
    assert np.isfinite(means[0])
    assert np.isfinite(robust[0])


def test_turn_color_likelihood_uses_global_profile_and_identity_prior():
    profiles, _grouped = render_script.build_global_color_profiles_from_samples([])
    identities = render_script.canonical_sticker_identities()
    red_index = render_script.GLOBAL_COLOR_NAMES.index("red")
    blue_index = render_script.GLOBAL_COLOR_NAMES.index("blue")
    red_prior = np.zeros(len(render_script.GLOBAL_COLOR_NAMES), dtype=np.float64)
    red_prior[red_index] = 1.0
    blue_prior = np.zeros(len(render_script.GLOBAL_COLOR_NAMES), dtype=np.float64)
    blue_prior[blue_index] = 1.0
    candidate = SimpleNamespace(
        sticker_mask_centers=(np.asarray((128.0, 128.0)),),
        sticker_mask_colors=(
            np.asarray(render_script.PURE_COLOR_RGB["red"], dtype=np.uint8),
        ),
    )

    likelihoods = render_script.turn_color_predictive_likelihoods(
        candidate,
        {
            identities[0].label: red_prior,
            identities[1].label: blue_prior,
        },
        profiles,
    )

    assert likelihoods is not None
    assert likelihoods.shape == (54, 1)
    assert likelihoods[0, 0] > likelihoods[1, 0]


def test_projected_sticker_identity_moves_with_layer_geometry():
    common = dict(
        camera_matrix=render_script.default_camera_matrix((256, 256), 1.2),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
    )
    rvec = np.asarray([[0.12], [-0.18], [0.08]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)
    static = SimpleNamespace(layer_rotations=(), **common)
    turned = SimpleNamespace(
        layer_rotations=(
            render_script.CubeLayerRotation(axis=0, side=-1, angle_degrees=90.0),
        ),
        **common,
    )
    static_faces = {
        identity.label: polygon
        for identity, _depth, polygon in render_script.projected_sticker_faces(
            static,
            rvec,
            tvec,
        )
    }
    turned_faces = {
        identity.label: polygon
        for identity, _depth, polygon in render_script.projected_sticker_faces(
            turned,
            rvec,
            tvec,
        )
    }
    moved_labels = set(static_faces) & set(turned_faces)

    assert moved_labels
    assert any(
        not np.allclose(static_faces[label], turned_faces[label])
        for label in moved_labels
    )


def test_gpu_turn_geometry_remaps_persistent_identities_after_state_move():
    _triangles, _triangle_centers, _faces, _centers, _cubie_centers, _normals, canonical = (
        render_script._turn_gpu_static_geometry()
    )
    state = render_script.CubeGeometryState.identity().apply_move("R'")
    remapped = render_script._turn_gpu_identity_indices_for_state(state, canonical)

    assert remapped.shape == canonical.shape
    assert not np.array_equal(remapped, canonical)
    assert sorted(remapped.tolist()) == sorted(canonical.tolist())


def test_gpu_turn_color_score_uses_identity_likelihoods_without_cpu_matching():
    torch = pytest.importorskip("torch")
    candidate = SimpleNamespace(
        sticker_mask_centers=(np.asarray((0.0, 0.0)),),
        sticker_mask_colors=(np.asarray((255, 0, 0), dtype=np.uint8),),
        sticker_mask_center_uncertainties=(1.0,),
    )
    likelihoods = np.full((54, 1), 0.01, dtype=np.float64)
    likelihoods[7, 0] = 0.90
    color_error, match_count = render_script._gpu_sticker_color_fit_metrics(
        candidate,
        torch,
        torch.as_tensor([[[0.0, 0.0], [20.0, 0.0]]], dtype=torch.float64),
        torch.as_tensor(
            [[
                [[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]],
                [[15.0, -5.0], [25.0, -5.0], [25.0, 5.0], [15.0, 5.0]],
            ]],
            dtype=torch.float64,
        ),
        torch.ones((1, 2), dtype=torch.bool),
        np.asarray((7, 8), dtype=np.int64),
        scale=100.0,
        max_error=10.0,
        color_likelihoods=likelihoods,
    )

    np.testing.assert_allclose(color_error, (-np.log(0.90),))
    assert match_count.tolist() == [1]


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


def test_sticker_mask_heatmap_can_hide_fill_without_hiding_centroid():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
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
        show_sticker_mask=False,
    )

    assert np.all(
        rendered[16, 16] == render_script.STICKER_MASK_CENTROID_COLOR_BGR
    )
    assert np.all(rendered[10, 10] == image[10, 10])


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

    render_script.recover_cached_pose_symmetries(
        [record, replace(record, frame_index=1)],
        render_script.StickerColorTrace.create(),
        refine_pose=True,
        recover_orientation=False,
        recognize_moves=True,
    )

    assert calls == [False, True, False, True]


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


def test_fitted_rotation_notation_uses_white_top_green_front_frame():
    assert render_script.cube_rotation_to_notation(
        render_script.CubeLayerRotation(0, 1, -90.0, 1)
    ) == "R"
    assert render_script.cube_rotation_to_notation(
        render_script.CubeLayerRotation(1, -1, -90.0, 1)
    ) == "F"
    assert render_script.cube_rotation_to_notation(
        render_script.CubeLayerRotation(2, 1, -90.0, 3)
    ) == "y"


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
    candidate = _solve_sticker_pnp(
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
    candidate = _solve_sticker_pnp(
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


def test_cached_pose_symmetry_recovery_uses_color_matching_after_continuity(capsys):
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
    candidate = _solve_sticker_pnp(
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
                STANDARD_FACE_COLOR_NAMES[identity.face]
            ],
            dtype=np.uint8,
        )
        for identity in trace.identities
    }
    known_color_names = {
        identity.label: STANDARD_FACE_COLOR_NAMES[identity.face]
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
    # The first visible frame establishes the color-based whole-cube frame.
    # Subsequent frames carry the same center-to-axis mapping forward.
    trace.observe(0, frame_one)
    trace.observe(1, frame_one)

    alternate_candidate = render_script.transform_candidate_by_symmetry(
        candidate,
        render_script.CUBE_SYMMETRY_MATRICES[1],
    )
    records = [
        SimpleNamespace(frame_index=0, pose_candidate=candidate),
        SimpleNamespace(frame_index=1, pose_candidate=alternate_candidate),
    ]
    render_script.recover_cached_pose_symmetries(
        records,
        trace,
        debug=True,
        known_initial_colors=known_color_names,
    )
    debug_output = capsys.readouterr().out
    assert "pose symmetry frame 1" in debug_output
    assert "tested orientations" in debug_output
    assert "symmetry=07" in debug_output

    recovered = records[1].pose_candidate
    expected_color_orientation = render_script.transform_rvec_by_symmetry(
        candidate.rvec,
        symmetry,
    )
    assert render_script.rotation_symmetry_distance(
        recovered.rvec,
        expected_color_orientation,
    ) < 1e-6


def test_cached_pose_recovery_rejects_core_displacement_with_motion_filter():
    def make_candidate(x_position):
        return render_script.StickerPnPCandidate(
            sticker_type="corner",
            sticker_average_confidence=0.9,
            image_points=np.zeros((4, 2), dtype=np.float32),
            object_points=np.zeros((4, 3), dtype=np.float32),
            projected_sticker_points=np.zeros((4, 2), dtype=np.float32),
            camera_matrix=np.eye(3, dtype=np.float64),
            dist_coeffs=np.zeros((4, 1), dtype=np.float64),
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.asarray([[x_position], [0.0], [200.0]], dtype=np.float64),
            reprojection_error=0.0,
            overall_reprojection_error=0.0,
        )

    first = make_candidate(0.0)
    second = make_candidate(41.0)
    third = make_candidate(82.0)
    records = [
        SimpleNamespace(frame_index=0, pose_candidate=first, pose_candidates=(first,)),
        SimpleNamespace(frame_index=1, pose_candidate=second, pose_candidates=(second,)),
        SimpleNamespace(frame_index=2, pose_candidate=third, pose_candidates=(third,)),
    ]
    render_script.recover_cached_pose_symmetries(
        records,
        render_script.StickerColorTrace.create(),
        recover_orientation=False,
    )

    assert records[0].pose_candidate is not None
    assert records[1].pose_candidate is None
    assert records[1].pose_diagnostic_candidate is not None
    assert records[2].pose_candidate is not None


def test_cached_pose_recovery_selects_temporally_closest_candidate(monkeypatch):
    monkeypatch.setattr(
        render_script,
        "pose_reprojection_fit_metric",
        lambda _candidate: 0.0,
    )

    def make_candidate(x_position):
        return render_script.StickerPnPCandidate(
            sticker_type="corner",
            sticker_average_confidence=0.9,
            image_points=np.zeros((4, 2), dtype=np.float32),
            object_points=np.zeros((4, 3), dtype=np.float32),
            projected_sticker_points=np.zeros((4, 2), dtype=np.float32),
            camera_matrix=np.eye(3, dtype=np.float64),
            dist_coeffs=np.zeros((4, 1), dtype=np.float64),
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.asarray([[x_position], [0.0], [200.0]], dtype=np.float64),
            reprojection_error=0.0,
            overall_reprojection_error=0.0,
        )

    previous = make_candidate(0.0)
    visually_first = make_candidate(100.0)
    temporally_close = make_candidate(4.0)
    records = [
        SimpleNamespace(
            frame_index=0,
            pose_candidate=previous,
            pose_candidates=(previous,),
        ),
        SimpleNamespace(
            frame_index=1,
            pose_candidate=visually_first,
            pose_candidates=(visually_first, temporally_close),
        ),
    ]

    render_script.recover_cached_pose_symmetries(
        records,
        render_script.StickerColorTrace.create(),
        recover_orientation=False,
    )

    np.testing.assert_allclose(
        records[1].pose_candidate.tvec,
        temporally_close.tvec,
    )


def test_cached_pose_recovery_accepts_large_jump_with_close_next_frame(monkeypatch):
    # Isolate the temporal look-ahead behavior from the shared reprojection
    # acceptance gate tested elsewhere.
    monkeypatch.setattr(
        render_script,
        "pose_reprojection_fit_metric",
        lambda _candidate: 0.0,
    )

    def make_candidate(x_position):
        return render_script.StickerPnPCandidate(
            sticker_type="corner",
            sticker_average_confidence=0.9,
            image_points=np.zeros((4, 2), dtype=np.float32),
            object_points=np.zeros((4, 3), dtype=np.float32),
            projected_sticker_points=np.zeros((4, 2), dtype=np.float32),
            camera_matrix=np.eye(3, dtype=np.float64),
            dist_coeffs=np.zeros((4, 1), dtype=np.float64),
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.asarray([[x_position], [0.0], [200.0]], dtype=np.float64),
            reprojection_error=0.0,
            overall_reprojection_error=0.0,
        )

    first = make_candidate(0.0)
    second = make_candidate(41.0)
    third = make_candidate(61.0)
    records = [
        SimpleNamespace(frame_index=0, pose_candidate=first, pose_candidates=(first,)),
        SimpleNamespace(frame_index=1, pose_candidate=second, pose_candidates=(second,)),
        SimpleNamespace(frame_index=2, pose_candidate=third, pose_candidates=(third,)),
    ]
    render_script.recover_cached_pose_symmetries(
        records,
        render_script.StickerColorTrace.create(),
        recover_orientation=False,
    )

    assert records[0].pose_candidate is not None
    assert records[1].pose_candidate is not None
    assert records[2].pose_candidate is not None


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


def test_sticker_identity_overlay_maps_predictive_likelihood_to_identity():
    candidate = SimpleNamespace(
        layer_rotations=(),
        camera_matrix=render_script.default_camera_matrix((256, 256), 1.2),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
    )
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64)
    faces = render_script.projected_sticker_faces(candidate, rvec, tvec)
    identity, _depth, polygon = faces[0]
    likelihoods = np.zeros((54, 1), dtype=np.float64)
    identity_index = render_script.canonical_sticker_identities().index(identity)
    likelihoods[identity_index, 0] = 0.91
    candidate.sticker_mask_centers = (np.mean(polygon, axis=0),)
    candidate.turn_color_likelihoods = likelihoods

    values = render_script.projected_turn_color_likelihoods_by_identity(
        candidate,
        rvec,
        tvec,
    )

    assert values[identity.label] == pytest.approx(0.91)


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
    assert render_script.build_parser().parse_args(["--initial-scramble", "", "--source", "frame.png"]).sticker_mask_erosion == 0
    args = render_script.build_parser().parse_args(
        ["--initial-scramble", "", "--source", "frame.png", "--sticker-mask-erosion", "1"]
    )
    assert args.sticker_mask_erosion == 1
    debug_args = render_script.build_parser().parse_args(
        ["--initial-scramble", "", "--source", "frame.png", "--debug-color-masks"]
    )
    assert debug_args.debug_color_masks is True
    hidden_args = render_script.build_parser().parse_args(
        ["--initial-scramble", "",
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
    candidate = _solve_sticker_pnp(
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


def test_solve_sticker_pnp_candidates_preserves_all_three_point_roots(monkeypatch):
    frame_points = np.asarray(
        [[40.0, 40.0], [80.0, 40.0], [80.0, 80.0]],
        dtype=np.float32,
    )
    point_class_ids = np.asarray([2, 2, 2], dtype=np.int32)
    match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2),
        sticker_type="center",
        average_confidence=0.95,
    )
    rvec = np.zeros((3, 1), dtype=np.float64)
    first_tvec = np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64)
    second_tvec = np.asarray([[0.0], [0.0], [190.0]], dtype=np.float64)

    monkeypatch.setattr(
        render_script,
        "sticker_correspondence_object_indices",
        lambda *_args: ((0, 1, 2),),
    )
    monkeypatch.setattr(
        render_script,
        "canonical_sticker_is_camera_facing",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        render_script.cv2,
        "solvePnPGeneric",
        lambda *_args, **_kwargs: (
            True,
            (rvec.copy(), rvec.copy()),
            (first_tvec.copy(), second_tvec.copy()),
            None,
        ),
    )

    def fake_evaluate(_rvec, _tvec, observed, _class_ids, _image_size, _camera, _dist, *_args):
        count = len(observed)
        return (
            1.0,
            np.arange(count, dtype=np.int32),
            np.arange(count, dtype=np.int32),
            np.asarray(observed, dtype=np.float64),
            np.zeros((count, 3), dtype=np.float64),
            np.zeros((count,), dtype=np.int32),
        )

    monkeypatch.setattr(render_script, "evaluate_pose_hypothesis", fake_evaluate)

    candidates = render_script.solve_sticker_pnp_candidates(
        frame_points,
        point_class_ids,
        match,
        (128, 128),
        1.2,
    )

    assert len(candidates) == 2
    assert [float(candidate.tvec[2, 0]) for candidate in candidates] == [180.0, 190.0]


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
    candidate = _solve_sticker_pnp(
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
    refined = render_script.solve_pnp_from_low_error_assignments(
        candidate,
    )

    assert not np.isfinite(refined.mask_fit_error)
    assert refined.mask_boundary_sample_count == 0


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
    base_candidate = _solve_sticker_pnp(
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
        return (
            replace(
                base_candidate,
                sticker_type=match.sticker_type,
                sticker_average_confidence=match.average_confidence,
                mask_fit_error=float(match.mask_index),
            ),
        )

    def fake_assignment_solve(candidate, **kwargs):
        refinement_flags.append(kwargs)
        return candidate

    monkeypatch.setattr(
        render_script,
        "solve_sticker_pnp_candidates",
        fake_sticker_solve,
    )
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


def test_recover_misaligned_layer_updates_assignment_geometry(monkeypatch):
    import torch
    monkeypatch.setattr(render_script, "_turn_torch_scoring_device", lambda _device: torch.device("cpu"))
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
    displacements = np.linalg.norm(
        rotated_objects[layer_indices] - visible_objects[layer_indices],
        axis=1,
    )
    source_indices = layer_indices[np.argsort(displacements)[-3:]]
    extra_points, _ = cv2.projectPoints(
        rotated_objects[source_indices],
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    all_points = np.concatenate((sticker_image_points, extra_points.reshape(-1, 2)))
    all_classes = np.concatenate(
        (
            np.asarray([0, 1, 2, 1], dtype=np.int32),
            visible_classes[source_indices].astype(np.int32),
        )
    )
    match = render_script.StickerKeypointMatch(
        mask_index=0,
        keypoint_indices=(0, 1, 2, 3),
        sticker_type="corner",
        average_confidence=0.95,
    )
    candidate = _solve_sticker_pnp(
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
    limit = (
        render_script.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION
        * render_script.projected_cube_diameter(candidate)
    )

    full_trials = render_script._score_turn_configurations_on_gpu(
        candidate, limit, torch.device("cpu"), materialize_projections=True,
    )
    deferred_trials = render_script._score_turn_configurations_on_gpu(
        candidate, limit, torch.device("cpu"), materialize_projections=False,
    )
    assert full_trials and deferred_trials
    assert len(full_trials) == len(deferred_trials)
    for full, deferred in zip(full_trials, deferred_trials, strict=True):
        assert full["label"] == deferred["label"]
        assert full["rank"] == deferred["rank"]
        assert full["robust_error"] == deferred["robust_error"]
        np.testing.assert_array_equal(
            full["assignment_projected_points"],
            deferred["assignment_projected_points"].detach().cpu().numpy(),
        )

    recovered = render_script.recover_misaligned_cube_layers(candidate, limit)
    logged = render_script.recover_misaligned_cube_layers(
        candidate,
        limit,
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
    searched_rotations = [
        rotation
        for debug_candidate in logged.turn_candidate_debug
        for rotation in debug_candidate.layer_rotations
    ]
    searched_layer_sides = {
        (rotation.axis, rotation.side)
        for rotation in searched_rotations
    }
    assert 0 < len(searched_layer_sides) < 6
    assert {rotation.layer_count for rotation in searched_rotations} == {1, 2}
    assert min(rotation.angle_degrees for rotation in searched_rotations) == -90.0
    assert max(rotation.angle_degrees for rotation in searched_rotations) == 90.0
    assert all(
        rotation.angle_degrees % 6.0 == 0.0
        for rotation in searched_rotations
    )
    assert len(logged.turn_candidate_scores) > render_script.PNP_TURN_DEBUG_CANDIDATE_COUNT
    assert len(logged.turn_candidate_debug) == len(logged.turn_candidate_scores)


def test_turn_acceptance_compares_against_matching_static_baseline():
    static_inliers = {
        (0, 1, 1): 19,
        (0, 1, 2): 19,
    }
    tied_turn = {
        "is_baseline": False,
        "layer_axis": 0,
        "layer_side": 1,
        "layer_count": 1,
        "inlier_count": 19,
    }
    improving_turn = {
        **tied_turn,
        "inlier_count": 20,
    }

    assert not render_script.turn_trial_exceeds_static_baseline(
        tied_turn,
        static_inliers,
    )
    assert render_script.turn_trial_exceeds_static_baseline(
        improving_turn,
        static_inliers,
    )


def test_completed_turn_still_requires_more_global_inliers_than_static():
    static = {
        "is_baseline": True,
        "layer_axis": 0,
        "layer_side": 1,
        "layer_count": 1,
        "inlier_count": 19,
        "rank": (1.0, -19.0, 4.0, -19.0),
    }
    completed_turn = {
        "is_baseline": False,
        "layer_axis": 0,
        "layer_side": 1,
        "layer_count": 1,
        "inlier_count": 19,
        "layer_inlier_count": 3,
        "rank": (0.9, -19.0, 4.0, -19.0),
        "layer_rotations": (
            render_script.CubeLayerRotation(0, 1, 90.0, 1),
        ),
    }
    partial_turn = {
        **completed_turn,
        "rank": (0.8, -19.0, 4.0, -19.0),
        "layer_rotations": (
            render_script.CubeLayerRotation(0, 1, 44.0, 1),
        ),
    }

    baseline = {(0, 1, 1): static}
    assert not render_script.turn_trial_exceeds_static_baseline(
        completed_turn,
        baseline,
    )
    assert not render_script.turn_trial_exceeds_static_baseline(
        partial_turn,
        baseline,
    )


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
    candidate = _solve_sticker_pnp(
        full_image_points[[2, 3, 0]],
        np.asarray([2, 1, 0], dtype=np.int32),
        match,
        (256, 256),
        1.2,
        sticker_contour=full_image_points,
    )
    candidate_without_mask = _solve_sticker_pnp(
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


def test_best_sticker_prefers_four_points_over_higher_confidence_three_points(monkeypatch):
    matches = (
        render_script.StickerKeypointMatch(0, (0, 1, 2), "corner", 0.999),
        render_script.StickerKeypointMatch(1, (0, 1, 2, 3), "edge", 0.900),
    )
    selected = []

    def fake_solve(_points, _classes, match, _image_size, _focal_scale, *_args):
        selected.append(match)
        return (
            SimpleNamespace(
                match=match,
                overall_reprojection_error=1.0,
                reprojection_error=0.0,
                sticker_average_confidence=match.average_confidence,
                assignment_image_points=None,
                assignment_projected_points=None,
                assignment_object_points=None,
            ),
        )

    monkeypatch.setattr(render_script, "solve_sticker_pnp_candidates", fake_solve)
    result = _best_sticker_pnp_candidate(
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
        return (
            SimpleNamespace(
                match=match,
                overall_reprojection_error={0: 12.0, 1: 4.0, 2: 8.0}[match.mask_index],
                reprojection_error=0.0,
                sticker_average_confidence=match.average_confidence,
                assignment_image_points=None,
                assignment_projected_points=None,
                assignment_object_points=None,
            ),
        )

    monkeypatch.setattr(render_script, "solve_sticker_pnp_candidates", fake_solve)
    result = _best_sticker_pnp_candidate(
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
        return (
            SimpleNamespace(
                match=match,
                overall_reprojection_error=float(3 - match.mask_index),
                reprojection_error=0.0,
                sticker_average_confidence=match.average_confidence,
            ),
        )

    monkeypatch.setattr(render_script, "solve_sticker_pnp_candidates", fake_solve)
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

    assert buffer.previous_candidates() == (second,)




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


def test_hypothesis_selection_uses_translation_for_candidates_under_shared_fit_gate(
    monkeypatch,
):
    monkeypatch.setattr(
        render_script,
        "pose_reprojection_fit_metric",
        lambda candidate: float(candidate.fit),
    )

    def candidate(position, fit, visual):
        return SimpleNamespace(
            fit=fit,
            visual=visual,
            camera_matrix=np.eye(3, dtype=np.float64),
            dist_coeffs=np.zeros((4, 1), dtype=np.float64),
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.asarray([[position], [0.0], [200.0]], dtype=np.float64),
            reprojection_error=0.0,
        )

    monkeypatch.setattr(
        render_script,
        "pose_hypothesis_visual_score",
        lambda value: float(value.visual),
    )
    visually_best_but_far = candidate(49.0, 0.036, 0.01)
    temporally_best = candidate(11.0, 0.040, 1.0)
    rejected = candidate(1.0, 0.051, 0.0)
    previous = SimpleNamespace(
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64),
    )

    selected = render_script.select_pose_hypothesis(
        (visually_best_but_far, temporally_best, rejected),
        None,
        (previous,),
    )

    assert selected is temporally_best
    assert render_script.PNP_POSE_REPROJECTION_ERROR_THRESHOLD == pytest.approx(0.05)
    assert not render_script.PNP_POSE_REPROJECTION_ERROR_REJECTION_ENABLED
    assert render_script.pose_reprojection_error_is_acceptable(rejected)


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


def test_hypothesis_selection_rejects_core_displacement_over_40mm():
    def candidate(position, error):
        return SimpleNamespace(
            overall_reprojection_error=error,
            reprojection_error=0.0,
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.asarray([[position], [0.0], [200.0]], dtype=np.float64),
        )

    selected = render_script.select_pose_hypothesis(
        (candidate(41.0, 0.1), candidate(20.0, 1.0)),
        (np.zeros((3, 1)), np.asarray([[0.0], [0.0], [200.0]])),
    )

    assert selected.tvec[0, 0] == 20.0
    assert render_script.select_pose_hypothesis(
        (candidate(41.0, 0.1),),
        (np.zeros((3, 1)), np.asarray([[0.0], [0.0], [200.0]])),
    ) is None


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


def test_cached_pipeline_refines_then_builds_one_shared_profile_then_recovers(monkeypatch):
    events = []
    initial_trace = render_script.StickerColorTrace.create()
    prepared_trace = render_script.StickerColorTrace.create()
    oriented_trace = render_script.StickerColorTrace.create()
    final_trace = render_script.StickerColorTrace.create()
    profiles = {"marker": object()}
    observations = {"marker": []}
    cache = []
    args = SimpleNamespace(
        show_pnp=True, recognize_moves=True, display=False,
        sticker_mask_erosion=0, color_samples_per_sticker=17,
        color_min_samples_per_sticker=3, debug_color_masks=False,
        debug_pose_symmetry=False, recover_pose_symmetry=True,
        turn_candidate_log=None,
    )
    colors = {"a": "red"}

    def recover(records, trace, **kwargs):
        assert records is cache
        if kwargs.get("refine_pose"):
            events.append("refine")
            assert trace is initial_trace
            assert not kwargs["recover_orientation"]
            assert kwargs["color_samples_per_sticker"] == 17
            return prepared_trace
        assert kwargs["pose_prepared"]
        assert kwargs["global_profiles"] is profiles
        assert kwargs["known_initial_colors"] is colors
        if kwargs["recover_orientation"]:
            events.append("orientation")
            assert trace is prepared_trace
            assert not kwargs["recognize_moves"]
            return oriented_trace
        events.append("moves")
        assert trace is oriented_trace
        assert kwargs["recognize_moves"]
        return final_trace

    def build(trace):
        events.append("profiles")
        assert trace is initial_trace
        return profiles, observations

    monkeypatch.setattr(render_script, "recover_cached_pose_symmetries", recover)
    monkeypatch.setattr(render_script, "build_global_color_profiles", build)
    result = render_script.process_cached_pose_and_profiles(cache, initial_trace, args, colors, None)
    assert events == ["refine", "profiles", "orientation", "moves"]
    assert result == (final_trace, profiles, observations)


def test_initial_scramble_is_required_and_cpu_is_default():
    parser = render_script.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--source", "video.mp4"])
    args = parser.parse_args(["--source", "video.mp4", "--initial-scramble", ""])
    assert args.initial_scramble == ()
    assert args.device == "cpu"
    with pytest.raises(SystemExit):
        parser.parse_args(["--source", "video.mp4", "--initial-scramble", "", "--device", "mps"])


def test_debug_colors_remain_independent_with_known_scramble(monkeypatch):
    trace = render_script.StickerColorTrace.create()
    args = SimpleNamespace(show_classified_colors=True, debug_independent_color_posteriors=True)
    monkeypatch.setattr(render_script, "build_sticker_global_profile_likelihoods", lambda *_: {0: {"C00": {"red": 1.0}}})
    monkeypatch.setattr(render_script, "classify_sticker_colors_by_frame_global_profile", lambda *_: {0: {"C00": "red"}})
    _, by_frame, names, _ = render_script.classify_cached_colors(
        trace, args, {}, {"C00": "blue"}, image=True,
    )
    assert by_frame[0]["C00"] == names["C00"] == "red"


def test_initial_orientation_applies_to_frames_before_first_color_match(monkeypatch):
    trace = render_script.StickerColorTrace.create()
    labels = [identity.label for identity in trace.identities[:4]]
    sample = np.asarray([200, 50, 30], dtype=np.uint8)
    trace.observe(0, {labels[0]: sample})
    trace.observe(1, {label: sample for label in labels})
    candidate = render_script.StickerPnPCandidate(
        sticker_type="corner", sticker_average_confidence=0.9,
        image_points=np.zeros((4, 2), dtype=np.float32),
        object_points=np.zeros((4, 3), dtype=np.float32),
        projected_sticker_points=np.zeros((4, 2), dtype=np.float32),
        camera_matrix=np.eye(3), dist_coeffs=np.zeros((4, 1)),
        rvec=np.zeros((3, 1)), tvec=np.asarray([[0.0], [0.0], [200.0]]),
        reprojection_error=0.0, overall_reprojection_error=0.0,
    )
    records = [SimpleNamespace(frame_index=i, pose_candidate=candidate, pose_candidates=(candidate,)) for i in range(2)]
    known = {identity.label: "red" for identity in trace.identities}
    permutations = render_script._symmetry_observation_permutations(trace.identities)
    wanted = permutations[7][1]
    transformed = []

    def score(observations, permutation, *_args, **_kwargs):
        return (1.0 if permutation == wanted else 0.0, len(observations), 0, observations, {})

    def transform(pose, symmetry):
        transformed.append(symmetry.copy())
        return pose

    monkeypatch.setattr(render_script, "_score_symmetry_observations", score)
    monkeypatch.setattr(render_script, "transform_candidate_by_symmetry", transform)
    monkeypatch.setattr(render_script, "pose_reprojection_error_is_acceptable", lambda _: True)
    render_script.recover_cached_pose_symmetries(
        records, trace, pose_prepared=True, recover_orientation=True,
        known_initial_colors=known, global_profiles={}, require_orientation=True,
    )
    assert len(transformed) == 2
    for symmetry in transformed:
        np.testing.assert_array_equal(symmetry, permutations[7][0])


def test_batched_keypoint_statistics_match_reference_scoring():
    import torch

    distances = np.array([
        [1.0, 2.0, 5.0, np.inf],
        [0.2, 0.4, 0.8, 1.6],
        [1.0, 2.0, 3.0, 4.0],
        [np.inf, np.inf, np.inf, np.inf],
    ], dtype=np.float32)
    objects = np.zeros((4, 4, 3), dtype=np.float32)
    objects[:, :, 0] = np.array([27.5, 27.5, -27.5, -27.5])
    classes = np.array([[0, 1, 2, 0], [0, 1, 2, 0], [1, 1, 2, 0], [0, 1, 2, 0]])
    observed_classes = np.array([0, 1, 2, 0])
    axes = torch.zeros(4, dtype=torch.long)
    sides = torch.tensor([1.0, -1.0, 1.0, 1.0])
    layers = torch.tensor([1.0, 1.0, 1.0, 1.0])
    values = render_script._gpu_keypoint_fit_metrics(
        torch, torch.tensor(distances), torch.tensor(objects), torch.tensor(classes),
        axes, sides, layers, torch.tensor(observed_classes), 10.0, 2.0,
    )
    means, robust, p90, inliers, layer_inliers = [v.numpy() for v in values]
    for index in (0, 1):
        valid = np.isfinite(distances[index])
        normalized = distances[index, valid] / 10.0
        assert means[index] == pytest.approx(float(np.mean(normalized)), abs=1e-7)
        assert p90[index] == pytest.approx(float(np.percentile(normalized, 90)), abs=1e-7)
        assert robust[index] == pytest.approx(float(np.median(normalized) + 0.5 * np.percentile(normalized, 90)), abs=1e-7)
        assert inliers[index] == int(np.count_nonzero(valid & (distances[index] <= 2.0)))
        layer = render_script.cube_layer_mask(objects[index], 0, int(sides[index]), 1)
        assert layer_inliers[index] == int(np.count_nonzero(valid & layer & (distances[index] <= 2.0)))
    assert np.all(np.isinf(means[2:]))
    assert np.all(np.isinf(robust[2:]))
    assert np.all(np.isinf(p90[2:]))
    assert np.array_equal(inliers[2:], [0, 0])
    assert np.array_equal(layer_inliers[2:], [0, 0])
