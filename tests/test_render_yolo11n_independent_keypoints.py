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


def test_explicit_state_move_overrides_partial_display_rotation():
    candidate = SimpleNamespace(
        layer_rotations=(render_script.CubeLayerRotation(0, 1, 30.0, 1),),
        turn_state_moves=("R",),
    )

    assert render_script.completed_candidate_moves(candidate) == ("R",)


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


def test_move_lookahead_weights_use_uniform_front_and_back_blocks():
    assert render_script.move_lookahead_weights(4) == pytest.approx((0.15, 0.15, 0.35, 0.35))
    assert render_script.move_lookahead_weights(11) == pytest.approx(
        (0.06,) * 5 + (0.70 / 6.0,) * 6,
    )
    assert render_script.move_lookahead_weighted_score(2.0, [4.0, 4.0, 4.0]) == pytest.approx(3.7)
    assert render_script.move_lookahead_weighted_score(2.0, []) == pytest.approx(2.0)
    # An observed frame with an impossible fit must invalidate the path.
    assert np.isinf(render_script.move_lookahead_weighted_score(2.0, [float("inf")]))
    assert np.isinf(render_script.move_lookahead_weighted_score(2.0, [float("nan")]))


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


def test_turn_color_weight_matches_each_geometry_term():
    assert render_script.PNP_TURN_COLOR_WEIGHT == pytest.approx(
        render_script.PNP_TURN_MASK_CENTER_WEIGHT
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


def _lookahead_test_candidate():
    return render_script.StickerPnPCandidate(
        sticker_type="corner", sticker_average_confidence=0.9,
        image_points=np.zeros((4, 2), dtype=np.float32),
        object_points=render_script.canonical_sticker_object_points("corner"),
        projected_sticker_points=np.zeros((4, 2), dtype=np.float32),
        camera_matrix=render_script.default_camera_matrix((256, 256), 1.2),
        dist_coeffs=np.zeros((4, 1)), rvec=np.zeros((3, 1)),
        tvec=np.asarray([[0.0], [0.0], [200.0]]), reprojection_error=0.0,
    )


def _lookahead_trial(score, moves=(), *, baseline=False):
    return dict(state_score=score, rank=(score,), state_moves=moves, is_baseline=baseline)


def _colors_for_scramble(scramble):
    state = render_script.CubeState.from_scramble(scramble)
    return {identity.label: state.color_at((identity.cubie_index, identity.axis, identity.side))
            for identity in render_script.canonical_sticker_identities()}


@pytest.mark.parametrize("case,solved_frame,fit_frames", [
    ("normal", 1, [0, 1]), ("rejected", 2, [0, 1, 2]), ("partial", 2, [0, 1, 2]),
    ("unreliable", 2, [0, 1, 2]),
])
def test_cached_search_stops_only_after_accepted_aligned_current_solution(
    tmp_path, monkeypatch, case, solved_frame, fit_frames,
):
    records = [SimpleNamespace(
        frame_index=frame, sticker_masks=None,
        pose_candidate=replace(_lookahead_test_candidate(), tvec=np.array([[float(frame)], [0.0], [200.0]])),
    ) for frame in range(5)]
    calls = []

    def fit(candidate, **kwargs):
        frame = int(candidate.tvec[0, 0])
        calls.append(frame)
        rotations = ((render_script.CubeLayerRotation(0, 1, 12.0),)
                     if case == "partial" and frame == 1 else ())
        moves = ("R'",) if frame else ()
        if case == "partial" and frame > 1:
            moves = ()
        return replace(candidate, turn_state_moves=moves,
                       layer_rotations=rotations, sticker_layer_rotations=None,
                       turn_color_unreliable=case == "unreliable" and frame == 1)

    monkeypatch.setattr(render_script, "_turn_torch_scoring_device", lambda _: None)
    monkeypatch.setattr(render_script, "solve_pnp_from_low_error_assignments", fit)
    monkeypatch.setattr(render_script, "pose_reprojection_error_is_acceptable", lambda candidate:
                        not (case == "rejected" and int(candidate.tvec[0, 0]) == 1))
    log = tmp_path / "scores.jsonl"
    render_script.recover_cached_pose_symmetries(
        records, render_script.StickerColorTrace.create(), pose_prepared=True,
        recover_orientation=False, recognize_moves=True, global_profiles={},
        known_initial_colors=_colors_for_scramble("R"), turn_candidate_log=log,
    )
    assert calls == fit_frames
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == len(records)
    assert rows[1]["selected_committed_moves"] == ([] if case == "unreliable" else ["R'"])
    assert not (rows[0]["search"] or {}).get("stopped")
    if case != "normal":
        assert not (rows[1]["search"] or {}).get("stopped")
    assert rows[solved_frame]["search"]["solved_frame_index"] == solved_frame
    for row, record in zip(rows[solved_frame + 1:], records[solved_frame + 1:]):
        assert row["search"]["status"] == "stopped_solved"
        assert row["search"]["evaluated_states"] == 0
        assert row["selected_committed_moves"] == []
        assert record.pose_candidate.cube_geometry_state.is_solved(_colors_for_scramble("R"))


def test_already_solved_current_fit_skips_lookahead_and_preserves_later_output(tmp_path, monkeypatch):
    records = [SimpleNamespace(frame_index=frame, sticker_masks=None,
                               pose_candidate=None if frame == 1 else _lookahead_test_candidate())
               for frame in range(4)]
    monkeypatch.setattr(render_script, "_turn_torch_scoring_device", lambda _: "cpu")
    monkeypatch.setattr(render_script, "_build_turn_geometry_frame", lambda *args: dict(options=()))
    monkeypatch.setattr(render_script, "_score_turn_color_offsets_on_gpu",
                        lambda *args, **kwargs: pytest.fail("Solved current fit must skip rollout"))
    calls = []
    monkeypatch.setattr(render_script, "solve_pnp_from_low_error_assignments",
                        lambda candidate, **kwargs: calls.append(candidate) or candidate)
    log = tmp_path / "scores.jsonl"
    render_script.recover_cached_pose_symmetries(
        records, render_script.StickerColorTrace.create(), pose_prepared=True,
        recover_orientation=False, recognize_moves=True, global_profiles={},
        known_initial_colors=_colors_for_scramble("x y"), turn_candidate_log=log,
    )
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 1
    assert len(rows) == 4
    assert rows[1]["has_pose"] is False
    for row in (rows[0], rows[2], rows[3]):
        assert row["has_pose"] is True
        if row["frame_index"]:
            assert row["search"]["status"] == "stopped_solved"
        assert row["search"]["solved_frame_index"] == 0
        assert row["selected_committed_moves"] == []


def test_solved_future_forecast_does_not_stop_current_unsolved_frame(tmp_path, monkeypatch):
    records = [SimpleNamespace(
        frame_index=frame, sticker_masks=None,
        pose_candidate=replace(_lookahead_test_candidate(), tvec=np.array([[float(frame)], [0.0], [200.0]])),
    ) for frame in range(3)]
    known = _colors_for_scramble("R")

    def trial(score, moves=()):
        return dict(_lookahead_trial(score, moves, baseline=not moves),
                    trial_id="solve" if moves else "hold", label="solve" if moves else "hold",
                    layer_axis=0, layer_side=1, layer_count=1, color_offset=90.0 if moves else 0.0,
                    layer_rotations=(), combined_mean_error=score, keypoint_error=0.0,
                    sticker_error=0.0, color_error=score)

    calls = []

    def score(pose, *args, **kwargs):
        frame = int(pose.tvec[0, 0])
        calls.append(frame)
        solved = pose.cube_geometry_state.is_solved(known)
        return [trial(0.0 if frame == 0 or solved else 100.0),
                trial(0.0 if frame > 0 and not solved else 100.0, ("R'",))]

    monkeypatch.setattr(render_script, "_turn_torch_scoring_device", lambda _: "cpu")
    monkeypatch.setattr(render_script, "_build_turn_geometry_frame", lambda *args:
                        dict(options=(trial(0.0),), trials=(trial(0.0),), static_trials=(trial(0.0),)))
    monkeypatch.setattr(render_script, "_score_turn_color_offsets_on_gpu", score)
    monkeypatch.setattr(render_script, "_score_turn_color_offsets_for_states_on_gpu",
                        lambda pose, options, limit, device, likelihoods, states, **kwargs:
                        [score(replace(pose, cube_geometry_state=state)) for state in states])
    monkeypatch.setattr(render_script, "turn_color_predictive_likelihoods", lambda *args: None)
    monkeypatch.setattr(render_script, "solve_pnp_from_low_error_assignments", lambda pose, **kwargs: pose)
    monkeypatch.setattr(render_script, "pose_reprojection_error_is_acceptable", lambda _: True)
    log = tmp_path / "scores.jsonl"
    render_script.recover_cached_pose_symmetries(
        records, render_script.StickerColorTrace.create(), pose_prepared=True,
        recover_orientation=False, recognize_moves=True, global_profiles={},
        known_initial_colors=known, turn_candidate_log=log,
    )
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    steps = rows[0]["search"]["selected_path"]["steps"]
    assert steps[0]["state_moves"] == []
    assert any(step["state_moves"] == ["R'"] for step in steps[1:])
    assert not rows[0]["search"].get("stopped")
    assert rows[1]["selected_committed_moves"] == ["R'"]
    assert rows[1]["search"]["solved_frame_index"] == 1
    assert rows[2]["search"]["status"] == "stopped_solved"


def test_search_diagnostics_explain_pruning_without_changing_the_search(tmp_path):
    state = render_script.CubeGeometryState.identity().apply_move("R")
    trials = tuple(dict(_lookahead_trial(score, moves, baseline=baseline), trial_id=name, label=name)
                   for name, score, moves, baseline in (
                       ("keep", 1.0, (), False), ("merge", 2.0, (), False),
                       ("beam", 3.0, ("U",), False), ("undo", 0.0, ("R'",), False),
                       ("invalid", float("inf"), (), False), ("static", 4.0, (), True),
                   ))

    def evaluate(frame, states):
        return ((_lookahead_trial(1.0), _lookahead_trial(0.0, ("R'",))),) * len(states)

    debug = {}
    paths = render_script.search_move_lookahead_paths(
        state, trials, (1,), evaluate, state_batch_size=1, previous_move="R", diagnostics=debug,
    )
    plain = render_script.search_move_lookahead_paths(
        state, trials, (1,), evaluate, state_batch_size=1, previous_move="R",
    )
    assert [(path.first_trial["trial_id"], path.future_scores) for path in paths] == [
        (path.first_trial["trial_id"], path.future_scores) for path in plain
    ]
    assert {root["trial_id"]: root["status"] for root in debug["roots"]} == {
        "keep": "complete", "merge": "merged", "beam": "complete",
        "undo": "complete", "invalid": "invalid_score", "static": "complete",
    }
    assert debug["depths"][0]["collapsed_holds"] == 2
    assert all(depth["beam_pruned_paths"] == 0 for depth in debug["depths"])
    assert debug["depths"][1]["immediate_undo"] == 0
    assert debug["depths"][1]["scored_states"] == 3
    assert debug["optimality"] == "proven"
    forecast = render_script.move_lookahead_path_log(paths[0], (100, 103))
    assert [step["frame_index"] for step in forecast["steps"]] == [100, 103]
    assert sum(step["weighted_score"] for step in forecast["steps"]) == pytest.approx(forecast["total_score"])
    assert all(not path.future_trials for path in plain)
    log = tmp_path / "scores.jsonl"
    render_script.append_turn_candidate_log(
        log, replace(_lookahead_test_candidate(), turn_search_diagnostics=debug), 100,
    )
    recorded = json.loads(log.read_text())
    invalid = next(trial for trial in recorded["candidates"] if trial["trial_id"] == "invalid")
    assert invalid["state_score"] is None
    assert invalid["search_status"] == "invalid_score"
    json.dumps(recorded, allow_nan=False)


def test_search_diagnostics_record_why_the_frontier_was_exhausted():
    debug = {}
    paths = render_script.search_move_lookahead_paths(
        render_script.CubeGeometryState.identity(), (_lookahead_trial(0.0, ("R",)),),
        (1,), lambda frame, states: ((_lookahead_trial(float("inf")),),) * len(states),
        diagnostics=debug,
    )
    assert not paths
    assert debug["depths"][1]["invalid_score"] == 1
    assert debug["depths"][1]["retained_paths"] == 0
    assert debug["roots"][0]["status"] == "no_legal_continuation"
    assert debug["roots"][0]["eliminated_depth"] == 1


def _write_search_debug_log(tmp_path, monkeypatch, turning_start_frame=None):
    candidate = _lookahead_test_candidate()
    initial = render_script.CubeGeometryState.identity().signature()
    records = []
    for cache_index, frame_index in enumerate((10, 12, 15)):
        pose = replace(candidate, tvec=np.asarray([[float(cache_index)], [0.0], [200.0]]))
        records.append(SimpleNamespace(frame_index=frame_index, pose_candidate=pose,
                                       pose_candidates=(pose,), sticker_masks=None))

    def trial(score, moves=()):
        return dict(_lookahead_trial(score, moves, baseline=not moves),
                    trial_id="0:-90" if moves else "0:+0", label="R offset -90" if moves else "static R",
                    layer_axis=0, layer_side=1, layer_count=1, color_offset=-90.0 if moves else 0.0,
                    layer_rotations=(), combined_mean_error=score, keypoint_error=0.0,
                    sticker_error=0.0, color_error=score)

    def score(pose, *args, **kwargs):
        frame = int(pose.tvec[0, 0])
        wrong = pose.cube_geometry_state.signature() == initial
        return [trial(0.0 if frame == 0 or not wrong else 2.0),
                trial(0.1 if frame == 0 else 5.0, ("R",))]

    monkeypatch.setattr(render_script, "_turn_torch_scoring_device", lambda _: "cpu")
    monkeypatch.setattr(render_script, "_build_turn_geometry_frame", lambda *args:
                        dict(options=(trial(0.0),), trials=(trial(0.0),), static_trials=(trial(0.0),)))
    monkeypatch.setattr(render_script, "_score_turn_color_offsets_on_gpu", score)
    monkeypatch.setattr(render_script, "_score_turn_color_offsets_for_states_on_gpu",
                        lambda pose, options, limit, device, likelihoods, states, **kwargs:
                        [score(replace(pose, cube_geometry_state=state)) for state in states])
    monkeypatch.setattr(render_script, "turn_color_predictive_likelihoods", lambda *args: None)
    monkeypatch.setattr(render_script, "solve_pnp_from_low_error_assignments", lambda pose, **kwargs: pose)
    monkeypatch.setattr(render_script, "pose_reprojection_error_is_acceptable", lambda _: True)
    log_path = tmp_path / "turn_candidate_scores.jsonl"
    render_script.recover_cached_pose_symmetries(
        records, render_script.StickerColorTrace.create(), pose_prepared=True,
        recover_orientation=False, recognize_moves=True, global_profiles={}, known_initial_colors={},
        turning_start_frame=turning_start_frame,
        turn_candidate_log=log_path,
    )
    return log_path


def test_move_search_log_records_actual_forecast_frames_and_selected_offset(tmp_path, monkeypatch):
    path = _write_search_debug_log(tmp_path, monkeypatch)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    first = records[0]
    assert first["log_schema_version"] == 2
    assert first["search"]["version"] == 15
    assert first["search"]["shortlist_rank"] == "geometry_plus_color"
    assert first["selected_trial_id"] == "0:-90"
    assert first["selected"] == "R offset -90"
    assert first["search"]["changed_local_choice"] is True
    assert first["search"]["local_best_trial_id"] == "0:+0"
    assert [step["frame_index"] for step in first["search"]["selected_path"]["steps"]] == [10, 12, 15]
    assert first["search"]["selected_path"]["total_score"] == pytest.approx(0.04)
    assert first["candidates"][0]["trial_id"] == "0:+0"
    assert first["candidates"][1]["trial_id"] == "0:-90"
    assert first["candidates"][1]["best_sequence_score"] == pytest.approx(0.04)
    assert records[1]["search"]["previous_move"] == "R"
    assert records[1]["search"]["reused_previous_forecast"]
    assert records[1]["search"]["seeds"][0]["source"] == "previous_forecast"
    assert all(row['search']['decision_evaluated_states'] <= render_script.PNP_TURN_LOOKAHEAD_STATE_BUDGET
               for row in records)
    assert first["search"]["geometry"]["retained_options"] == 1
    assert first["search"]["evaluation_cache"]["gpu_batches"] == 4
    assert first["search"]["algorithm"] == "adaptive_branch_and_bound"
    assert first["search"]["optimality"] == "proven"
    assert first["lookahead_beam_width"] is None
    assert first["search"]["prune_state_revisits"] is False
    assert first["search"]["prune_immediate_undo"] is False
    assert first["search"]["prune_root_state_returns"] is True
    assert first["search"]["merge_key"] == "cube_state"
    assert first["search"]["no_turn_path"]["move_count"] == 0
    assert all(not step["state_moves"] for step in first["search"]["no_turn_path"]["steps"])
    assert first["search"]["improvement_over_no_turn"] > 0
    json.dumps(records, allow_nan=False)


def test_move_lookahead_keeps_a_locally_worse_future_to_choose_the_right_first_move():
    state = render_script.CubeGeometryState.identity()
    initial = state.signature()
    bad = state.apply_moves(("U",)).signature()
    good = state.apply_moves(("F",)).signature()
    baseline = _lookahead_trial(0.0, baseline=True)
    action = _lookahead_trial(0.2, ("R",))

    def evaluate(frame, states):
        trials = []
        for future_state in states:
            signature = future_state.signature()
            if frame == 1 and signature == initial:
                trials.append((_lookahead_trial(0.0, ("U",)), _lookahead_trial(1.0, ("F",))))
            else:
                score = 10.0 if signature == bad else (0.0 if signature == good else 0.3)
                trials.append((_lookahead_trial(score),))
        return tuple(trials)

    paths = render_script.search_move_lookahead_paths(
        state, (baseline, action), tuple(range(1, 11)), evaluate, state_batch_size=3,
    )
    winner = min(paths, key=lambda path: render_script.move_lookahead_weighted_score(
        path.first_trial["state_score"], path.future_scores,
    ))
    assert winner.first_trial is baseline
    assert winner.state.signature() == good
    assert winner.future_scores == (1.0,) + (0.0,) * 9
    assert render_script.move_lookahead_weighted_score(
        winner.first_trial["state_score"], winner.future_scores,
    ) == pytest.approx(0.06)


def test_move_lookahead_rate_cap_allows_only_every_other_frame():
    state = render_script.CubeGeometryState.identity()
    hold = _lookahead_trial(1.0, baseline=True)
    change = dict(
        _lookahead_trial(0.0, ("R",)),
        layer_axis=0, layer_side=1, layer_count=1,
    )
    later_change = dict(
        _lookahead_trial(0.0, ("U",)),
        layer_axis=1, layer_side=1, layer_count=1,
    )
    final_change = dict(
        _lookahead_trial(0.0, ("F",)),
        layer_axis=2, layer_side=1, layer_count=1,
    )

    paths = render_script.search_move_lookahead_paths(
        state,
        (change, hold),
        (101, 102, 103),
        lambda frame, states: tuple(
            (final_change if frame == 103 else later_change, hold)
            for _state in states
        ),
        frame_indices=(100, 101, 102, 103),
        min_state_update_gap_frames=2,
        retain_forecast=True,
    )
    winner = min(paths, key=render_script.move_lookahead_path_score)
    selected = (winner.first_trial, *winner.future_trials)

    assert [frame for frame, trial in zip((100, 101, 102, 103), selected)
            if trial["state_moves"]] == [100, 102]


def test_move_lookahead_rate_cap_allows_one_same_layer_direction_continuation():
    state = render_script.CubeGeometryState.identity()
    hold = _lookahead_trial(1.0, baseline=True)
    first = dict(_lookahead_trial(0.0, ("R",)), layer_axis=0, layer_side=1, layer_count=1)
    same_layer_direction = dict(
        _lookahead_trial(0.0, ("R",)),
        layer_axis=0, layer_side=1, layer_count=1,
    )
    different_layer = dict(
        _lookahead_trial(0.0, ("U",)),
        layer_axis=1, layer_side=1, layer_count=1,
    )

    def evaluate(frame, states):
        trial = same_layer_direction if frame == 101 else different_layer
        return tuple((trial, hold) for _state in states)

    paths = render_script.search_move_lookahead_paths(
        state,
        (first, hold),
        (101, 102),
        evaluate,
        frame_indices=(100, 101, 102),
        min_state_update_gap_frames=2,
        retain_forecast=True,
    )
    winner = min(paths, key=render_script.move_lookahead_path_score)
    selected = (winner.first_trial, *winner.future_trials)

    assert [frame for frame, trial in zip((100, 101, 102), selected)
            if trial["state_moves"]] == [100, 101]


def test_turn_lookahead_frame_count_scales_from_30_fps():
    assert render_script.turn_lookahead_frame_count(30.0) == 10
    assert render_script.turn_lookahead_frame_count(60.0) == 20
    assert render_script.turn_lookahead_frame_count(15.0) == 5
    assert render_script.turn_lookahead_frame_count(None) == 10


@pytest.mark.parametrize('fps, expected', [
    (23.976, 1), (30., 1), (59.94, 2), (60., 2), (120., 4),
    (None, 1), (0., 1), (float('nan'), 1),
])
def test_turn_state_update_spacing_scales_with_source_fps(fps, expected):
    assert render_script.turn_state_update_gap_frames(fps) == expected


@pytest.mark.parametrize('fps, next_move', [(23.976, ('F',)), (60., ())])
def test_next_layer_after_double_turn_is_allowed_at_low_fps(fps, next_move):
    # Xuanyi's second R2 quarter at 205 must not exclude the next move at 206.
    # At 60 FPS the same frame indices intentionally require a hold at 206.
    quarter = dict(_lookahead_trial(0., ('R',)), layer_axis=0, layer_side=1, layer_count=1)
    next_layer = dict(_lookahead_trial(0., ('F',)), layer_axis=1, layer_side=-1, layer_count=1)
    hold = _lookahead_trial(10., baseline=True)
    paths = render_script.search_move_lookahead_paths(
        render_script.CubeGeometryState.identity(), (quarter,), (206,),
        lambda _frame, states: ((next_layer, hold),) * len(states),
        frame_indices=(205, 206), previous_state_update_frame=204,
        previous_state_update_key=render_script.state_update_layer_direction(quarter),
        previous_state_update_streak=1,
        min_state_update_gap_frames=render_script.turn_state_update_gap_frames(fps),
        retain_forecast=True,
    )
    winner = min(paths, key=render_script.move_lookahead_path_score)
    assert winner.first_trial['state_moves'] == ('R',)
    assert winner.future_trials[0]['state_moves'] == next_move


def test_move_lookahead_rate_cap_applies_across_window_boundary():
    state = render_script.CubeGeometryState.identity()
    hold = _lookahead_trial(0.0, baseline=True)
    change = _lookahead_trial(-1.0, ("R",))
    debug = {}

    paths = render_script.search_move_lookahead_paths(
        state,
        (change, hold),
        (102,),
        lambda _frame, states: tuple((change, hold) for _state in states),
        frame_indices=(101, 102),
        previous_state_update_frame=100,
        min_state_update_gap_frames=2,
        diagnostics=debug,
    )
    winner = min(paths, key=render_script.move_lookahead_path_score)

    assert not winner.first_trial["state_moves"]
    assert debug["depths"][0]["state_update_too_soon"] == 1
    assert debug["roots"][0]["status"] == "state_update_too_soon"


def test_move_lookahead_matches_exhaustive_search_and_scores_merged_states_once():
    from itertools import product

    state = render_script.CubeGeometryState.identity()
    target = state.apply_moves(("U", "R"))
    moves = ((), ("U",), ("R",), ("R'",))
    evaluated = []
    evaluated_states = {}

    def trials(frame, before):
        result = []
        for action in moves:
            after = before.apply_moves(action)
            wrong = sum(after.facelets[key] != target.facelets[key] for key in after.facelets)
            result.append(_lookahead_trial(wrong / 54.0 + 0.1 * len(action), action,
                                           baseline=not action))
        return tuple(result)

    def evaluate(frame, states):
        signatures = [value.signature() for value in states]
        assert len(signatures) == len(set(signatures))
        previous = evaluated_states.setdefault(frame, set())
        assert not previous.intersection(signatures)
        previous.update(signatures)
        evaluated.append((frame, len(states)))
        return tuple(trials(frame, before) for before in states)

    paths = render_script.search_move_lookahead_paths(
        state, trials(0, state), (1, 2, 3), evaluate, state_batch_size=128,
    )
    actual = min(render_script.move_lookahead_path_score(path) for path in paths)
    exhaustive = []
    for choices in product(range(len(moves)), repeat=4):
        before = state
        scores = []
        move_count = 0
        for frame, choice in enumerate(choices):
            trial = trials(frame, before)[choice]
            scores.append(trial["state_score"])
            before = before.apply_moves(trial["state_moves"])
            move_count += len(trial["state_moves"])
            if trial["state_moves"] and before.signature() == state.signature():
                break
        else:
            exhaustive.append(render_script.move_lookahead_weighted_score(scores[0], scores[1:])
                              + render_script.PNP_TURN_LOOKAHEAD_MOVE_COST * move_count)
    assert actual == pytest.approx(min(exhaustive))
    assert set(evaluated_states) == {1, 2, 3}
    assert all(count > 0 for _, count in evaluated)


def test_move_lookahead_rejects_impossible_future_paths():
    state = render_script.CubeGeometryState.identity()
    baseline = _lookahead_trial(0.0, baseline=True)
    action = _lookahead_trial(1.0, ("R",))

    def evaluate(frame, states):
        return tuple((
            _lookahead_trial(float("inf") if before.signature() == state.signature() else 0.0),
        ) for before in states)

    paths = render_script.search_move_lookahead_paths(state, (baseline, action), (1,), evaluate)
    assert len(paths) == 1
    assert paths[0].first_trial is action


@pytest.mark.parametrize("move,undo", [
    ("R", "R'"), ("R'", "R"), ("Rw", "Rw'"), ("r", "Rw'"), ("R2", "R2"),
])
@pytest.mark.parametrize("neutral_frames", [0, 2, 9])
def test_move_lookahead_prunes_returns_to_start_even_after_neutral_frames(move, undo, neutral_frames):
    state = render_script.CubeGeometryState.identity()
    first = _lookahead_trial(0.0, (move,))

    def evaluate(frame, states):
        trials = ((_lookahead_trial(0.0),) if frame <= neutral_frames else (
            _lookahead_trial(0.0, (undo,)), _lookahead_trial(1.0, ("U",)),
        ))
        return (trials,) * len(states)

    debug = {}
    paths = render_script.search_move_lookahead_paths(
        state, (first,), tuple(range(1, neutral_frames + 2)), evaluate,
        diagnostics=debug,
    )
    winner = min(paths, key=lambda path: render_script.move_lookahead_weighted_score(
        path.first_trial["state_score"], path.future_scores))
    assert winner.last_move == "U"
    assert winner.future_scores == (0.0,) * neutral_frames + (1.0,)
    assert winner.state.signature() == state.apply_moves((move, "U")).signature()
    assert debug["depths"][neutral_frames + 1]["root_return"] == 1
    assert debug["roots"][0]["pruning_counts"]["root_return"] == 1
    assert debug["roots"][0]["root_return_examples"][0]["state_moves"] == (undo,)


def test_move_lookahead_allows_current_undo_and_repeated_turns():
    state = render_script.CubeGeometryState.identity().apply_move("R")
    inverse = _lookahead_trial(0.0, ("R'",))
    neutral = _lookahead_trial(1.0, baseline=True)
    repeated = _lookahead_trial(2.0, ("R",))
    paths = render_script.search_move_lookahead_paths(
        state, (inverse, neutral, repeated), (), lambda *args: (), previous_move="R",
    )
    assert any(path.first_trial is inverse for path in paths)
    assert any(path.first_trial is repeated for path in paths)
    assert next(path for path in paths if path.first_trial is neutral).last_move == "R"


def test_move_lookahead_prunes_a_multi_move_trial_returning_to_start():
    state = render_script.CubeGeometryState.identity()
    inverse = _lookahead_trial(0.0, ("R", "R'"))
    valid = _lookahead_trial(1.0, ("R", "U", "R'"))
    debug = {}
    paths = render_script.search_move_lookahead_paths(
        state, (inverse, valid), (), lambda *args: (), diagnostics=debug,
    )
    assert len(paths) == 1
    assert paths[0].first_trial is valid
    assert debug["roots"][0]["status"] == "root_return"
    assert debug["depths"][0]["root_return"] == 1


def test_move_lookahead_merges_distinct_last_moves_when_cube_states_merge():
    state = render_script.CubeGeometryState.identity()
    first_r = _lookahead_trial(1.0, ("R",))
    first_l = _lookahead_trial(0.0, ("L",))
    r_state = state.apply_move("R").signature()
    frontier_sizes = []

    def evaluate(frame, states):
        frontier_sizes.append(len(states))
        if frame == 1:
            return tuple((
                _lookahead_trial(0.0, ("L",) if before.signature() == r_state else ("R",)),
            ) for before in states)
        return ((_lookahead_trial(0.0, ("R'",)),),) * len(states)

    paths = render_script.search_move_lookahead_paths(
        state, (first_r, first_l), (1, 2), evaluate,
    )
    # R L and L R have the same future options now that history does not
    # exclude inverses. The cheaper L R prefix correctly dominates.
    assert frontier_sizes == [2, 1]
    assert len(paths) == 1
    assert paths[0].first_trial is first_l
    assert paths[0].state.signature() == state.apply_move("L").signature()


def test_move_lookahead_prunes_full_cycles_but_allows_half_turns():
    state = render_script.CubeGeometryState.identity()
    debug = {}
    paths = render_script.search_move_lookahead_paths(
        state, (_lookahead_trial(0.0, ("R",)),), (1, 2, 3),
        lambda frame, states: ((_lookahead_trial(0.0, ("R",)), _lookahead_trial(1.0)),) * len(states),
        diagnostics=debug,
    )
    assert paths
    assert debug["depths"][3]["state_revisit"] == 0
    assert debug["depths"][3]["root_return"] > 0
    assert all(path.state.signature() != state.signature() for path in paths)
    half = render_script.search_move_lookahead_paths(
        state, (_lookahead_trial(0.0, ("R",)),), (1,),
        lambda frame, states: ((_lookahead_trial(0.0, ("R",)),),) * len(states),
    )
    assert half[0].state.signature() == state.apply_moves(("R", "R")).signature()


def test_move_lookahead_does_not_block_recent_states_in_a_new_window():
    initial = render_script.CubeGeometryState.identity()
    past = tuple(initial.apply_moves(("R",) * turns).signature() for turns in range(3))
    current = initial.apply_moves(("R",) * 3)
    cycle = _lookahead_trial(0.0, ("R",))
    hold = _lookahead_trial(1.0, baseline=True)
    debug = {}
    paths = render_script.search_move_lookahead_paths(
        current, (cycle, hold), (), lambda *args: (), previous_move="R",
        previous_states=past, diagnostics=debug,
    )
    assert len(paths) == 2
    assert paths[0].first_trial is cycle
    assert debug["roots"][0]["status"] == "complete"


def test_move_lookahead_catches_an_intermediate_loop_when_the_window_advances():
    initial = render_script.CubeGeometryState.identity()
    after_r = initial.apply_move("R")
    # From the R frame, U U' returns to an intermediate state and is allowed.
    paths = render_script.search_move_lookahead_paths(
        initial, (_lookahead_trial(0.0, ("R",)),), (1, 2),
        lambda frame, states: ((_lookahead_trial(0.0, ("U",) if frame == 1 else ("U'",)),),) * len(states),
    )
    assert paths[0].state.signature() == after_r.signature()
    # Once U is the current frame, U' returns to that window's root.
    debug = {}
    paths = render_script.search_move_lookahead_paths(
        after_r, (_lookahead_trial(0.0, ("U",)),), (2,),
        lambda frame, states: ((_lookahead_trial(0.0, ("U'",)), _lookahead_trial(1.0)),) * len(states),
        diagnostics=debug,
    )
    assert paths[0].state.signature() == after_r.apply_move("U").signature()
    assert debug["depths"][1]["root_return"] == 1


def test_move_lookahead_preserves_holds_and_offsets_at_the_starting_state():
    initial = render_script.CubeGeometryState.identity()
    tilted_hold = dict(_lookahead_trial(0.0, baseline=True), layer_rotations=(
        render_script.CubeLayerRotation(0, 1, 21.0, 1),))
    debug = {}
    paths = render_script.search_move_lookahead_paths(
        initial, (tilted_hold,), tuple(range(1, 11)),
        lambda frame, states: ((tilted_hold,),) * len(states), diagnostics=debug,
        seed_greedy=True, state_evaluation_budget=11,
    )
    assert paths
    assert all(path.state.signature() == initial.signature() and path.move_count == 0 for path in paths)
    assert all(depth["root_return"] == 0 for depth in debug["depths"])
    assert debug["seeds"][0]["root_return_pruned"] == 0


@pytest.mark.parametrize("seed", ["warm_start", "greedy"])
def test_move_lookahead_seed_cannot_bypass_root_return_pruning(seed):
    initial = render_script.CubeGeometryState.identity()
    first = _lookahead_trial(0.0, ("R",))
    debug = {}

    def evaluate(frame, states):
        trials = ((_lookahead_trial(0.0),) if frame < 10 else (
            _lookahead_trial(0.0, ("R'",)), _lookahead_trial(0.5, ("U",)), _lookahead_trial(1.0),
        ))
        return (trials,) * len(states)

    paths = render_script.search_move_lookahead_paths(
        initial, (first,), tuple(range(1, 11)), evaluate, diagnostics=debug,
        state_evaluation_budget=11, state_change_limit=1,
        warm_start_moves=(("R",),) + ((),) * 9 + (("R'",),) if seed == "warm_start" else None,
        seed_greedy=seed == "greedy",
    )
    winner = min(paths, key=render_script.move_lookahead_path_score)
    assert winner.state.signature() == initial.apply_moves(("R", "U")).signature()
    assert winner.future_trials[-1]["state_moves"] == ("U",)
    assert all(path.state.signature() != initial.signature() for path in paths)
    assert debug["seeds"][0]["root_return_pruned"] == 1
    assert debug["seeds"][0]["repaired_steps"] == (1 if seed == "warm_start" else 0)
    assert debug["evaluated_states"] <= 11


def test_root_return_rejections_and_examples_are_saved_in_candidate_log(tmp_path):
    initial = render_script.CubeGeometryState.identity()
    debug = {}
    render_script.search_move_lookahead_paths(
        initial, (_lookahead_trial(0.0, ("R",)),), (1,),
        lambda frame, states: ((_lookahead_trial(0.0, ("R'",)),),) * len(states),
        diagnostics=debug,
    )
    assert debug["roots"][0]["status"] == "root_return"
    log = tmp_path / "scores.jsonl"
    render_script.append_turn_candidate_log(
        log, replace(_lookahead_test_candidate(), turn_search_diagnostics=debug), 10,
    )
    record = json.loads(log.read_text())
    assert record["search"]["prune_root_state_returns"] is True
    assert record["candidates"][0]["pruning_counts"]["root_return"] == 1
    assert record["candidates"][0]["root_return_examples"][0]["state_moves"] == ["R'"]


def test_move_lookahead_merges_distinct_visited_histories_with_the_same_state_and_last_move():
    initial = render_script.CubeGeometryState.identity()
    first_r, first_l = _lookahead_trial(1.0, ("R",)), _lookahead_trial(0.0, ("L",))
    r_state = initial.apply_move("R").signature()

    def evaluate(frame, states):
        if frame == 1:
            return tuple((_lookahead_trial(0.0, ("L",) if state.signature() == r_state else ("R",)),)
                         for state in states)
        return ((_lookahead_trial(0.0, ("R",) if frame == 2 else ("R2",)),),) * len(states)

    paths = render_script.search_move_lookahead_paths(initial, (first_r, first_l), (1, 2, 3), evaluate)
    # Visited states no longer affect future availability. The cheaper
    # L R prefix can be merged with R L and may revisit L later.
    assert len(paths) == 1
    assert paths[0].first_trial is first_l
    assert paths[0].state.signature() == initial.apply_move("L").signature()


@pytest.mark.parametrize("last_frame", [9, 11])
def test_state_revisit_is_allowed_through_neutral_frames_and_fallback(monkeypatch, last_frame):
    candidate = _lookahead_test_candidate()
    turns = (0, 3, 6, last_frame)
    records = []
    for frame in range(last_frame + 1):
        pose = replace(candidate, tvec=np.asarray([[float(frame)], [0.0], [200.0]]))
        records.append(SimpleNamespace(frame_index=frame, pose_candidate=pose, pose_candidates=(pose,), sticker_masks=None))
    monkeypatch.setattr(render_script, "_turn_torch_scoring_device", lambda _: "cpu")
    monkeypatch.setattr(render_script, "_build_turn_geometry_frame", lambda *args: None)
    monkeypatch.setattr(render_script, "solve_pnp_from_low_error_assignments",
                        lambda pose, **kwargs: replace(pose, turn_state_moves=("R",) if int(pose.tvec[0, 0]) in turns else ()))
    monkeypatch.setattr(render_script, "pose_reprojection_error_is_acceptable", lambda _: True)
    render_script.recover_cached_pose_symmetries(
        records, render_script.StickerColorTrace.create(), pose_prepared=True,
        recover_orientation=False, recognize_moves=True, global_profiles={}, known_initial_colors={},
    )
    assert [render_script.completed_candidate_moves(records[index].pose_candidate) for index in turns] == [
        ("R",), ("R",), ("R",), ("R",),
    ]


def test_live_accumulator_retains_recent_departed_states_through_neutral_frames():
    accumulator = render_script.MoveAccumulator()
    initial = accumulator._geometry_state.signature()
    candidate = replace(_lookahead_test_candidate(), cube_geometry_state=accumulator._geometry_state.copy(),
                        turn_state_moves=("R",))
    accumulator.observe(candidate)
    for _ in range(9):
        accumulator.observe(None)
    assert initial in accumulator.recent_geometry_states()
    accumulator.observe(None)
    assert initial not in accumulator.recent_geometry_states()


@pytest.mark.parametrize("geometry_mode", ["available", "missing", "forced_return"])
def test_move_search_allows_undo_of_the_previous_committed_window_or_fallback(monkeypatch, geometry_mode):
    candidate = _lookahead_test_candidate()
    records = []
    for frame in range(3):
        pose = replace(candidate, tvec=np.asarray([[float(frame)], [0.0], [200.0]]))
        records.append(SimpleNamespace(
            frame_index=frame, pose_candidate=pose, pose_candidates=(pose,), sticker_masks=None,
        ))
    previous_moves = []

    def local_search(pose, **kwargs):
        frame = int(pose.tvec[0, 0])
        previous_moves.append(kwargs["turn_previous_move"])
        return replace(pose, turn_state_moves=("R",) if frame == 0 else (("R'",) if frame == 2 else ()))

    def score(pose, *args, **kwargs):
        frame = int(pose.tvec[0, 0])
        if frame == 0:
            return [_lookahead_trial(0.0, ("R",))]
        if frame == 1:
            return [_lookahead_trial(0.0, baseline=True)]
        trials = [_lookahead_trial(0.0, ("R'",))]
        if geometry_mode == "available":
            trials.append(_lookahead_trial(1.0, baseline=True))
        return trials

    monkeypatch.setattr(render_script, "_turn_torch_scoring_device", lambda _: "cpu")
    monkeypatch.setattr(render_script, "_build_turn_geometry_frame", lambda *args:
                        None if geometry_mode == "missing" else {"options": ({},)})
    monkeypatch.setattr(render_script, "turn_color_predictive_likelihoods", lambda *args: None)
    monkeypatch.setattr(render_script, "_score_turn_color_offsets_on_gpu", score)
    monkeypatch.setattr(render_script, "_score_turn_color_offsets_for_states_on_gpu",
                        lambda pose, options, limit, device, likelihoods, states, **kwargs: [score(pose) for state in states])
    monkeypatch.setattr(render_script, "solve_pnp_from_low_error_assignments", local_search)
    monkeypatch.setattr(render_script, "pose_reprojection_error_is_acceptable", lambda _: True)
    render_script.recover_cached_pose_symmetries(
        records, render_script.StickerColorTrace.create(), pose_prepared=True,
        recover_orientation=False, recognize_moves=True, global_profiles={}, known_initial_colors={},
    )
    if geometry_mode == "forced_return":
        # With no alternate continuation, the first forecast's only path
        # returns to its root and is rejected. Fallback holds instead.
        assert previous_moves == [None, None, None]
        assert [render_script.completed_candidate_moves(record.pose_candidate) for record in records] == [
            (), (), ("R'",),
        ]
        assert records[-1].pose_candidate.cube_geometry_state.signature() == render_script.CubeGeometryState.identity().signature()
    else:
        assert previous_moves == [None, "R", "R"]
        assert [render_script.completed_candidate_moves(record.pose_candidate) for record in records] == [
            ("R",), (), ("R'",),
        ]
        assert records[-1].pose_candidate.cube_geometry_state.signature() == (
            render_script.CubeGeometryState.identity().apply_move("R").signature()
        )


def test_cached_geometry_move_permutations_match_cube_state():
    geometry = render_script.CubeGeometryState.identity()
    reference = render_script.CubeState(dict(geometry.facelets))
    for base in ("R", "L", "U", "D", "F", "B", "Rw", "Lw", "Uw", "Dw", "Fw", "Bw", "x", "y", "z"):
        for suffix in ("", "'", "2"):
            move = base + suffix
            geometry = geometry.apply_move(move)
            reference.apply_move(move)
            assert geometry.facelets == reference.colors


@pytest.mark.parametrize("layer_count", [1, 2])
@pytest.mark.parametrize("offset", [-90.0, 0.0, 90.0])
@pytest.mark.parametrize("partial_angle", [0.0, 15.0])
def test_selected_candidate_projects_the_sticker_identities_used_by_color_scoring(layer_count, offset, partial_angle):
    candidate = replace(
        _lookahead_test_candidate(),
        rvec=np.asarray([[0.18], [-0.25], [0.12]]),
        cube_geometry_state=render_script.CubeGeometryState.identity().apply_moves(("R", "U")),
    )
    partial = render_script.CubeLayerRotation(2, -1, partial_angle, layer_count)
    geometry_rotations = (partial,) if partial_angle else ()
    full = replace(partial, angle_degrees=partial.angle_degrees + offset)
    notation = render_script.cube_rotation_to_notation(replace(partial, angle_degrees=offset))
    trial = dict(
        layer_rotations=geometry_rotations, layer_axis=2, layer_side=-1, layer_count=layer_count,
        color_offset=offset, state_moves=(notation,) if notation else (), is_baseline=False,
    )
    # A local trial can refine the diagnostic pose before lookahead picks a
    # different one. Its camera must not leak into the selected cached trial.
    diagnostic = replace(candidate, tvec=candidate.tvec + np.asarray([[10.0], [0.0], [0.0]]))
    selected = render_script._candidate_with_turn_trial(diagnostic, candidate, trial)
    expected = replace(candidate, layer_rotations=(full,))
    projected = render_script.projected_sticker_faces(selected, selected.rvec, selected.tvec)
    reference = render_script.projected_sticker_faces(expected, expected.rvec, expected.tvec)
    assert [entry[0].label for entry in projected] == [entry[0].label for entry in reference]
    for (identity, depth, polygon), (_, expected_depth, expected_polygon) in zip(projected, reference):
        np.testing.assert_allclose(polygon, expected_polygon, atol=1e-5)
        assert depth == pytest.approx(expected_depth)
        assert render_script.sticker_view_reliability_weight(selected, identity) == pytest.approx(
            render_script.sticker_view_reliability_weight(expected, identity),
        )
    assert selected.layer_rotations == geometry_rotations
    np.testing.assert_array_equal(selected.rvec, candidate.rvec)
    np.testing.assert_array_equal(selected.tvec, candidate.tvec)
    assert selected.cube_geometry_state.signature() == candidate.cube_geometry_state.signature()
    assert render_script.completed_candidate_moves(selected) == trial["state_moves"]
    # Replay still consumes this pre-move model and commits exactly once.
    accumulator = render_script.MoveAccumulator()
    accumulator._geometry_state = candidate.cube_geometry_state.copy()
    accumulator.observe(selected)
    assert accumulator._geometry_state.signature() == candidate.cube_geometry_state.apply_moves(
        trial["state_moves"],
    ).signature()


@pytest.mark.parametrize("has_colors", [True, False])
def test_batched_color_offsets_match_cpu_identity_projection_and_share_geometry(monkeypatch, has_colors):
    torch = pytest.importorskip("torch")
    candidate = replace(_lookahead_test_candidate(), rvec=np.asarray([[0.18], [-0.25], [0.12]]))
    faces = render_script.projected_sticker_faces(candidate, candidate.rvec, candidate.tvec)
    centers = tuple(polygon.mean(axis=0) for identity, depth, polygon in faces)
    candidate = replace(candidate, sticker_mask_centers=centers,
                        sticker_mask_colors=(np.asarray([255, 0, 0], dtype=np.uint8),) * len(centers))
    option = dict(
        layer_axis=2, layer_side=-1, layer_count=1, layer_rotations=(),
        label="static D", is_baseline=True, combined_mean_error=0.03,
        assignment_projected_points=None, assignment_object_points=None,
        assignment_projected_class_ids=None,
    )
    states = (render_script.CubeGeometryState.identity(),
              render_script.CubeGeometryState.identity().apply_moves(("R", "U")))
    likelihoods = np.random.default_rng(12).uniform(0.01, 0.99, (54, len(centers))) if has_colors else None
    project = render_script._torch_project_turn_points
    calls = []

    def count_projection(*args, **kwargs):
        calls.append(1)
        return project(*args, **kwargs)

    monkeypatch.setattr(render_script, "_torch_project_turn_points", count_projection)
    scores = render_script._score_turn_color_offsets_for_states_on_gpu(
        candidate, (option,), 10.0, torch.device("cpu"), likelihoods, states,
    )
    assert calls == [1]
    assert len(scores) == 2
    for state, trials in zip(states, scores):
        for trial in trials:
            rotation = render_script.CubeLayerRotation(2, -1, trial["color_offset"], 1)
            expected_error, expected_count = render_script.sticker_mask_color_turn_fit_metrics(
                replace(candidate, cube_geometry_state=state), candidate.rvec, candidate.tvec,
                (rotation,), 10.0, likelihoods,
            )
            assert trial["color_error"] == pytest.approx(expected_error, abs=1e-6)
            assert trial["color_match_count"] == expected_count


def test_partial_angles_survive_geometry_stage_until_color_scoring(monkeypatch):
    trials = []
    for axis in range(3):
        for side in (-1, 1):
            for width in (1, 2):
                for angle, score in ((0., .2), (-15., .05), (45., .1)):
                    trials.append(dict(
                        layer_axis=axis, layer_side=side, layer_count=width,
                        is_baseline=angle == 0, rank=(score,),
                        layer_rotations=() if angle == 0 else (
                            render_script.CubeLayerRotation(axis, side, angle, width),),
                    ))
    monkeypatch.setattr(render_script, '_score_turn_configurations_on_gpu', lambda *args, **kwargs: trials)
    result = render_script._build_turn_geometry_frame(_lookahead_test_candidate(), 10., 'cpu')
    assert len(result['options']) == 36
    assert len(result['static_trials']) == 12
    correct_layer = [o for o in result['options']
                     if (o['layer_axis'], o['layer_side'], o['layer_count']) == (1, -1, 1)]
    assert [o['layer_rotations'][0].angle_degrees if o['layer_rotations'] else 0.
            for o in correct_layer] == [0., -15., 45.]


def test_partial_angle_selection_uses_each_states_combined_fit(monkeypatch):
    torch = pytest.importorskip('torch')
    options = tuple(dict(
        layer_axis=1, layer_side=-1, layer_count=1, is_baseline=False,
        label=f'Y-{angle:+g}w1', combined_mean_error=geometry,
        layer_rotations=(render_script.CubeLayerRotation(1, -1, angle, 1),),
        assignment_projected_points=None, assignment_object_points=None,
        assignment_projected_class_ids=None,
    ) for angle, geometry in ((-15., .05), (45., .1)))
    states = (render_script.CubeGeometryState.identity(),
              render_script.CubeGeometryState.identity().apply_move('R'))

    def colors(candidate, torch, centers, *args, **kwargs):
        # Configuration order is three offsets for -15, then three for +45.
        errors = centers.new_tensor([[2., 5., 4., 1., .2, 3.],
                                     [.1, .1, .1, 2., 2., 2.]])
        return errors, torch.ones_like(errors, dtype=torch.int32)

    monkeypatch.setattr(render_script, '_gpu_sticker_color_fit_metrics', colors)
    scores = render_script._score_turn_color_offsets_for_states_on_gpu(
        _lookahead_test_candidate(), options, 10., torch.device('cpu'), None, states,
    )
    # Retain only one angle per offset/state, but choose after color scoring.
    assert [len(trials) for trials in scores] == [3, 3]
    assert all(t['layer_rotations'][0].angle_degrees == 45. for t in scores[0])
    assert all(t['layer_rotations'][0].angle_degrees == -15. for t in scores[1])
    assert next(t for t in scores[0] if t['color_offset'] == 0)['state_score'] == pytest.approx(.3)


@pytest.mark.parametrize("color_error", [0.0, 0.7, float("inf"), float("nan")])
def test_color_offset_scores_preserve_invalid_color_fits(monkeypatch, color_error):
    torch = pytest.importorskip("torch")
    candidate = _lookahead_test_candidate()
    option = dict(
        layer_axis=0, layer_side=1, layer_count=1, layer_rotations=(),
        label="static R", is_baseline=True, combined_mean_error=0.03,
        robust_error=0.04, assignment_projected_points=None,
        assignment_object_points=None, assignment_projected_class_ids=None,
    )

    def color_metrics(_candidate, _torch, centers, *args, **kwargs):
        return (
            centers.new_full((len(centers),), color_error),
            torch.zeros(len(centers), dtype=torch.int32),
        )

    monkeypatch.setattr(render_script, "_gpu_sticker_color_fit_metrics", color_metrics)
    trials = render_script._score_turn_color_offsets_on_gpu(
        candidate, (option,), 10.0, torch.device("cpu"), None,
    )
    assert len(trials) == 3
    for trial in trials:
        if np.isfinite(color_error):
            assert trial["state_score"] == pytest.approx(0.03 + color_error)
            assert trial["robust_error"] == pytest.approx(0.04 + color_error)
        else:
            assert trial["state_score"] == float("inf")
            assert trial["robust_error"] == float("inf")


def test_all_invalid_color_trials_do_not_commit_a_move(monkeypatch):
    candidate = _lookahead_test_candidate()
    monkeypatch.setattr(render_script, "_turn_torch_scoring_device", lambda _: "cpu")
    monkeypatch.setattr(render_script, "turn_color_predictive_likelihoods", lambda *args: None)
    monkeypatch.setattr(
        render_script, "_score_turn_color_offsets_on_gpu",
        lambda *args: [dict(state_score=float("inf"), rank=(float("inf"),),
                           state_moves=("R",), is_baseline=False)],
    )
    result = render_script.recover_misaligned_cube_layers_from_geometry(
        candidate, 10.0, dict(options=({},), static_trials=({},)),
    )
    assert result.turn_state_moves == ()
    assert result.layer_rotations == ()


@pytest.mark.parametrize("prevent_cycle", [False, True])
def test_single_frame_search_allows_inverse_even_with_recent_state_history(monkeypatch, prevent_cycle):
    torch = pytest.importorskip("torch")
    initial = render_script.CubeGeometryState.identity()
    candidate = replace(_lookahead_test_candidate(), cube_geometry_state=(
        initial.apply_moves(("R",) * 3) if prevent_cycle else initial
    ))
    option = dict(
        layer_axis=0, layer_side=1, layer_count=1, layer_rotations=(),
        label="static R", is_baseline=True, combined_mean_error=0.03,
        keypoint_error=0.01, sticker_error=0.02, robust_error=0.04,
        assignment_projected_points=None, assignment_object_points=None,
        assignment_projected_class_ids=None,
    )
    trials = render_script._score_turn_color_offsets_on_gpu(
        candidate, (option,), 10.0, torch.device("cpu"), None,
    )
    for trial in trials:
        score = {("R'",): 0.0, ("R",): 1.0, (): 2.0}[trial["state_moves"]]
        trial.update(state_score=score, combined_mean_error=score, rank=(score,))
    monkeypatch.setattr(render_script, "_turn_torch_scoring_device", lambda _: "cpu")
    monkeypatch.setattr(render_script, "turn_color_predictive_likelihoods", lambda *args: None)
    monkeypatch.setattr(render_script, "_score_turn_color_offsets_on_gpu", lambda *args: trials)
    result = render_script.recover_misaligned_cube_layers_from_geometry(
        candidate, 10.0, dict(options=(option,), static_trials=(option,)), previous_move="R",
        previous_states=tuple(initial.apply_moves(("R",) * turns).signature() for turns in range(3))
        if prevent_cycle else (),
    )
    assert result.turn_state_moves == ("R'",)


@pytest.mark.parametrize("change", ["profiles", "priors"])
def test_lookahead_recalculates_cached_scores_after_color_model_changes(monkeypatch, change):
    candidate = _lookahead_test_candidate()
    trace = render_script.StickerColorTrace.create()
    label = trace.identities[0].label
    count = 4 if change == "profiles" else 3
    records = []
    for frame in range(count):
        pose = replace(candidate, tvec=np.asarray([[float(frame)], [0.0], [200.0]]))
        records.append(SimpleNamespace(
            frame_index=frame, pose_candidate=pose, pose_candidates=(pose,),
            sticker_masks=None,
        ))
        trace.observe(frame, {label: np.asarray([180, 40, 20], dtype=np.uint8)})
    generation = 0
    scores = []
    bound_generations = []

    def update_model(*args, **kwargs):
        nonlocal generation
        generation += 1
        return {"red": object()}, {}

    def score(pose, *args, **kwargs):
        frame = int(pose.tvec[0, 0])
        scores.append((frame, pose.cube_geometry_state.signature(), generation))
        moves = ("R",) if change == "profiles" and frame == 1 else ()
        return [dict(
            state_score=0.0, rank=(0.0,), state_moves=moves,
            is_baseline=not moves, layer_rotations=(),
        )]

    def lower_bound(pose, *args, **kwargs):
        bound_generations.append((int(pose.tvec[0, 0]), generation))
        return -float("inf")

    monkeypatch.setattr(render_script, "_turn_color_frame_score_lower_bound", lower_bound)
    monkeypatch.setattr(render_script, "_turn_torch_scoring_device", lambda _: "cpu")
    monkeypatch.setattr(render_script, "_build_turn_geometry_frame", lambda *args: {"options": ({},)})
    monkeypatch.setattr(render_script, "_score_turn_color_offsets_on_gpu", score)
    monkeypatch.setattr(
        render_script, "_score_turn_color_offsets_for_states_on_gpu",
        lambda pose, options, limit, device, likelihoods, states, **kwargs: [
            score(replace(pose, cube_geometry_state=state)) for state in states
        ],
    )
    monkeypatch.setattr(render_script, "turn_color_predictive_likelihoods", lambda *args: None)
    monkeypatch.setattr(render_script, "solve_pnp_from_low_error_assignments", lambda pose, **kwargs: pose)
    monkeypatch.setattr(render_script, "pose_reprojection_error_is_acceptable", lambda _: True)
    monkeypatch.setattr(render_script, "refine_global_color_profiles_from_initial_state", update_model)
    monkeypatch.setattr(render_script, "_update_sticker_color_posteriors", update_model)
    render_script.recover_cached_pose_symmetries(
        records, trace, pose_prepared=True, recover_orientation=False,
        recognize_moves=True, global_profiles={"red": object()},
        known_initial_colors={label: "red"} if change == "profiles" else None,
        calibrate_initial_colors=change == "profiles",
    )
    # Frame 2 was cached before inspection closed at frame 1; frame 1 was cached
    # before learning from frame 0. Both must be evaluated under the new model.
    target = 2 if change == "profiles" else 1
    signature = render_script.CubeGeometryState.identity().apply_moves(
        ("R",) if change == "profiles" else (),
    ).signature()
    versions = {version for frame, state, version in scores
                if frame == target and state == signature}
    assert 0 in versions
    assert 1 in versions
    bound_target = 3 if change == "profiles" else 2
    assert {version for frame, version in bound_generations if frame == bound_target} == {0, 1}


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


def test_pose_continuity_log_records_y_quarter_turn_and_all_alternatives(tmp_path):
    candidate = replace(
        _lookahead_test_candidate(), rvec=np.asarray([[0.0], [np.pi / 2], [0.0]]),
    )
    tracker = render_script.CubeSymmetryTracker()
    tracker.observe(np.zeros((3, 1)), 147)
    canonical = render_script.canonicalize_pose_symmetry(candidate, tracker)
    diagnostic = canonical.pose_continuity_diagnostics
    assert diagnostic["reference_frame_index"] == 147
    assert diagnostic["input_distance_degrees"] == pytest.approx(90.0)
    assert diagnostic["selected_distance_degrees"] == pytest.approx(0.0, abs=1e-6)
    assert diagnostic["runner_up_margin_degrees"] == pytest.approx(90.0)
    assert len(diagnostic["alternatives"]) == 24
    for alternative in diagnostic["alternatives"]:
        matrix = alternative["symmetry_matrix"]
        expected_rotation = cv2.Rodrigues(candidate.rvec)[0] @ matrix.T
        np.testing.assert_allclose(
            cv2.Rodrigues(alternative["rvec"])[0], expected_rotation, atol=1e-7,
        )
        assert alternative["distance_degrees"] == pytest.approx(np.degrees(
            np.linalg.norm(cv2.Rodrigues(expected_rotation)[0])
        ), abs=1e-5)
    chosen = diagnostic["alternatives"][diagnostic["selected_symmetry_index"]]
    np.testing.assert_allclose(canonical.rvec, chosen["rvec"], atol=1e-7)

    # Color orientation and forced static holds must preserve the historical
    # continuity frame while selected_pose reflects the later orientation.
    colored = render_script.transform_candidate_by_symmetry(
        canonical, render_script.CUBE_SYMMETRY_MATRICES[7],
    )
    held = render_script._hold_before_turning_start(colored, 148, 198)
    path = tmp_path / "scores.jsonl"
    render_script.append_turn_candidate_log(path, held, 148)
    render_script.append_turn_candidate_log(path, None, 149)
    record, missing = map(json.loads, path.read_text().splitlines())
    assert record["candidate_count"] == 0
    assert record["pose_continuity"]["reference_frame_index"] == 147
    np.testing.assert_allclose(record["pose_continuity"]["input_rvec"], candidate.rvec)
    np.testing.assert_allclose(record["pose_continuity"]["output_rvec"], canonical.rvec)
    np.testing.assert_allclose(record["selected_pose"]["rvec"], colored.rvec)
    assert missing["pose_continuity"] is None


def test_pose_continuity_log_exposes_ambiguous_symmetries_and_tracker_reset():
    candidate = replace(
        _lookahead_test_candidate(), rvec=np.asarray([[0.0], [np.pi / 4], [0.0]]),
    )
    tracker = render_script.CubeSymmetryTracker()
    tracker.observe(np.zeros((3, 1)), 147)
    ambiguous = render_script.canonicalize_pose_symmetry(candidate, tracker)
    diagnostic = ambiguous.pose_continuity_diagnostics
    assert diagnostic["selection_reason"] == "closest_reference"
    assert diagnostic["selected_distance_degrees"] == pytest.approx(45.0)
    assert diagnostic["runner_up_margin_degrees"] == pytest.approx(0.0, abs=1e-6)

    tracker.reset()
    first = render_script.canonicalize_pose_symmetry(candidate, tracker)
    diagnostic = first.pose_continuity_diagnostics
    assert diagnostic["selection_reason"] == "no_reference_identity"
    assert diagnostic["reference_frame_index"] is None
    assert diagnostic["reference_rvec"] is None
    assert diagnostic["selected_symmetry_index"] == 0
    assert diagnostic["input_distance_degrees"] is None
    assert diagnostic["runner_up_margin_degrees"] is None
    assert all(item["distance_degrees"] is None for item in diagnostic["alternatives"])
    np.testing.assert_allclose(first.rvec, candidate.rvec, atol=1e-7)


def test_xuanyi_frame_148_color_and_motion_select_correct_y_orientation(monkeypatch):
    # Real mask-sampled RGB evidence and accepted-frame appearance history.
    # The regression target was identified visually, independently of scoring.
    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          "xuanyi_frame_148_continuity.json").read_text())
    evidence_by_rgb = {
        tuple(rgb): np.asarray(fixture["evidence"][label])
        for label, rgb in fixture["observations"].items()
    }
    calls = []

    def evidence(rgb, _profiles):
        calls.append(tuple(rgb))
        return evidence_by_rgb[tuple(rgb)]

    monkeypatch.setattr(render_script, "_rgb_observation_color_evidence", evidence)
    scores = render_script.pose_continuity_color_scores(
        fixture["observations"], fixture["observation_weights"],
        {label: np.asarray(prior) for label, prior in fixture["priors"].items()},
        {"available": True},
        render_script._symmetry_observation_permutations(render_script.canonical_sticker_identities()),
    )
    assert len(calls) == len(fixture["observations"])
    candidate = replace(_lookahead_test_candidate(), rvec=np.array(fixture["input_rvec"]))
    tracker = render_script.CubeSymmetryTracker()
    tracker.observe(np.array(fixture["reference_rvec"]), fixture["reference_frame_index"])
    nearest = render_script.canonicalize_pose_symmetry(candidate, tracker)
    assert nearest.pose_continuity_diagnostics["selected_symmetry_index"] == 19
    for weight in (render_script.POSE_CONTINUITY_COLOR_WEIGHT, 0.25, 0.5, 1.0, 2.0, 4.0):
        monkeypatch.setattr(render_script, "POSE_CONTINUITY_COLOR_WEIGHT", weight)
        corrected = render_script.canonicalize_pose_symmetry(
            candidate, tracker, color_scores=scores,
            frame_index=fixture["frame_index"], source_fps=fixture["source_fps"],
        )
        diagnostic = corrected.pose_continuity_diagnostics
        assert diagnostic["selected_symmetry_index"] == fixture["expected_symmetry"]
        assert diagnostic["selection_reason"] == "angular_and_color"
        assert diagnostic["angular_winner_symmetry_index"] == 19
        assert np.degrees(render_script.rotation_symmetry_distance(
            nearest.rvec, corrected.rvec,
        )) == pytest.approx(90.0)


@pytest.mark.parametrize('frame', (207, 212))
def test_xuanyi_balanced_weight_preserves_camera_orientation(monkeypatch, frame):
    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          f"xuanyi_frame_{frame}_continuity.json").read_text())
    candidate = replace(_lookahead_test_candidate(), rvec=np.array(fixture['input_rvec']))
    tracker = render_script.CubeSymmetryTracker()
    tracker.observe(np.array(fixture['reference_rvec']), fixture['reference_frame_index'])

    def choose():
        return render_script.canonicalize_pose_symmetry(
            candidate, tracker, color_scores=tuple(fixture['color_scores']),
            frame_index=fixture['frame_index'], source_fps=fixture['source_fps'],
        ).pose_continuity_diagnostics['selected_symmetry_index']

    # Visually confirmed yellow top / orange front; test the production default.
    assert choose() == fixture['expected_symmetry']
    monkeypatch.setattr(render_script, 'POSE_CONTINUITY_COLOR_WEIGHT', 1.)
    assert choose() == fixture['previous_weight_symmetry']


def test_pose_continuity_motion_cost_accounts_for_frame_gaps_and_fps():
    tracker = render_script.CubeSymmetryTracker()
    tracker.observe(np.zeros((3, 1)), 10)
    candidate = replace(_lookahead_test_candidate(), rvec=np.array([[0.0], [0.2], [0.0]]))
    diagnostics = []
    for frame, fps in ((11, 30), (12, 30), (12, 60)):
        diagnostics.append(render_script.canonicalize_pose_symmetry(
            candidate, tracker, frame_index=frame, source_fps=fps, color_scores=(1.0,) * 24,
        ).pose_continuity_diagnostics)
    assert diagnostics[1]["alternatives"][0]["angular_score"] == pytest.approx(
        diagnostics[0]["alternatives"][0]["angular_score"] / 2,
    )
    assert diagnostics[2]["alternatives"][0]["angular_score"] == pytest.approx(
        diagnostics[0]["alternatives"][0]["angular_score"],
    )
    invalid = render_script.canonicalize_pose_symmetry(
        candidate, tracker, color_scores=(float("nan"),) * 24,
    )
    assert invalid.pose_continuity_diagnostics["selection_reason"] == "closest_reference"


def test_cached_pose_continuity_log_survives_prepared_pass_and_missing_frame(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(render_script, "pose_reprojection_error_is_acceptable", lambda _: True)
    monkeypatch.setattr(render_script, "pose_reprojection_fit_metric", lambda _: 0.0)
    candidate = _lookahead_test_candidate()
    alternate = replace(candidate, rvec=np.asarray([[0.0], [np.pi / 2], [0.0]]))
    records = [
        SimpleNamespace(frame_index=146, pose_candidate=candidate),
        SimpleNamespace(frame_index=147, pose_candidate=None),
        SimpleNamespace(frame_index=148, pose_candidate=alternate),
    ]
    trace = render_script.StickerColorTrace.create()
    render_script.recover_cached_pose_symmetries(records, trace, recover_orientation=False)
    diagnostic = records[2].pose_candidate.pose_continuity_diagnostics
    assert diagnostic["reference_frame_index"] == 146
    assert diagnostic["input_distance_degrees"] == pytest.approx(90.0)
    path = tmp_path / "scores.jsonl"
    render_script.recover_cached_pose_symmetries(
        records, trace, pose_prepared=True, recover_orientation=False,
        recognize_moves=True, global_profiles={}, turning_start_frame=198,
        turn_candidate_log=path,
    )
    record = json.loads(path.read_text().splitlines()[2])
    assert record["frame_index"] == 148
    assert record["pose_continuity"]["reference_frame_index"] == 146
    assert record["pose_continuity"]["input_distance_degrees"] == pytest.approx(90.0)
    assert record["pose_continuity"]["selected_distance_degrees"] == pytest.approx(0.0, abs=1e-6)


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


def test_cached_pose_recovery_rejects_core_displacement_with_motion_filter(monkeypatch):
    monkeypatch.setattr(render_script, "POSE_TWO_SIDED_MOTION_REJECTION_ENABLED", True)
    monkeypatch.setattr(render_script, "pose_reprojection_error_is_acceptable", lambda _: True)
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


def test_rgb_plot_exposes_whole_color_observations_and_profiles(monkeypatch, tmp_path):
    go = pytest.importorskip("plotly.graph_objects")
    trace = render_script.StickerColorTrace.create()
    trace.observe(0, {trace.identities[0].label: np.array([190, 40, 20])})
    profiles, groups = render_script.build_global_color_profiles_from_samples([
        ('unlabeled', frame, np.array([190, 40, 20])) for frame in range(20)
    ])
    profiles, groups = render_script.refine_global_color_profiles_from_initial_state(
        trace, {trace.identities[0].label: "red"}, profiles, None,
    )
    saved = []
    monkeypatch.setattr(go.Figure, "write_html", lambda figure, *args, **kwargs: saved.append(figure))
    render_script.save_sticker_rgb_html_plot(
        trace, tmp_path / "colors.html", title="Colors",
        global_profiles=profiles, global_observations=groups,
    )
    figure = saved[0]
    whole = [item for item in figure.data if item.legendgroup == "global-colors"]
    assert len(whole) == 2 * len(render_script.GLOBAL_COLOR_NAMES)
    assert all(item.showlegend is not False for item in whole)
    assert all(item.legendgrouptitle.text == "Whole colors and regressions" for item in whole)
    assert next(item for item in whole if item.name == "red global profile regression").x.size > 0
    assert next(item for item in whole if item.name == "blue global profile prior").x.size > 0
    assert figure.layout.updatemenus[0].buttons[0].label == "All stickers and colors"


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


def test_colored_intersection_masks_use_original_pixels_and_global_profiles(monkeypatch):
    source = np.zeros((32, 32, 3), dtype=np.uint8)
    source[4:28, 2:16] = (30, 40, 190)
    source[4:28, 16:30] = (190, 60, 35)
    canvas = np.zeros_like(source)
    cube = np.zeros((32, 32), dtype=bool)
    cube[4:24, :] = True
    predicted = ([np.ones((32, 32), dtype=bool)], {}, np.array([0]), cube)
    identities = render_script.canonical_sticker_identities()
    faces = [
        (identities[0], 1, np.array([[2, 4], [15, 4], [15, 27], [2, 27]])),
        (identities[1], 1, np.array([[16, 4], [29, 4], [29, 27], [16, 27]])),
    ]
    monkeypatch.setattr(render_script, "projected_sticker_faces", lambda *_: faces)
    profiles, _ = render_script.build_global_color_profiles_from_samples([])
    pose = SimpleNamespace(rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)))
    output = render_script.draw_projected_classified_masks(
        canvas, source, (0, 0, 32, 32), predicted, pose, profiles, opacity=1.0,
    )
    np.testing.assert_array_equal(output[10, 8], [0, 0, 255])
    np.testing.assert_array_equal(output[10, 22], [255, 0, 0])
    np.testing.assert_array_equal(output[26, 8], [0, 0, 0])


def test_colored_masks_option_uses_final_cached_profile(monkeypatch):
    seen = []
    monkeypatch.setattr(render_script, "draw_context", lambda image, *args: image)
    monkeypatch.setattr(render_script, "draw_projected_classified_masks",
                        lambda image, source, roi, masks, pose, profiles, **kw: seen.append((source, pose, profiles)) or image)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    record = render_script.NeuralFrameData(
        frame_index=0, frame=frame, roi_xyxy=(0, 0, 16, 16), roi_source="tracker",
        sticker_masks=None, local_points=np.empty((0, 2)), point_confidences=np.empty(0),
        class_ids=np.empty(0), box_confidences=np.empty(0), pose_ready=True,
    )
    args = SimpleNamespace(show_colored_masks=True, view_size=16, show_keypoints=False,
                           show_pnp=False, show_status=False)
    profiles = {"marker": object()}
    render_script.render_frame(frame, None, None, None, args, neural_data=record,
                               cached_final_only=True, global_profiles=profiles)
    assert len(seen) == 1
    assert seen[0][0] is frame and seen[0][2] is profiles


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
    colored_args = render_script.build_parser().parse_args(
        ["--initial-scramble", "", "--source", "frame.png", "--show-colored-masks"]
    )
    assert colored_args.show_colored_masks is True
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


def test_turn_acceptance_uses_more_unified_inliers_as_a_waiver():
    static = {
        "is_baseline": True,
        "layer_axis": 0,
        "layer_side": 1,
        "layer_count": 1,
        "inlier_count": 19,
        "color_error": 2.0,
    }
    color_only_turn = {
        "is_baseline": False,
        "layer_axis": 0,
        "layer_side": 1,
        "layer_count": 1,
        "inlier_count": 20,
        "color_error": 1.5,
    }

    assert render_script.turn_trial_exceeds_static_baseline(
        color_only_turn,
        {(0, 1, 1): static},
    )


def test_turn_without_color_improvement_is_rejected():
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


def test_turn_color_rescue_accepts_a_strong_color_improvement():
    static = {
        "is_baseline": True,
        "layer_axis": 0,
        "layer_side": 1,
        "layer_count": 1,
        "inlier_count": 19,
        "color_error": 2.0,
    }
    color_rescued_turn = {
        "is_baseline": False,
        "layer_axis": 0,
        "layer_side": 1,
        "layer_count": 1,
        "inlier_count": 19,
        "color_error": 1.0,
    }

    assert render_script.turn_trial_exceeds_static_baseline(
        color_rescued_turn,
        {(0, 1, 1): static},
    )


def test_turn_color_rescue_rejects_a_small_color_improvement():
    static = {
        "is_baseline": True,
        "layer_axis": 0,
        "layer_side": 1,
        "layer_count": 1,
        "inlier_count": 19,
        "color_error": 2.0,
    }
    weak_color_turn = {
        "is_baseline": False,
        "layer_axis": 0,
        "layer_side": 1,
        "layer_count": 1,
        "inlier_count": 19,
        "color_error": 1.01,
    }

    assert not render_script.turn_trial_exceeds_static_baseline(
        weak_color_turn,
        {(0, 1, 1): static},
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
    temporally_best = candidate(11.0, 0.080, 1.0)
    rejected = candidate(1.0, 0.101, 0.0)
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
    assert render_script.PNP_POSE_REPROJECTION_ERROR_THRESHOLD == pytest.approx(0.1)
    assert render_script.PNP_POSE_REPROJECTION_ERROR_REJECTION_ENABLED
    assert not render_script.pose_reprojection_error_is_acceptable(rejected)


def test_two_sided_translation_rejection_is_temporarily_disabled(monkeypatch):
    candidate = render_script.StickerPnPCandidate(
        sticker_type="corner", sticker_average_confidence=0.9,
        image_points=np.zeros((4, 2), dtype=np.float32),
        object_points=np.zeros((4, 3), dtype=np.float32),
        projected_sticker_points=np.zeros((4, 2), dtype=np.float32),
        camera_matrix=np.eye(3), dist_coeffs=np.zeros((4, 1)),
        rvec=np.zeros((3, 1)), tvec=np.array([[0.0], [0.0], [200.0]]),
        reprojection_error=0.0, overall_reprojection_error=0.0,
    )
    monkeypatch.setattr(render_script, "select_pose_hypothesis", lambda candidates, *_: candidates[0])
    monkeypatch.setattr(render_script, "canonicalize_pose_symmetry", lambda pose, _: pose)
    monkeypatch.setattr(render_script, "pose_reprojection_error_is_acceptable", lambda _: True)
    assert not render_script.POSE_TWO_SIDED_MOTION_REJECTION_ENABLED
    for enabled in (False, True):
        monkeypatch.setattr(render_script, "POSE_TWO_SIDED_MOTION_REJECTION_ENABLED", enabled)
        poses = [replace(candidate, tvec=np.array([[x], [0.0], [200.0]])) for x in (0.0, 100.0, 0.0)]
        records = [SimpleNamespace(frame_index=i, pose_candidate=pose, pose_candidates=(pose,))
                   for i, pose in enumerate(poses)]
        render_script.recover_cached_pose_symmetries(
            records, render_script.StickerColorTrace.create(), recover_orientation=False,
        )
        assert (records[1].pose_candidate is None) is enabled


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


def test_cached_pipeline_builds_pose_free_profiles_before_continuity(monkeypatch):
    events = []
    initial_trace = render_script.StickerColorTrace.create()
    prepared_trace = render_script.StickerColorTrace.create()
    oriented_trace = render_script.StickerColorTrace.create()
    final_trace = render_script.StickerColorTrace.create()
    profiles = {"marker": object()}
    observations = {"marker": []}
    cache = [SimpleNamespace(
        frame_index=12, frame=np.zeros((8, 8, 3), dtype=np.uint8),
        roi_xyxy=(0, 0, 8, 8), sticker_masks="masks", pose_candidate=None,
    )]
    args = SimpleNamespace(
        show_pnp=True, recognize_moves=True, display=False,
        sticker_mask_erosion=0, color_samples_per_sticker=17,
        color_min_samples_per_sticker=3, debug_color_masks=False,
        debug_pose_symmetry=False, recover_pose_symmetry=True,
        turn_candidate_log=None, turning_start_frame=380,
    )
    colors = {"a": "red"}

    def recover(records, trace, **kwargs):
        assert records is cache
        if kwargs.get("refine_pose"):
            events.append("refine")
            assert trace is initial_trace
            assert not kwargs["recover_orientation"]
            assert kwargs["color_samples_per_sticker"] == 17
            assert kwargs["global_profiles"] is profiles
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
        assert kwargs["turning_start_frame"] == 380
        assert kwargs["use_calibrated_solve_continuity"]
        assert kwargs["color_samples_per_sticker"] == 17
        # Inspection-label output must not replace the full-video EM groups.
        assert "calibration_observations" not in kwargs
        return final_trace

    def build(samples):
        events.append("profiles")
        assert len(samples) == 1
        assert samples[0][:2] == ("M3", 12)
        np.testing.assert_array_equal(samples[0][2], [120, 60, 30])
        return profiles, observations

    monkeypatch.setattr(render_script, "sticker_mask_rgb_observations",
                        lambda *_: [(3, np.array([120, 60, 30], dtype=np.uint8))])
    monkeypatch.setattr(render_script, "recover_cached_pose_symmetries", recover)
    monkeypatch.setattr(render_script, "build_global_color_profiles_from_samples", build)
    result = render_script.process_cached_pose_and_profiles(cache, initial_trace, args, colors, None)
    assert events == ["profiles", "refine", "orientation", "moves"]
    assert result == (final_trace, profiles, observations)


def test_calibrated_solve_continuity_preserves_inspection_and_uses_labeled_profiles(monkeypatch):
    identities = render_script.canonical_sticker_identities()
    target = identities[0].label
    permutations = render_script._symmetry_observation_permutations(identities)
    symmetry, permutation = permutations[15]
    initial = replace(_lookahead_test_candidate(), rvec=np.zeros((3, 1)))
    moving = replace(initial, rvec=render_script.cv2.Rodrigues(symmetry)[0])
    records = [SimpleNamespace(
        frame_index=frame, pose_candidate=pose, pose_candidates=(pose,),
        frame=np.zeros((4, 4, 3), dtype=np.uint8), roi_xyxy=(0, 0, 4, 4), sticker_masks="masks",
    ) for frame, pose in ((10, initial), (11, moving), (12, None))]
    profiles, _ = render_script.build_global_color_profiles_from_samples([])
    profiles['red'] = replace(profiles['red'], labeled_samples=((190., 40., 20.),),
                              labeled_sample_weights=(1.,))
    observations = {10: {target: np.array([190, 40, 20], dtype=np.uint8)}}
    weights = {10: {target: 1.}}
    red = np.eye(6)[render_script.GLOBAL_COLOR_NAMES.index('red')]
    sampled = []

    def sample(frame, roi, masks, pose, **kwargs):
        sampled.append(pose)
        return {permutation[target]: np.array([190, 40, 20], dtype=np.uint8)}

    def evidence(rgb, actual_profiles):
        assert actual_profiles is profiles
        assert actual_profiles['red'].labeled_samples
        return red

    monkeypatch.setattr(render_script, 'sample_projected_sticker_colors', sample)
    monkeypatch.setattr(render_script, 'sticker_view_reliability_weight', lambda *_: 1.)
    monkeypatch.setattr(render_script, '_rgb_observation_color_evidence', evidence)
    render_script.recover_calibrated_solve_continuity(
        records, observations, weights, profiles, 10, source_fps=30., calibration_generation=2,
    )

    assert records[0].pose_candidate is initial
    assert sampled == [moving]
    corrected = records[1].pose_candidate
    np.testing.assert_allclose(render_script.cv2.Rodrigues(corrected.rvec)[0], np.eye(3), atol=1e-7)
    assert records[1].pose_candidates == (corrected,)
    assert records[2].pose_candidate is None
    assert target in observations[11]
    diagnostic = corrected.pose_continuity_diagnostics
    assert diagnostic['color_profile_source'] == 'inspection_labeled'
    assert diagnostic['color_calibration_generation'] == 2
    assert diagnostic['reference_frame_index'] == 10
    assert diagnostic['coordinate_frame'] == 'scramble_anchored_after_color_orientation'


def test_initial_state_profile_refinement_uses_labels_and_excludes_turn_frame():
    trace = render_script.StickerColorTrace.create()
    red, orange = (identity.label for identity in trace.identities[:2])
    trace.observe(0, {red: np.array([190, 80, 30]), orange: np.array([200, 60, 20])})
    trace.observe(1, {red: np.array([10, 240, 10])})
    initial, _ = render_script.build_global_color_profiles_from_samples([])

    refined, grouped = render_script.refine_global_color_profiles_from_initial_state(
        trace, {red: "red", orange: "orange"}, initial, 1,
    )

    assert [entry[0] for entry in grouped["red"]] == [red]
    assert [entry[0] for entry in grouped["orange"]] == [orange]
    assert refined["red"].sample_count == initial["red"].sample_count
    assert refined["orange"].sample_count == initial["orange"].sample_count
    assert refined["red"].fitted is initial["red"].fitted
    assert refined["orange"].fitted is initial["orange"].fitted
    assert refined["blue"] is initial["blue"]
    assert refined["red"].labeled_samples == ((190.0, 80.0, 30.0),)
    assert refined["orange"].labeled_samples == ((200.0, 60.0, 20.0),)


def test_inspection_labels_do_not_refit_or_replace_full_video_regressions(monkeypatch):
    trace = render_script.StickerColorTrace.create()
    label = trace.identities[0].label
    for frame in range(30):
        trace.observe(frame, {label: np.array([190, 40, 20])})
    profiles, unsupervised_groups = render_script.build_global_color_profiles(trace)
    assert profiles['red'].fitted is not None
    before = dict(profiles)
    monkeypatch.setattr(render_script, 'fit_rgb_color_profile',
                        lambda *args, **kwargs: pytest.fail('Inspection labels must not refit regressions'))
    labeled, groups = render_script.refine_global_color_profiles_from_initial_state(
        trace, {label: 'red'}, profiles, 3,
    )
    assert [f for _, f, _ in groups['red']] == [0, 1, 2]
    assert len(labeled['red'].labeled_samples) == 3
    assert max(f for entries in unsupervised_groups.values() for _, f, _ in entries) == 29
    for color in render_script.GLOBAL_COLOR_NAMES:
        assert labeled[color].fitted is before[color].fitted
        assert labeled[color].prior is before[color].prior
        assert labeled[color].sample_count == before[color].sample_count


@pytest.mark.parametrize('missing', [(), (493,), (493, 494), (0, 492, 493)])
def test_explicit_inspection_freezes_labels_before_search_and_never_uses_solving_frames(monkeypatch, missing, tmp_path):
    trace = render_script.StickerColorTrace.create()
    label = trace.identities[0].label
    frames = (0, 492, 493, 494, 495, 496, 497, 516)
    candidate = _lookahead_test_candidate()
    records = []
    for frame in frames:
        rgb = np.array([190, 40, 20] if frame <= 493 else [20, 190, 40])
        trace.observe(frame, {label: rgb})
        pose = None if frame in missing else replace(candidate, tvec=np.array([[float(frame)], [0.0], [200.0]]))
        records.append(SimpleNamespace(frame_index=frame, pose_candidate=pose,
                                       pose_candidates=(pose,) if pose is not None else (), sticker_masks=None))
    profiles, unsupervised = render_script.build_global_color_profiles(trace)
    base = dict(profiles)
    labeled_observations = {}
    searches = []
    expected = [frame for frame in frames if frame <= 493 and frame not in missing]
    continuity_calls = []

    def calibrated_continuity(records, raw, weights, actual_profiles, boundary, **kwargs):
        assert not searches
        assert actual_profiles is profiles
        assert boundary == 493
        assert len(profiles['red'].labeled_samples) == len(expected)
        continuity_calls.append(boundary)

    def search(pose, **kwargs):
        frame = int(pose.tvec[0, 0])
        profile = kwargs['turn_global_profiles']['red']
        searches.append((frame, profile))
        assert len(profile.labeled_samples) == len(expected)
        moves = ('R',) if frame == 495 else (("R'",) if frame == 496 else ())
        return replace(pose, turn_state_moves=moves)

    monkeypatch.setattr(render_script, '_turn_torch_scoring_device', lambda _: 'cpu')
    monkeypatch.setattr(render_script, 'recover_calibrated_solve_continuity', calibrated_continuity)
    monkeypatch.setattr(render_script, '_build_turn_geometry_frame', lambda *args: None)
    monkeypatch.setattr(render_script, 'solve_pnp_from_low_error_assignments', search)
    monkeypatch.setattr(render_script, 'pose_reprojection_error_is_acceptable', lambda _: True)
    path = tmp_path / 'scores.jsonl'
    corrected = render_script.recover_cached_pose_symmetries(
        records, trace, pose_prepared=True, recover_orientation=False, recognize_moves=True,
        turning_start_frame=493, global_profiles=profiles, known_initial_colors={label: 'red'},
        calibrate_initial_colors=True, calibration_observations=labeled_observations, turn_candidate_log=path,
        use_calibrated_solve_continuity=True,
    )
    assert continuity_calls == ([493] if expected else [])
    assert searches
    assert all(f > 493 and p is profiles['red'] for f, p in searches)
    assert [f for _, f, _ in labeled_observations.get('red', [])] == expected
    for color in render_script.GLOBAL_COLOR_NAMES:
        assert profiles[color].fitted is base[color].fitted
        assert profiles[color].sample_count == base[color].sample_count
    assert max(f for entries in unsupervised.values() for _, f, _ in entries) == 516
    assert any(f == 516 for f, _ in corrected.observations[label])
    logs = [json.loads(line) for line in path.read_text().splitlines()]
    moving = [r['search'] for r in logs if r['frame_index'] > 493 and r['has_pose']]
    assert all(s['inspection_calibration_complete'] for s in moving)
    assert len({s['color_calibration_generation'] for s in moving}) == 1


def test_labeled_sample_evidence_is_bounded_and_falls_back_at_distance():
    profiles, _ = render_script.build_global_color_profiles_from_samples([])
    red_index = render_script.GLOBAL_COLOR_NAMES.index("red")
    point = np.array([200, 50, 50], dtype=np.uint8)
    baseline = render_script._rgb_observation_color_evidence(point, profiles)[red_index]
    single = dict(profiles)
    single["red"] = replace(profiles["red"], labeled_samples=((200., 50., 50.),),
                            labeled_sample_weights=(1.,))
    supported = dict(profiles)
    supported["red"] = replace(profiles["red"], labeled_samples=((200., 50., 50.),) * 3,
                               labeled_sample_weights=(1.,) * 3)
    single_evidence = render_script._rgb_observation_color_evidence(point, single)[red_index]
    supported_evidence = render_script._rgb_observation_color_evidence(point, supported)[red_index]
    assert baseline < single_evidence < supported_evidence <= 1.0
    distant = np.array([255, 255, 255], dtype=np.uint8)
    assert render_script._rgb_observation_color_evidence(distant, supported)[red_index] == pytest.approx(
        render_script._rgb_observation_color_evidence(distant, profiles)[red_index], abs=1e-5,
    )


@pytest.mark.parametrize("first_turn", [0, 1, None, "reversal"])
def test_move_search_calibrates_once_without_replaying_frames(monkeypatch, first_turn, capsys):
    trace = render_script.StickerColorTrace.create()
    label = trace.identities[0].label
    candidate = render_script.StickerPnPCandidate(
        sticker_type="corner", sticker_average_confidence=0.9,
        image_points=np.zeros((4, 2), dtype=np.float32),
        object_points=np.zeros((4, 3), dtype=np.float32),
        projected_sticker_points=np.zeros((4, 2), dtype=np.float32),
        camera_matrix=np.eye(3), dist_coeffs=np.zeros((4, 1)),
        rvec=np.zeros((3, 1)), tvec=np.asarray([[0.0], [0.0], [200.0]]),
        reprojection_error=0.0, overall_reprojection_error=0.0,
    )
    records = []
    for frame in range(5 if first_turn == "reversal" else 3):
        trace.observe(frame, {label: np.array([180, 40, 20])})
        records.append(SimpleNamespace(frame_index=frame, pose_candidate=candidate,
                                       pose_candidates=(candidate,), sticker_masks=None))
    profiles, observations = render_script.build_global_color_profiles_from_samples([])
    original = profiles["red"]
    searches = []

    def search(pose, **kwargs):
        index = len(searches)
        searches.append(kwargs["turn_global_profiles"]["red"])
        angle = ({0: 30.0, 1: -30.0, 3: 60.0}.get(index, 0.0)
                 if first_turn == "reversal" else (60.0 if index == first_turn else 0.0))
        rotations = (render_script.CubeLayerRotation(0, 1, angle),) if angle else ()
        return replace(pose, layer_rotations=rotations)

    monkeypatch.setattr(render_script, "solve_pnp_from_low_error_assignments", search)
    monkeypatch.setattr(render_script, "pose_reprojection_error_is_acceptable", lambda _: True)
    render_script.recover_cached_pose_symmetries(
        records, trace, pose_prepared=True, recover_orientation=False,
        recognize_moves=True, global_profiles=profiles, known_initial_colors={label: "red"},
        calibrate_initial_colors=True, calibration_observations=observations,
    )
    message = capsys.readouterr().out
    assert message.count("Color calibration stopped:") == 1
    assert ("end of input without a committed move" if first_turn is None
            else f"first committed move at frame {3 if first_turn == 'reversal' else first_turn}") in message
    assert len(searches) == (5 if first_turn == "reversal" else 3)
    if first_turn == "reversal":
        assert all(profile is original for profile in searches[:4])
        assert searches[4] is profiles["red"]
        assert profiles["red"] is not original
        assert [frame for _, frame, _ in observations["red"]] == [2]
    elif first_turn == 1:
        assert searches[:2] == [original, original]
        assert searches[2] is profiles["red"]
        assert profiles["red"] is not original
        assert [frame for _, frame, _ in observations["red"]] == [0]
    elif first_turn == 0:
        assert all(profile is original for profile in searches)
        assert profiles["red"] is original
    else:
        assert all(profile is original for profile in searches)
        assert [frame for _, frame, _ in observations["red"]] == [0, 1, 2]


@pytest.mark.parametrize("inverse_frame", [2, 21, 22])
def test_calibration_does_not_reopen_after_a_committed_move_is_reversed(monkeypatch, inverse_frame, capsys):
    trace = render_script.StickerColorTrace.create()
    label = trace.identities[0].label
    candidate = render_script.StickerPnPCandidate(
        sticker_type="corner", sticker_average_confidence=0.9,
        image_points=np.zeros((4, 2), dtype=np.float32),
        object_points=np.zeros((4, 3), dtype=np.float32),
        projected_sticker_points=np.zeros((4, 2), dtype=np.float32),
        camera_matrix=np.eye(3), dist_coeffs=np.zeros((4, 1)),
        rvec=np.zeros((3, 1)), tvec=np.asarray([[0.0], [0.0], [200.0]]),
        reprojection_error=0.0, overall_reprojection_error=0.0,
    )
    records = []
    for frame in range(25):
        trace.observe(frame, {label: np.array([180, 40, 20])})
        records.append(SimpleNamespace(frame_index=frame, pose_candidate=candidate,
                                       pose_candidates=(candidate,), sticker_masks=None))
    profiles, observations = render_script.build_global_color_profiles_from_samples([])
    searches = []

    def search(pose, **kwargs):
        index = len(searches)
        searches.append(kwargs["turn_global_profiles"]["red"])
        # Opposite faces commute: this restores the initial state after the
        # first turn, but must never make these solving frames label sources.
        moves = ("R", "L") if index == 1 else (("R'", "L'") if index == inverse_frame else ())
        return replace(pose, turn_state_moves=moves)

    monkeypatch.setattr(render_script, "solve_pnp_from_low_error_assignments", search)
    monkeypatch.setattr(render_script, "pose_reprojection_error_is_acceptable", lambda _: True)
    render_script.recover_cached_pose_symmetries(
        records, trace, pose_prepared=True, recover_orientation=False,
        recognize_moves=True, global_profiles=profiles, known_initial_colors={label: "red"},
        calibrate_initial_colors=True, calibration_observations=observations,
    )
    output = capsys.readouterr().out
    assert len(searches) == 25
    assert "moves canceled" not in output
    assert "first committed move at frame 1" in output
    assert [frame for _, frame, _ in observations["red"]] == [0]
    assert searches[1] is not profiles["red"]
    assert all(profile is profiles["red"] for profile in searches[2:])


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
    means, robust, p90, inliers, layer_inliers, valid_trials = [v.numpy() for v in values]
    for index in (0, 1):
        finite = np.isfinite(distances[index])
        normalized = distances[index, finite] / 10.0
        assert means[index] == pytest.approx(float(np.mean(normalized)), abs=1e-7)
        assert p90[index] == pytest.approx(float(np.percentile(normalized, 90)), abs=1e-7)
        assert robust[index] == pytest.approx(float(np.median(normalized) + 0.5 * np.percentile(normalized, 90)), abs=1e-7)
        assert inliers[index] == int(np.count_nonzero(finite & (distances[index] <= 2.0)))
        layer = render_script.cube_layer_mask(objects[index], 0, int(sides[index]), 1)
        assert layer_inliers[index] == int(np.count_nonzero(finite & layer & (distances[index] <= 2.0)))
    assert np.all(np.isinf(means[2:]))
    assert np.all(np.isinf(robust[2:]))
    assert np.all(np.isinf(p90[2:]))
    assert np.array_equal(inliers[2:], [0, 0])
    assert np.array_equal(layer_inliers[2:], [0, 0])
    assert np.array_equal(valid_trials, [True, True, False, False])


def test_move_lookahead_preserves_a_complete_no_turn_path_until_later_evidence():
    initial = render_script.CubeGeometryState.identity()
    static = _lookahead_trial(2.0, baseline=True)
    action = _lookahead_trial(0.0, ("R",))
    debug = {}

    def evaluate(frame, states):
        if frame == 1:
            return tuple((
                (_lookahead_trial(2.0, baseline=True), _lookahead_trial(0.0, ("L",)))
                if state.signature() == initial.signature() else (_lookahead_trial(0.0),)
            ) for state in states)
        return tuple((_lookahead_trial(0.0 if state.signature() == initial.signature() else 10.0),)
                     for state in states)

    paths = render_script.search_move_lookahead_paths(
        initial, (static, action), (1, 2), evaluate, state_batch_size=1, diagnostics=debug,
    )
    winner = min(paths, key=lambda path: render_script.move_lookahead_weighted_score(
        path.first_trial["state_score"], path.future_scores))
    assert winner.first_trial is static
    assert winner.move_count == 0
    assert winner.future_scores == (2.0, 0.0)
    assert debug["initial_no_turn_score"] == pytest.approx(render_script.move_lookahead_path_score(winner))
    assert debug["optimality"] == "proven"
    assert all(not trial["state_moves"] for trial in winner.future_trials)


def test_no_turn_comparison_does_not_depend_on_exact_static_geometry():
    initial = render_script.CubeGeometryState.identity()
    partial_hold = _lookahead_trial(2.0)  # Non-baseline geometry, zero state offset.
    paths = render_script.search_move_lookahead_paths(
        initial, (partial_hold, _lookahead_trial(0.0, ("R",))), (1,),
        lambda frame, states: tuple((_lookahead_trial(
            0.0 if state.signature() == initial.signature() else 10.0),) for state in states),
        state_batch_size=1,
    )
    winner = min(paths, key=lambda path: render_script.move_lookahead_weighted_score(
        path.first_trial["state_score"], path.future_scores))
    assert winner.first_trial is partial_hold
    assert winner.move_count == 0


def test_sequence_move_cost_rejects_tiny_gains_without_excluding_a_move():
    initial = render_script.CubeGeometryState.identity()
    static = _lookahead_trial(1.0, baseline=True)
    action = _lookahead_trial(0.999, ("R",))
    paths = render_script.search_move_lookahead_paths(
        initial, (static, action), (), lambda *args: (),
    )
    assert any(path.first_trial is action for path in paths)
    assert min(paths, key=render_script.move_lookahead_path_score).first_trial is static


def test_future_evidence_can_pay_for_a_move_with_a_worse_current_fit():
    initial = render_script.CubeGeometryState.identity()
    static = _lookahead_trial(1.0, baseline=True)
    action = _lookahead_trial(1.2, ("R",))
    paths = render_script.search_move_lookahead_paths(
        initial, (static, action), (1, 2),
        lambda frame, states: tuple((_lookahead_trial(
            1.0 if state.signature() == initial.signature() else 0.0),) for state in states),
    )
    winner = min(paths, key=render_script.move_lookahead_path_score)
    assert winner.first_trial is action
    assert render_script.move_lookahead_path_score(winner) == pytest.approx(0.36 + 0.01)


def test_move_cost_is_charged_once_per_turn_outside_temporal_weights():
    initial = render_script.CubeGeometryState.identity()
    paths = render_script.search_move_lookahead_paths(
        initial, (_lookahead_trial(0.0, ("R",)),), (1, 2),
        lambda frame, states: ((_lookahead_trial(0.0, ("U",) if frame == 2 else ()),),) * len(states),
        diagnostics={},
    )
    path = paths[0]
    assert path.move_count == 2
    assert path.state.signature() == initial.apply_moves(("R", "U")).signature()
    forecast = render_script.move_lookahead_path_log(path, (70, 71, 72))
    assert forecast["observation_score"] == 0.0
    assert forecast["move_cost"] == pytest.approx(0.02)
    assert forecast["total_score"] == pytest.approx(0.02)
    assert [step["move_cost"] for step in forecast["steps"]] == pytest.approx([0.01, 0.0, 0.01])
    assert sum(step["weighted_score"] for step in forecast["steps"]) == pytest.approx(forecast["total_score"])


@pytest.mark.parametrize("state_batch_size", [1, 5, 32])
def test_adaptive_search_can_choose_the_24th_ranked_change_without_shortlist(state_batch_size):
    initial = render_script.CubeGeometryState.identity()
    moves = [base + suffix for base in ("R", "L", "U", "D", "F", "B", "Rw", "Lw", "Uw", "Dw", "Fw", "Bw")
             for suffix in ("", "'")]
    target = initial.apply_move(moves[-1]).signature()
    static = dict(_lookahead_trial(0.2, baseline=True), color_error=0.2)
    changes = tuple(dict(_lookahead_trial(0.4 + i / 100.0, (move,)), color_error=i / 100.0)
                    for i, move in enumerate(moves))
    debug = {}
    paths = render_script.search_move_lookahead_paths(
        initial, (static, *changes), (1,),
        lambda frame, states: tuple((_lookahead_trial(0.0 if state.signature() == target else 5.0),)
                                    for state in states),
        state_batch_size=state_batch_size, state_change_limit=None,
        frame_score_lower_bounds=(0.0, 0.0), diagnostics=debug,
    )
    winner = min(paths, key=render_script.move_lookahead_path_score)
    assert winner.first_trial is changes[-1]
    assert winner.first_trial["state_moves"] == ("Bw'",)
    assert debug["optimality"] == "proven"


def test_adaptive_search_collapses_12_holds_before_expanding():
    initial = render_script.CubeGeometryState.identity()
    holds = tuple(dict(_lookahead_trial(0.2 + i / 10.0, baseline=i == 0),
                       trial_id=f"{i}:+0", color_error=0.2) for i in range(12))
    evaluated = []

    def evaluate(frame, states):
        evaluated.append((frame, len(states)))
        return ((_lookahead_trial(0.1),),) * len(states)

    debug = {}
    paths = render_script.search_move_lookahead_paths(
        initial, holds, (1,), evaluate, diagnostics=debug, frame_score_lower_bounds=(0.0, 0.0),
    )
    assert min(paths, key=render_script.move_lookahead_path_score).first_trial is holds[0]
    assert debug["depths"][0]["collapsed_holds"] == 11
    assert debug["depths"][0]["expanded_paths"] == 1
    assert evaluated == [(1, 1)]


@pytest.mark.parametrize("bounds", [None, (0.0, -2.0)])
def test_adaptive_bound_allows_negative_future_error_to_rescue_a_prefix(bounds):
    initial = render_script.CubeGeometryState.identity()
    hold = _lookahead_trial(0.0, baseline=True)
    action = _lookahead_trial(1.0, ("R",))
    paths = render_script.search_move_lookahead_paths(
        initial, (hold, action), (1,),
        lambda frame, states: tuple((_lookahead_trial(
            0.0 if state.signature() == initial.signature() else -2.0),) for state in states),
        frame_score_lower_bounds=bounds,
    )
    assert min(paths, key=render_script.move_lookahead_path_score).first_trial is action


def test_adaptive_bound_prunes_only_when_a_complete_solution_cannot_be_beaten():
    initial = render_script.CubeGeometryState.identity()
    hold = _lookahead_trial(0.0, baseline=True)
    action = _lookahead_trial(1.0, ("R",))
    evaluated = []

    def evaluate(frame, states):
        evaluated.extend(state.signature() for state in states)
        return ((_lookahead_trial(0.0),),) * len(states)

    debug = {}
    paths = render_script.search_move_lookahead_paths(
        initial, (hold, action), (1,), evaluate,
        frame_score_lower_bounds=(0.0, 0.0), diagnostics=debug,
    )
    assert min(paths, key=render_script.move_lookahead_path_score).first_trial is hold
    assert evaluated == [initial.signature()]
    assert debug["roots"][1]["status"] == "bound_pruned"
    example = debug["roots"][1]["pruning_examples"][0]
    assert example["lower_bound"] > example["incumbent_score"]
    assert debug["optimality"] == "proven"


def test_optimistic_color_bound_is_valid_for_every_state_and_offset():
    torch = pytest.importorskip("torch")
    candidate = replace(_lookahead_test_candidate(), rvec=np.asarray([[0.18], [-0.25], [0.12]]))
    faces = render_script.projected_sticker_faces(candidate, candidate.rvec, candidate.tvec)
    centers = tuple(polygon.mean(axis=0) for _, _, polygon in faces)
    candidate = replace(candidate, sticker_mask_centers=centers,
                        sticker_mask_colors=(np.asarray([255, 0, 0], dtype=np.uint8),) * len(centers))
    options = tuple(dict(layer_axis=axis, layer_side=side, layer_count=width,
                         layer_rotations=(render_script.CubeLayerRotation(axis, side, angle, width),) if angle else (),
                         label=f"option-{i}", is_baseline=not angle, combined_mean_error=0.03 + i / 100.0,
                         assignment_projected_points=None, assignment_object_points=None,
                         assignment_projected_class_ids=None)
                    for i, (axis, side, width, angle) in enumerate(((2, -1, 1, 0.0), (0, 1, 2, 15.0), (1, -1, 1, -21.0))))
    initial = render_script.CubeGeometryState.identity()
    states = (initial, initial.apply_moves(("R", "U")), initial.apply_moves(("Bw", "L'")))
    # Densities greater than one deliberately exercise negative color errors.
    likelihoods = np.random.default_rng(12).uniform(0.01, 2.0, (54, len(centers)))
    lower = render_script._turn_color_frame_score_lower_bound(
        candidate, options, 10.0, torch.device("cpu"), likelihoods,
    )
    scores = render_script._score_turn_color_offsets_for_states_on_gpu(
        candidate, options, 10.0, torch.device("cpu"), likelihoods, states,
    )
    assert np.isfinite(lower)
    assert all(lower <= trial["state_score"] for trials in scores for trial in trials)


def test_combined_shortlist_keeps_hold_and_geometry_favored_root_change():
    initial = render_script.CubeGeometryState.identity()
    moves = [base + suffix for base in ("R", "L", "U", "D", "F", "B", "Rw", "Lw", "Uw", "Dw", "Fw", "Bw")
             for suffix in ("", "'")]
    target = initial.apply_move(moves[-1]).signature()
    hold = dict(_lookahead_trial(0.2, baseline=True), color_error=3.9)
    changes = tuple(dict(_lookahead_trial(0.4 + i / 100.0, (move,)), color_error=i / 100.0)
                    for i, move in enumerate(moves))
    # Last by color, but best after including geometry: it must survive.
    changes = (*changes[:-1], dict(_lookahead_trial(0.3, (moves[-1],)), color_error=0.23))
    debug = {}
    paths = render_script.search_move_lookahead_paths(
        initial, (hold, *changes), (1,),
        lambda frame, states: tuple((_lookahead_trial(0.0 if state.signature() == target else 5.0),)
                                    for state in states),
        frame_score_lower_bounds=(0.0, 0.0), diagnostics=debug,
    )
    assert min(paths, key=render_script.move_lookahead_path_score).first_trial is changes[-1]
    assert debug["state_change_limit"] == 10
    assert debug["shortlist_rank"] == "geometry_plus_color"
    assert debug["depths"][0]["expanded_paths"] == 11
    assert debug["depths"][0]["shortlist_pruned_paths"] == 14
    assert sum(root["status"] == "shortlist_pruned" for root in debug["roots"]) == 14
    assert debug["optimality"] == "proven_within_shortlist"


def test_combined_shortlist_applies_again_in_future_frames():
    initial = render_script.CubeGeometryState.identity()
    moves = [base + suffix for base in ("R", "L", "U", "D", "F", "B", "Rw", "Lw", "Uw", "Dw", "Fw", "Bw")
             for suffix in ("", "'")]
    hold = dict(_lookahead_trial(10.0), color_error=0.0)
    changes = tuple(dict(_lookahead_trial(4.0 if i < 10 else 0.3, (move,)), color_error=i / 100.0)
                    for i, move in enumerate(moves))
    debug = {}
    paths = render_script.search_move_lookahead_paths(
        initial, (_lookahead_trial(0.0, baseline=True),), (1,),
        lambda frame, states: ((hold, *changes),) * len(states), diagnostics=debug,
    )
    winner = min(paths, key=render_script.move_lookahead_path_score)
    assert winner.future_trials[0]["state_moves"] == ("B",)
    assert debug["depths"][1]["expanded_paths"] == 11
    assert debug["depths"][1]["shortlist_pruned_paths"] == 14
    assert debug["optimality"] == "proven_within_shortlist"


@pytest.mark.parametrize("known_colors", [True, False])
def test_unreliable_color_holds_do_not_train_color_models(monkeypatch, known_colors):
    candidate = _lookahead_test_candidate()
    trace = render_script.StickerColorTrace.create()
    label = trace.identities[0].label
    records = []
    for frame in range(2):
        trace.observe(frame, {label: np.asarray([180, 40, 20], dtype=np.uint8)})
        records.append(SimpleNamespace(frame_index=frame, pose_candidate=candidate,
                                       pose_candidates=(candidate,), sticker_masks=None))
    learned = []
    monkeypatch.setattr(render_script, "_turn_torch_scoring_device", lambda _: "cpu")
    monkeypatch.setattr(render_script, "_build_turn_geometry_frame", lambda *args: None)
    monkeypatch.setattr(render_script, "solve_pnp_from_low_error_assignments", lambda pose, **kwargs:
                        replace(pose, turn_state_moves=("R",), turn_fit_evaluated=True, turn_fit_color_score=float("inf")))
    monkeypatch.setattr(render_script, "pose_reprojection_error_is_acceptable", lambda _: True)
    monkeypatch.setattr(render_script, "_update_sticker_color_posteriors", lambda *args, **kwargs: learned.append("posterior"))
    monkeypatch.setattr(render_script, "refine_global_color_profiles_from_initial_state",
                        lambda *args, **kwargs: learned.append("profiles"))
    render_script.recover_cached_pose_symmetries(
        records, trace, pose_prepared=True, recover_orientation=False, recognize_moves=True,
        global_profiles={}, known_initial_colors={label: "red"} if known_colors else None,
        calibrate_initial_colors=known_colors,
    )
    assert all(record.pose_candidate.turn_color_unreliable for record in records)
    assert learned == []


def test_selected_color_score_tracks_the_trial_instead_of_the_static_reference():
    candidate = replace(_lookahead_test_candidate(), turn_fit_color_score=10.0, turn_color_unreliable=True)
    selected = render_script._candidate_with_turn_trial(candidate, candidate, dict(
        trial_id="0:-90", is_baseline=False, layer_axis=0, layer_side=1, layer_count=1,
        layer_rotations=(), color_offset=-90.0, state_moves=("R",), color_error=0.4,
    ))
    assert selected.turn_fit_color_score == 0.4
    assert selected.turn_color_unreliable is False


def test_local_accepted_hold_clears_an_old_unreliable_color_flag(monkeypatch):
    torch = pytest.importorskip("torch")
    candidate = replace(_lookahead_test_candidate(), turn_color_unreliable=True)
    option = dict(layer_axis=0, layer_side=1, layer_count=1, layer_rotations=(),
                  label="static R", is_baseline=True, combined_mean_error=0.03,
                  keypoint_error=0.01, sticker_error=0.02, robust_error=0.04,
                  assignment_projected_points=None, assignment_object_points=None,
                  assignment_projected_class_ids=None)
    monkeypatch.setattr(render_script, "_turn_torch_scoring_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(render_script, "turn_color_predictive_likelihoods", lambda *args: None)
    result = render_script.recover_misaligned_cube_layers_from_geometry(
        candidate, 10.0, dict(options=(option,), static_trials=(option,)),
    )
    assert result.turn_state_moves == ()
    assert result.turn_color_unreliable is False


def test_state_budget_returns_a_complete_forecast_and_honest_bound():
    initial = render_script.CubeGeometryState.identity()
    hold = _lookahead_trial(5.0, baseline=True)
    change = _lookahead_trial(0.0, ("R",))
    evaluated = []

    def evaluate(frame, states):
        evaluated.extend((frame, state.signature()) for state in states)
        return ((_lookahead_trial(5.0),),) * len(states)

    debug = {}
    paths = render_script.search_move_lookahead_paths(
        initial, (hold, change), (1, 2), evaluate, diagnostics=debug,
        state_evaluation_budget=3, frame_score_lower_bounds=(0.0, 0.0, 0.0),
    )
    winner = min(paths, key=render_script.move_lookahead_path_score)
    assert winner.first_trial is hold
    assert len(winner.future_scores) == len(winner.future_trials) == 2
    assert debug['budget_exhausted']
    assert debug['optimality'] == 'budget_exhausted'
    assert debug['termination_reason'] == 'state_budget'
    assert debug['evaluated_states'] == 1 + len(evaluated) == 3
    assert debug['best_lower_bound'] <= render_script.move_lookahead_path_score(winner)
    assert debug['optimality_gap'] > 0


def test_greedy_seed_supplies_complete_path_without_any_hold():
    initial = render_script.CubeGeometryState.identity()
    debug = {}
    paths = render_script.search_move_lookahead_paths(
        initial, (_lookahead_trial(1.0, ('R',)),), (1, 2),
        lambda frame, states: ((_lookahead_trial(1.0, ('U',)),),) * len(states),
        diagnostics=debug, state_evaluation_budget=3, seed_greedy=True,
    )
    winner = min(paths, key=render_script.move_lookahead_path_score)
    assert winner.move_count == 3
    assert len(winner.future_trials) == 2
    assert winner.state.signature() == initial.apply_moves(('R', 'U', 'U')).signature()
    assert debug['seeds'][0]['status'] == 'complete'
    assert debug['evaluated_states'] <= 3


@pytest.mark.parametrize('seed', ('warm_start', 'greedy'))
def test_seeds_use_combined_shortlist_and_repair_excluded_warm_start(seed):
    initial = render_script.CubeGeometryState.identity()
    hold = dict(_lookahead_trial(5.0, baseline=True), color_error=0.1)
    first = dict(_lookahead_trial(1.0, ('R',)), color_error=0.2)
    excluded = dict(_lookahead_trial(2.0, ('U',)), color_error=0.1)
    debug = {}
    paths = render_script.search_move_lookahead_paths(
        initial, (hold, first, excluded), (1,),
        lambda frame, states: ((_lookahead_trial(1.0),),) * len(states),
        warm_start_moves=(('U',), ()) if seed == 'warm_start' else None,
        seed_greedy=seed == 'greedy', state_change_limit=1, diagnostics=debug,
    )
    winner = min(paths, key=render_script.move_lookahead_path_score)
    assert winner.first_trial is first
    assert all(path.first_trial is not excluded for path in paths)
    assert debug['seeds'][0]['source'] == ('previous_forecast' if seed == 'warm_start' else 'greedy')
    assert debug['seeds'][0]['repaired_steps'] == (1 if seed == 'warm_start' else 0)
    assert debug['optimality'] == 'proven_within_shortlist'


def test_budget_stop_without_complete_path_is_not_reported_as_proven_failure():
    debug = {}
    paths = render_script.search_move_lookahead_paths(
        render_script.CubeGeometryState.identity(),
        (_lookahead_trial(0.0, ('R',)), _lookahead_trial(1.0, ('U',))), (1,),
        lambda frame, states: ((dict(_lookahead_trial(0.0), color_error=float("inf")),),) * len(states),
        seed_greedy=True, state_evaluation_budget=2, diagnostics=debug,
    )
    assert not paths
    assert debug['budget_exhausted']
    assert debug['optimality'] == 'budget_exhausted_no_path'
    assert debug['evaluated_states'] == 2
    assert any(root['status'] == 'budget_unfinished' for root in debug['roots'])


def test_search_budget_parser_and_validation():
    parser = render_script.build_parser()
    args = ['--source', 'frame.png', '--initial-scramble', '']
    assert parser.parse_args(args).turn_search_state_budget == 4096
    assert parser.parse_args([*args, '--turn-search-state-budget', '256']).turn_search_state_budget == 256
    assert parser.parse_args([*args, '--turn-search-state-budget', 'unlimited']).turn_search_state_budget is None
    for bad in ('0', '10', '-1', '1.5', 'nan'):
        with pytest.raises(SystemExit):
            parser.parse_args([*args, '--turn-search-state-budget', bad])
    with pytest.raises(ValueError, match='at least one state'):
        render_script.search_move_lookahead_paths(
            render_script.CubeGeometryState.identity(), (), (1, 2), lambda *args: (), state_evaluation_budget=2,
        )


def test_turning_start_frame_parser_is_optional_and_nonnegative():
    parser = render_script.build_parser()
    args = ['--source', 'frame.png', '--initial-scramble', '']
    assert parser.parse_args(args).turning_start_frame is None
    assert parser.parse_args([*args, '--turning-start-frame', '0']).turning_start_frame == 0
    assert parser.parse_args([*args, '--turning-start-frame', '380']).turning_start_frame == 380
    for bad in ('-1', '1.5', 'nan', 'unlimited'):
        with pytest.raises(SystemExit):
            parser.parse_args([*args, '--turning-start-frame', bad])
    with pytest.raises(ValueError, match='Turning start frame'):
        render_script.recover_cached_pose_symmetries([], render_script.StickerColorTrace.create(), turning_start_frame=-1)


def test_turning_start_frame_overrides_lookahead_through_inclusive_boundary(tmp_path, monkeypatch):
    path = _write_search_debug_log(tmp_path, monkeypatch, turning_start_frame=12)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r['frame_index'] for r in rows] == [10, 12, 15]
    for row in rows[:2]:
        assert row['selected_committed_moves'] == []
        assert row['selected_layer_rotations'] == []
        assert row['selected_color_offset_degrees'] == 0.0
        assert row['search']['status'] == 'forced_static'
        assert row['search']['turning_start_frame'] == 12
        assert row['search']['evaluated_states'] == 0
    assert rows[2]['search']['status'] == 'searched'
    assert rows[2]['search']['turning_start_frame'] == 12
    assert rows[2]['search']['previous_move'] is None


def test_turning_start_frame_preserves_state_skips_fitting_and_allows_later_moves(tmp_path, monkeypatch):
    candidate = replace(_lookahead_test_candidate(),
                        layer_rotations=(render_script.CubeLayerRotation(0, 1, 21.0, 1),),
                        sticker_layer_rotations=(render_script.CubeLayerRotation(0, 1, 111.0, 1),),
                        turn_state_moves=('U',), turn_color_offset_degrees=90.0)
    records = []
    for f in (0, 378, 379, 380, 381, 383):
        pose = None if f == 378 else replace(candidate, tvec=np.asarray([[float(f)], [0.0], [200.0]]))
        records.append(SimpleNamespace(frame_index=f, pose_candidate=pose,
                                       pose_candidates=(pose,) if pose is not None else (), sticker_masks=None))
    fitted = []
    prepared = []

    def fit(pose, **kwargs):
        fitted.append(int(pose.tvec[0, 0]))
        assert kwargs['recover_layer_turns']
        return replace(pose, turn_state_moves=('R',))

    monkeypatch.setattr(render_script, '_turn_torch_scoring_device', lambda _: 'cpu')
    monkeypatch.setattr(render_script, '_build_turn_geometry_frame',
                        lambda pose, *args: prepared.append(int(pose.tvec[0, 0])))
    monkeypatch.setattr(render_script, 'solve_pnp_from_low_error_assignments', fit)
    monkeypatch.setattr(render_script, 'pose_reprojection_error_is_acceptable', lambda _: True)
    log = tmp_path / 'scores.jsonl'
    render_script.recover_cached_pose_symmetries(
        records, render_script.StickerColorTrace.create(), pose_prepared=True,
        recover_orientation=False, recognize_moves=True, turning_start_frame=380,
        global_profiles={}, known_initial_colors={}, turn_candidate_log=log,
    )
    initial = render_script.CubeGeometryState.identity()
    assert fitted == [381, 383]
    assert prepared and all(f > 380 for f in prepared)
    for record in records[:4]:
        pose = record.pose_candidate
        if pose is None:
            continue
        assert render_script.completed_candidate_moves(pose) == ()
        assert pose.layer_rotations == pose.sticker_layer_rotations == ()
        assert pose.turn_color_offset_degrees == 0.0
        assert pose.cube_geometry_state.signature() == initial.signature()
    assert records[4].pose_candidate.cube_geometry_state.signature() == initial.signature()
    assert records[5].pose_candidate.cube_geometry_state.signature() == initial.apply_move('R').signature()
    assert render_script.completed_candidate_moves(records[4].pose_candidate) == ('R',)
    logged = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(logged) == len(records)
    assert logged[1]['has_pose'] is False


@pytest.mark.parametrize('frame_index,turning_start_frame,allowed', [(0, 0, False), (380, 380, False), (381, 380, True), (0, None, True)])
def test_live_turning_start_frame_controls_turn_fitting(monkeypatch, frame_index, turning_start_frame, allowed):
    args = render_script.build_parser().parse_args([
        '--source', 'frame.png', '--initial-scramble', '', '--recognize-moves', '--no-show-pnp', '--no-show-keypoints',
    ])
    args.turning_start_frame = turning_start_frame
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    pose = _lookahead_test_candidate()
    record = render_script.NeuralFrameData(
        frame_index=frame_index, frame=frame, roi_xyxy=(0, 0, 32, 32), roi_source='tracker',
        sticker_masks=None, local_points=np.asarray([[16.0, 16.0]]), point_confidences=np.asarray([0.9]),
        class_ids=np.asarray([0]), box_confidences=np.asarray([0.9]),
        pose_candidate=pose, pose_candidates=(pose,), pose_ready=True,
    )
    calls = []

    def fit(candidate, **kwargs):
        calls.append(kwargs['recover_layer_turns'])
        return replace(candidate, turn_state_moves=('R',)) if kwargs['recover_layer_turns'] else candidate

    monkeypatch.setattr(render_script, 'solve_pnp_from_low_error_assignments', fit)
    monkeypatch.setattr(render_script, 'select_pose_hypothesis', lambda candidates, *args, **kw: candidates[0])
    monkeypatch.setattr(render_script, 'draw_context', lambda image, *args: image)
    monkeypatch.setattr(render_script, 'draw_sticker_mask_overlay', lambda image, *args, **kw: image)
    render_script.render_frame(frame, None, None, render_script.RoiTracker(), args,
                               frame_index=frame_index, neural_data=record)
    assert calls == ([False, True] if allowed else [False])
    assert render_script.completed_candidate_moves(record.pose_candidate) == (('R',) if allowed else ())
    if not allowed:
        assert record.pose_candidate.layer_rotations == record.pose_candidate.sticker_layer_rotations == ()
        assert record.pose_candidate.turn_search_diagnostics['status'] == 'forced_static'


@pytest.mark.parametrize("scramble,search_frames", [("R", [0]), ("", [0])])
def test_live_search_stops_on_current_solution_without_needing_color_display(
    monkeypatch, scramble, search_frames,
):
    args = render_script.build_parser().parse_args([
        '--source', '0', '--initial-scramble', scramble,
        '--recognize-moves', '--no-show-pnp', '--no-show-keypoints',
    ])
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    accumulator = render_script.MoveAccumulator()
    searches = []

    def fit(candidate, **kwargs):
        if kwargs['recover_layer_turns']:
            searches.append(int(candidate.tvec[0, 0]))
            return replace(candidate, turn_state_moves=("R'",) if scramble else ())
        return candidate

    monkeypatch.setattr(render_script, 'solve_pnp_from_low_error_assignments', fit)
    monkeypatch.setattr(render_script, 'select_pose_hypothesis', lambda candidates, *args, **kw: candidates[0])
    monkeypatch.setattr(render_script, 'pose_reprojection_error_is_acceptable', lambda _: True)
    monkeypatch.setattr(render_script, 'draw_context', lambda image, *args: image)
    monkeypatch.setattr(render_script, 'draw_sticker_mask_overlay', lambda image, *args, **kw: image)
    for index in range(3):
        pose = replace(_lookahead_test_candidate(), tvec=np.array([[float(index)], [0.0], [200.0]]))
        record = render_script.NeuralFrameData(
            frame_index=index, frame=frame, roi_xyxy=(0, 0, 32, 32), roi_source='tracker',
            sticker_masks=None, local_points=np.asarray([[16.0, 16.0]]), point_confidences=np.asarray([0.9]),
            class_ids=np.asarray([0]), box_confidences=np.asarray([0.9]),
            pose_candidate=pose, pose_candidates=(pose,), pose_ready=True,
        )
        render_script.render_frame(
            frame, None, None, render_script.RoiTracker(), args,
            frame_index=index, neural_data=record, move_accumulator=accumulator,
        )
        assert record.pose_candidate is not None
        assert record.pose_candidate.turn_search_diagnostics['solved_frame_index'] == 0
        if index:
            assert record.pose_candidate.turn_search_diagnostics['status'] == 'stopped_solved'
            assert render_script.completed_candidate_moves(record.pose_candidate) == ()
    assert searches == search_frames
    assert accumulator.moves == (["R'"] if scramble else [])
    assert accumulator._state.is_solved()
    assert accumulator.solved_frame_index == 0


@pytest.mark.parametrize('kind', ['normal', 'zero', 'outside', 'empty', 'no_colors'])
def test_precomputed_color_assignments_match_explicit_offset_projections(monkeypatch, kind):
    torch = pytest.importorskip('torch')
    candidate = replace(_lookahead_test_candidate(), rvec=np.asarray([[0.18], [-0.25], [0.12]]))
    centers = tuple(polygon.mean(axis=0) for _, _, polygon in
                    render_script.projected_sticker_faces(candidate, candidate.rvec, candidate.tvec))
    if kind == 'outside':
        centers = tuple(np.asarray([-10000., -10000.]) for _ in centers)
    elif kind == 'empty':
        centers = ()
    candidate = replace(candidate, sticker_mask_centers=centers,
                        sticker_mask_colors=(np.asarray([255, 0, 0]),) * len(centers),
                        sticker_mask_center_uncertainties=tuple(1. + i % 3 for i in range(len(centers))))
    options = tuple(dict(
        layer_axis=axis, layer_side=side, layer_count=width,
        layer_rotations=(render_script.CubeLayerRotation(axis, side, angle, width),) if angle else (),
        label=f'{axis}:{side}:{width}:{angle}', is_baseline=not angle,
        combined_mean_error=.03 + i * .0001, inlier_count=10, p90_error=.1,
        assignment_projected_points=None, assignment_object_points=None,
        assignment_projected_class_ids=None)
        for i, (axis, side, width, angle) in enumerate(
            (axis, side, width, angle) for axis in range(3) for side in (-1, 1)
            for width in (1, 2) for angle in (0., -33., 21., 45.)))
    likelihoods = np.random.default_rng(86).uniform(.001, 2., (54, len(centers)))
    if kind == 'zero':
        likelihoods[::2] = 0.
    elif kind == 'no_colors':
        likelihoods = None
    initial = render_script.CubeGeometryState.identity()
    states = (initial, initial.apply_moves(('R', 'U', "F'")),
              initial.apply_moves(("Rw'", 'Uw', 'Bw', 'Lw', "Dw'", 'Fw')))
    score = render_script._score_turn_color_offsets_for_states_on_gpu
    cache = {}
    actual = score(candidate, options, 10., 'cpu', likelihoods, states, projection_cache=cache)
    assert actual is not None
    assert cache['projection_count'] == 37  # One shared static pose and 36 partial poses.
    assert cache['configuration_count'] == 144  # Retain every option/offset for scoring.
    plan = render_script._turn_color_projection_plan

    def explicit_projections(configurations):
        # Reference path projects base angle + offset separately, as before.
        return (configurations, np.arange(len(configurations)),
                np.tile(np.arange(54), (len(configurations), 1)), plan(configurations)[3])

    monkeypatch.setattr(render_script, '_turn_color_projection_plan', explicit_projections)
    expected = score(candidate, options, 10., 'cpu', likelihoods, states)
    assert expected is not None
    for actual_trials, expected_trials in zip(actual, expected, strict=True):
        by_transition = lambda trials: {
            (t['layer_axis'], t['layer_side'], t['layer_count'], t['color_offset']): t for t in trials}
        a, b = by_transition(actual_trials), by_transition(expected_trials)
        assert a.keys() == b.keys()
        for key in a:
            assert a[key]['rank'] == pytest.approx(b[key]['rank'], abs=1e-10)
            assert a[key]['color_error'] == pytest.approx(b[key]['color_error'], abs=1e-10)
            assert a[key]['color_match_count'] == b[key]['color_match_count']
            assert a[key]['state_moves'] == b[key]['state_moves']
            if kind in ('normal', 'zero'):
                assert a[key]['trial_id'] == b[key]['trial_id']


@pytest.mark.parametrize('change', ['likelihoods', 'observations', 'uncertainty', 'pose', 'geometry', 'limit'])
def test_projection_assignment_cache_matches_fresh_scoring_and_invalidates(change):
    torch = pytest.importorskip('torch')
    candidate = replace(_lookahead_test_candidate(), rvec=np.asarray([[0.18], [-0.25], [0.12]]))
    faces = render_script.projected_sticker_faces(candidate, candidate.rvec, candidate.tvec)
    centers = tuple(polygon.mean(axis=0) for _, _, polygon in faces)
    candidate = replace(candidate, sticker_mask_centers=centers,
                        sticker_mask_colors=(np.asarray([255, 0, 0], dtype=np.uint8),) * len(centers),
                        sticker_mask_center_uncertainties=(1.0,) * len(centers))
    options = (dict(layer_axis=2, layer_side=-1, layer_count=1, layer_rotations=(),
                    label='static', is_baseline=True, combined_mean_error=0.03,
                    assignment_projected_points=None, assignment_object_points=None,
                    assignment_projected_class_ids=None),)
    initial = render_script.CubeGeometryState.identity()
    states = (initial, initial.apply_moves(('R', 'U')))
    likelihoods = np.random.default_rng(19).uniform(0.01, 2.0, (54, len(centers)))
    cache = {}
    score = render_script._score_turn_color_offsets_for_states_on_gpu
    score(candidate, options, 10.0, torch.device('cpu'), likelihoods, states[:1], projection_cache=cache)
    score(candidate, options, 10.0, torch.device('cpu'), likelihoods, states[1:], projection_cache=cache)
    assert cache['builds'] == cache['assignments']['builds'] == 1
    assert cache['hits'] == cache['assignments']['hits'] == 1
    assert cache['likelihoods']['builds'] == cache['likelihoods']['hits'] == 1
    limit = 10.0
    if change == 'likelihoods':
        likelihoods = likelihoods[:, ::-1].copy()
    elif change == 'observations':
        candidate = replace(candidate, sticker_mask_centers=tuple(center + [1.0, -2.0] for center in centers))
    elif change == 'uncertainty':
        candidate = replace(candidate, sticker_mask_center_uncertainties=tuple(1.0 + i % 3 for i in range(len(centers))))
    elif change == 'pose':
        candidate = replace(candidate, rvec=candidate.rvec + 0.02)
    elif change == 'geometry':
        options = (dict(options[0], layer_rotations=(render_script.CubeLayerRotation(2, -1, 15.0, 1),)),)
    elif change == 'limit':
        limit = 1.0
    cached = score(candidate, options, limit, torch.device('cpu'), likelihoods, states, projection_cache=cache)
    fresh = score(candidate, options, limit, torch.device('cpu'), likelihoods, states)
    np.testing.assert_allclose([[t['state_score'] for t in ts] for ts in cached],
                               [[t['state_score'] for t in ts] for ts in fresh], rtol=0, atol=1e-12)
    assert [[t['color_match_count'] for t in ts] for ts in cached] == [[t['color_match_count'] for t in ts] for ts in fresh]
    assert cache['builds'] == (2 if change in ('pose', 'geometry') else 1)
    if change in ('observations', 'uncertainty', 'limit'):
        assert cache['assignments']['builds'] == 2
    if change == 'likelihoods':
        assert cache['assignments']['builds'] == 1
        assert cache['likelihoods']['builds'] == 2
    else:
        assert cache['likelihoods']['builds'] == 1


def test_wrapper_state_budget_keeps_complete_forecasts_and_json_safe_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(render_script, 'PNP_TURN_LOOKAHEAD_STATE_BUDGET', 3)
    path = _write_search_debug_log(tmp_path, monkeypatch)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    first = records[0]['search']
    assert first['budget_exhausted']
    assert first['decision_evaluated_states'] <= 3
    assert first['state_evaluation_budget'] == 3
    assert len(first['selected_path']['steps']) == 3
    assert first['optimality'] == 'budget_exhausted'
    assert records[1]['search']['reused_previous_forecast']
    json.dumps(records, allow_nan=False)


def test_default_color_scoring_keeps_large_finite_errors_and_full_horizon(monkeypatch, tmp_path):
    candidate = _lookahead_test_candidate()
    colors = (0.8, 24.2, 0.79)
    records = [SimpleNamespace(frame_index=f, pose_candidate=replace(candidate,
                tvec=np.asarray([[float(f)], [0.0], [200.0]])), pose_candidates=(), sticker_masks=None)
               for f in range(3)]

    def score(pose, *args, **kwargs):
        color = colors[int(pose.tvec[0, 0])]
        return [dict(_lookahead_trial(color, baseline=True), color_error=color)]

    monkeypatch.setattr(render_script, '_turn_torch_scoring_device', lambda _: 'cpu')
    monkeypatch.setattr(render_script, '_build_turn_geometry_frame', lambda *args: dict(options=({},)))
    monkeypatch.setattr(render_script, 'turn_color_predictive_likelihoods', lambda *args: None)
    monkeypatch.setattr(render_script, '_score_turn_color_offsets_on_gpu', score)
    monkeypatch.setattr(render_script, '_score_turn_color_offsets_for_states_on_gpu',
                        lambda pose, options, limit, device, likelihoods, states, **kwargs: [score(pose) for _ in states])
    monkeypatch.setattr(render_script, 'solve_pnp_from_low_error_assignments', lambda pose, **kwargs: pose)
    monkeypatch.setattr(render_script, 'pose_reprojection_error_is_acceptable', lambda _: True)
    log = tmp_path / 'scores.jsonl'
    render_script.recover_cached_pose_symmetries(
        records, render_script.StickerColorTrace.create(), pose_prepared=True,
        recover_orientation=False, recognize_moves=True, global_profiles={}, known_initial_colors={},
        turn_candidate_log=log,
    )
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[0]['search']['frames'] == [0, 1, 2]
    assert [step['color_error'] for step in rows[0]['search']['selected_path']['steps']] == list(colors)
    assert rows[1]['turn_fit_color_score'] == 24.2
    assert all(not row['color_evidence_unreliable'] for row in rows)
    assert all(depth['invalid_color'] == 0 for row in rows for depth in row['search']['depths'])
    json.dumps(rows, allow_nan=False)


@pytest.mark.parametrize('moves', [(), ('R',)])
@pytest.mark.parametrize('color', [4.01, 1e6])
def test_large_finite_color_scores_do_not_disqualify_hold_or_turn(moves, color):
    trial = dict(_lookahead_trial(color, moves, baseline=not moves), color_error=color)
    debug = {}
    paths = render_script.search_move_lookahead_paths(
        render_script.CubeGeometryState.identity(), (trial,), (), lambda *args: (), diagnostics=debug,
    )
    assert paths
    assert min(paths, key=render_script.move_lookahead_path_score).first_trial is trial
    assert debug['depths'][0]['invalid_color'] == 0


@pytest.mark.parametrize('color', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('moves', [(), ('R',)])
def test_available_nonfinite_color_fit_is_still_invalid(color, moves):
    trial = dict(_lookahead_trial(0.0, moves, baseline=not moves), color_error=color)
    debug = {}
    paths = render_script.search_move_lookahead_paths(
        render_script.CubeGeometryState.identity(), (trial,), (), lambda *args: (),
        diagnostics=debug, seed_greedy=True,
    )
    assert not paths
    assert debug['depths'][0]['invalid_color'] == 1
    assert debug['roots'][0]['status'] == 'invalid_color'


def test_removed_color_cutoff_is_not_a_cli_option():
    parser = render_script.build_parser()
    args = ['--source', 'frame.png', '--initial-scramble', '']
    parsed = parser.parse_args(args)
    assert not hasattr(parsed, 'turn_color_error_ceiling')
    with pytest.raises(SystemExit):
        parser.parse_args([*args, '--turn-color-error-ceiling', '4.0'])


def test_failed_future_color_does_not_trigger_evidence_omission(monkeypatch, tmp_path):
    candidate = _lookahead_test_candidate()
    records = [SimpleNamespace(frame_index=f, pose_candidate=replace(candidate,
                tvec=np.asarray([[float(f)], [0.0], [200.0]])), pose_candidates=(), sticker_masks=None)
               for f in range(3)]

    def score(pose, *args, **kwargs):
        color = float('inf') if int(pose.tvec[0, 0]) == 1 else 0.5
        return [dict(_lookahead_trial(color, baseline=True), color_error=color)]

    monkeypatch.setattr(render_script, '_turn_torch_scoring_device', lambda _: 'cpu')
    monkeypatch.setattr(render_script, '_build_turn_geometry_frame', lambda *args: dict(options=({},)))
    monkeypatch.setattr(render_script, 'turn_color_predictive_likelihoods', lambda *args: None)
    monkeypatch.setattr(render_script, '_score_turn_color_offsets_on_gpu', score)
    monkeypatch.setattr(render_script, '_score_turn_color_offsets_for_states_on_gpu',
                        lambda pose, options, limit, device, likelihoods, states, **kwargs: [score(pose) for _ in states])
    monkeypatch.setattr(render_script, 'solve_pnp_from_low_error_assignments', lambda pose, **kwargs: pose)
    monkeypatch.setattr(render_script, 'pose_reprojection_error_is_acceptable', lambda _: True)
    log = tmp_path / 'scores.jsonl'
    render_script.recover_cached_pose_symmetries(
        records, render_script.StickerColorTrace.create(), pose_prepared=True,
        recover_orientation=False, recognize_moves=True, global_profiles={}, known_initial_colors={},
        turn_candidate_log=log,
    )
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[0]['search']['frames'] == [0, 1, 2]
    assert rows[0]['search']['skipped_frames'] == []
    assert rows[0]['search']['fallback_reason'] == 'no_surviving_path'
    assert rows[0]['search']['optimality'] == 'no_finite_path'
    assert all(not row['selected_committed_moves'] for row in rows)
    json.dumps(rows, allow_nan=False)


@pytest.mark.parametrize('inherited_color', [4.999, 5., 5.001])
def test_turn_orientation_rescue_threshold_is_per_state_and_strict(monkeypatch, inherited_color):
    initial = render_script.CubeGeometryState.identity()
    other = initial.apply_move('U')
    calls = []

    def trial(color):
        return dict(trial_id='0:-90', label='static R offset -90', is_baseline=False,
                    layer_axis=0, layer_side=1, layer_count=1, layer_rotations=(),
                    color_offset=-90., state_moves=('R',), color_error=color,
                    state_score=color + .1, rank=(color + .1,))

    def score(candidate, options, limit, device, likelihoods, states, **kwargs):
        calls.append(states)
        return [[trial(inherited_color)], [trial(1.)]] if len(calls) == 1 else [[trial(.5)]]

    monkeypatch.setattr(render_script, '_score_turn_color_offsets_for_states_on_gpu', score)
    result = render_script._score_turn_color_offsets_with_orientation_rescue(
        _lookahead_test_candidate(), (), 10., 'cpu', np.ones((54, 1)), (initial, other),
    )
    assert len(calls) == (24 if inherited_color > 5 else 1)
    assert all(len(states) == 1 for states in calls[1:])
    assert result[1][0]['color_error'] == 1.
    assert 'orientation_rescue' not in result[1][0]
    if inherited_color > 5:
        assert min(t['state_score'] for t in result[0]) == pytest.approx(.6)
        assert all(t['orientation_rescue']['inherited_color_error'] == inherited_color
                   and t['orientation_rescue']['tested_orientations'] == 24 for t in result[0])
    else:
        assert result[0][0]['color_error'] == inherited_color
        assert 'orientation_rescue' not in result[0][0]


@pytest.mark.parametrize('symmetry_index', range(24))
@pytest.mark.parametrize('layer_count,offset', [(1, -90.), (1, 0.), (2, 90.)])
def test_rescued_pose_projects_the_same_physical_identities_and_commits_only_the_mapped_move(
    symmetry_index, layer_count, offset,
):
    symmetry = render_script.CUBE_SYMMETRY_MATRICES[symmetry_index]
    state = render_script.CubeGeometryState.identity().apply_moves(('R', 'Uw', 'F'))
    relabeled = render_script._geometry_state_under_symmetry(state, symmetry.T)
    candidate = replace(_lookahead_test_candidate(), rvec=np.array([[.18], [-.25], [.12]]),
                        cube_geometry_state=state)
    partial = render_script.CubeLayerRotation(2, -1, 15., layer_count)
    move = render_script.cube_rotation_to_notation(replace(partial, angle_degrees=offset))
    trial = dict(trial_id='0', layer_axis=2, layer_side=-1, layer_count=layer_count,
                 layer_rotations=(partial,), color_offset=offset,
                 state_moves=(move,) if move else (), is_baseline=False)
    mapped = render_script._turn_trial_under_pose_symmetry(trial, symmetry, symmetry_index)
    selected = render_script._candidate_with_turn_trial(candidate, candidate, mapped)
    reference = replace(candidate, cube_geometry_state=relabeled,
                        layer_rotations=(replace(partial, angle_degrees=15. + offset),))
    actual = {i.label: polygon for i, _, polygon in render_script.projected_sticker_faces(
        selected, selected.rvec, selected.tvec)}
    expected = {i.label: polygon for i, _, polygon in render_script.projected_sticker_faces(
        reference, reference.rvec, reference.tvec)}
    assert actual.keys() == expected.keys()
    for label in actual:
        # A coordinate relabeling can rotate/reverse a polygon's vertex order.
        distances = np.linalg.norm(actual[label][:, None] - expected[label][None, :], axis=-1)
        assert distances.min(axis=1).max() < 1e-4
    assert selected.cube_geometry_state.signature() == state.signature()
    mapped_after = state.apply_moves(mapped['state_moves'])
    expected_after = render_script._geometry_state_under_symmetry(
        relabeled.apply_moves(trial['state_moves']), symmetry,
    )
    assert mapped_after.signature() == expected_after.signature()
    assert render_script.completed_candidate_moves(selected) == mapped['state_moves']
    assert all(move[0] not in 'xyz' for move in mapped['state_moves'])


def test_xuanyi_frame_213_rescues_correct_one_layer_move_without_reprojecting_each_orientation(tmp_path):
    torch = pytest.importorskip('torch')
    fixture = json.loads((Path(__file__).parent / 'fixtures/xuanyi_frame_213_turn_orientation.json').read_text())
    p = fixture['pose']
    state = render_script.CubeGeometryState({
        (tuple(v[0]), v[1], v[2]): k for k, v in fixture['state_before_frame212'].items()
    }).apply_move(fixture['initial_correct_move'])
    candidate = replace(
        _lookahead_test_candidate(), sticker_type=p['sticker_type'],
        camera_matrix=np.array(p['camera_matrix']), dist_coeffs=np.array(p['dist_coeffs']),
        rvec=np.array(p['rvec']), tvec=np.array(p['tvec']), image_size=tuple(p['image_size']),
        cube_geometry_state=state,
        sticker_mask_centers=tuple(np.array(x) for x in p['sticker_mask_centers']),
        sticker_mask_center_uncertainties=tuple(p['sticker_mask_center_uncertainties']),
        sticker_mask_colors=tuple(np.zeros(3) for _ in p['sticker_mask_centers']),
    )
    options = tuple(dict(
        o, combined_mean_error=o['keypoint_mean_error'] + o['heatmap_mean_error'],
        keypoint_error=o['keypoint_mean_error'], sticker_error=o['heatmap_mean_error'],
        p90_error=o['keypoint_p90_error'],
        layer_rotations=tuple(render_script.CubeLayerRotation(**x) for x in o['layer_rotations']),
        assignment_projected_points=None, assignment_object_points=None, assignment_projected_class_ids=None,
    ) for o in fixture['options'])
    likelihoods = np.array(fixture['likelihoods'])
    limit = render_script.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * render_script.projected_cube_diameter(candidate)
    cache = {}
    inherited = render_script._score_turn_color_offsets_for_states_on_gpu(
        candidate, options, limit, torch.device('cpu'), likelihoods, (state,), projection_cache=cache,
    )[0]
    assert min(inherited, key=lambda t: t['rank'])['color_error'] == pytest.approx(9.28534836)
    rescued = render_script._score_turn_color_offsets_with_orientation_rescue(
        candidate, options, limit, torch.device('cpu'), likelihoods, (state,), projection_cache=cache,
    )[0]
    best = min(rescued, key=lambda t: t['rank'])
    assert best['state_moves'] == ('F',)
    assert best['layer_count'] == 1
    assert best['state_score'] == pytest.approx(.95944329)
    assert best['orientation_symmetry_index'] == 15
    assert best['orientation_rescue']['tested_orientations'] == 24
    assert cache['builds'] == 1
    assert cache['assignments']['builds'] == 1
    assert len(rescued) == 36
    selected = render_script._candidate_with_turn_trial(candidate, candidate, best)
    color, _ = render_script.sticker_mask_color_turn_fit_metrics(
        selected, selected.rvec, selected.tvec, render_script.physical_sticker_rotations(selected),
        limit, likelihoods,
    )
    assert color == pytest.approx(best['color_error'], abs=1e-6)
    lower = render_script._turn_color_frame_score_lower_bound(
        candidate, options, limit, torch.device('cpu'), likelihoods, projection_cache=cache,
    )
    assert lower <= best['state_score']
    payload = render_script.turn_trial_log_payload(best)
    assert payload['orientation_symmetry_index'] == 15
    log = tmp_path / 'scores.jsonl'
    selected = replace(selected, turn_candidate_diagnostics=(payload,),
                       turn_fit_search_rvec=candidate.rvec, turn_fit_search_tvec=candidate.tvec)
    render_script.append_turn_candidate_log(log, selected, 213)
    logged = json.loads(log.read_text())
    assert logged['candidates'][0]['orientation_symmetry_index'] == 15
    assert logged['candidates'][0]['orientation_rescue']['threshold'] == 5.
    np.testing.assert_allclose(logged['selected_pose']['rvec'], selected.rvec)
    json.dumps(logged, allow_nan=False)


def test_orientation_only_static_rescue_updates_pose_without_committing_a_turn(monkeypatch):
    torch = pytest.importorskip('torch')
    candidate = replace(_lookahead_test_candidate(), cube_geometry_state=render_script.CubeGeometryState.identity())
    option = dict(layer_axis=0, layer_side=1, layer_count=1, layer_rotations=(),
                  label='static R', is_baseline=True, combined_mean_error=.03,
                  keypoint_error=.01, sticker_error=.02, robust_error=.04,
                  assignment_projected_points=None, assignment_object_points=None,
                  assignment_projected_class_ids=None)
    trials = render_script._score_turn_color_offsets_for_states_on_gpu(
        candidate, (option,), 10., torch.device('cpu'), None, (candidate.cube_geometry_state,),
    )[0]
    hold = next(t for t in trials if not t['state_moves'])
    symmetry = render_script.CUBE_SYMMETRY_MATRICES[15]
    rescued = render_script._turn_trial_under_pose_symmetry(hold, symmetry, 15)
    monkeypatch.setattr(render_script, '_turn_torch_scoring_device', lambda _: 'cpu')
    monkeypatch.setattr(render_script, 'turn_color_predictive_likelihoods', lambda *args: None)
    monkeypatch.setattr(render_script, '_score_turn_color_offsets_on_gpu', lambda *args, **kwargs: [rescued])
    selected = render_script.recover_misaligned_cube_layers_from_geometry(
        candidate, 10., dict(options=(option,), static_trials=(option,)),
    )
    expected = render_script.transform_rvec_by_symmetry(candidate.rvec, symmetry)
    np.testing.assert_allclose(selected.rvec, expected)
    assert render_script.completed_candidate_moves(selected) == ()
    assert selected.cube_geometry_state.signature() == candidate.cube_geometry_state.signature()
    assert selected.turn_selected_trial_id == rescued['trial_id']
    assert selected.turn_candidate_diagnostics[0]['orientation_symmetry_index'] == 15


def test_cached_orientation_rescue_resamples_colors_in_the_selected_pose(monkeypatch):
    candidate = _lookahead_test_candidate()
    trace = render_script.StickerColorTrace.create()
    label = trace.identities[0].label
    trace.observe(0, {label: np.array([255, 0, 0], dtype=np.uint8)})
    record = SimpleNamespace(frame_index=0, frame=np.zeros((64, 64, 3), np.uint8),
                             roi_xyxy=(0, 0, 64, 64), sticker_masks=object(),
                             pose_candidate=candidate, pose_candidates=(candidate,))
    symmetry = render_script.CUBE_SYMMETRY_MATRICES[15]
    trial = dict(_lookahead_trial(.2, baseline=True), trial_id='hold@pose15',
                 layer_rotations=(), layer_axis=0, layer_side=1, layer_count=1,
                 color_offset=0., color_error=.2, orientation_symmetry=symmetry.tolist())
    calls = []

    def sample(frame, roi, masks, pose, **kwargs):
        calls.append(pose.rvec.copy())
        return {label: np.array([0, 0, 255], dtype=np.uint8)}

    monkeypatch.setattr(render_script, '_turn_torch_scoring_device', lambda _: 'cpu')
    monkeypatch.setattr(render_script, '_build_turn_geometry_frame', lambda *args: dict(options=({},)))
    monkeypatch.setattr(render_script, 'sticker_mask_rgb_observations', lambda *args: [(0, np.zeros(3))])
    monkeypatch.setattr(render_script, 'turn_color_predictive_likelihoods', lambda *args: None)
    monkeypatch.setattr(render_script, '_score_turn_color_offsets_on_gpu', lambda *args, **kwargs: [trial])
    monkeypatch.setattr(render_script, 'solve_pnp_from_low_error_assignments', lambda pose, **kwargs: pose)
    monkeypatch.setattr(render_script, 'sample_projected_sticker_colors', sample)
    monkeypatch.setattr(render_script, 'sticker_view_reliability_weight', lambda *args: 1.)
    result = render_script.recover_cached_pose_symmetries(
        [record], trace, pose_prepared=True, recover_orientation=False, recognize_moves=True,
        global_profiles={}, known_initial_colors={},
    )
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], render_script.transform_rvec_by_symmetry(candidate.rvec, symmetry))
    np.testing.assert_array_equal(result.observations[label][0][1], [0, 0, 255])
