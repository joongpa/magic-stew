import importlib.util
import json
from dataclasses import replace
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
    color_matches = np.zeros((54, 1), dtype=np.float64)
    color_matches[:2, 0] = (0.8, 0.2)
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
        turn_color_matches=color_matches,
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
    assert records[0]["turn_color_matches"][:2] == [[0.8], [0.2]]
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
    likelihoods = np.zeros((54, 1), dtype=np.float64)
    likelihoods[7, 0] = 1.0
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
        color_matches=likelihoods,
    )

    np.testing.assert_allclose(color_error, (0.0,))
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
        records,
        render_script.StickerColorTrace.create(),
        pose_prepared=True,
        recover_orientation=False,
        recognize_moves=True,
        known_initial_colors=_colors_for_scramble("R"),
        turn_candidate_log=log,
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
        records,
        render_script.StickerColorTrace.create(),
        pose_prepared=True,
        recover_orientation=False,
        recognize_moves=True,
        known_initial_colors=_colors_for_scramble("x y"),
        turn_candidate_log=log,
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
    monkeypatch.setattr(render_script, "turn_color_match_matrix", lambda *args: None)
    monkeypatch.setattr(render_script, "solve_pnp_from_low_error_assignments", lambda pose, **kwargs: pose)
    monkeypatch.setattr(render_script, "pose_reprojection_error_is_acceptable", lambda _: True)
    log = tmp_path / "scores.jsonl"
    render_script.recover_cached_pose_symmetries(
        records,
        render_script.StickerColorTrace.create(),
        pose_prepared=True,
        recover_orientation=False,
        recognize_moves=True,
        known_initial_colors=known,
        turn_candidate_log=log,
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
    monkeypatch.setattr(render_script, "turn_color_match_matrix", lambda *args: None)
    monkeypatch.setattr(render_script, "solve_pnp_from_low_error_assignments", lambda pose, **kwargs: pose)
    monkeypatch.setattr(render_script, "pose_reprojection_error_is_acceptable", lambda _: True)
    log_path = tmp_path / "turn_candidate_scores.jsonl"
    render_script.recover_cached_pose_symmetries(
        records,
        render_script.StickerColorTrace.create(),
        pose_prepared=True,
        recover_orientation=False,
        recognize_moves=True,
        known_initial_colors={},
        turning_start_frame=turning_start_frame,
        turn_candidate_log=log_path,
    )
    return log_path


def test_move_search_log_records_actual_forecast_frames_and_selected_offset(tmp_path, monkeypatch):
    path = _write_search_debug_log(tmp_path, monkeypatch)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    first = records[0]
    assert first["log_schema_version"] == 3
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
        records,
        render_script.StickerColorTrace.create(),
        pose_prepared=True,
        recover_orientation=False,
        recognize_moves=True,
        known_initial_colors={},
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
    monkeypatch.setattr(render_script, "turn_color_match_matrix", lambda *args: None)
    monkeypatch.setattr(render_script, "_score_turn_color_offsets_on_gpu", score)
    monkeypatch.setattr(render_script, "_score_turn_color_offsets_for_states_on_gpu",
                        lambda pose, options, limit, device, likelihoods, states, **kwargs: [score(pose) for state in states])
    monkeypatch.setattr(render_script, "solve_pnp_from_low_error_assignments", local_search)
    monkeypatch.setattr(render_script, "pose_reprojection_error_is_acceptable", lambda _: True)
    render_script.recover_cached_pose_symmetries(
        records,
        render_script.StickerColorTrace.create(),
        pose_prepared=True,
        recover_orientation=False,
        recognize_moves=True,
        known_initial_colors={},
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
    likelihoods = np.random.default_rng(12).integers(0, 2, (54, len(centers))).astype(float) if has_colors else None
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


def test_color_cutoff_applies_before_choosing_partial_angle(monkeypatch):
    torch = pytest.importorskip('torch')
    options = tuple(dict(layer_axis=1, layer_side=-1, layer_count=1, is_baseline=False,
        label=f'angle {angle}', combined_mean_error=keypoint+sticker, keypoint_error=keypoint, sticker_error=sticker,
        layer_rotations=(render_script.CubeLayerRotation(1, -1, angle, 1),),
        assignment_projected_points=None, assignment_object_points=None, assignment_projected_class_ids=None,
    ) for angle, keypoint, sticker in ((-15., 0., 0.), (45., .05, .05)))

    def colors(candidate, torch, centers, *args, **kwargs):
        errors = centers.new_tensor([[2./3.]*3 + [1.-2./3.]*3])
        return errors, torch.ones_like(errors, dtype=torch.int32)

    monkeypatch.setattr(render_script, '_gpu_sticker_color_fit_metrics', colors)
    scores = render_script._score_turn_color_offsets_for_states_on_gpu(
        _lookahead_test_candidate(), options, 10., torch.device('cpu'), None,
        (render_script.CubeGeometryState.identity(),), balance_geometry_color=True, minimum_color_accuracy=2./3.,
    )
    assert len(scores[0]) == 3
    # The invalid angle has the lower combined cost; retain the valid alternative.
    assert all(t['layer_rotations'][0].angle_degrees == 45. for t in scores[0])
    assert all(t['state_score'] == pytest.approx(2./3.) for t in scores[0])


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
    monkeypatch.setattr(render_script, "turn_color_match_matrix", lambda *args: None)
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
    monkeypatch.setattr(render_script, "turn_color_match_matrix", lambda *args: None)
    monkeypatch.setattr(render_script, "_score_turn_color_offsets_on_gpu", lambda *args: trials)
    result = render_script.recover_misaligned_cube_layers_from_geometry(
        candidate, 10.0, dict(options=(option,), static_trials=(option,)), previous_move="R",
        previous_states=tuple(initial.apply_moves(("R",) * turns).signature() for turns in range(3))
        if prevent_cycle else (),
    )
    assert result.turn_state_moves == ("R'",)


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
        SimpleNamespace(),
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
    np.testing.assert_allclose(projected[2], projected[0], atol=1e-5)  # White: model +Z
    assert projected[3, 1] < projected[0, 1]  # Green: model -Y points up


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
    candidate.turn_color_matches = likelihoods

    values = render_script.projected_turn_color_matches_by_identity(
        candidate,
        rvec,
        tvec,
    )

    assert values[identity.label] == pytest.approx(0.91)


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
        "solveP3P",
        lambda *_args, **_kwargs: (False, (), ()),
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


def test_turn_occlusion_allows_shallow_seam_hits_but_keeps_deeper_hits():
    torch = pytest.importorskip("torch")
    m = render_script
    # This view puts two seam vertices just behind the turning face and a
    # third seam vertex well behind it.
    rvec = np.array([1.06345, -0.83809, -0.39880])
    tvec = np.array([-66.84, -35.86, 956.41])
    camera = np.array([[2304., 0., 540.], [0., 2304., 960.], [0., 0., 1.]])
    distortion = np.zeros(4)
    indices = [m.CUBE_LATTICE_COORDS.index(point)
               for point in ((2, 0, 1), (1, 0, 1), (0, 2, 1))]
    objects = m.CUBE_LATTICE_OBJECT_POINTS_MM[indices].astype(np.float64)
    classes = m.CUBE_LATTICE_CLASS_IDS[indices]
    observed, _ = cv2.projectPoints(objects, rvec, tvec, camera, distortion)
    observed = observed.reshape(-1, 2)
    candidate = m.StickerPnPCandidate(
        sticker_type="edge", sticker_average_confidence=1.0,
        image_points=np.empty((0, 2)), object_points=np.empty((0, 3)),
        projected_sticker_points=np.empty((0, 2)), reprojection_error=0.0,
        camera_matrix=camera, dist_coeffs=distortion, rvec=rvec, tvec=tvec,
        assignment_image_points=observed, assignment_projected_points=observed,
        assignment_object_points=objects, assignment_observed_class_ids=classes,
        image_size=(1080, 1920),
    )
    rotation = m.CubeLayerRotation(2, -1, -45.0, 1)
    limit = 0.05 * m.projected_cube_diameter(candidate)

    def assignments(**kwargs):
        trials = m._score_turn_configurations_on_gpu(
            candidate, limit, torch.device("cpu"), angle_values=np.array([-45.0]),
            enforce_layer_observation_gate=False, enforce_validity_gate=False, **kwargs,
        )
        return next(trial["assignment_projected_points"] for trial in trials
                    if trial["layer_rotations"] == (rotation,))

    strict = np.linalg.norm(assignments(seam_clearance_mm=0.0) - observed, axis=1)
    relaxed = np.linalg.norm(assignments() - observed, axis=1)
    assert np.all(strict[:2] > limit)
    np.testing.assert_allclose(relaxed[:2], 0.0, atol=1e-6)
    assert relaxed[2] > limit


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


def test_hypothesis_selection_uses_translation_for_candidates_under_shared_fit_gate(
    monkeypatch,
):
    monkeypatch.setattr(render_script, "PNP_POSE_REPROJECTION_ERROR_REJECTION_ENABLED", True)
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
    monkeypatch.setattr(render_script, "pose_reprojection_error_is_acceptable", lambda _: True)
    assert not render_script.POSE_TWO_SIDED_MOTION_REJECTION_ENABLED
    for enabled in (False, True):
        monkeypatch.setattr(render_script, "POSE_TWO_SIDED_MOTION_REJECTION_ENABLED", enabled)
        poses = [replace(candidate, tvec=np.array([[x], [0.0], [200.0]])) for x in (0.0, 100.0, 0.0)]
        records = [SimpleNamespace(frame_index=i, pose_candidate=pose, pose_candidates=(pose,))
                   for i, pose in enumerate(poses)]
        render_script.recover_cached_pose_symmetries(
            records,
            render_script.StickerColorTrace.create(),
            recover_orientation=False,
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


def test_initial_scramble_is_required_and_cpu_is_default():
    parser = render_script.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--source", "video.mp4"])
    args = parser.parse_args(["--source", "video.mp4", "--initial-scramble", ""])
    assert args.initial_scramble == ()
    assert args.device == "cpu"
    with pytest.raises(SystemExit):
        parser.parse_args(["--source", "video.mp4", "--initial-scramble", "", "--device", "mps"])


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
    monkeypatch.setattr(render_script, "turn_color_match_matrix", lambda *args: None)
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
    assert parser.parse_args(args).turn_search_state_budget is None
    assert parser.parse_args(args).branch_widths == (3,)
    assert parser.parse_args([*args, '--turn-search-state-budget', '256']).turn_search_state_budget == 256
    assert parser.parse_args([*args, '--max-explored-states', '1']).turn_search_state_budget == 1
    assert parser.parse_args([*args, '--max-explored-states', 'unlimited']).turn_search_state_budget is None
    for bad in ('0', '-1', '1.5', 'nan'):
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
        render_script.recover_cached_pose_symmetries(
            [],
            render_script.StickerColorTrace.create(),
            turning_start_frame=-1,
        )


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


def test_turning_start_frame_preserves_state_skips_fitting_and_allows_later_moves(tmp_path, monkeypatch, capsys):
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
        records,
        render_script.StickerColorTrace.create(),
        pose_prepared=True,
        recover_orientation=False,
        recognize_moves=True,
        progress=True,
        turning_start_frame=380,
        known_initial_colors={},
        turn_candidate_log=log,
    )
    initial = render_script.CubeGeometryState.identity()
    output = capsys.readouterr().out
    assert "static holds:" in output
    assert output.count("Starting move search") == 1
    assert "Starting move search at frame 381" in output
    assert output.index("static holds:") < output.index("Starting move search at frame 381")
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
    render_script.render_frame(
        frame,
        None,
        None,
        render_script.RoiTracker(),
        args,
        frame_index=frame_index,
        neural_data=record,
    )
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
    for index in range(3):
        pose = replace(_lookahead_test_candidate(), tvec=np.array([[float(index)], [0.0], [200.0]]))
        record = render_script.NeuralFrameData(
            frame_index=index, frame=frame, roi_xyxy=(0, 0, 32, 32), roi_source='tracker',
            sticker_masks=None, local_points=np.asarray([[16.0, 16.0]]), point_confidences=np.asarray([0.9]),
            class_ids=np.asarray([0]), box_confidences=np.asarray([0.9]),
            pose_candidate=pose, pose_candidates=(pose,), pose_ready=True,
        )
        render_script.render_frame(
            frame,
            None,
            None,
            render_script.RoiTracker(),
            args,
            frame_index=index,
            neural_data=record,
            move_accumulator=accumulator,
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
    pytest.importorskip('torch')
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
    likelihoods = np.random.default_rng(86).integers(0, 2, (54, len(centers))).astype(float)
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
        def by_transition(trials):
            return {(t['layer_axis'], t['layer_side'], t['layer_count'], t['color_offset']): t
                    for t in trials}
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
    likelihoods = np.random.default_rng(19).integers(0, 2, (54, len(centers))).astype(float)
    cache = {}
    score = render_script._score_turn_color_offsets_for_states_on_gpu
    score(candidate, options, 10.0, torch.device('cpu'), likelihoods, states[:1], projection_cache=cache)
    score(candidate, options, 10.0, torch.device('cpu'), likelihoods, states[1:], projection_cache=cache)
    assert cache['builds'] == cache['assignments']['builds'] == 1
    assert cache['hits'] == cache['assignments']['hits'] == 1
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
    if change == 'observations':
        assert cache['assignments']['builds'] == 2
    if change == 'likelihoods':
        assert cache['assignments']['builds'] == 1


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
    monkeypatch.setattr(render_script, 'turn_color_match_matrix', lambda *args: None)
    monkeypatch.setattr(render_script, '_score_turn_color_offsets_on_gpu', score)
    monkeypatch.setattr(render_script, '_score_turn_color_offsets_for_states_on_gpu',
                        lambda pose, options, limit, device, likelihoods, states, **kwargs: [score(pose) for _ in states])
    monkeypatch.setattr(render_script, 'solve_pnp_from_low_error_assignments', lambda pose, **kwargs: pose)
    monkeypatch.setattr(render_script, 'pose_reprojection_error_is_acceptable', lambda _: True)
    log = tmp_path / 'scores.jsonl'
    render_script.recover_cached_pose_symmetries(
        records,
        render_script.StickerColorTrace.create(),
        pose_prepared=True,
        recover_orientation=False,
        recognize_moves=True,
        known_initial_colors={},
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
    monkeypatch.setattr(render_script, 'turn_color_match_matrix', lambda *args: None)
    monkeypatch.setattr(render_script, '_score_turn_color_offsets_on_gpu', score)
    monkeypatch.setattr(render_script, '_score_turn_color_offsets_for_states_on_gpu',
                        lambda pose, options, limit, device, likelihoods, states, **kwargs: [score(pose) for _ in states])
    monkeypatch.setattr(render_script, 'solve_pnp_from_low_error_assignments', lambda pose, **kwargs: pose)
    monkeypatch.setattr(render_script, 'pose_reprojection_error_is_acceptable', lambda _: True)
    log = tmp_path / 'scores.jsonl'
    render_script.recover_cached_pose_symmetries(
        records,
        render_script.StickerColorTrace.create(),
        pose_prepared=True,
        recover_orientation=False,
        recognize_moves=True,
        known_initial_colors={},
        turn_candidate_log=log,
    )
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[0]['search']['frames'] == [0, 1, 2]
    assert rows[0]['search']['skipped_frames'] == []
    assert rows[0]['search']['fallback_reason'] == 'no_surviving_path'
    assert rows[0]['search']['optimality'] == 'no_finite_path'
    assert all(not row['selected_committed_moves'] for row in rows)
    json.dumps(rows, allow_nan=False)


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
    monkeypatch.setattr(render_script, 'turn_color_match_matrix', lambda *args: None)
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
