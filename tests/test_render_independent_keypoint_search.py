import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "render_search_tests", Path(__file__).resolve().parents[1] /
    "scripts/render_independent_keypoint_search.py")
video = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = video
SPEC.loader.exec_module(video)


@pytest.fixture(scope="module")
def m():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    return video.replay.load_renderer()


def record(m):
    rvec = np.array([[.18], [-.25], [.12]])
    tvec = np.array([[0.], [0.], [200.]])
    camera = m.default_camera_matrix((256, 256), 1.2)
    objects, classes = m.visible_cube_lattice_points(rvec, tvec)
    projected, _ = cv2.projectPoints(objects, rvec, tvec, camera, np.zeros((4, 1)))
    return dict(has_pose=True, search_pose=dict(
        sticker_type="corner", rvec=rvec.tolist(), tvec=tvec.tolist(),
        camera_matrix=camera.tolist(), dist_coeffs=np.zeros((4, 1)).tolist(), image_size=[256, 256],
        assignment_image_points=projected.reshape(-1, 2).tolist(),
        assignment_observed_class_ids=classes.tolist(), sticker_mask_centers=[],
        sticker_mask_center_uncertainties=[], sticker_color_names=[]))


def step(m, angle=0., offset=90., symmetry=None):
    rotation = m.CubeLayerRotation(0, 1, angle, 1)
    if symmetry is not None:
        rotation = m.transform_layer_rotation_by_symmetry(rotation, symmetry)
    move = m.cube_rotation_to_notation(m.CubeLayerRotation(
        rotation.axis, rotation.side, offset, rotation.layer_count))
    result = dict(trial_id="0:+90", geometry_option_index=0,
        label="saved trial", is_baseline=False, layer_axis=rotation.axis, layer_side=rotation.side,
        layer_count=1, layer_rotations=[] if not angle else [dict(axis=rotation.axis,
            side=rotation.side, angle_degrees=angle, layer_count=1)], color_offset=offset,
        state_moves=[move] if move else [], state_score=.1,
        keypoint_mean_error=.01, heatmap_mean_error=.01)
    if symmetry is not None:
        result["orientation_symmetry"] = symmetry.tolist()
    return result


def test_saved_rescue_partial_angle_and_color_offset_render_in_physical_coordinates(m):
    state = m.CubeGeometryState.identity().apply_moves(("U", "Rw'"))
    before = state.signature()
    symmetry = next(s for s in m.CUBE_SYMMETRY_MATRICES if not np.array_equal(s, np.eye(3)))
    trial = step(m, 30., 90., symmetry)
    colors = video.replay.initial_sticker_colors(m, "R U")
    source = record(m)
    candidate = video.candidate_for_step(m, source, trial, state, colors)
    expected_rvec = m.transform_rvec_by_symmetry(np.asarray(source['search_pose']['rvec']), symmetry)
    np.testing.assert_allclose(cv2.Rodrigues(candidate.rvec)[0], cv2.Rodrigues(expected_rvec)[0])
    assert candidate.layer_rotations[0].angle_degrees == 30.
    assert candidate.layer_rotations[0].axis == trial['layer_axis']
    assert candidate.sticker_layer_rotations[0].angle_degrees == 120.
    assert candidate.turn_state_moves == tuple(trial['state_moves'])
    assert state.signature() == before == candidate.cube_geometry_state.signature()
    centers = {i.label: colors[i.label] for i in m.canonical_sticker_identities() if i.category == 'center'}
    assert tuple(centers[label] for label in candidate.orientation_axis_center_labels) == ('red', 'white', 'green')
    assert candidate.assignment_projected_points.shape == candidate.assignment_image_points.shape


def test_static_frames_use_selected_pose_when_search_pose_is_not_populated(m):
    source = record(m)
    source['selected_pose'] = source['search_pose']
    source['search_pose'] = dict(source['search_pose'], rvec=None, tvec=None)
    candidate = video.candidate_for_step(m, source,
        dict(state_moves=[], observation_status='turning_start_frame'),
        m.CubeGeometryState.identity(), video.replay.initial_sticker_colors(m, 'R U'))
    assert candidate is not None
    np.testing.assert_allclose(candidate.rvec, source['selected_pose']['rvec'])
    assert candidate.turn_state_moves == ()


def test_video_replays_consecutive_moves_once_and_freezes_solved_tail(m, tmp_path, monkeypatch):
    source, output, log = (tmp_path / name for name in ('input.avi', 'output.avi', 'turns.jsonl'))
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 25., (256, 256))
    assert writer.isOpened()
    for i in range(5):
        writer.write(np.full((256, 256, 3), 30 + 20*i, dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(m, 'add_quicktime_frame_counter', lambda _: False)
    move = dict(step(m),soft_window_penalty=.5,soft_window_penalty_until_frame=1,
                supported_move_count=5,soft_window_protected_frame=1,
                color_scoring='directional_confusion_cost',color_error=.25,color_match_count=4,
                color_correct_count=2,color_confusion_count=2,color_forbidden_count=0,color_confusion_cost=.5,
                ilds_budget=2,ilds_choice_index=3,ilds_discrepancy=1,ilds_discrepancies_used=2,
                compensatory_beam_id=1,compensatory_beam_start_frame=1,
                compensatory_beam_end_frame=3,compensatory_beam_level=1,
                compensatory_beam_phase='local',compensatory_beam_trigger='better_geometry_worse_color')
    result = dict(status='solved', solved_frame_index=2, config=dict(initial_scramble='R2', fps=25.),
        steps=[dict(frame_index=0, state_moves=[], observation_status='turning_start_frame'),
               dict(move, frame_index=1), dict(move, frame_index=2)])
    report = video.render_result(m, {i: record(m) for i in range(5)}, result, source,
                                output_path=output, turn_log=log)
    assert report['frames'] == 5 and report['fps'] == 25.
    assert report['virtual_cube_panel']
    assert report['committed_moves'] == move['state_moves'] * 2
    assert not report['model_inference_used'] and not report['move_search_used']
    capture = cv2.VideoCapture(str(output))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    assert capture.get(cv2.CAP_PROP_FPS) == 25.
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 512
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 256
    ok, rendered = capture.read()
    assert ok
    # The side panel contains the virtual cube, beyond the header and footer.
    assert np.count_nonzero(np.ptp(rendered[60:210, 256:], axis=2) > 30) > 200
    capture.release()
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r['frame_index'] for r in records] == list(range(5))
    assert [r['selected_committed_moves'] for r in records] == [[], move['state_moves'], move['state_moves'], [], []]
    assert records[3]['cube_geometry_state_before'] == records[4]['cube_geometry_state_before']
    assert records[0]['search']['status'] == 'forced_static'
    assert records[3]['search']['status'] == 'solved_hold'
    assert records[1]['search']['soft_window_penalty'] == .5
    assert records[1]['search']['soft_window_penalty_until_frame'] == 1
    assert records[1]['search']['ilds_budget'] == 2
    assert records[1]['search']['ilds_discrepancies_used'] == 2
    assert records[1]['search']['ilds_choice_index'] == 3
    assert records[1]['search']['compensatory_beam_id'] == 1
    assert records[1]['search']['compensatory_beam_level'] == 1
    assert records[1]['search']['compensatory_beam_phase'] == 'local'
    assert records[1]['search']['compensatory_beam_start_frame'] == 1
    assert records[1]['search']['compensatory_beam_end_frame'] == 3
    assert records[1]['search']['compensatory_beam_trigger'] == 'better_geometry_worse_color'
    assert records[1]['search']['supported_move_count'] == 5
    assert records[1]['color_scoring'] == 'directional_confusion_cost'
    assert records[1]['candidates'][0]['color_correct_count'] == 2
    assert records[1]['candidates'][0]['color_confusion_count'] == 2


def test_unsolved_results_are_not_presented_as_winning_paths(m):
    with pytest.raises(ValueError, match='solved result'):
        video.render_result(m, {}, dict(status='exhausted'), 'missing.mp4')


def test_render_handoff_uses_post_commit_colors_and_logs_move_once(m, tmp_path):
    source = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 25., (256, 256))
    assert writer.isOpened()
    writer.write(np.zeros((256, 256, 3), dtype=np.uint8))
    writer.release()
    turn = dict(step(m, angle=33., offset=0.), frame_index=0, state_moves=["U'"],
                handoff_commit=True, turn_handoff_bonus=.02, queue_priority=.08)
    config = dict(initial_scramble="U", fps=25.)
    before = m.CubeGeometryState.identity()
    colors = video.replay.initial_sticker_colors(m, config["initial_scramble"])
    candidate = video.candidate_for_step(m, record(m), turn, before, colors)
    assert candidate.cube_geometry_state.signature() == before.apply_moves(("U'",)).signature()
    assert candidate.sticker_layer_rotations[0].angle_degrees == 33.
    log = tmp_path / "turns.jsonl"
    result = dict(status="solved", solved_frame_index=0, config=config, steps=[turn])
    report = video.render_result(m, {0: record(m)}, result, source, turn_log=log)
    assert report["committed_moves"] == ["U'"]
    logged = json.loads(log.read_text())
    assert logged["cube_geometry_state_before"] == before.to_json()
    assert logged["selected_committed_moves"] == ["U'"]
    assert logged["search"]["handoff_commit"]


def test_unobserved_bridge_move_keeps_video_and_log_state_in_sync(m, tmp_path, monkeypatch):
    source, output, log = (tmp_path/name for name in ['in.avi', 'out.avi', 'turns.jsonl'])
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 25., (256, 256))
    for _ in range(4):
        writer.write(np.zeros((256, 256, 3), dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(m, 'add_quicktime_frame_counter', lambda _: False)
    move = step(m)
    skipped = dict(move, frame_index=1, state_score=None, queue_priority=None,
                   bridge_inferred=True, observation_status='unreliable_frame_bridge', frame_quality={})
    skipped.pop('geometry_option_index')
    records = {i: record(m) for i in range(4)}
    records[1] = dict(has_pose=False, search_pose=None, selected_pose=None)
    result = dict(status='solved', solved_frame_index=2, config=dict(initial_scramble='R2', fps=25.),
        steps=[dict(frame_index=0, state_moves=[]), skipped, dict(move, frame_index=2)])
    report = video.render_result(m, records, result, source, output_path=output, turn_log=log)
    assert report['committed_moves'] == move['state_moves']*2
    rows = [json.loads(s) for s in log.read_text().splitlines()]
    assert rows[1]['selected_committed_moves'] == move['state_moves']
    assert rows[1]['search']['status'] == 'unreliable_frame_bridge'
    assert rows[1]['cube_geometry_state_before'] == m.CubeGeometryState.identity().to_json()
