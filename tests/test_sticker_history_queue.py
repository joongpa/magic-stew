import importlib.util
from pathlib import Path
import pickle
import sqlite3
import sys

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('sticker_queue_tests',
    Path(__file__).resolve().parents[1]/'scripts/search_independent_keypoint_frame.py')
f = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = f
SPEC.loader.exec_module(f)


@pytest.mark.parametrize('backend',['sqlite','buffered'])
def test_exact_lexicographic_queue_rank_survives_disk_spills_and_reopen(tmp_path,backend):
    cls = f.storage.MeanQueue if backend=='sqlite' else f.storage.BufferedMeanQueue
    kwargs = {} if backend=='sqlite' else dict(frontier_cache_size=1,write_batch_size=1)
    path = tmp_path/'queue.sqlite3'
    q = cls(path,**kwargs)
    for name,count,fit in [('bad',1,0.),('worse_fit',0,900.),('best',0,5.)]:
        node = f.storage.MeanNode(4,name,fit,(None,None,0),recent=(1, 2),
            priority=count,secondary_priority=fit)
        q.offer(name.encode(),node,pickle.dumps(name),dict(state_moves=()))
    q.close()
    q = cls(path,**kwargs)
    try:
        assert q.minimum() == 0 and q.minimum_secondary() == 5.
        nodes = [q.pop(pickle.loads) for _ in range(3)]
        assert [n.state for n in nodes] == ['best','worse_fit','bad']
        assert all(n.recent == (1, 2) for n in nodes)
        assert [n.secondary_priority for n in nodes] == [5.,900.,0.]
    finally:
        q.close()


def test_old_queue_schema_retains_zero_secondary_and_can_be_migrated(tmp_path):
    path = tmp_path/'old.sqlite3'
    c = sqlite3.connect(path)
    c.execute('CREATE TABLE contexts (key BLOB PRIMARY KEY,total REAL,priority REAL,depth INTEGER,serial INTEGER,pending INTEGER,payload BLOB) WITHOUT ROWID')
    payload = f.storage.MeanQueue.pack((pickle.dumps('old'),(None,None,0),0,()))
    c.execute('INSERT INTO contexts VALUES (?,?,?,?,?,?,?)',(b'old',.3,.3,1,1,1,payload))
    c.commit()
    c.close()
    q = f.storage.MeanQueue(path)
    try:
        node = q.pop(pickle.loads)
        assert node.state=='old' and node.priority==.3 and node.secondary_priority==0.
    finally:
        q.close()


def test_duplicate_points_add_one_physical_sticker_vote_and_any_match_wins():
    identities = np.arange(54)[::-1]
    nearest = np.array([[0,0,1,2]])
    valid = np.array([[True,True,True,False]])
    matrix = np.zeros((54,4))
    matrix[53,1] = 1.  # First point is wrong; its duplicate is correct.
    observed,correct = f.replay.load_renderer().sticker_color_observation_masks(identities,nearest,valid,matrix)
    assert int(observed[0]) == (1<<53)|(1<<52)
    assert int(correct[0]) == 1<<53


@pytest.mark.parametrize('cls',[f.replay.ReplayScorer,f.replay.PreparedReplayScorer])
def test_static_opening_collects_physical_history_without_move_fitting(monkeypatch,cls):
    m = f.replay.load_renderer()
    colors = f.replay.initial_sticker_colors(m,"R U F2")
    state = m.CubeGeometryState.identity().apply_moves(('Rw',"U'"))
    candidate = m.StickerPnPCandidate(sticker_type='corner',sticker_average_confidence=1.,
        image_points=np.empty((0,2)),object_points=np.empty((0,3)),
        projected_sticker_points=np.empty((0,2)),reprojection_error=0.,
        rvec=np.asarray([[.18],[-.25],[.12]]),tvec=np.asarray([[0.],[0.],[200.]]),
        camera_matrix=m.default_camera_matrix((256,256),1.2),dist_coeffs=np.zeros((4,1)),
        image_size=(256,256),cube_geometry_state=state)
    faces = m.projected_sticker_faces(candidate,candidate.rvec,candidate.tvec)
    names = [colors[i.label] for i,_,_ in faces]
    names[0] = next(color for color in m.PURE_COLOR_RGB if color!=names[0])
    pose = dict(sticker_type='corner',rvec=candidate.rvec.tolist(),tvec=candidate.tvec.tolist(),
        camera_matrix=candidate.camera_matrix.tolist(),dist_coeffs=candidate.dist_coeffs.tolist(),
        image_size=[256,256],sticker_mask_centers=[polygon.mean(axis=0).tolist() for _,_,polygon in faces],
        sticker_color_names=names)
    records = {i:dict(has_pose=True,selected_pose=pose,search=dict(status='forced_static')) for i in range(4)}
    scorer = cls(m,records,'cpu',initial_colors=colors,turning_start_frame=3,track_sticker_observations=True)
    def forbidden(*args,**kwargs):
        pytest.fail('The static prefix must not fit moves or perform orientation rescue')
    monkeypatch.setattr(m,'_score_turn_configurations_on_gpu',forbidden)
    monkeypatch.setattr(m,'_score_turn_color_offsets_for_states_on_gpu',forbidden)
    target = next(i for i, identity in enumerate(m.canonical_sticker_identities())
                  if identity.label == faces[0][0].label)
    for frame in range(4):
        choices = scorer(frame, state)
        assert len(choices) == 1 and choices[0].payload['state_moves'] == ()
        seen = int(choices[0].payload['sticker_observed_mask_hex'], 16)
        matched = int(choices[0].payload['sticker_correct_mask_hex'], 16)
        assert seen & ~matched == 1 << target
    assert scorer.scoring_calls == 0
