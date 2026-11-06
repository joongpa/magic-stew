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
h = f.history


def observation(correct, sticker=0):
    return dict(sticker_observed_mask_hex=f'{1<<sticker:014x}',
                sticker_correct_mask_hex=f'{(1<<sticker) if correct else 0:014x}')


def transition(state,score,correct,**extra):
    return f.replay.Transition(state,score,(score,),dict(state_moves=(),**observation(correct),**extra))


def test_last_four_observations_include_current_frame_and_expire_old_errors():
    recent = h.EMPTY
    for _ in range(3):
        recent = h.observe(recent,observation(False,53))
        assert h.mismatch_count(recent) == 0
    assert h.observe(recent,{}) == recent  # Unseen frames add no vote.
    recent = h.observe(recent,observation(True,53))
    assert h.mismatch_count(recent) == 1
    assert h.summary(recent)['sticker_mismatched_mask_hex'] == f'{1<<53:014x}'
    assert h.summary(recent)['sticker_history_qualified_count'] == 1
    recent = h.observe(recent,observation(True,53))
    assert h.mismatch_count(recent) == 0  # Two old errors remain, no longer 3/4.


@pytest.mark.parametrize('backend',['sqlite','buffered'])
def test_fewer_mismatches_beats_arbitrarily_better_fit_in_global_queue(backend):
    def choices(frame,state):
        if frame == 0:
            return (transition('bad',.001,False),transition('good',100.,True))
        return (transition(state,.001 if state=='bad' else 100.,state=='good'),)
    q = (f.storage.MeanQueue() if backend=='sqlite' else
         f.storage.BufferedMeanQueue(frontier_cache_size=1,write_batch_size=1))
    visited = []
    try:
        result = f.frame_full_solve('root',(0,1,2,3),choices,lambda _:False,str,
            queue=q,sticker_history=True,sticker_history_weight=None,progress=lambda e:visited.append(e))
        # Both endpoints were popped; their rank must prefer good (0 mismatches,
        # fit 100) before bad (1 mismatch, fit .001). Greater depth cannot override it.
        terminal = [e for e in visited if e['event']=='furthest' and e['furthest_frame_index']==3]
        assert terminal[0]['current_sticker_mismatch_count'] == 0
        assert terminal[0]['current_frame_score'] == 100.
        assert result['status'] == 'exhausted'
    finally:
        q.close()


@pytest.mark.parametrize('backend',['sqlite','buffered'])
@pytest.mark.parametrize('weight',[None,.1])
def test_different_recent_histories_at_same_cube_state_cannot_merge(backend,weight):
    def choices(frame,state):
        if frame == 0:
            return (transition('a',.08,False,branch='a'),transition('b',.01,True,branch='b'))
        if frame < 3:
            return (transition(state,.08 if state=='a' else .01,
                               (frame>=2) if state=='a' else frame<2),)
        if frame == 3:
            return (transition('merged',.08 if state=='a' else .01,state=='a'),)
        return (transition('goal',.2,False),)
    q = (f.storage.MeanQueue() if backend=='sqlite' else
         f.storage.BufferedMeanQueue(frontier_cache_size=1,write_batch_size=1))
    try:
        result = f.frame_full_solve('root',tuple(range(5)),choices,lambda s:s=='goal',str,
            queue=q,sticker_history=True,sticker_history_weight=weight)
    finally:
        q.close()
    assert result['status'] == 'solved'
    assert result['steps'][0]['branch'] == 'a'
    assert result['steps'][-1]['sticker_mismatch_count'] == 0


def test_graded_penalty_starts_at_first_error_and_rolls_over_observations():
    recent = h.EMPTY
    for errors in range(1,5):
        recent = h.observe(recent,observation(False,53))
        assert h.penalty(recent) == (errors/4)**2
    assert h.observe(recent,{}) == recent
    for errors in range(3,-1,-1):
        recent = h.observe(recent,observation(True,53))
        assert h.penalty(recent) == (errors/4)**2
    # Independent stickers add; correct duplicate points still vote once.
    recent = h.observe(recent,observation(False,0))
    recent = h.observe(recent,observation(False,53))
    assert h.penalty(recent) == .125
    assert h.mismatch_count(recent) == 0


@pytest.mark.parametrize('backend',['sqlite','buffered'])
def test_graded_queue_penalizes_errors_before_binary_flag_without_pruning(backend):
    def choices(frame,state):
        if frame == 0:
            return (transition('bad',.001,False),transition('good',.02,True))
        return (transition(state,.001 if state=='bad' else .02,state=='good'),)
    q = (f.storage.MeanQueue() if backend=='sqlite' else
         f.storage.BufferedMeanQueue(frontier_cache_size=1,write_batch_size=1))
    visited = []
    try:
        result = f.frame_full_solve('root',(0,1),choices,lambda _:False,str,
            queue=q,sticker_history=True,minimum_color_accuracy=None,
            progress=lambda e:visited.append(e))
        terminal = [e for e in visited if e['event']=='furthest' and e['furthest_frame_index']==1]
        assert terminal[0]['current_frame_score'] == .02  # Better history beats .001 fit.
        assert terminal[0]['current_sticker_mismatch_count'] == 0
        assert terminal[0]['current_queue_score'] == .02
        assert result['unsolved_endpoints'] == 2  # Both branches retained and expanded.
    finally:
        q.close()


def test_soft_color_mode_retains_below_two_thirds_and_logs_combined_score():
    choice = transition('goal',.45,False,color_error=.9)
    result = f.frame_full_solve('root',(0,),lambda *_:(choice,),lambda s:s=='goal',str,
        sticker_history=True,minimum_color_accuracy=None,sticker_history_weight=.2)
    assert result['status'] == 'solved'
    assert result['color_pruned'] == 0
    assert result['steps'][0]['state_score'] == .45
    assert result['steps'][0]['sticker_history_penalty'] == .0625
    assert result['steps'][0]['queue_priority'] == pytest.approx(.45+.2*.0625)


@pytest.mark.parametrize('backend',['sqlite','buffered'])
def test_exact_lexicographic_queue_rank_survives_disk_spills_and_reopen(tmp_path,backend):
    cls = f.storage.MeanQueue if backend=='sqlite' else f.storage.BufferedMeanQueue
    kwargs = {} if backend=='sqlite' else dict(frontier_cache_size=1,write_batch_size=1)
    path = tmp_path/'queue.sqlite3'
    q = cls(path,**kwargs)
    for name,count,fit in [('bad',1,0.),('worse_fit',0,900.),('best',0,5.)]:
        node = f.storage.MeanNode(4,name,fit,(None,None,0),recent=h.EMPTY,
            priority=count,secondary_priority=fit)
        q.offer(name.encode(),node,pickle.dumps(name),dict(state_moves=()))
    q.close()
    q = cls(path,**kwargs)
    try:
        assert q.minimum() == 0 and q.minimum_secondary() == 5.
        nodes = [q.pop(pickle.loads) for _ in range(3)]
        assert [n.state for n in nodes] == ['best','worse_fit','bad']
        assert all(n.recent == h.EMPTY for n in nodes)
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
    recent = h.EMPTY
    payload = dict(sticker_observed_mask_hex=f'{int(observed[0]):014x}',
                   sticker_correct_mask_hex=f'{int(correct[0]):014x}')
    for _ in range(4):
        recent = h.observe(recent,payload)
    assert h.summary(recent)['sticker_mismatched_mask_hex'] == f'{1<<52:014x}'


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
    recent = h.EMPTY
    for frame in range(4):
        choices = scorer(frame,state)
        assert len(choices)==1 and choices[0].payload['state_moves']==()
        recent = h.observe(recent,choices[0].payload)
    target = next(i for i,identity in enumerate(m.canonical_sticker_identities()) if identity.label==faces[0][0].label)
    assert h.summary(recent)['sticker_mismatched_mask_hex']==f'{1<<target:014x}'
    assert scorer.scoring_calls==0


@pytest.mark.parametrize('with_window',[False,True])
def test_long_unreliable_gap_preserves_histories_without_observation_votes(with_window):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(2)
    m = f.replay.load_renderer()
    initial = m.CubeGeometryState.identity()
    target = initial.apply_move('R')
    class Scorer:
        def __init__(self):
            self.m = m
            self.records = {i:dict(search=dict(geometry=dict(options=[dict(
                reprojection_fit_error=.2 if i in (3,4,5) else .01)]))) for i in range(7)}
            self.calls = []
        def hold_reason(self,frame):return None
        def __call__(self,frame,state):
            self.calls.append(frame)
            error = 0. if frame<6 or state.signature()==target.signature() else .9
            return (f.replay.Transition(state,.2,(.2,),dict(state_moves=(),color_error=error,
                **observation(False,53))),)
    scorer = Scorer()
    def key(state):
        return pickle.dumps(state.signature())
    bridge = f.FrameObservationBridge(scorer,tuple(range(7)),fps=60.,advance_update=f.advance_update,
        canonical_update=f.canonical_update,transition_type=f.replay.Transition)
    result = f.frame_full_solve(initial,tuple(range(7)),scorer,
        lambda state:state.signature()==target.signature(),key,bridge=bridge,sticker_history=True,
        merge_histories=with_window,soft_window=f.SoftMoveWindow(key) if with_window else None)
    assert result['status']=='solved' and result['solved_frame_index']==6
    assert set(scorer.calls)=={0,1,2,6}
    assert [s['sticker_mismatch_count'] for s in result['steps']]==[0,0,0,0,0,0,1]
    assert result['steps'][-1]['sticker_history_qualified_count']==1
