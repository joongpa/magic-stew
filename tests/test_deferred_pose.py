from dataclasses import replace
import numpy as np
import pytest
import torch

from test_independent_keypoint_pose_geometry import observation
from scripts import render_yolo11n_independent_keypoints as renderer
from scripts import independent_keypoint_pose_geometry as geometry
from scripts.independent_keypoint_deferred_pose import static_candidates,buffered_cameras,_camera_grid,fit_buffered


def test_static_candidates_ignore_saved_pose_and_articulation(observation):
    base=observation[0]
    a,stats=static_candidates(renderer,base,beam=10)
    poisoned=replace(base,rvec=np.full((3,1),np.nan),tvec=np.full((3,1),np.nan),
        layer_rotations=(renderer.CubeLayerRotation(0,1,44.),))
    b,_=static_candidates(renderer,poisoned,beam=10)
    assert len(a)==len(b)>0
    for p,q in zip(a,b,strict=True):
        np.testing.assert_array_equal(p.parameters,q.parameters)
        assert p.score==q.score
    assert stats['raw_roots']>0
    assert stats['best_landmark_support']>=4


def test_static_stage_does_not_call_articulated_fitter(observation,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('articulation in static stage')
    monkeypatch.setattr(geometry,'fit_frame',forbidden)
    monkeypatch.setattr(geometry,'_refine',forbidden)
    monkeypatch.setattr(renderer,'parallel_turn_frame_geometry',forbidden)
    proposals,_=static_candidates(renderer,observation[0],beam=8)
    assert proposals
    selected=buffered_cameras(renderer,proposals,proposals,proposals,budget=6)
    assert 1<=len(selected)<=6


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
def test_multi_camera_grid_matches_individual_full_sweep(observation):
    base=observation[0]
    proposals,_=static_candidates(renderer,base,beam=8)
    cameras=proposals[:2]
    bank,stats=_camera_grid(renderer,base,cameras,device='mps')
    expected=[]
    for proposal in cameras:
        x=proposal.parameters
        camera=replace(base,rvec=x[:3].reshape(3,1),tvec=x[3:6].reshape(3,1))
        expected+=geometry.swept_geometries(renderer,base,camera,sweep_device='mps')
    assert len(bank)>0 and stats['configurations']==len(cameras)*1026
    poisoned=[replace(p,axis=(p.axis+1)%3,layer=(p.layer+1)%3) for p in cameras]
    independent,_=_camera_grid(renderer,base,poisoned,device='mps')
    assert len(independent)==len(bank)
    for first,second in zip(bank,independent,strict=True):
        assert first.axis==second.axis and first.angles==second.angles
        np.testing.assert_allclose(first.score,second.score,atol=1e-7)
    for fit in bank:
        differences=[abs(fit.score-e.score) for e in expected if fit.axis==e.axis and fit.angles==e.angles
            and np.allclose(fit.candidate.rvec,e.candidate.rvec,atol=1e-8)
            and np.allclose(fit.candidate.tvec,e.candidate.tvec,atol=1e-8)]
        assert differences and min(differences)<2e-5
