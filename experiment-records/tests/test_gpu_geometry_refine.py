import numpy as np
import pytest
import torch

from test_independent_keypoint_pose_geometry import observation
from scripts import render_yolo11n_independent_keypoints as renderer
from scripts.independent_keypoint_pose_geometry import ArticulatedProjection
from scripts.independent_keypoint_gpu_refine import BatchedArticulatedProjection, refine_batch


@pytest.mark.parametrize('distorted',[False,True])
def test_batched_residual_and_analytic_jacobian_match_cpu(observation,distorted):
    from dataclasses import replace
    candidate = observation[0]
    if distorted:
        candidate = replace(candidate,dist_coeffs=np.array([.02,-.01,.001,-.002,.003,.0001,-.0002,.0003,.0001,.0002,-.0001,.0003]))
    axes = [0,1,2,0,1,2]
    seeds = np.stack([np.r_[candidate.rvec.ravel()+.01*i,candidate.tvec.ravel(),10.-i,-12.+i] for i in range(len(axes))])
    model = BatchedArticulatedProjection(renderer,candidate,axes,device='cpu')
    residual,jac = model.evaluate(model.tensor(seeds))
    for i,axis in enumerate(axes):
        expected_residual,expected_jac = ArticulatedProjection(renderer,candidate,axis).evaluate(seeds[i])
        np.testing.assert_allclose(residual[i],expected_residual,atol=1e-8)
        np.testing.assert_allclose(jac[i],expected_jac,atol=1e-8,rtol=1e-8)


def test_batched_solver_preserves_fixed_angles_and_reduces_objective(observation):
    candidate = observation[0]
    seeds = np.stack([np.r_[candidate.rvec.ravel(),candidate.tvec.ravel(),*angles] for angles in [(0,0),(10,0),(0,-10),(10,-10)]])
    result,diagnostics = refine_batch(renderer,candidate,[0,1,2,0],seeds,device='cpu',return_diagnostics=True)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[:,6:][seeds[:,6:]==0],0.)
    assert (np.abs(result[:,6:])<=45).all()
    assert (diagnostics['final_cost']<=diagnostics['initial_cost']+1e-9).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
@pytest.mark.parametrize('distorted',[False,True])
@pytest.mark.parametrize('zero_rotation',[False,True])
def test_mps_refinement_matches_tensor_objective_and_keeps_fixed_slabs(observation,distorted,zero_rotation):
    from dataclasses import replace
    candidate = observation[0]
    if distorted:
        candidate = replace(candidate,dist_coeffs=np.array([.02,-.01,.001,-.002,.003,.0001,-.0002,.0003,.0001,.0002,-.0001,.0003]))
    if zero_rotation:
        candidate = replace(candidate,rvec=np.zeros((3,1)))
    seeds = np.stack([np.r_[candidate.rvec.ravel(),candidate.tvec.ravel(),*angles] for angles in [(0,0),(10,0),(0,-10),(10,-10)]])
    model = BatchedArticulatedProjection(renderer,candidate,[0,1,2,0],device='mps')
    values,jac = model.evaluate(model.tensor(seeds))
    expected = BatchedArticulatedProjection(renderer,candidate,[0,1,2,0],device='cpu')
    expected_values,expected_jac = expected.evaluate(expected.tensor(seeds))
    np.testing.assert_allclose(values.cpu(),expected_values,atol=5e-4,rtol=5e-4)
    np.testing.assert_allclose(jac.cpu(),expected_jac,atol=5e-4,rtol=5e-4)
    from scripts.independent_keypoint_metal_refine import MetalArticulatedProjection
    metal = MetalArticulatedProjection(renderer,candidate,[0,1,2,0])
    raw = metal.evaluate(metal.tensor(seeds.astype(np.float32))).cpu().numpy()
    np.testing.assert_allclose(raw[:,:,:2].reshape(len(seeds),-1),expected_values,atol=5e-4,rtol=5e-4)
    np.testing.assert_allclose(raw[:,:,2:].reshape(len(seeds),-1,8),expected_jac,atol=5e-4,rtol=5e-4)
    result,diagnostics = refine_batch(renderer,candidate,[0,1,2,0],seeds,return_diagnostics=True)
    np.testing.assert_array_equal(result[:,6:][seeds[:,6:]==0],0.)
    assert (diagnostics['final_cost']<=diagnostics['initial_cost']+1e-4).all()
