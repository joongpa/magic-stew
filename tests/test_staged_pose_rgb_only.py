"""RGB-only handoff/joint routing and differentiable full-prediction geometry."""
from types import SimpleNamespace

import pytest
import torch

from live_recognition.staged_pose.fields import CubePoseFields
from live_recognition.staged_pose.kinematics import rotation_exp
from live_recognition.staged_pose.loss import StagedPoseLoss, visible_observations
from live_recognition.staged_pose.phase import decode, decode_hypotheses
from scripts.export_direct_cube_pose_dataset import symmetry_manifest
from scripts.train_yolo26_staged_cube_pose import build_parser, validate_args, selection_score, fixed_split


@pytest.fixture
def criterion(monkeypatch):
    monkeypatch.setattr('live_recognition.staged_pose.loss.independent_pose_criterion',
        lambda _: lambda raw, batch: (batch['img'].sum()*0, None))
    return StagedPoseLoss(SimpleNamespace(detector=None), symmetry_manifest())


def example(criterion):
    truth = dict(rotation_matrix=rotation_exp(torch.tensor([[.3, -.7, .2]])),
        center=torch.tensor([[.51, .49]]), scale=torch.tensor([.12]),
        perspective=torch.tensor([.2]), principal=torch.tensor([[.5, .5]]),
        angles=torch.tensor([[23., -11., 0., 0., 0., 0.]]))
    projected, visible, types, _ = criterion.reference.template.geometry.project(truth, torch.tensor([0]))
    uv=projected[0, visible[0]].detach();classes=types[0, visible[0]]
    batch=dict(img=torch.zeros(1, 3, 128, 128), batch_idx=torch.zeros(len(uv)),
        cls=classes[:, None].float(), keypoints=torch.cat((uv, torch.ones(len(uv), 1)), -1)[:, None])
    target=dict(truth, rotation=truth['rotation_matrix'], eligible=torch.ones(1, 6, dtype=torch.bool),
        enabled=torch.ones(1, dtype=torch.bool),
        visible_xy=criterion.reference.project(truth, torch.tensor([0])), visible_mask=torch.ones(1,110,dtype=torch.bool))
    return truth, batch, target


def test_all_axis_decodes_match_deployed_decoder_and_axis_gradients():
    torch.manual_seed(31)
    logits=torch.randn(4,3,9,9,requires_grad=True)
    angles,probability=decode_hypotheses(logits)
    deployed,_,selected=decode(logits)
    torch.testing.assert_close(angles[torch.arange(4),selected],deployed)
    for axis in range(3):
        masked=torch.full_like(logits,-10000.);masked[:,axis]=logits[:,axis]
        torch.testing.assert_close(angles[:,axis],decode(masked)[0])
    ((probability*torch.tensor([1.,2.,4.])).sum()+angles.square().mean()).backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad.abs().sum((0,2,3))>0).all()


def test_full_geometry_alone_reaches_rotation_camera_angles_and_axis(criterion):
    truth,batch,target=example(criterion)
    spin=torch.tensor([[.35,-.76,.23]],requires_grad=True)
    pose={k:v.detach().clone().requires_grad_(True) for k,v in truth.items() if k!='rotation_matrix'}
    pose['rotation_matrix']=rotation_exp(spin)
    logits=torch.zeros(1,3,9,9,requires_grad=True)
    loss,rmse,active=criterion.predicted_geometry_loss(pose,logits,batch,target['enabled'],128)
    loss.backward()
    assert active.all() and torch.isfinite(rmse).all()
    for value in [spin,pose['center'],pose['scale'],pose['perspective'],pose['principal'],logits]:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum()>0
    # The axis marginal itself gets a geometric gradient, not only local bins.
    assert (logits.grad.sum((-1,-2)).abs()>1e-7).any()


@pytest.mark.parametrize('phase',['handoff','joint'])
def test_autonomous_forward_and_metrics_never_use_gt_conditioning(criterion,phase):
    truth,batch,target=example(criterion)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.logits=torch.nn.Parameter(torch.randn(1,3,9,9)*.1)
            self.fields=CubePoseFields();self.detector=None;self.calls=0
        def encode(self,images,amp):
            return dict(static=truth,raw_static=self.logits.flatten(1),encoded=dict(raw={}))
        def articulate(self,context,base=None):
            assert base is None, 'Ground truth entered the prediction path'
            self.calls+=1
            return self.logits,None
    model=Model().train()
    # joint needs the raw invariant-orientation field shape as well.
    original=model.encode
    model.encode=lambda *a:dict(original(*a),raw_static=torch.cat((model.fields.encode_rotation(truth['rotation_matrix']),torch.zeros(1,6)),1))
    _,parts,context=criterion(model,batch,target,phase)
    metrics=criterion.metrics(model,context,target)
    assert model.calls==1
    assert metrics['teacher_observed_moving_turn_deg']==(0.,0)
    assert parts['replay']==0
    baseline=parts['geometry'].detach().clone()
    changed=dict(target,angles=torch.tensor([[0.,0.,-29.,17.,0.,0.]]),
        rotation=rotation_exp(torch.tensor([[1.,.4,-.2]])),center=torch.tensor([[.8,.1]]))
    _,parts2,context2=criterion(model,batch,changed,phase)
    torch.testing.assert_close(parts2['geometry'],baseline)
    torch.testing.assert_close(context2['actual_logits'],context['actual_logits'])


def test_observation_packing_preserves_visibility1_and2_and_detaches_coordinates():
    points=torch.tensor([[[.2,.3,1]],[[.4,.5,0]],[[.6,.7,2]]],requires_grad=True)
    batch=dict(batch_idx=torch.tensor([1.,0.,1.]),cls=torch.tensor([[0.],[1.],[3.]]),keypoints=points)
    xy,classes,mask=visible_observations(batch,torch.tensor([True,True]))
    assert not xy.requires_grad
    assert mask.sum(-1).tolist()==[0,2]
    assert classes[1].tolist()==[0,3]


def test_geometry_first_schedule_and_checkpoint_selection():
    args=validate_args(build_parser().parse_args([]))
    assert args.handoff_epochs==30 and args.joint_epochs==60
    assert args.pose_aux_weight==.25 and args.geometry_weight==1
    assert selection_score(dict(observed_fit_rmse_px=2.1,moving_turn_deg=13), 'joint')==2.1
    for flag,value in [('--geometry-weight','0'),('--pose-aux-weight','-1'),('--geometry-weight','nan')]:
        with pytest.raises(ValueError):validate_args(build_parser().parse_args([flag,value]))


def test_explicit_continuation_split_preserves_holdout_and_rejects_leakage(tmp_path):
    import json
    path=tmp_path/'split.json';path.write_text(json.dumps(dict(train=['b','a'],val=['c'])))
    entries=[dict(key=k) for k in ['a','b','c']]
    train,val=fixed_split(entries,path,dict(training_keys=['a','b']))
    assert [e['key'] for e in train]==['b','a'] and [e['key'] for e in val]==['c']
    with pytest.raises(ValueError,match='previously used'):
        fixed_split(entries,path,dict(training_keys=['c']))
    with pytest.raises(ValueError,match='cannot be combined'):
        validate_args(build_parser().parse_args(['--split',str(path),'--limit','4']))
