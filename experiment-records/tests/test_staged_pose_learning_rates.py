"""Joint adaptation rates, exact resume policy and actual optimizer steps."""
import copy
import math

import pytest
import torch

from scripts.train_yolo26_staged_cube_pose import (
    apply_resumed_joint_rates, build_parser, learning_rate_cli_overrides,
    optimizer_for, restore_learning_rate_plan, validate_args,
)


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.static_head = torch.nn.Linear(2, 2)
        self.articulation_head = torch.nn.Linear(2, 2)
        self.source = torch.nn.Linear(2, 2)
        for p in self.parameters():
            torch.nn.init.ones_(p)

    def configure_phase(self, phase):
        self.static_head.requires_grad_(phase in ('static', 'joint'))
        self.source.requires_grad_(phase in ('static', 'joint'))
        self.articulation_head.requires_grad_(phase != 'static')


def test_new_joint_defaults_make_larger_actual_updates_in_both_heads():
    model = SmallModel()
    old = copy.deepcopy(model)
    optimizer = optimizer_for(model, 'joint')
    previous = optimizer_for(old, 'joint', joint_static_lr=1e-5,
        joint_articulation_lr=5e-5, joint_backbone_lr=2e-6)
    assert {g['label']: g['lr'] for g in optimizer.param_groups} == dict(static=1e-4, source=1e-5, turn=3e-4)
    before = {n: p.detach().clone() for n,p in model.named_parameters()}
    for instance,opt in [(model,optimizer),(old,previous)]:
        sum(p.sum() for p in instance.parameters()).backward()
        opt.step()
    for name,param in model.named_parameters():
        delta = (param.detach()-before[name]).abs().mean()
        previous_delta = (dict(old.named_parameters())[name].detach()-before[name]).abs().mean()
        assert torch.isfinite(delta) and delta > previous_delta * 4


def test_component_phases_keep_existing_rates_and_freeze_masks():
    model=SmallModel()
    assert {g['label']:g['lr'] for g in optimizer_for(model,'static').param_groups} == dict(static=1e-4,source=2e-5)
    for phase in ['articulation','handoff']:
        assert {g['label']:g['lr'] for g in optimizer_for(model,phase).param_groups} == dict(turn=5e-4)
    optimizer=optimizer_for(model,'joint',joint_static_lr=2e-4,joint_backbone_lr=3e-5,joint_articulation_lr=4e-4)
    assert {g['label']:g['lr'] for g in optimizer.param_groups} == dict(static=2e-4,source=3e-5,turn=4e-4)


def test_resume_preserves_saved_rates_or_legacy_rates_unless_explicit():
    args=build_parser().parse_args([])
    restore_learning_rate_plan(args,{}, {})
    assert (args.joint_static_lr,args.joint_articulation_lr,args.joint_backbone_lr)==(1e-5,5e-5,2e-6)
    saved=dict(joint_static_lr=7e-5,joint_articulation_lr=2e-4,joint_backbone_lr=8e-6)
    argv=['--joint-static-lr=0.0002']
    args=build_parser().parse_args(argv)
    restore_learning_rate_plan(args,saved,learning_rate_cli_overrides(args,argv))
    assert (args.joint_static_lr,args.joint_articulation_lr,args.joint_backbone_lr)==(2e-4,2e-4,8e-6)


def test_resumed_override_keeps_cosine_epoch_and_adam_moments():
    model=SmallModel();optimizer=optimizer_for(model,'joint')
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,60,eta_min=1e-6)
    for _ in range(17):
        optimizer.zero_grad();sum(p.square().sum() for p in model.parameters()).backward()
        optimizer.step();scheduler.step()
    moments=[s['exp_avg'].clone() for s in optimizer.state.values()]
    previous_backbone=optimizer.param_groups[1]['lr']
    apply_resumed_joint_rates(optimizer,scheduler,dict(joint_static_lr=2e-4,joint_articulation_lr=5e-4))
    assert scheduler.last_epoch==17 and optimizer.param_groups[1]['lr']==previous_backbone
    for old,state in zip(moments,optimizer.state.values(),strict=True):
        torch.testing.assert_close(old,state['exp_avg'])
    optimizer.step();scheduler.step()
    for group,base in zip(optimizer.param_groups,[2e-4,1e-5,5e-4],strict=True):
        expected=1e-6+(base-1e-6)*.5*(1+math.cos(math.pi*18/60))
        assert group['lr']==pytest.approx(expected,abs=1e-12)


@pytest.mark.parametrize('flag',['--joint-static-lr','--joint-articulation-lr','--joint-backbone-lr'])
@pytest.mark.parametrize('value',['0','-0.01','nan','inf'])
def test_invalid_learning_rates_rejected(flag,value):
    with pytest.raises(ValueError,match='positive and finite'):
        validate_args(build_parser().parse_args([flag,value]))
