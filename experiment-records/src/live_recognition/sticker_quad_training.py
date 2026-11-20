"""Ultralytics pose training with clockwise C4 symmetry for center stickers.

Keeps the native YOLO11/YOLO26 location, presence, detection and RLE losses.
Only center target routing and center OKS matching accept cyclic shifts.
Requires the Ultralytics loss API introduced by 8.4.185.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.nn.modules.head import Pose26
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import ops
from ultralytics.utils.loss import E2ELoss, PoseLoss26, v8PoseLoss
from ultralytics.utils.metrics import kpt_iou
from ultralytics.utils.tal import PoseTaskAlignedAssigner, TaskAlignedAssigner

from .sticker_quad_schema import CLASS_NAMES


def is_center(classes):
    return (classes.long() % 3 == 2) & (classes >= 0) & (classes < 18)


def cyclic_keypoint_iou(gt, pred, classes, area, sigma):
    """Pairwise OKS; center rows alone accept four clockwise target rotations."""
    score = kpt_iou(gt, pred, area, sigma)
    center = is_center(classes)
    for shift in range(1, 4):
        alternate = kpt_iou(gt.roll(-shift, dims=1), pred, area, sigma)
        score = torch.where(center[:, None], torch.maximum(score, alternate), score)
    return score


class CyclicPoseAssigner(PoseTaskAlignedAssigner):
    def pair_overlaps(self, indices, gt_bboxes, pd_bboxes, pd_kpts, gt_kpts):
        b, g, a = indices
        iou = TaskAlignedAssigner.pair_overlaps(self, indices, gt_bboxes, pd_bboxes)
        gt = gt_kpts[b, g]
        pred = pd_kpts[b, a].float()
        area = (gt_bboxes[b, g, 2:] - gt_bboxes[b, g, :2]).prod(-1, keepdim=True) * 0.53
        center = self.center_mask[b, g]
        best = torch.zeros_like(iou)
        vis = gt[..., 2] > 0
        for shift in range(4):
            candidate = gt.roll(-shift, dims=1)
            visible = candidate[..., 2] > 0
            d = (pred - candidate[..., :2]).square().sum(-1)
            oks = ((-d / ((2 * self.sigmas).square() * (area + self.eps) * 2)).exp() * visible).sum(
                -1
            ) / visible.sum(-1).clamp(min=1)
            best = oks if shift == 0 else torch.where(center, torch.maximum(best, oks), best)
        return torch.where(vis.any(-1), (iou + best.to(iou.dtype)) / 2, iou)


def select_cyclic_targets(pred, gt, centers, area, sigmas, pose_gain, presence_gain, rle_cost=None):
    """Choose a single target permutation per assigned object; never average XY.

    Native OKS and presence reductions are reproduced per object. Optional
    RLE costs are already normalized into per-object contributions.
    """
    candidates = torch.stack([gt.roll(-i, dims=1) for i in range(4)], dim=1)
    visible = candidates[..., 2] != 0
    d = (pred[:, None, :, :2] - candidates[..., :2]).square().sum(-1)
    e = d / ((2 * sigmas).square()[None, None] * (area[:, None] + 1e-9) * 2)
    location = ((1 - (-e).exp()) * visible).sum(-1) / (visible.sum(-1) + 1e-9)
    presence = F.binary_cross_entropy_with_logits(
        pred[:, None, :, 2].expand_as(visible), visible.float(), reduction="none"
    ).mean(-1)
    score = pose_gain * location + presence_gain * presence
    if rle_cost is not None:
        score = score + rle_cost
    winner = score.detach().argmin(-1)
    winner = torch.where(centers, winner, torch.zeros_like(winner))
    return candidates[torch.arange(len(gt), device=gt.device), winner]


class CyclicLossMixin:
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        if tuple(self.kpt_shape) != (4, 3):
            raise ValueError("Sticker quad loss requires kpt_shape=[4, 3]")
        if isinstance(self.assigner, PoseTaskAlignedAssigner):
            self.assigner.__class__ = CyclicPoseAssigner

    def loss(self, preds, batch):
        # Mirror Ultralytics' per-image object packing used by target_gt_idx.
        batch_size = preds["kpts"].shape[0]
        indices = batch["batch_idx"].to(self.device).flatten().long()
        counts = torch.bincount(indices, minlength=batch_size)
        classes = torch.full(
            (batch_size, int(counts.max())), -1, device=self.device, dtype=torch.long
        )
        slots = (
            torch.arange(len(indices), device=self.device) - (counts.cumsum(0) - counts)[indices]
        )
        classes[indices, slots] = batch["cls"].flatten().to(self.device).long()
        self.center_mask = is_center(classes)
        if isinstance(self.assigner, CyclicPoseAssigner):
            self.assigner.center_mask = self.center_mask
        return super().loss(preds, batch)

    @torch.no_grad()
    def _candidate_rle(self, pred, gt):
        """Per-instance contributions to native global mean RLE for C4 routing."""
        n, k = gt.shape[:2]
        candidates = torch.stack([gt.roll(-i, dims=1) for i in range(4)], dim=1)
        visible = candidates[..., 2] != 0
        sigma = pred[..., -2:].float().sigmoid().clamp_min(torch.finfo(pred.dtype).tiny)
        sigma = sigma[:, None].expand(-1, 4, -1, -1)
        error = ((pred[:, None, :, :2] - candidates[..., :2]) / (sigma + 1e-9)).clamp(-100, 100)
        log_phi = self.flow_model.log_prob(error.reshape(-1, 2)).reshape(n, 4, k)
        terms = (sigma.log() - log_phi[..., None] + (sigma * 2).log() + error.abs()).sum(-1)
        weights = self.target_weights[None, None]
        # Native RLE sums over visible points and divides by their total count.
        return (
            (terms * visible * weights).sum(-1)
            * n
            / visible[:, 0].sum().clamp(min=1)
            * self.hyp.rle
        )

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, stride_tensor, target_bboxes, pred_kpts
    ):
        batch_indices = masks.nonzero(as_tuple=True)[0]
        target_indices = target_gt_idx[masks]
        gt = keypoints[batch_indices, target_indices].clone()
        strides = stride_tensor.view(1, -1).expand(masks.shape[0], -1)[masks]
        gt[..., :2] /= strides[:, None, None]
        boxes = target_bboxes[masks] / strides[:, None]
        area = (boxes[:, 2:] - boxes[:, :2]).prod(-1, keepdim=True)
        pred = pred_kpts[masks]
        rle = getattr(self, "rle_loss", None)
        rle_cost = (
            self._candidate_rle(pred, gt) if rle is not None and pred.shape[-1] == 5 else None
        )
        gt = select_cyclic_targets(
            pred,
            gt,
            self.center_mask[batch_indices, target_indices],
            area,
            self.keypoint_loss.sigmas,
            self.hyp.pose,
            self.hyp.kobj,
            rle_cost,
        )
        visible = gt[..., 2] != 0
        location = self.keypoint_loss(pred, gt, visible, area)
        presence = self.bce_pose(pred[..., 2], visible.float())
        if isinstance(self, PoseLoss26):
            rle_value = (
                self.calculate_rle_loss(pred, gt, visible).clamp(min=0)
                if rle is not None
                else pred[..., :0].sum()
            )
            return location, presence, rle_value
        return location, presence


class StickerQuadLoss(CyclicLossMixin, v8PoseLoss):
    pass


class StickerQuadLoss26(CyclicLossMixin, PoseLoss26):
    pass


class StickerQuadModel(PoseModel):
    def init_criterion(self):
        loss = StickerQuadLoss26 if isinstance(self.model[-1], Pose26) else StickerQuadLoss
        return (
            E2ELoss(self, loss)
            if getattr(self.model[-1], "one2one_cv2", None) is not None
            else loss(self)
        )


class StickerQuadValidator(PoseValidator):
    def _process_batch(self, preds, batch):
        result = DetectionValidator._process_batch(self, preds, batch)
        if len(batch["cls"]) and len(preds["cls"]):
            area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
            oks = cyclic_keypoint_iou(
                batch["keypoints"], preds["keypoints"], batch["cls"], area, self.sigma
            )
            result["tp_p"] = self.match_predictions(preds["cls"], batch["cls"], oks).cpu().numpy()
        else:
            result["tp_p"] = np.zeros((len(preds["cls"]), self.niou), dtype=bool)
        return result


class StickerQuadTrainer(PoseTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        names = self.data["names"]
        if tuple(names[i] for i in range(len(names))) != CLASS_NAMES or self.data["kpt_shape"] != [
            4,
            3,
        ]:
            raise ValueError("Expected the 18-class sticker_quads_v1 dataset")
        model = StickerQuadModel(
            cfg, ch=self.data.get("channels", 3), nc=18, data_kpt_shape=[4, 3], verbose=verbose
        )
        model.args = self.args
        model.names = dict(enumerate(CLASS_NAMES))
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        # Current YOLO26 may add an RLE term; keep its native loss names.
        criterion = self.model.init_criterion()
        branch = getattr(criterion, "one2one", criterion)
        self.loss_names = branch.loss_names
        return StickerQuadValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy.copy(self.args),
            _callbacks=self.callbacks,
        )
