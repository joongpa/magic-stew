"""A geometry-guided, articulated pose branch for the independent YOLO detector.

The nine original classes retain the installed Ultralytics pose criterion.
This branch uses the feature pyramid and predicted class-scored point tokens;
ground-truth observations are never inputs. Its quality score learns projected
pose accuracy. It is neither a detection score nor keypoint-presence confidence.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

CLASS_NAMES = ("corner", "corner-edge", "corner-edge-center", "white", "yellow",
               "orange", "red", "blue", "green")
FACE_ORDER = ("U", "D", "F", "B", "R", "L")
FACE_COLORS = ("white", "yellow", "green", "blue", "red", "orange")
FACE_AXES = ((0, 0, 1), (0, 0, -1), (0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0))
CHECKPOINT_FORMAT = "direct_cube_pose_checkpoint_v2"


def rotation_from_6d(value: Tensor) -> Tensor:
    """Two columns with Gram-Schmidt; finite proper rotations at degenerate inputs."""
    first, second = value[..., :3], value[..., 3:]
    fallback = torch.zeros_like(first)
    fallback[..., 0] = 1
    first = torch.where(first.norm(dim=-1, keepdim=True) > 1e-6, first, fallback)
    first = F.normalize(first, dim=-1, eps=1e-6)
    second = second - (first * second).sum(-1, keepdim=True) * first
    alternative = F.one_hot(first.abs().argmin(-1), 3).to(first.dtype)
    alternative = alternative - (first * alternative).sum(-1, keepdim=True) * first
    second = torch.where(second.norm(dim=-1, keepdim=True) > 1e-6, second, alternative)
    second = F.normalize(second, dim=-1, eps=1e-6)
    return torch.stack((first, second, torch.linalg.cross(first, second, dim=-1)), dim=-1)


def project_points(points: Tensor, prediction: dict[str, Tensor]) -> Tensor:
    """Normalized exact pinhole-equivalent projection, preserving principal shift."""
    rotated = points @ prediction["rotation_matrix"].transpose(-1, -2)
    denominator = 1 + prediction["perspective"][:, None, None] * rotated[..., 2:] / 3
    principal = prediction["principal"][:, None]
    return principal + (prediction["center"][:, None] - principal
                        + prediction["scale"][:, None, None] * rotated[..., :2]) / denominator.clamp_min(.02)


def articulated_landmarks(rest: Tensor, owners: Tensor, angles: Tensor) -> Tensor:
    """Clockwise face offsets; matches the generator's moving-owner selection.

    Dataset targets have at most opposite outer layers active (corner cuts use
    the dominant angle). Therefore face-order rotations commute at every target.
    """
    points = rest.unsqueeze(0).expand(len(angles), -1, -1)
    centers = owners.unsqueeze(0).expand_as(points)
    for index, axis_values in enumerate(FACE_AXES):
        axis = angles.new_tensor(axis_values)
        theta = -angles[:, index] * math.pi / 180
        cosine, sine = theta.cos()[:, None, None], theta.sin()[:, None, None]

        def turn(value):
            return (value * cosine + torch.linalg.cross(axis.expand_as(value), value, dim=-1) * sine
                    + axis * (value * axis).sum(-1, keepdim=True) * (1 - cosine))

        selected = (torch.round((centers * axis).sum(-1)) == 1).unsqueeze(-1)
        points = torch.where(selected, turn(points), points)
        centers = torch.where(selected, turn(centers), centers)
    return points


class DirectPoseHead(nn.Module):
    """Fuse spatial pyramid features and multiple predicted points per class."""

    def __init__(self, channels, width=64, topk=16):
        super().__init__()
        self.topk = int(topk)
        self.projections = nn.ModuleList(nn.Sequential(
            nn.Conv2d(channel + 2, width, 1, bias=False), nn.GroupNorm(8, width), nn.SiLU(),
        ) for channel in channels)
        self.token_encoder = nn.Sequential(nn.Linear(12, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.regressor = nn.Sequential(
            nn.Linear(len(channels) * width * 16 + 9 * width * 2, 512),
            nn.LayerNorm(512), nn.SiLU(), nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 31),
        )
        final = self.regressor[-1]
        nn.init.normal_(final.weight, std=.001)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[:6] = final.bias.new_tensor((1, 0, 0, 0, 1, 0))
            final.bias[6:8] = .5
            final.bias[8] = math.log(.12)
            final.bias[9] = math.log(.2 / .8)
            final.bias[10:12] = .5
            final.bias[18:30] = .5

    def forward(self, features, point_xy, class_scores):
        spatial = []
        for feature, projection in zip(features, self.projections):
            height, width = feature.shape[-2:]
            y, x = torch.meshgrid(torch.linspace(-1, 1, height, device=feature.device, dtype=feature.dtype),
                                  torch.linspace(-1, 1, width, device=feature.device, dtype=feature.dtype), indexing="ij")
            coords = torch.stack((x, y)).unsqueeze(0).expand(len(feature), -1, -1, -1)
            spatial.append(F.adaptive_avg_pool2d(projection(torch.cat((feature, coords), 1)), (4, 4)).flatten(1))
        # These weights are pose-object detection/class scores, NOT the third
        # keypoint channel. Keep multiple candidates for every landmark/color.
        scores, indexes = class_scores.topk(min(self.topk, class_scores.shape[-1]), dim=-1)
        xy = point_xy[:, None].expand(-1, 9, -1, -1).gather(
            2, indexes[..., None].expand(-1, -1, -1, 2))
        classes = torch.eye(9, device=xy.device, dtype=xy.dtype)[None, :, None].expand(len(xy), -1, xy.shape[2], -1)
        tokens = self.token_encoder(torch.cat((xy, scores[..., None], classes), -1))
        weighted_mean = (tokens * scores[..., None]).sum(2) / scores.sum(2, keepdim=True).clamp_min(1e-6)
        weighted_max = (tokens * scores[..., None]).amax(2)
        raw = self.regressor(torch.cat((*spatial, weighted_mean.flatten(1), weighted_max.flatten(1)), 1))
        with torch.autocast(device_type=raw.device.type, enabled=False):
            return self.decode(raw.float())

    @staticmethod
    def decode(raw):
        """Keep rotations and the projector in FP32 even with mixed precision."""
        output = {"rotation_matrix": rotation_from_6d(raw[:, :6]), "rotation_6d": raw[:, :6],
                  "center": raw[:, 6:8], "scale": raw[:, 8].clamp(-7, 2).exp(),
                  "perspective": raw[:, 9].sigmoid(), "principal": raw[:, 10:12],
                  "angles": raw[:, 12:18].tanh() * 45,
                  "center_keypoints": raw[:, 18:30].reshape(-1, 6, 2), "quality_logit": raw[:, 30]}
        output["quality"] = output["quality_logit"].sigmoid()
        centers = raw.new_tensor(FACE_AXES).mul(1.5).unsqueeze(0).expand(len(raw), -1, -1)
        output["projected_centers"] = project_points(centers, output)
        return output


class DirectCubePoseModel(nn.Module):
    """Preserve independent YOLO predictions and attach the articulated branch."""

    def __init__(self, detector, width=64, topk=16):
        super().__init__()
        self.detector = detector
        head = detector.model[-1]
        if head.nc != 9 or list(head.kpt_shape) != [1, 3]:
            raise ValueError("Direct pose needs nine independent classes and kpt_shape=[1,3]")
        if head.cv2 is None:
            raise ValueError("Use an unfused training checkpoint, with the one-to-many head intact")
        channels = [block[0].conv.in_channels for block in head.cv2]
        self.head_config = {"width": width, "topk": topk}
        self.pose_head = DirectPoseHead(channels, **self.head_config)
        self.base_frozen = False

    def freeze_detector(self, frozen=True):
        self.base_frozen = frozen
        for name, parameter in self.detector.named_parameters():
            # Match Ultralytics' permanent freeze of the DFL integral weights.
            parameter.requires_grad_(not frozen and ".dfl" not in name)
        if frozen:
            self.detector.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.base_frozen:
            self.detector.eval()
        return self

    def encode_observations(self, images):
        """Return model-predicted inputs; a frozen encoder can cache these."""
        from ultralytics.nn.modules.head import Pose26
        from ultralytics.utils.loss import PoseLoss26, v8PoseLoss
        from ultralytics.utils.tal import make_anchors

        independent = self.detector(images)
        raw = independent[1] if isinstance(independent, tuple) else independent
        if not isinstance(raw, dict):
            raise TypeError("This branch requires Ultralytics' dictionary pose outputs (tested with 8.4.51)")
        pyramid_branch = raw.get("one2many", raw)
        token_branch = raw.get("one2one", pyramid_branch)
        if "kpts" not in token_branch or "scores" not in token_branch:
            raise ValueError("YOLO output is missing independent keypoints or class logits")
        head = self.detector.model[-1]
        anchors, strides = make_anchors(token_branch["feats"], head.stride, .5)
        # Use the installed TRAINING decoder: YOLO26 and YOLO11 have different
        # XY formulas. The inference decoder applies an in-place sigmoid to
        # presence, which is not safe to differentiate through during training.
        decoder = PoseLoss26.kpts_decode if isinstance(head, Pose26) else v8PoseLoss.kpts_decode
        keypoints = token_branch["kpts"].transpose(1, 2).reshape(len(images), -1, 1, 3)
        points = decoder(anchors, keypoints)[:, :, 0, :2] * strides[None]
        points = points / images.new_tensor((images.shape[-1], images.shape[-2]))
        return independent, pyramid_branch["feats"], points, token_branch["scores"].sigmoid()

    def forward(self, images):
        independent, features, points, scores = self.encode_observations(images)
        return {"independent": independent, "pose": self.pose_head(features, points, scores)}


def make_detector(model_source, epochs=100):
    """Load a trained independent detector or adapt a stock pose architecture."""
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import PoseModel

    if isinstance(model_source, dict) or str(model_source).endswith((".yaml", ".yml")):
        detector = PoseModel(copy.deepcopy(model_source), nc=9, data_kpt_shape=[1, 3], verbose=False)
    else:
        pretrained = YOLO(str(model_source)).model.float()
        if pretrained.model[-1].nc == 9 and list(pretrained.model[-1].kpt_shape) == [1, 3]:
            names = [pretrained.names[index] for index in range(9)]
            if tuple(names) != CLASS_NAMES:
                raise ValueError(f"Checkpoint class order differs from the dataset: {names}")
            detector = pretrained
        else:
            detector = PoseModel(copy.deepcopy(pretrained.yaml), nc=9, data_kpt_shape=[1, 3], verbose=False)
            detector.load(pretrained)
    detector.names = dict(enumerate(CLASS_NAMES))
    detector.args = get_cfg(overrides={"epochs": epochs})
    for name, parameter in detector.named_parameters():
        parameter.requires_grad_(".dfl" not in name)
    return detector


class IndependentPoseCriterion:
    """Dispatch the shipped pose losses using the actual prediction structure.

    Runtime head flags and checkpoint flags can disagree across Ultralytics
    versions or change between warmup and fine-tuning. Never send a container
    of one2many/one2one branches to the flat loss, which expects preds['kpts'].
    """

    def __init__(self, detector):
        from ultralytics.nn.modules.head import Pose26
        from ultralytics.utils.loss import E2ELoss, PoseLoss26, v8PoseLoss

        loss_class = PoseLoss26 if isinstance(detector.model[-1], Pose26) else v8PoseLoss
        # Pose26's stock loss also keeps its matching XY decoder and RLE term
        # when a loaded checkpoint disables end-to-end detection.
        self.flat = loss_class(detector)
        self.end_to_end = E2ELoss(detector, loss_class)
        self.last_layout = None

    @staticmethod
    def raw_predictions(predictions):
        while isinstance(predictions, tuple) and len(predictions) == 2:
            predictions = predictions[1]
        if not isinstance(predictions, dict):
            raise ValueError(f"Expected raw YOLO pose dictionaries, got {type(predictions).__name__}")
        return predictions

    @staticmethod
    def is_pose_branch(value):
        return isinstance(value, dict) and all(key in value for key in ("kpts", "boxes", "scores", "feats"))

    def __call__(self, predictions, batch):
        raw = self.raw_predictions(predictions)
        if self.is_pose_branch(raw):
            criterion, inputs, layout = self.flat, raw, "flat pose output"
        else:
            many, one = raw.get("one2many"), raw.get("one2one")
            if self.is_pose_branch(many) and self.is_pose_branch(one):
                criterion, inputs, layout = self.end_to_end, raw, "one2many + one2one pose outputs"
            elif self.is_pose_branch(many):
                criterion, inputs, layout = self.flat, many, "one2many pose output"
            elif self.is_pose_branch(one):
                criterion, inputs, layout = self.flat, one, "one2one pose output"
            else:
                structure = {key: sorted(value) if isinstance(value, dict) else type(value).__name__
                             for key, value in raw.items()}
                raise ValueError(f"YOLO outputs contain no complete pose branch (need kpts/boxes/scores/feats): {structure}")
        if layout != self.last_layout:
            print(f"Independent pose loss: {type(criterion).__name__} ({layout})", flush=True)
            self.last_layout = layout
        return criterion(inputs, batch)

    def update(self):
        self.end_to_end.update()

    def kpts_decode(self, *args, **kwargs):
        return self.flat.kpts_decode(*args, **kwargs)


def independent_pose_criterion(detector):
    """Keep stock pose losses, dispatching flat versus E2E from real outputs."""
    return IndependentPoseCriterion(detector)


class SymmetryPoseLoss(nn.Module):
    """Choose a single allowed symmetry for all structured losses together."""

    def __init__(self, manifest, rest_points, owners, *, rotation=1., projection=1., angles=1.,
                 centers=10., geometry=10., quality=.25, quality_sigma=.02):
        super().__init__()
        candidates = manifest["candidates"]
        self.register_buffer("rotations", torch.tensor([c["candidate_to_reference"] for c in candidates], dtype=torch.float32))
        self.register_buffer("center_inverse", torch.tensor([c["reference_to_candidate_center_index"] for c in candidates]).argsort(-1))
        self.register_buffer("landmark_inverse", torch.tensor([c["reference_to_candidate_landmark_index"] for c in candidates]).argsort(-1))
        self.register_buffer("rest_points", torch.as_tensor(rest_points, dtype=torch.float32))
        self.register_buffer("owners", torch.as_tensor(owners, dtype=torch.float32))
        self.weights = dict(rotation=rotation, projection=projection, angles=angles, centers=centers, geometry=geometry, quality=quality)
        self.quality_sigma = quality_sigma

    def forward(self, prediction, target, image_size=160):
        enabled = target["enabled"].float()
        count = enabled.sum().clamp_min(1)
        target_rotations = target["rotation"][:, None] @ self.rotations[None]
        target_angles = target["angles"][:, self.center_inverse]
        target_centers = target["centers"][:, self.center_inverse]
        target_landmarks = target["landmarks"][:, self.landmark_inverse]
        articulated = articulated_landmarks(self.rest_points, self.owners, prediction["angles"])
        projected = project_points(articulated, prediction)

        def smooth(value, expected, beta=.02):
            return F.smooth_l1_loss(value.expand_as(expected), expected, reduction="none", beta=beta)

        components = {
            "rotation": (prediction["rotation_matrix"][:, None] - target_rotations).square().mean((-1, -2)),
            "angles": smooth(prediction["angles"][:, None] / 45, target_angles / 45, beta=.1).mean(-1),
            "centers": (smooth(prediction["center_keypoints"][:, None], target_centers).mean((-1, -2))
                        + smooth(prediction["projected_centers"][:, None], target_centers).mean((-1, -2))),
            "geometry": smooth(projected[:, None], target_landmarks).mean((-1, -2)),
        }
        projection_loss = (
            F.smooth_l1_loss(prediction["center"], target["center"], reduction="none", beta=.02).mean(-1)
            + F.smooth_l1_loss(prediction["scale"].log(), target["scale"].log(), reduction="none", beta=.1)
            + F.smooth_l1_loss(prediction["perspective"], target["perspective"], reduction="none", beta=.1)
            + F.smooth_l1_loss(prediction["principal"], target["principal"], reduction="none", beta=.02).mean(-1)
        )
        candidate_loss = sum(self.weights[name] * values for name, values in components.items())
        candidate_loss = candidate_loss + self.weights["projection"] * projection_loss[:, None]
        selected = candidate_loss.masked_fill(~target["allowed"], float("inf")).argmin(-1)
        rows = torch.arange(len(enabled), device=enabled.device)
        chosen = {name: values[rows, selected] for name, values in components.items()}
        chosen["projection"] = projection_loss
        losses = {name: (values * enabled).sum() / count for name, values in chosen.items()}
        landmark_error = (projected - target_landmarks[rows, selected]).square().sum(-1).mean(-1).clamp_min(1e-12).sqrt()
        observed = target["observed_landmarks"][:, self.landmark_inverse][rows, selected]
        observed_mask = target["observed_mask"][:, self.landmark_inverse][rows, selected].float()
        observed_count = observed_mask.sum(-1)
        observed_error = (((projected - observed).square().sum(-1) * observed_mask).sum(-1)
                          / observed_count.clamp_min(1)).clamp_min(1e-12).sqrt()
        # A detached, projection-error-derived quality target prevents the
        # confidence output from reducing geometric supervision by itself.
        quality_error = torch.maximum(landmark_error, observed_error)
        quality_target = torch.exp(-.5 * (quality_error.detach() / self.quality_sigma).square()) * enabled
        losses["quality"] = F.binary_cross_entropy_with_logits(prediction["quality_logit"], quality_target)
        losses["total"] = sum(self.weights[name] * value for name, value in losses.items())
        with torch.no_grad():
            relative = prediction["rotation_matrix"].transpose(-1, -2) @ target_rotations[rows, selected]
            rotation_degrees = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1).acos() * 180 / math.pi
            center_error = (prediction["projected_centers"] - target_centers[rows, selected]).square().sum(-1).mean(-1).sqrt()
            angle_error = (prediction["angles"] - target_angles[rows, selected]).abs().mean(-1)
            metrics = {"rotation_deg": (rotation_degrees * enabled).sum() / count,
                       "center_rmse_px": (center_error * enabled).sum() / count * image_size,
                       "landmark_rmse_px": (landmark_error * enabled).sum() / count * image_size,
                       "observed_landmark_rmse_px": (observed_error * enabled * (observed_count > 0)).sum()
                       / (enabled * (observed_count > 0)).sum().clamp_min(1) * image_size,
                       "angle_mae_deg": (angle_error * enabled).sum() / count,
                       "quality_target": quality_target.mean(), "quality_predicted": prediction["quality"].mean()}
            metrics["per_image"] = {"rotation_deg": rotation_degrees, "center_rmse_px": center_error * image_size,
                                    "landmark_rmse_px": landmark_error * image_size, "angle_mae_deg": angle_error,
                                    "observed_landmark_rmse_px": observed_error * image_size}
        return losses, metrics, selected


def load_direct_pose_checkpoint(path, device="cpu"):
    """Load the training/inference wrapper, with no dependency on its source .pt."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected a v2 direct cube pose checkpoint, which records the detector's end-to-end mode")
    detector = make_detector(checkpoint["detector_yaml"], checkpoint["train_args"]["epochs"])
    # The trained YOLO26 checkpoint can override its YAML's end2end setting.
    # This flag selects different prediction branches and is not a state_dict
    # tensor. Losing it changes the custom head's inputs after reloading.
    detector.end2end = checkpoint["detector_end2end"]
    model = DirectCubePoseModel(detector, **checkpoint["head_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint
