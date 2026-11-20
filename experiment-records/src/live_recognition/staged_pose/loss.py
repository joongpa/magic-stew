"""Static, circular articulation and retained-visible reprojection objectives."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from live_recognition.direct_cube_pose import independent_pose_criterion

from .data import PhysicalReference
from .kinematics import rotation_exp
from .phase import decode, decode_hypotheses, loss_function


OBJECTIVE_VERSION = "rgb_only_handoff_joint_geometry_v1"


def visible_observations(batch, enabled):
    """Pack original visible YOLO rows as loss targets, without physical IDs.

    Visibility 1 and 2 both count. Observation coordinates are fixed labels,
    never a trainable target derived from the detector being optimized.
    """
    ids = batch["batch_idx"].long()
    order = ids.argsort(stable=True)
    ids = ids[order]
    counts = torch.bincount(ids, minlength=len(enabled))
    maximum = max(int(counts.max()), 1)
    slot = torch.arange(len(ids), device=ids.device) - (counts.cumsum(0)-counts)[ids]
    points = batch["keypoints"][order, 0].detach()
    classes = batch["cls"][order, 0].long().detach()
    xy = points.new_zeros(len(enabled), maximum, 2)
    types = classes.new_zeros(len(enabled), maximum)
    mask = torch.zeros(len(enabled), maximum, device=ids.device, dtype=torch.bool)
    xy[ids, slot] = points[:, :2]
    types[ids, slot] = classes
    mask[ids, slot] = ((points[:, 2] > 0) & torch.isfinite(points).all(-1)
        & (points[:, :2] >= 0).all(-1) & (points[:, :2] <= 1).all(-1)
        & (classes >= 0) & (classes < 9) & enabled[ids])
    # Avoid NaNs even in a masked, malformed observation.
    return torch.where(mask[..., None], xy.nan_to_num(nan=0., posinf=0., neginf=0.), 0.), types, mask


class StagedPoseLoss(nn.Module):
    def __init__(self, model, symmetries, *, geometry_weight=1., pose_aux_weight=.25):
        super().__init__()
        self.geometry_weight = geometry_weight
        self.pose_aux_weight = pose_aux_weight
        self.reference = PhysicalReference()
        self.register_buffer(
            "rotations",
            torch.tensor(
                [v["candidate_to_reference"] for v in symmetries["candidates"]], dtype=torch.float32
            ),
        )
        self.register_buffer(
            "faces",
            torch.tensor(
                [v["reference_to_candidate_center_index"] for v in symmetries["candidates"]],
                dtype=torch.long,
            ),
        )
        # Output candidate-face -> reference-face, as in the verified loss.
        self.faces.copy_(self.faces.argsort(-1))
        rest = self.reference.rest
        owner = self.reference.owner
        permutations = []
        for Q in self.rotations:
            distance = ((rest @ Q)[:, None] - rest[None]).square().sum(-1) + (
                (owner @ Q)[:, None] - owner[None]
            ).square().sum(-1)
            distance.masked_fill_(self.reference.types[:, None] != self.reference.types[None], 1e6)
            error, permutation = distance.min(-1)
            if float(error.max()) > 1e-6:
                raise ValueError("Invalid physical symmetry orbit")
            permutations.append(permutation.argsort(-1))
        self.register_buffer("inverse_slots", torch.stack(permutations))
        self.independent = independent_pose_criterion(model.detector)

    def predicted_geometry_loss(self, base, logits, batch, enabled, image_size):
        """Score complete RGB predictions; target pose/axis never enters here."""
        angles, probability = decode_hypotheses(logits)
        xy, classes, mask = visible_observations(batch, enabled)
        pose = {k:v.repeat_interleave(3, 0) for k,v in base.items()}
        pose["angles"] = angles.flatten(0, 1)
        family = torch.arange(3, device=logits.device).repeat(len(logits))
        value, rmse, _ = self.reference.template.geometry(
            pose, xy.repeat_interleave(3, 0), classes.repeat_interleave(3, 0),
            mask.repeat_interleave(3, 0), family, beta=2./image_size)
        value = value.reshape(-1, 3) * image_size / 4
        selected = probability.argmax(-1)
        row = torch.arange(len(logits), device=logits.device)
        # Half directly scores the deployed hard choice. Half scores the
        # probability-weighted alternatives, giving axis selection a geometric
        # gradient. Each term scores a complete shape, never a blended shape.
        per_image = .5 * (value[row, selected] + (value * probability).sum(-1))
        active = mask.any(-1)
        loss = per_image[active].mean() if active.any() else logits.sum()*0 + value.sum()*0
        return loss, rmse.reshape(-1, 3)[row, selected] * image_size, active

    def gauge(self, base, target):
        truth = target["rotation"][:, None] @ self.rotations[None]
        qi = (base["rotation_matrix"].detach()[:, None] * truth).sum((-1, -2)).argmax(-1)
        face = self.faces[qi]
        pose = {
            k: target["rotation" if k == "rotation_matrix" else k]
            for k in ["rotation_matrix", "center", "scale", "perspective", "principal", "angles"]
        }
        pose = dict(
            pose,
            rotation_matrix=target["rotation"] @ self.rotations[qi],
            angles=target["angles"].gather(1, face),
        )
        return pose, target["eligible"].gather(1, face), qi

    @staticmethod
    def visible_loss(projected, xy, mask, enabled, image_size):
        known = mask & enabled[:, None]
        active = known.any(-1)
        distance = (projected - xy).norm(dim=-1) * image_size
        value = (
            F.smooth_l1_loss(distance, torch.zeros_like(distance), beta=2.0, reduction="none") / 4
        )
        per_image = (value * known).sum(-1) / known.sum(-1).clamp_min(1)
        return per_image[active].mean() if active.any() else projected.sum() * 0

    def reframe_targets(self, qi, target):
        slot = self.inverse_slots[qi]
        return target["visible_xy"].gather(1, slot[..., None].expand(-1, -1, 2)), target[
            "visible_mask"
        ].gather(1, slot)

    def joint_visible_loss(self, pose, truth, xy, mask, enabled, image_size, metric=False):
        family = truth["angles"].abs().reshape(-1, 3, 2).sum(-1).argmax(-1)
        axis = self.reference.template.axis[family]
        face = family[:, None] * 2 + torch.arange(2, device=family.device)[None]
        actual = truth["angles"].gather(1, face)
        current = pose["angles"].detach().gather(1, face)

        def slab_frames(R, angle):
            theta = torch.stack((angle[:, 1], torch.zeros_like(angle[:, 0]), -angle[:, 0]), -1)
            spin = rotation_exp(
                (axis[:, None] * theta[..., None] * (math.pi / 180)).flatten(0, 1)
            ).reshape(-1, 3, 3, 3)
            return R[:, None] @ spin

        with torch.no_grad():
            quarter = rotation_exp(
                (
                    axis[:, None]
                    * (torch.arange(4, device=axis.device)[None, :, None] * math.pi / 2)
                ).flatten(0, 1)
            ).reshape(-1, 4, 3, 3)
            gt = slab_frames(truth["rotation_matrix"], actual)
            pred = slab_frames(pose["rotation_matrix"].detach(), current)
            selected = (
                (pred[:, :, None] * (gt[:, :, None] @ quarter[:, None])).sum((-1, -2)).argmax(-1)
            )
            row = torch.arange(len(family), device=family.device)
            quarter = quarter[row[:, None], selected]
            layer = (self.reference.owner[None] * axis[:, None]).sum(-1).round().long() + 1
            spin = quarter[row[:, None], layer]
            rest = torch.einsum("ni,bnij->bnj", self.reference.rest, spin)
            owner = torch.einsum("ni,bnij->bnj", self.reference.owner, spin)
            distance = (rest[:, :, None] - self.reference.rest[None, None]).square().sum(-1) + (
                owner[:, :, None] - self.reference.owner[None, None]
            ).square().sum(-1)
            distance.masked_fill_(
                self.reference.types[None, :, None] != self.reference.types[None, None], 1e6
            )
            index = distance.argmin(-1)
        prediction_family = (
            pose["angles"].abs().reshape(-1, 3, 2).sum(-1).argmax(-1) if metric else family
        )
        projected = self.reference.project(pose, prediction_family).gather(
            1, index[..., None].expand(-1, -1, 2)
        )
        if metric:
            return (projected - xy).norm(dim=-1) * image_size
        return self.visible_loss(projected, xy, mask, enabled, image_size)

    def forward(self, model, batch, target, phase):
        context = model.encode(batch["img"], getattr(model, "use_amp", False))
        base = context["static"]
        context["training_phase"] = phase
        autonomous = phase in ("handoff", "joint")
        # Complete the inference path before inspecting any target pose.
        actual_logits = model.articulate(context)[0] if autonomous else None
        if autonomous:
            context["actual_logits"] = actual_logits
        enabled = target["enabled"]
        size = batch["img"].shape[-1]
        truth, eligible, qi = self.gauge(base, target)
        xy, mask = self.reframe_targets(qi, target)
        zero = context["raw_static"].sum() * 0
        static = geometry = yolo = phase_loss = replay = zero
        if phase in ("static", "joint"):
            expected = model.fields.encode_rotation(target["rotation"])
            field = (context["raw_static"][:, :9] - expected).square().mean(-1) / (
                model.fields.orientation_norm.square() / 9
            )
            static = field[enabled].mean() if enabled.any() else zero
            if phase == "static":
                camera_pose = dict(
                    truth, **{k: base[k] for k in ["center", "scale", "perspective", "principal"]}
                )
                family = truth["angles"].abs().reshape(-1, 3, 2).sum(-1).argmax(-1)
                geometry = self.visible_loss(
                    self.reference.project(camera_pose, family), xy, mask, enabled, size
                )
            with torch.autocast(
                device_type=batch["img"].device.type, enabled=getattr(model, "use_amp", False)
            ):
                raw = context["encoded"]["raw"]
                # Match the staged45 detector's configured one-to-many path;
                # the unused one-to-one branch stays frozen. This is still the
                # installed default Pose26 loss, not a custom YOLO objective.
                components, _ = self.independent(raw.get("one2many", raw), batch)
                yolo = components.sum() / len(enabled)
        logits = None
        if phase != "static":
            conditioning = base
            if model.training and phase == "articulation":
                # Geometry augmentation affects only loss-time conditioning;
                # actual validation ALWAYS uses the RGB-predicted static pose.
                conditioning = {k: v.detach().clone() for k, v in truth.items()}
                probability = torch.rand(len(enabled), device=enabled.device)
                jitter = probability >= 0.5
                strength = 0.5 + torch.rand(len(enabled), device=enabled.device)
                spin = rotation_exp(
                    torch.randn(len(enabled), 3, device=enabled.device) * (0.12 * strength[:, None])
                )
                conditioning["rotation_matrix"] = torch.where(
                    jitter[:, None, None],
                    spin @ conditioning["rotation_matrix"],
                    conditioning["rotation_matrix"],
                )
                for name, magnitude in [("center", 0.10), ("principal", 0.10)]:
                    conditioning[name] += (
                        jitter[:, None]
                        * conditioning["scale"][:, None]
                        * torch.randn_like(conditioning[name])
                        * (magnitude * strength[:, None])
                    )
                conditioning["scale"] *= torch.where(
                    jitter, (torch.randn_like(strength) * 0.10 * strength).exp(), 1.0
                )
                p = (
                    conditioning["perspective"].clamp(0.0001, 0.9999).logit()
                    + jitter * torch.randn_like(strength) * 0.30 * strength
                ).sigmoid()
                conditioning["perspective"] = p
            logits = actual_logits if autonomous else model.articulate(context, conditioning)[0]
            phase_loss = loss_function(logits, truth["angles"], eligible, enabled)
            if autonomous:
                geometry, fit_rmse, fit_active = self.predicted_geometry_loss(
                    base, logits, batch, enabled, size)
                context["observed_fit_rmse_px"] = fit_rmse.detach()
                context["observed_fit_active"] = fit_active
        total = (self.pose_aux_weight * (static + phase_loss) + self.geometry_weight * geometry + .1*yolo
                 if autonomous else static + geometry + .1*yolo + phase_loss)
        return (
            total,
            {
                "static": static,
                "geometry": geometry,
                "yolo": yolo,
                "phase": phase_loss,
                "replay": replay,
            },
            context,
        )

    @torch.no_grad()
    def metrics(self, model, context, target, image_size=128):
        logits = context.get("actual_logits")
        if logits is None:
            logits, _ = model.articulate(context)
        pose = dict(context["static"], angles=decode(logits)[0])
        truth, eligible, qi = self.gauge(pose, target)
        enabled = target["enabled"]
        angles = truth["angles"]
        error = (pose["angles"] - angles + 45).remainder(90).sub(45).abs()
        cosine = ((pose["rotation_matrix"] * truth["rotation_matrix"]).sum((-1, -2)) - 1) / 2
        rotation = cosine.clamp(-1, 1).acos() * 180 / math.pi
        result = {}

        def add(name, value, mask):
            result[name] = (float(value[mask].sum()), int(mask.sum()))

        add("static_rotation_deg", rotation, enabled)
        moving = (angles.abs() > 1e-4) & enabled[:, None]
        add("moving_turn_deg", error, moving)
        add("observed_moving_turn_deg", error, moving & eligible)
        add("stationary_turn_deg", error, (angles.abs() <= 1e-4) & enabled[:, None])
        add("moving_within5", (error <= 5).float(), moving)
        xy, mask = self.reframe_targets(qi, target)
        known = mask & enabled[:, None]
        distance = self.joint_visible_loss(pose, truth, xy, mask, enabled, image_size, metric=True)
        rmse = ((distance.square() * known).sum(-1) / known.sum(-1).clamp_min(1)).sqrt()
        add("visible_keypoint_rmse_px", rmse, known.any(-1))
        if "observed_fit_rmse_px" in context:
            add("observed_fit_rmse_px", context["observed_fit_rmse_px"], context["observed_fit_active"])
        # The early known-static phase retains its explicit teacher diagnostic.
        # Handoff/joint NEVER run an additional GT-conditioned forward pass.
        if context.get("training_phase", getattr(model, "phase", None)) not in ("handoff", "joint"):
            teacher, _ = model.articulate(context, truth)
            teacher_error = (decode(teacher)[0] - angles + 45).remainder(90).sub(45).abs()
            add("teacher_observed_moving_turn_deg", teacher_error, moving & eligible)
        else:
            result["teacher_observed_moving_turn_deg"] = (0., 0)
        return result
