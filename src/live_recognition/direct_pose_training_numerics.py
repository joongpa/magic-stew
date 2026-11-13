"""Precision boundaries and diagnostics for the direct-pose training loop."""

from __future__ import annotations

import torch


def floating_to_float32(value):
    """Cast raw predictions without detaching their backward graph."""
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: floating_to_float32(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(floating_to_float32(item) for item in value)
    return value


def floating_tensors(value, path=""):
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from floating_tensors(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from floating_tensors(item, f"{path}[{index}]")


def nonfinite_fields(value):
    fields = list(floating_tensors(value))
    if not fields:
        return []
    # One device synchronization, rather than one per prediction tensor.
    finite = torch.stack([torch.isfinite(tensor).all() for _, tensor in fields]).detach().cpu().tolist()
    return [name for (name, _), valid in zip(fields, finite) if not valid]


def compute_loss_attempt(model, batch, target, independent_loss, direct_loss, device,
                         independent_weight=1., amp_enabled=False, amp_dtype=None):
    with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
        prediction = model(batch["img"])
    # Network convolutions can use AMP. Assignment, RLE/log probabilities,
    # reductions and the geometry losses must not inherit that FP16 context.
    with torch.autocast(device_type=device.type, enabled=False):
        raw = independent_loss.raw_predictions(prediction["independent"])
        outputs = {"independent": floating_to_float32(raw), "pose": floating_to_float32(prediction["pose"])}
        invalid = nonfinite_fields(outputs)
        if invalid:
            return {"outputs": outputs, "nonfinite": invalid}
        components, items = independent_loss(outputs["independent"], batch)
        independent = components.sum() / len(batch["img"])
        losses, metrics, _ = direct_loss(outputs["pose"], target, batch["img"].shape[-1])
        total = losses["total"] + independent_weight * independent
        result = {"outputs": outputs, "independent": independent, "independent_components": components,
                  "independent_items": items, "pose_losses": losses, "metrics": metrics, "total": total}
        result["nonfinite"] = nonfinite_fields({"yolo": components, "pose": losses, "total": total})
        return result


def batchnorm_snapshot(model):
    result = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training and module.track_running_stats:
            result.append((module, {key: value.detach().clone() for key in (
                "running_mean", "running_var", "num_batches_tracked") if (value := getattr(module, key)) is not None}))
    return result


def restore_batchnorm(snapshot):
    with torch.no_grad():
        for module, values in snapshot:
            for key, value in values.items():
                getattr(module, key).copy_(value)


def attempt_diagnostics(attempt):
    def safe_number(value):
        number = float(value.detach())
        return number if torch.isfinite(value) else str(number)

    result = {"nonfinite_fields": attempt["nonfinite"], "outputs": {}}
    for name, value in floating_tensors(attempt["outputs"]):
        detached = value.detach()
        finite = torch.isfinite(detached)
        valid = detached[finite]
        result["outputs"][name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                                   "nonfinite_values": int((~finite).sum()),
                                   "finite_min": float(valid.min()) if valid.numel() else None,
                                   "finite_max": float(valid.max()) if valid.numel() else None}
    if "independent_components" in attempt:
        names = ("box", "keypoint_xy", "keypoint_presence", "class", "dfl", "rle")
        result["yolo_components"] = {name: safe_number(value) for name, value in zip(names, attempt["independent_components"])}
        result["pose_components"] = {name: safe_number(value) for name, value in attempt["pose_losses"].items()}
    return result
