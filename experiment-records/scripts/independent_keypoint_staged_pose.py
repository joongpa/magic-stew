"""Bridge the learned staged pose to the independent renderer's camera and slabs.

The network sees the same square RGB crop as training. Camera conversion is
analytic; this module does not run PnP, angle sweeps, or numerical refinement.
"""
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np


DEFAULT_STAGED_MODEL = Path(__file__).resolve().parents[1] / "runs/staged_cube_pose/weights/best.pt"
# Training face order: U, D, F, B, R, L. Both models use X=R, Y=B, Z=U.
FACE_AXIS_SIDES = ((2, 1), (2, -1), (1, -1), (1, 1), (0, 1), (0, -1))


def add_pose_method_arguments(parser):
    parser.add_argument("--pose-method", choices=("manual", "staged"), default="manual",
                        help="Pose source: existing geometric fitting or the trained staged network.")
    parser.add_argument("--staged-model", type=Path, default=DEFAULT_STAGED_MODEL,
                        help="Staged training checkpoint, used with --pose-method staged.")


def pose_checkpoint_path(args):
    return Path(args.staged_model if getattr(args, "pose_method", "manual") == "staged" else args.model)


@dataclass
class StagedPrediction:
    pose: dict
    local_points: np.ndarray
    presence_scores: np.ndarray
    class_ids: np.ndarray
    object_scores: np.ndarray


class StagedPosePredictor:
    def __init__(self, checkpoint, device="cpu"):
        import torch
        from live_recognition.staged_pose.model import model_from_checkpoint

        self.checkpoint = Path(checkpoint)
        device = str(device or "cpu")
        self.device = torch.device(f"cuda:{device}" if device.isdigit() else device)
        self.model, saved = model_from_checkpoint(self.checkpoint, device=self.device)
        self.image_size = int(saved.get("train_args", {}).get("imgsz", 128))
        del saved  # Training optimizer state is not retained by the renderer.
        self.model.eval().requires_grad_(False)

    def predict_crop(self, crop_bgr, args):
        import torch
        from live_recognition.staged_pose.phase import decode

        # Match StagedPoseDataset preprocessing, independently of overlay size.
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        images = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None]
        images = images.to(device=self.device, dtype=torch.float32) / 255
        with torch.inference_mode():
            context = self.model.encode(images)
            logits, _ = self.model.articulate(context)
            pose = dict(context["static"], angles=decode(logits)[0])
            encoded = context["encoded"]
            # These are the actual NMS-selected per-object/class probabilities.
            # Keep all available independent objects, not just the 64 head tokens.
            scores, classes = encoded["object_scores"][0].max(0)
            floor = args.pose_conf if args.pose_conf is not None else min(args.pose_class_conf.values())
            index = (scores > 0).logical_and(scores >= floor).nonzero().flatten()
            index = index[scores[index].argsort(descending=True)[:args.max_det]]
            branch = encoded["raw"].get("one2many", encoded["raw"])
            # Pose26 kpts_decode uses sigmoid on channel 2 of this one-keypoint
            # object. This is presence, separate from the detection/class score.
            presence = branch["kpts"][0, 2].sigmoid()[index]
            return StagedPrediction(
                pose={k:v[0].cpu().numpy().copy() for k,v in pose.items()},
                local_points=encoded["points"][0, index].cpu().numpy() * args.view_size,
                presence_scores=presence.cpu().numpy(),
                class_ids=classes[index].cpu().numpy().astype(np.int32),
                object_scores=scores[index].cpu().numpy(),
            )


def candidate_from_prediction(renderer, prediction, roi, image_size, args):
    """Preserve the network's normalized pinhole projection in full-frame pixels."""
    p = prediction.pose
    if not all(np.isfinite(value).all() for value in p.values()):
        return None
    scale, perspective = float(p["scale"]), float(p["perspective"])
    if scale <= 0 or perspective <= 0:
        return None
    x1, y1, x2, y2 = roi
    size = np.asarray((x2-x1, y2-y1), dtype=float)
    principal = np.asarray(p["principal"], dtype=float) * size + (x1, y1)
    focal = 3 * scale / perspective * size
    camera = np.array([[focal[0], 0, principal[0]], [0, focal[1], principal[1]], [0, 0, 1.]])
    translation = np.r_[(p["center"]-p["principal"])/scale, 3/perspective]
    rotations = tuple(renderer.CubeLayerRotation(axis, side, -float(angle))
        for (axis, side), angle in zip(FACE_AXIS_SIDES, p["angles"], strict=True) if angle != 0)
    points, _ = renderer.local_to_frame_points(prediction.local_points, roi, args.view_size)
    usable = renderer.usable_pose_points(prediction.local_points, prediction.presence_scores,
        prediction.class_ids, prediction.object_scores, args.keypoint_conf, args.pose_class_conf)
    landmarks = usable & (prediction.class_ids < 3) & (prediction.class_ids >= 0)
    stickers = usable & (prediction.class_ids >= 3)
    result = renderer.StickerPnPCandidate(
        sticker_type="staged", sticker_average_confidence=(float(prediction.object_scores[stickers].mean()) if stickers.any() else 0.),
        image_points=np.empty((0, 2)), object_points=np.empty((0, 3)), projected_sticker_points=np.empty((0, 2)),
        camera_matrix=camera, dist_coeffs=np.zeros((5, 1)),
        rvec=cv2.Rodrigues(np.asarray(p["rotation_matrix"], dtype=float))[0],
        tvec=translation.reshape(3, 1)*renderer.CUBE_LAYER_SIZE_MM, reprojection_error=float("inf"),
        image_size=image_size, layer_rotations=rotations, sticker_layer_rotations=rotations,
        assignment_image_points=points[landmarks], assignment_observed_class_ids=prediction.class_ids[landmarks],
        assignment_fixed_mask=np.zeros(int(landmarks.sum()), dtype=bool), turn_state_moves=(),
        joint_pose_diagnostics=dict(pose_method="staged", geometric_refinement=False))
    return renderer.attach_color_keypoints(result, points[stickers],
        tuple(renderer.CLASS_NAMES[int(i)] for i in prediction.class_ids[stickers]))


def cache_predicted_geometry(renderer, candidate):
    """Score/cache only the predicted shape for overlays and optional state search."""
    if candidate is None:
        return None
    rotations = candidate.layer_rotations
    axis = rotations[0].axis if rotations else 0
    angles = renderer.parallel_layer_angles(rotations)
    limit = renderer.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * renderer.projected_cube_diameter(candidate)
    options, nearest, valid = renderer.parallel_turn_frame_geometry(candidate, limit, axis, angles=angles)
    if not options:
        return candidate
    option = options[0]
    # The predicted rvec, tvec and layer rotations are preserved exactly.
    option["layer_rotations"] = rotations
    payload = renderer.turn_trial_log_payload(option)
    payload.update(joint_pose_fit=True, pose_proposal_source="staged_network",
        pose_rvec=candidate.rvec.tolist(), pose_tvec=candidate.tvec.tolist(),
        cached_sticker_indices=nearest[0].tolist(), cached_sticker_valid=valid[0].tolist(),
        assignment_projected_points=option["assignment_projected_points"].tolist(),
        assignment_object_points=option["assignment_object_points"].tolist(),
        assignment_projected_class_ids=option["assignment_projected_class_ids"].tolist())
    return replace(candidate, assignment_projected_points=option["assignment_projected_points"],
        assignment_object_points=option["assignment_object_points"],
        assignment_projected_class_ids=option["assignment_projected_class_ids"],
        overall_reprojection_error=float(np.linalg.norm(
            candidate.assignment_image_points-option["assignment_projected_points"], axis=-1).mean()),
        joint_geometry_candidates=(payload,), turn_fit_search_rvec=candidate.rvec.copy(),
        turn_fit_search_tvec=candidate.tvec.copy(),
        turn_search_diagnostics=dict(status="staged_pose", geometry=dict(joint_refined=False,
            scope="network_prediction", options=[payload])))
