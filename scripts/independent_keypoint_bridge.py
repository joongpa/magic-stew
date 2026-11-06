"""State-independent frame quality and bounded discrete motion through weak frames."""
import math
from dataclasses import replace


def reprojection_options(m, record):
    """Recover the displayed fit for old caches without rerunning inference.

    New caches record the metric with every geometric option. Older caches
    retain the camera, landmark observations and anchor projections, so only
    the state-independent geometric projection pass needs repeating.
    """
    options = ((record.get('search') or {}).get('geometry') or {}).get('options') or []
    if all('reprojection_fit_error' in o for o in options):
        return options
    import cv2
    import numpy as np

    pose = record.get('search_pose') or {}
    if pose.get('rvec') is None:
        pose = record.get('selected_pose') or {}
    if pose.get('rvec') is None or not pose.get('assignment_image_points'):
        return []
    observed = np.asarray(pose['assignment_image_points'], dtype=float).reshape(-1, 2)
    classes = np.asarray(pose.get('assignment_observed_class_ids') or [-1]*len(observed))
    candidate = m.StickerPnPCandidate(sticker_type=pose['sticker_type'], sticker_average_confidence=1.,
        image_points=np.empty((0, 2)), object_points=np.empty((0, 3)),
        projected_sticker_points=np.empty((0, 2)), reprojection_error=float('inf'),
        rvec=np.asarray(pose['rvec']), tvec=np.asarray(pose['tvec']),
        camera_matrix=np.asarray(pose['camera_matrix']), dist_coeffs=np.asarray(pose['dist_coeffs']),
        image_size=tuple(pose['image_size']), assignment_image_points=observed,
        assignment_observed_class_ids=classes)
    objects, projected_classes = m.visible_cube_lattice_points(candidate.rvec, candidate.tvec)
    projected, _ = cv2.projectPoints(objects, candidate.rvec, candidate.tvec,
                                    candidate.camera_matrix, candidate.dist_coeffs)
    projected = projected.reshape(-1, 2)
    selected = record.get('selected_pose') or {}
    anchor_projections = selected.get('assignment_projected_points')
    if (anchor_projections is not None and np.shape(anchor_projections) == observed.shape
            and np.allclose(selected.get('rvec'), pose['rvec'])
            and np.allclose(selected.get('tvec'), pose['tvec'])):
        # Preserve the anchor correspondences used when preparing this cache.
        targets = np.asarray(anchor_projections)
    else:
        targets = observed
    _, rows, columns = m.observed_keypoint_reprojection_nearest_assignment(
        targets, projected, candidate.image_size,
        observed_class_ids=classes, projected_class_ids=projected_classes)
    if len(rows) != len(observed):
        return []
    fixed = np.asarray(pose.get('assignment_fixed_mask') or [False]*len(observed), dtype=bool)
    candidate = replace(candidate, assignment_object_points=objects[columns],
        assignment_projected_points=projected[columns], assignment_fixed_mask=fixed,
        assignment_projected_class_ids=projected_classes[columns],
        image_points=observed[fixed], object_points=objects[columns][fixed],
        projected_sticker_points=projected[columns][fixed])
    angles = sorted({float(rotation['angle_degrees']) for o in options
                     for rotation in o.get('layer_rotations', ())})
    trials = m._score_turn_configurations_on_gpu(candidate,
        m.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION*m.projected_cube_diameter(candidate), 'cpu',
        angle_values=np.asarray(angles), enforce_layer_observation_gate=False, enforce_validity_gate=False)
    def key(o):
        rotations = o.get('layer_rotations') or ()
        angle = (rotations[0]['angle_degrees'] if isinstance(rotations[0], dict)
                 else rotations[0].angle_degrees) if rotations else 0.
        return o['layer_axis'], o['layer_side'], o['layer_count'], angle
    fits = {key(t): m.pose_reprojection_fit_metric(replace(candidate,
            assignment_projected_points=t['assignment_projected_points'])) for t in trials or ()}
    return [dict(o, reprojection_fit_error=fits.get(key(o), float('inf'))) for o in options]


class FrameObservationBridge:
    def __init__(self, scorer, frames, *, fps, advance_update, canonical_update,
                 transition_type, max_frames=2, max_seconds=.1,
                 max_reprojection=.1, min_inlier_fraction=.5):
        self.scorer, self.frames = scorer, frames
        self.fps, self.max_frames, self.max_seconds = fps, max_frames, max_seconds
        self.advance_update, self.canonical_update = advance_update, canonical_update
        self.transition_type = transition_type
        self.stats = dict(bridge_batches=0, bridged_frame_visits=0,
                          bridge_generated_states=0, bridge_merged_states=0, bridge_peak_contexts=0)
        self.quality = {}
        m = scorer.m
        for frame in frames:
            reason = scorer.hold_reason(frame)
            record = scorer.records[frame]
            pose = record.get('search_pose') or {}
            if pose.get('rvec') is None:
                pose = record.get('selected_pose') or {}
            observations = pose.get('assignment_image_points')
            count = len(observations) if observations is not None else None
            options = reprojection_options(m, record) if reason is None else []
            costs, supports, joint = [], [], []
            for option in options:
                fit = option.get('reprojection_fit_error')
                if fit is None:
                    continue
                cost = float(fit)
                if not math.isfinite(cost):
                    continue
                support = (option.get('keypoint_inlier_count', 0)/count
                           if count and 'keypoint_inlier_count' in option else None)
                costs.append(cost)
                if support is not None:
                    supports.append(support)
                if support is None or support >= min_inlier_fraction:
                    joint.append(cost)
            reliable = (reason == 'turning_start_frame' or reason is None and bool(joint)
                        and min(joint) <= max_reprojection and count != 0)
            self.quality[frame] = dict(reliable=reliable, best_reprojection_fit=min(costs) if costs else None,
                best_supported_reprojection_fit=min(joint) if joint else None,
                max_keypoint_inlier_fraction=max(supports) if supports else None,
                observed_landmarks=count, reprojection_limit=max_reprojection,
                reprojection_metric='median + 0.5*p90 landmark residual / projected cube diameter (display REPROJ fit)',
                minimum_inlier_fraction=min_inlier_fraction, hold_reason=reason)
        self.options = [dict(state_moves=(), layer_rotations=[])]
        for axis in range(3):
            for side in (-1, 1):
                for width in (1, 2):
                    for angle in (-90., 90.):
                        move = m.cube_rotation_to_notation(m.CubeLayerRotation(axis, side, angle, width))
                        self.options.append(dict(state_moves=(move,), layer_axis=axis,
                            layer_side=side, layer_count=width, color_offset=angle, layer_rotations=[]))

    def endpoint(self, depth):
        """Stop at a reliable frame or a bounded unobserved work batch.

        A batch cap never makes weak observations usable again. The caller
        queues unfinished batches using the last reliable frame's priority.
        Always advance at least one frame, even below one frame's time budget.
        """
        end = depth
        while end < len(self.frames) and not self.quality[self.frames[end]]['reliable']:
            count = end-depth+1
            if end > depth and (count > self.max_frames or count/self.fps > self.max_seconds):
                break
            end += 1
        return end if depth < end else None

    def expand(self, depth, state, update, gap, state_key, heartbeat):
        end = self.endpoint(depth)
        if end is None:
            return None
        self.stats['bridge_batches'] += 1
        self.stats['bridged_frame_visits'] += end-depth
        front = {(state_key(state), self.canonical_update(update, self.frames[depth], gap)): (state, update, ())}
        for index in range(depth, end):
            frame = self.frames[index]
            following = {}
            for state, update, prefix in front.values():
                for option in self.options:
                    heartbeat()
                    choice = self.transition_type(state, 0., (), option,
                                                  self.scorer.m.state_update_layer_direction(option))
                    changed = self.advance_update(update, choice, frame, gap)
                    if changed is None:
                        continue
                    self.stats['bridge_generated_states'] += 1
                    apply = getattr(self.scorer, 'apply_moves', None)
                    after = apply(state, option['state_moves']) if apply else state.apply_moves(option['state_moves'])
                    next_frame = self.frames[index+1] if index+1 < len(self.frames) else None
                    key = (state_key(after), self.canonical_update(changed, next_frame, gap))
                    if key in following:
                        self.stats['bridge_merged_states'] += 1
                        continue
                    step = dict(option, frame_index=frame, state_score=None, queue_priority=None,
                        observation_status='unreliable_frame_bridge', bridge_inferred=True,
                        geometry_score_used=False, color_score_used=False, frame_quality=self.quality[frame])
                    following[key] = (after, changed, prefix+(step,))
            front = following
            self.stats['bridge_peak_contexts'] = max(self.stats['bridge_peak_contexts'], len(front))
        return end, tuple(front.values())
