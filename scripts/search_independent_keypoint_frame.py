#!/usr/bin/env python3
"""Full-solve search with soft local ranks and preceding-frame layer activity.

The default queue uses the current 50/50 geometric + weighted color fit.
Color confusions affect scores; color errors do not hard-prune by default.
Sticker-history ordering and soft move windows remain optional experiments.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from functools import lru_cache
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "frame_search_storage", ROOT / "scripts/search_independent_keypoint_mean.py")
storage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = storage
SPEC.loader.exec_module(storage)
replay = storage.replay
BRIDGE_SPEC = importlib.util.spec_from_file_location(
    "frame_observation_bridge", ROOT / "scripts/independent_keypoint_bridge.py")
bridge_module = importlib.util.module_from_spec(BRIDGE_SPEC)
sys.modules[BRIDGE_SPEC.name] = bridge_module
BRIDGE_SPEC.loader.exec_module(bridge_module)
FrameObservationBridge = bridge_module.FrameObservationBridge
HISTORY_SPEC = importlib.util.spec_from_file_location(
    "frame_sticker_history", ROOT / "scripts/independent_keypoint_sticker_history.py")
history = importlib.util.module_from_spec(HISTORY_SPEC)
sys.modules[HISTORY_SPEC.name] = history
HISTORY_SPEC.loader.exec_module(history)
WINDOW_SPEC = importlib.util.spec_from_file_location(
    'frame_soft_window', ROOT / 'scripts/independent_keypoint_soft_window.py')
window_module = importlib.util.module_from_spec(WINDOW_SPEC)
sys.modules[WINDOW_SPEC.name] = window_module
WINDOW_SPEC.loader.exec_module(window_module)
SoftMoveWindow = window_module.SoftMoveWindow
ACTIVITY_SPEC = importlib.util.spec_from_file_location(
    'frame_layer_activity', ROOT / 'scripts/independent_keypoint_activity.py')
activity = importlib.util.module_from_spec(ACTIVITY_SPEC)
sys.modules[ACTIVITY_SPEC.name] = activity
ACTIVITY_SPEC.loader.exec_module(activity)


def canonical_update(update, next_frame, gap):
    last, _key, _streak = update
    if gap <= 1 or last is None or next_frame is None or next_frame - last >= gap:
        return None, None, 0
    return update


def advance_update(update, choice, frame, gap):
    if not choice.payload.get("state_moves"):
        return update
    last, key, streak = update
    consecutive = last is not None and frame - last == 1
    continuation = (consecutive and streak < 2 and choice.update_key is not None
                    and choice.update_key == key)
    if last is not None and frame - last < gap and not continuation:
        return None
    return frame, choice.update_key, streak + 1 if consecutive and choice.update_key == key else 1


def rotation_invariant_state_key(scorer):
    """Ignore whole-cube rotations only for measuring supported move progress."""
    np = scorer.np
    sources = getattr(scorer, 'orientation_sources', None)
    if sources is None:
        indices = {position:i for i,position in enumerate(scorer.positions)}
        sources = []
        for symmetry in scorer.m.CUBE_SYMMETRY_MATRICES:
            mapping = dict(scorer.m._geometry_symmetry_permutation(tuple(int(v) for v in symmetry.ravel())))
            sources.append([indices[mapping[position]] for position in scorer.positions])
        sources = np.asarray(sources)

    @lru_cache(maxsize=16384)
    def canonical(encoded):
        return min(row.tobytes() for row in np.frombuffer(encoded, dtype=np.uint8)[sources])
    return lambda state: canonical(scorer.key(state))


def frame_full_solve(initial_state, frames, transitions, solved, state_key, *,
                     min_update_gap=2, queue=None, serialize_state=pickle.dumps,
                     deserialize_state=pickle.loads, progress=None, progress_seconds=10.,
                     minimum_color_accuracy=None, bridge=None, sticker_history=False,
                     sticker_history_weight=.1, merge_histories=False, soft_window=None,
                     color_confusion_pruning=False, soft_top_k=0, outside_top_k_penalty=.25,
                     require_layer_activity=False):
    """Expand the global minimum, testing goals on pop.

    With sticker_history, rank by fit + weight * history penalty; weight=None
    restores lexicographic mismatch-count ordering. merge_histories keeps one
    complete history per physical context, minimizing fit + history penalty.
    Soft windows deprioritize new old forks without deleting any alternatives.
    soft_top_k adds a noncumulative queue penalty by physical-successor rank.
    Layer activity is evidence from frame-1; evaluators must preserve distinct
    hold activity fits when this changes the legal next moves.
    """
    if (not frames or any(b != a + 1 for a, b in zip(frames, frames[1:]))
            or min_update_gap < 1 or not math.isfinite(progress_seconds) or progress_seconds <= 0
            or soft_top_k < 0 or not isinstance(soft_top_k, int)
            or not math.isfinite(outside_top_k_penalty) or outside_top_k_penalty < 0
            or minimum_color_accuracy is not None and
                (not math.isfinite(minimum_color_accuracy) or not 0. <= minimum_color_accuracy <= 1.)
            or sticker_history_weight is not None and
                (not math.isfinite(sticker_history_weight) or sticker_history_weight < 0)
            or (merge_histories or soft_window is not None) and
                (not sticker_history or sticker_history_weight is None)):
        raise ValueError("Require consecutive frames, a positive move gap and progress interval")
    if queue is None:
        queue = storage.BufferedMeanQueue()
        try:
            return frame_full_solve(initial_state, frames, transitions, solved, state_key,
                min_update_gap=min_update_gap, queue=queue, serialize_state=serialize_state,
                deserialize_state=deserialize_state, progress=progress, progress_seconds=progress_seconds,
                minimum_color_accuracy=minimum_color_accuracy, bridge=bridge, sticker_history=sticker_history,
                sticker_history_weight=sticker_history_weight,
                merge_histories=merge_histories, soft_window=soft_window,
                color_confusion_pruning=color_confusion_pruning, soft_top_k=soft_top_k,
                outside_top_k_penalty=outside_top_k_penalty, require_layer_activity=require_layer_activity)
        finally:
            queue.close()
    if queue.context_count or queue.deferred_count:
        raise ValueError("Start in a fresh search database")
    stats = dict(evaluated_states=0, attempted_transitions=0, invalid_pruned=0, color_pruned=0,
        spacing_pruned=0, dominated_pruned=0, reopened_states=0, improved_pending=0,
        unsolved_endpoints=0, furthest_frames=0, furthest_frame_index=None,
        current_frames=0, current_frame_index=None, current_frame_score=None,
        peak_pending=0, error_gate_enabled=False, minimum_color_accuracy=minimum_color_accuracy,
        sticker_history_order=sticker_history, current_sticker_mismatch_count=None,
        sticker_history_weight=sticker_history_weight if sticker_history else None,
        current_queue_score=None, current_sticker_history_penalty=None,
        merge_sticker_histories=merge_histories, current_soft_window_penalty=None,
        color_confusion_pruning=color_confusion_pruning, soft_top_k=soft_top_k,
        outside_top_k_penalty=outside_top_k_penalty, outside_top_k_offered=0,
        require_layer_activity=require_layer_activity, activity_pruned=0,
        current_sticker_history_qualified_count=None)
    started = last_heartbeat = time.monotonic()

    def report(event, **extra):
        if bridge is not None:
            stats.update(bridge.stats)
        if soft_window is not None:
            stats.update(soft_window.stats)
        if progress:
            pending = queue.minimum()
            progress(dict(event=event, elapsed_seconds=time.monotonic() - started, **stats,
                queue_pending=queue.pending_count, stored_contexts=queue.context_count,
                best_pending_frame_score=(queue.minimum_frame_score() if soft_top_k else
                                          queue.minimum_secondary() if sticker_history else pending),
                best_pending_queue_score=pending,
                best_pending_sticker_mismatches=(int(pending) if sticker_history and
                    sticker_history_weight is None and pending is not None else None),
                **(queue.cache_stats() if hasattr(queue, "cache_stats") else {}), **extra))

    def heartbeat():
        nonlocal last_heartbeat
        if time.monotonic() - last_heartbeat >= progress_seconds:
            last_heartbeat = time.monotonic()
            report("progress")

    def context(node, step=None):
        next_frame = frames[node.depth] if node.depth < len(frames) else None
        evidence = (activity.previous(step if step is not None else queue.last_step(node.path_id), next_frame)
                    if require_layer_activity and next_frame is not None else ())
        return pickle.dumps(("sticker-best-history-v1" if merge_histories else
                             "sticker-history-graded-v1" if sticker_history and sticker_history_weight is not None else
                             "sticker-history-v1" if sticker_history else
                             "latest-reliable-frame-v3" if bridge else "latest-frame-v1",
            node.depth, state_key(node.state),
            canonical_update(node.update, next_frame, min_update_gap),
            node.recent if sticker_history and not merge_histories else (), evidence), protocol=5)

    def stickers(recent):
        return recent[0] if soft_window is not None else recent

    def history_summary(recent):
        return history.summary(stickers(recent))

    def offer(child, step, preceding_steps=()):
        status = queue.offer(context(child, step), child, serialize_state(child.state), step,
                             preceding_steps=preceding_steps)
        if status is None:
            stats["dominated_pruned"] += 1
        elif status == "reopened":
            stats["reopened_states"] += 1
        elif status == "improved_pending":
            stats["improved_pending"] += 1
        stats["peak_pending"] = max(stats["peak_pending"], queue.pending_count)

    def outcome(status, node=None):
        report(status)
        return dict(status=status, elapsed_seconds=time.monotonic() - started, **stats,
            queue_pending=queue.pending_count, stored_contexts=queue.context_count,
            steps=queue.steps(node.path_id) if node else [],
            solved_frame_index=frames[node.depth - 1] if node and node.depth else None)

    if solved(initial_state):
        return outcome("already_solved")
    root = storage.MeanNode(0, initial_state, 0., (None, None, 0), priority=0.,
                            recent=(soft_window.initial(history.EMPTY, initial_state) if soft_window is not None else
                                    history.EMPTY if sticker_history else ()))
    queue.offer(context(root), root, serialize_state(initial_state))
    stats["peak_pending"] = 1
    report("started")
    while queue.pending_count:
        heartbeat()
        node = queue.pop(deserialize_state)
        incoming = queue.last_step(node.path_id)
        frame = frames[node.depth - 1] if node.depth else None
        node_history = history_summary(node.recent) if sticker_history else {}
        if soft_window is not None:
            soft_window.establish(node.recent)
        stats.update(current_frames=node.depth, current_frame_index=frame,
                     current_local_rank=incoming.get('local_frame_rank'),
                     current_top_k_penalty=incoming.get('outside_top_k_penalty'),
                     current_frame_score=(incoming.get('state_score') if soft_top_k else
                                          node.secondary_priority if sticker_history else node.priority)
                     if node.depth else None,
                     current_queue_score=node.priority if node.depth else None,
                     current_soft_window_penalty=(node.priority-node.secondary_priority-
                        sticker_history_weight*node_history['sticker_history_penalty']-
                        incoming.get('outside_top_k_penalty', 0.)
                        if soft_window is not None else None),
                     current_supported_move_count=(sum(good for _,_,good in node.recent[1])
                                                   if soft_window is not None else None),
                     current_sticker_mismatch_count=node_history.get('sticker_mismatch_count'),
                     current_sticker_history_penalty=node_history.get('sticker_history_penalty'),
                     current_sticker_history_qualified_count=node_history.get('sticker_history_qualified_count'))
        if node.depth > stats["furthest_frames"]:
            stats.update(furthest_frames=node.depth, furthest_frame_index=frame)
            steps = queue.steps(node.path_id)
            report("furthest", last_score=node.priority,
                   frontier_moves=[dict(frame_index=s["frame_index"], state_moves=s["state_moves"])
                                   for s in steps if s.get("state_moves")])
        # A newly generated solved branch must compete with the entire queue.
        if solved(node.state) and (bridge is None or frame is None or bridge.quality[frame]['reliable']):
            return outcome("solved", node)
        if node.depth == len(frames):
            stats["unsolved_endpoints"] += 1
            continue
        stats["evaluated_states"] += 1
        expanded = (bridge.expand(node.depth, node.state, node.update, min_update_gap,
                                  state_key, heartbeat) if bridge is not None else None)
        if expanded is None:
            depth, parents = node.depth, ((node.state, node.update, ()),)
        else:
            depth, parents = expanded
            if depth == len(frames) or not bridge.quality[frames[depth]]['reliable']:
                # The next observation is still weak (or absent). Preserve
                # all legal discrete states without consulting this gap's
                # geometry/colors; continue with the last reliable priority.
                for parent_state, parent_update, prefix_steps in parents:
                    recent = node.recent
                    if soft_window is not None:
                        recent = (stickers(recent), soft_window.prefix_trail(recent[1], node.state, prefix_steps), recent[2])
                    child = storage.MeanNode(depth, parent_state, node.total_cost,
                        parent_update, node.path_id, recent=recent, priority=node.priority,
                        secondary_priority=node.secondary_priority)
                    prefix = tuple(dict(s, **history_summary(node.recent)) for s in prefix_steps) if sticker_history else prefix_steps
                    step = dict(prefix[-1], inherited_queue_priority=node.priority,
                                inherited_queue_secondary_score=node.secondary_priority)
                    offer(child, step, prefix[:-1])
                continue
        frame = frames[depth]
        for parent_state, parent_update, prefix_steps in parents:
            children = []
            evidence = activity.previous(prefix_steps[-1] if prefix_steps else incoming, frame)
            parent_trail = (soft_window.prefix_trail(node.recent[1], node.state, prefix_steps)
                            if soft_window is not None else None)
            choices, skipped = replay.transitions_for_update_context(
                transitions, frame, parent_state, parent_update, min_update_gap)
            stats["spacing_pruned"] += skipped
            family_scores = {}
            for choice in choices:
                if not math.isfinite(choice.score) or advance_update(parent_update, choice, frame, min_update_gap) is None:
                    continue
                if color_confusion_pruning and choice.payload.get('color_forbidden_count', 0):
                    continue
                error = choice.payload.get('color_error')
                if error is not None and (not math.isfinite(error) or
                        minimum_color_accuracy is not None and error > 1.-minimum_color_accuracy):
                    continue
                key = state_key(choice.state)
                family_scores[key] = min(family_scores.get(key, float('inf')), choice.score)
            ranked_scores = sorted(family_scores.values())
            for choice in choices:
                heartbeat()
                stats["attempted_transitions"] += 1
                if require_layer_activity and not activity.permits(evidence, choice.payload):
                    stats['activity_pruned'] += 1
                    continue
                if color_confusion_pruning and choice.payload.get('color_forbidden_count', 0):
                    stats['color_pruned'] += 1
                    continue
                if not math.isfinite(choice.score):
                    stats["invalid_pruned"] += 1
                    continue
                color_error = choice.payload.get("color_error")
                if color_error is not None:
                    if not math.isfinite(float(color_error)):
                        stats["invalid_pruned"] += 1
                        continue
                    if minimum_color_accuracy is not None and float(color_error) > 1. - minimum_color_accuracy:
                        stats["color_pruned"] += 1
                        continue
                update = advance_update(parent_update, choice, frame, min_update_gap)
                if update is None:
                    stats["spacing_pruned"] += 1
                    continue
                recent = node.recent
                priority, secondary, extra = choice.score, 0., {}
                prefix = prefix_steps
                if sticker_history:
                    if (choice.payload.get('color_match_count', 0) and
                            'sticker_observed_mask_hex' not in choice.payload):
                        raise ValueError('Sticker-history ordering requires physical observation masks')
                    recent = history.observe(stickers(recent), choice.payload)
                    extra = history.summary(recent)
                    priority = (extra['sticker_mismatch_count'] if sticker_history_weight is None else
                                choice.score + sticker_history_weight * extra['sticker_history_penalty'])
                    secondary = choice.score
                    prefix = tuple(dict(s, **history_summary(node.recent)) for s in prefix_steps)
                if soft_window is not None:
                    supported = soft_window.is_supported(choice.payload,
                        node_history['sticker_history_penalty'], extra['sticker_history_penalty'],
                        reliable=bridge is None or bridge.quality[frame]['reliable'])
                    trail = soft_window.trail_after(parent_trail, choice.state, frame,
                        choice.payload.get('state_moves', ()), supported=supported)
                    recent = recent, trail, node.recent[2]
                child = storage.MeanNode(depth + 1, choice.state, priority if merge_histories else choice.score,
                    update, node.path_id, recent=recent, priority=priority, secondary_priority=secondary)
                step = dict(choice.payload, frame_index=frame, state_score=choice.score,
                            queue_priority=priority, queue_secondary_score=secondary, **extra)
                if require_layer_activity:
                    step.update(activity.log_fields(evidence, choice.payload))
                if soft_top_k:
                    step['local_frame_rank'] = 1+bisect_left(ranked_scores, family_scores[state_key(choice.state)]-1e-9)
                    step['local_candidate_count'] = len(family_scores)
                children.append((child, step, prefix))
            if children:
                for child, step, _ in children:
                    local_rank = step.get('local_frame_rank', 1)
                    penalty = (outside_top_k_penalty if soft_top_k and local_rank > soft_top_k else 0.)
                    child.priority += penalty
                    if soft_top_k:
                        child.total_cost = child.priority
                        stats['outside_top_k_offered'] += int(penalty > 0)
                        step.update(outside_top_k_penalty=penalty, queue_priority=child.priority)
                best = min(children, key=lambda c:(c[0].priority, c[0].secondary_priority))[0]
                best_step = next(step for child, step, _ in children if child is best)
                best_context = context(best, best_step)
                for child, step, prefix in children:
                    if soft_window is not None:
                        penalty, until = soft_window.fork_cost(frame, child.recent[2],
                            new_fork=context(child, step) != best_context)
                        child.recent = child.recent[0], child.recent[1], until
                        child.priority += penalty
                        if soft_top_k:
                            child.total_cost = child.priority
                        step.update(queue_priority=child.priority, **soft_window.describe(child.recent, penalty))
                    offer(child, step, prefix)
    return outcome("exhausted")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    replay.add_nine_class_input_arguments(parser)
    parser.add_argument("--initial-scramble", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, help="Last source frame; defaults to the end of the input.")
    parser.add_argument("--video-frame-count", type=int)
    parser.add_argument("--fps", type=float, help="Source rate; inferred from video/cache metadata, otherwise 60.")
    parser.add_argument("--min-state-update-gap", type=int)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--scorer", choices=("auto", "prepared", "legacy"), default="auto")
    parser.add_argument("--score-cache-size", type=int, default=4096)
    parser.add_argument("--frontier-cache-size", type=int, default=4096)
    parser.add_argument("--cost-cache-size", type=int, default=131072)
    parser.add_argument("--sqlite-cache-mib", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-seconds", type=float, default=10.)
    parser.add_argument("--console-timezone", default="America/Chicago")
    parser.add_argument('--queue-order', choices=('sticker-history', 'sticker-mismatches', 'frame-score'),
                        default='frame-score',
                        help='Default: current 50/50 geometry + weighted color fit; no history/window ordering.')
    parser.add_argument('--soft-top-k', type=int, default=8,
                        help='Prefer the best K scored physical successors per parent/frame, including boundary ties; 0 disables.')
    parser.add_argument('--outside-top-k-penalty', type=float, default=.25,
                        help='Queue penalty for choices outside the local top K; never accumulated along the path (default: 0.25).')
    parser.add_argument('--require-layer-activity', action=argparse.BooleanOptionalAction, default=True,
                        help='A move requires a nonzero partial turn or state change on that layer in frame-1 (default: on).')
    parser.add_argument('--color-confusion-scoring', action=argparse.BooleanOptionalAction, default=True,
                        help='Reduce the color cost for listed direct confusions (default: on).')
    parser.add_argument('--color-confusion-pruning', action=argparse.BooleanOptionalAction, default=False,
                        help='Optional legacy hard rejection of unlisted color confusions (default: off).')
    parser.add_argument('--color-confusion-cost', type=float, default=.5,
                        help='Cost per directly permitted confusion, relative to an exact match cost of zero (default: 0.5).')
    parser.add_argument('--sticker-history-weight', type=float, default=.1,
                        help='Weight of sum((wrong observations in last four / 4)^2); default: 0.1.')
    parser.add_argument('--joint-orientation-rescue', action=argparse.BooleanOptionalAction, default=True,
                        help='Choose rescue orientation jointly with partial-turn fit per physical successor (default: on).')
    parser.add_argument('--merge-sticker-histories', action=argparse.BooleanOptionalAction, default=True,
                        help='Keep the lowest fit + history cost per physical context in graded mode (default: on).')
    parser.add_argument('--soft-backtrack-moves', type=int, default=4,
                        help='Supported moves kept open before penalizing distant new forks; 0 disables.')
    parser.add_argument('--late-fork-penalty', type=float, default=.5,
                        help='Temporary queue penalty for new forks behind the supported window (default: 0.5).')
    parser.add_argument("--bridge-unreliable-frames", action=argparse.BooleanOptionalAction, default=False,
                        help="Ignore globally weak observations and carry discrete state transitions (experimental).")
    parser.add_argument("--bridge-max-frames", type=int, default=2, help="Weak frames expanded per work batch.")
    parser.add_argument("--bridge-max-seconds", type=float, default=.1, help="Source duration per work batch.")
    parser.add_argument("--bridge-max-reprojection", type=float, default=.1,
                        help="Maximum best REPROJ fit, using the same metric as the display.")
    parser.add_argument("--bridge-min-inlier-fraction", type=float, default=.5)
    parser.add_argument("--save-final-video", type=Path,
                        help="Render the winning path after solving, with pose and move overlays.")
    parser.add_argument("--display", action="store_true", help="Display the winning path after solving.")
    parser.add_argument("--render-source", type=Path,
                        help="Original video for rendering when observation metadata has no source path.")
    parser.add_argument("--show-keypoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-orientation-axes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-virtual-cube", action=argparse.BooleanOptionalAction, default=True,
                        help="Show the colored virtual cube beside the source video (default: on).")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if (args.start_frame < 0 or args.end_frame is not None and args.end_frame < args.start_frame
            or args.turning_start_frame is not None and args.turning_start_frame < 0
            or args.min_state_update_gap is not None and args.min_state_update_gap < 1
            or args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0)
            or args.frontier_cache_size < 1 or args.cost_cache_size < 0 or args.score_cache_size < 0
            or args.sqlite_cache_mib < 1 or not math.isfinite(args.progress_seconds) or args.progress_seconds <= 0
            or args.bridge_max_frames < 1 or not math.isfinite(args.bridge_max_seconds) or args.bridge_max_seconds <= 0
            or not math.isfinite(args.sticker_history_weight) or args.sticker_history_weight < 0
            or not math.isfinite(args.color_confusion_cost) or not 0 < args.color_confusion_cost < 1
            or args.soft_backtrack_moves < 0 or not math.isfinite(args.late_fork_penalty) or args.late_fork_penalty < 0
            or args.soft_top_k < 0 or not math.isfinite(args.outside_top_k_penalty) or args.outside_top_k_penalty < 0
            or not math.isfinite(args.bridge_max_reprojection) or args.bridge_max_reprojection < 0
            or not math.isfinite(args.bridge_min_inlier_fraction) or not 0 <= args.bridge_min_inlier_fraction <= 1):
        parser.error("Invalid frame interval, turning boundary, FPS, gap, cache size, or progress interval")
    try:
        ZoneInfo(args.console_timezone)
    except (KeyError, ValueError):
        parser.error("Unknown console timezone")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in
           ("config.json", "progress.jsonl", "search.sqlite3", "observations.jsonl")):
        parser.error("Use a fresh output directory; previous inputs and searches are preserved")
    import torch
    torch.set_num_threads(2)
    track_history = args.queue_order != 'frame-score'
    history_weight = args.sticker_history_weight if args.queue_order == 'sticker-history' else None
    merge_histories = args.merge_sticker_histories and args.queue_order == 'sticker-history'
    minimum_color_accuracy = None
    confusion_scoring = args.color_confusion_scoring or args.color_confusion_pruning
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; use --device cpu")
    renderer = replay.load_renderer()
    try:
        records, metadata = replay.load_nine_class_records(args, renderer)
        render_source = args.render_source or args.source or metadata.get("source")
        if (args.save_final_video or args.display) and not render_source:
            raise ValueError("Rendering needs the original video; specify --render-source")
        if args.save_final_video and Path(render_source).resolve() == args.save_final_video.resolve():
            raise ValueError("Output video must differ from the source video")
        colors = replay.initial_sticker_colors(renderer, args.initial_scramble)
        end = max(records) if args.end_frame is None else args.end_frame
        frames = tuple(range(args.start_frame, end + 1))
        if not frames or any(frame not in records for frame in frames):
            raise ValueError("Source observations do not cover the requested consecutive frame interval")
        video_count = args.video_frame_count if args.video_frame_count is not None else max(records) + 1
        if video_count < max(records) + 1 or not math.isfinite(args.fps) or args.fps <= 0:
            raise ValueError("Invalid whole-video frame count or FPS metadata")
        mode = ("prepared" if args.device == "cpu" else "legacy") if args.scorer == "auto" else args.scorer
        cls = replay.PreparedReplayScorer if mode == "prepared" else replay.ReplayScorer
        scorer = cls(renderer, records, args.device, cache_size=args.score_cache_size,
                     initial_colors=colors, turning_start_frame=args.turning_start_frame,
                     balance_geometry_color=True, minimum_color_accuracy=minimum_color_accuracy,
                     track_sticker_observations=track_history,
                     joint_orientation_rescue=args.joint_orientation_rescue,
                     color_confusion_pruning=args.color_confusion_pruning, color_confusion_cost=args.color_confusion_cost,
                     color_confusion_scoring=confusion_scoring,
                     preserve_layer_activity=args.require_layer_activity)
        for frame in frames:
            scorer.prepare(frame)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    initial = scorer.state_from_key(scorer.key(renderer.CubeGeometryState.identity()))
    soft_window = (SoftMoveWindow(rotation_invariant_state_key(scorer), apply_moves=scorer.apply_moves,
        moves=args.soft_backtrack_moves, fork_penalty=args.late_fork_penalty)
        if args.queue_order == 'sticker-history' and args.soft_backtrack_moves else None)
    gap = (renderer.turn_state_update_gap_frames(args.fps)
           if args.min_state_update_gap is None else args.min_state_update_gap)
    bridge = (FrameObservationBridge(scorer, frames, fps=args.fps, advance_update=advance_update,
        canonical_update=canonical_update, transition_type=replay.Transition,
        max_frames=args.bridge_max_frames, max_seconds=args.bridge_max_seconds,
        max_reprojection=args.bridge_max_reprojection, min_inlier_fraction=args.bridge_min_inlier_fraction)
        if args.bridge_unreliable_frames else None)
    if bridge is not None:
        (args.output_dir / 'frame_quality.json').write_text(json.dumps(bridge.quality, indent=2)+'\n')
        print(f"Frame bridge: {sum(not q['reliable'] for q in bridge.quality.values())} weak frames; "
              f"up to {args.bridge_max_frames} frames / {args.bridge_max_seconds:g}s per work batch.", flush=True)
    config = dict(algorithm="best_first_latest_frame", log=str(args.log.resolve()),
        log_sha256=hashlib.sha256(args.log.read_bytes()).hexdigest(), observation_input=metadata,
        initial_scramble=args.initial_scramble, start_frame=args.start_frame, end_frame=end,
        turning_start_frame=args.turning_start_frame,
        priority_statistic="latest_frame_geometric_plus_color_cost", priority_ties="greater depth, then insertion order",
        color_scoring=('directional_confusion_cost' if confusion_scoring else 'direct_match_fraction'),
        sticker_keypoint_min_confidence=.25,
        score_weights=dict(geometry=.5, color=.5),
        geometric_component_weights=dict(landmarks=.5, sticker_positions=.5),
        geometry_normalization=dict(landmarks=renderer.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION,
                                    sticker_positions=renderer.PNP_TURN_CENTER_PENALTY_SCALE, range=[0., 1.]),
        minimum_color_accuracy=minimum_color_accuracy,
        color_pruning=('disabled; color errors affect fit and history priority only' if minimum_color_accuracy is None else
                       'after orientation rescue and before partial-angle/state collapse'),
        error_gate=None, threshold=None, forecast_frames=0, past_path_priority_weight=0.,
        frame_or_state_budget=None, time_budget=None, beam_width=None, state_change_limit=None,
        video_frame_count=video_count, fps=args.fps, min_state_update_gap_frames=gap,
        scorer=mode, device=args.device,
        geometry_scope="all retained source geometry and discrete offsets; no top-k cutoff",
        orientation_policy=('24-way joint orientation and partial-turn rescue below 2/3 static state accuracy'
                            if args.joint_orientation_rescue else
                            '24-way state-color rescue below 2/3 accuracy before move search'),
        orientation_rescue_minimum_color_accuracy=2./3.,
        joint_orientation_rescue=args.joint_orientation_rescue,
        orientation_rescue_selection='lowest geometric + color fit per physical successor; history affects queue ordering only',
        merge_key="frame,physical_state,unexpired_move_spacing_context",
        dominance_policy="lowest latest-frame score; past path costs are not used",
        goal="physical cube colors solved in any whole-cube orientation", reference_solution_used=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    config['observation_bridge'] = dict(enabled=bridge is not None, max_frames=args.bridge_max_frames,
        max_seconds=args.bridge_max_seconds, max_reprojection_fit=args.bridge_max_reprojection,
        min_keypoint_inlier_fraction=args.bridge_min_inlier_fraction,
        quality_scope='state-independent display REPROJ fit; same hypothesis must meet reprojection and coverage limits',
        overlong_gap_policy='unscored transition batches; inherit last reliable queue priority; no unobserved goal acceptance',
        intermediate_scores='unscored null; no color pruning, orientation rescue, or partial-angle search',
        priority=('unchanged history penalty and latest reliable fit; inherit combined queue score' if track_history
                  else 'latest reliable endpoint only'), source_frame_moves_preserved=True)
    if bridge is not None:
        config.update(priority_statistic='latest_reliable_frame_geometric_plus_color_cost',
            color_pruning=(config['color_pruning'] if minimum_color_accuracy is None else
                           'reliable observed frames only; never inside unreliable gaps'),
            goal='physical colors solved on an observed reliable endpoint; never inside an unscored gap')
    if track_history:
        config.update(algorithm=('best_first_graded_sticker_history' if history_weight is not None else
                                 'best_first_sticker_mismatch_count'),
            priority_statistic=('latest_reliable_fit_plus_weighted_sticker_history_penalty' if history_weight is not None else
                                'physical_stickers_incorrect_in_at_least_3_of_their_last_4_observations'),
            priority_ties='latest reliable geometric + color score, then greater depth, then insertion order',
            merge_key='frame,physical_state,unexpired_move_spacing_context,all_54_sticker_histories',
            dominance_policy='lowest fit score only within identical physical state and sticker histories')
    config['sticker_history'] = dict(enabled=track_history, weight=history_weight,
        penalty='sum((wrong_observations_in_last_four / 4)^2)',
        minimum_penalty_observations=1, missing_history='zero contribution; fixed denominator four',
        sticker_count=54, window_observations=4, minimum_observations=4, incorrect_threshold=3,
        binary_mismatch_count='diagnostic only in graded mode; requires four observations',
        includes_current_frame=True, includes_static_opening=True,
        unobserved='retain history without adding an observation',
        identities=[dict(label=i.label,color=colors[i.label]) for i in renderer.canonical_sticker_identities()])
    config['color_confusion_pruning'] = dict(enabled=args.color_confusion_pruning,
        direction='expected_to_predicted', allowed={k:sorted(v) for k,v in renderer.COLOR_CLASS_CONFUSIONS.items()},
        exact_cost=0., permitted_cost=args.color_confusion_cost,
        forbidden=('immediate rejection before fit collapse' if args.color_confusion_pruning else 'cost one; no rejection'),
        transitive=False, duplicate_keypoints='minimum cost per projected sticker; any exact match wins',
        denominator='observed projected stickers only')
    if args.color_confusion_pruning:
        config['color_pruning'] = 'reject any projected sticker with no exact or directly permitted predicted color'
    config['color_confusion_scoring'] = dict(enabled=confusion_scoring,
        exact_cost=0., permitted_cost=args.color_confusion_cost, unlisted_cost=1., transitive=False)
    config['soft_local_top_k'] = dict(enabled=args.soft_top_k > 0, k=args.soft_top_k,
        outside_penalty=args.outside_top_k_penalty, ranking='50/50 geometry + weighted color, after spacing checks and before activity rejection; one rank per physical successor',
        ties='same treatment for equal scores at the boundary', hard_pruning=False,
        penalty_scope='latest incoming choice only; not accumulated; outside choices remain interleaved')
    config['layer_activity'] = dict(enabled=args.require_layer_activity, preceding_frames=1,
        minimum_activity_frames=2, includes_current_state_change=True, minimum_partial_angle='any nonzero',
        match='physical axis, side and layer width; direction ignored',
        static_hold='no activity; clears previous evidence', state_change='counts even with zero residual angle',
        bridge='unobserved activity; existing blind-frame transitions remain unconstrained',
        merge_key_includes_activity=True, retains_distinct_hold_activity_fits=True)
    if args.soft_top_k:
        config.update(priority_statistic='latest_frame_fit_plus_local_rank_penalty',
                      dominance_policy='lowest queue priority within the same legal state context')
    if args.require_layer_activity:
        config['merge_key'] += ',preceding_frame_layer_activity'
    if merge_histories:
        config.update(merge_key='frame,physical_sticker_arrangement,unexpired_move_spacing_context',
            dominance_policy='lowest current fit + history penalty; keep the entire actual path and history; reopen on improvement',
            algorithm='best_first_graded_sticker_history_merged')
    config['history_merging'] = dict(enabled=merge_histories,
        approximation='future histories can rank differently; alternate histories are discarded, legal continuations retained')
    config['soft_move_window'] = dict(enabled=soft_window is not None,
        supported_moves=args.soft_backtrack_moves, late_fork_penalty=args.late_fork_penalty,
        support_geometry_max=.5, support_color_accuracy_min=2./3.,
        support_history='penalty must not increase on a supported move',
        progress='loop-erased physical states modulo whole-cube rotations; holds and unobserved moves earn no support',
        protected_boundary='moves older than the configured supported-move window on an expanded path',
        scheduling='penalize newly generated nonbest children behind boundary; existing branches and best continuation stay eligible',
        recovery='penalty expires after catching up past the boundary recorded when the fork was created',
        hard_pruning=False)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    queue = storage.BufferedMeanQueue(args.output_dir / "search.sqlite3",
        frontier_cache_size=args.frontier_cache_size, cost_cache_size=args.cost_cache_size,
        sqlite_cache_mib=args.sqlite_cache_mib)
    latest = {}
    with (args.output_dir / "progress.jsonl").open("w", buffering=1) as log:
        def progress(item):
            now = datetime.now(timezone.utc)
            item.update(timestamp_utc=now.isoformat(), total_video_frames=video_count,
                furthest_video_frame=item["furthest_frame_index"] + 1 if item["furthest_frame_index"] is not None else 0,
                scoring_calls=scorer.scoring_calls, score_cache_hits=scorer.cache_hits,
                joint_rescue_calls=getattr(scorer, 'joint_rescue_calls', None),
                orientation_candidates_scored=getattr(scorer, 'orientation_candidates_scored', None),
                color_configurations_scored=getattr(scorer, 'color_configurations_scored', None),
                child_states_built=getattr(scorer, 'child_states_built', None))
            item.update(confusion_pruned_fits=scorer.confusion_pruned_fits,
                        confusion_dead_end_states=scorer.confusion_dead_end_states)
            latest.clear()
            latest.update(item)
            log.write(json.dumps(item, allow_nan=False) + "\n")
            if item["event"] in {"started", "progress", "solved", "already_solved", "exhausted"}:
                print(storage.format_console_progress(item, args.console_timezone, now), flush=True)
            temporary = args.output_dir / "status.tmp"
            temporary.write_text(json.dumps(item, indent=2, allow_nan=False))
            temporary.replace(args.output_dir / "status.json")
            if (args.output_dir / "STOP").exists():
                raise KeyboardInterrupt
        try:
            result = frame_full_solve(initial, frames, scorer, scorer.solved_predicate(colors), scorer.key,
                min_update_gap=gap, queue=queue, serialize_state=scorer.key,
                deserialize_state=scorer.state_from_key, progress=progress, progress_seconds=args.progress_seconds,
                minimum_color_accuracy=minimum_color_accuracy,
                bridge=bridge, sticker_history=track_history, sticker_history_weight=history_weight,
                merge_histories=merge_histories, soft_window=soft_window,
                color_confusion_pruning=args.color_confusion_pruning, soft_top_k=args.soft_top_k,
                outside_top_k_penalty=args.outside_top_k_penalty,
                require_layer_activity=args.require_layer_activity)
            result.update(confusion_pruned_fits=scorer.confusion_pruned_fits,
                          confusion_dead_end_states=scorer.confusion_dead_end_states)
            (args.output_dir / "result.json").write_text(json.dumps(dict(config=config, **result), indent=2))
            if result["status"] == "exhausted":
                print("No solution found: all surviving branches were exhausted. Search stopped.", flush=True)
        except KeyboardInterrupt:
            item = dict(latest, event="interrupted", status="interrupted")
            result = item
            log.write(json.dumps(item) + "\n")
            (args.output_dir / "status.json").write_text(json.dumps(item, indent=2))
            (args.output_dir / "result.json").write_text(json.dumps(dict(config=config, **item), indent=2))
            print("Search interrupted; queued branches and paths are preserved.", flush=True)
        finally:
            queue.close()
    if (args.save_final_video or args.display) and result.get("status") in {"solved", "already_solved"}:
        spec = importlib.util.spec_from_file_location(
            "full_search_video", ROOT / "scripts/render_independent_keypoint_search.py")
        video = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = video
        spec.loader.exec_module(video)
        report = video.render_result(renderer, records, dict(config=config, **result), render_source,
            output_path=args.save_final_video, display=args.display, show_keypoints=args.show_keypoints,
            show_orientation_axes=args.show_orientation_axes,
            show_virtual_cube=args.show_virtual_cube,
            turn_log=args.output_dir / "rendered_turn_candidate_scores.jsonl")
        (args.output_dir / "render.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
