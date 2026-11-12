#!/usr/bin/env python3
"""Search independent odd and even frames using the latest geometry and color fit."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "frame_search_storage", ROOT / "scripts/search_independent_keypoint_mean.py")
storage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = storage
SPEC.loader.exec_module(storage)
replay = storage.replay
TURN_HANDOFF_MIN_DEGREES = 12.
COLOR_MISMATCH_WINDOW_FRAMES = 8
BRANCH_STAGE_EXPANSIONS_PER_FRAME = 100


class SearchTimeLimit(Exception):
    pass


@lru_cache(maxsize=8192)
def sticker_mask_bits(mask):
    """Decode the scorer's physical-sticker mask in identity order."""
    packed = np.frombuffer(int(mask, 16).to_bytes(7, 'little'), dtype=np.uint8)
    return np.unpackbits(packed, bitorder='little')[:54]


def update_bottom_quartile(recent, step, counter_dtype):
    """Add one scored observation and average the lowest quarter of seen identities."""
    seen = sticker_mask_bits(step['sticker_observed_mask_hex'])
    matched = sticker_mask_bits(step['sticker_correct_mask_hex'])
    counts = np.frombuffer(recent, dtype=counter_dtype).reshape(2, 54).copy()
    counts[0] += seen
    counts[1] += matched
    observed = counts[0] > 0
    rates = counts[1, observed] / counts[0, observed]
    size = (len(rates) + 3) // 4
    rate = float(np.partition(rates, size - 1)[:size].mean()) if size else None
    return counts.tobytes(), rate


def supported_completion(renderer, step):
    rotations = step.get('layer_rotations', ())
    if step.get('parallel_layers'):
        axis = step['layer_axis']
        angles = renderer.parallel_layer_angles(tuple(renderer.CubeLayerRotation(**r) for r in rotations))
        offsets = renderer.parallel_turn_actions(axis).get(tuple(step.get('state_moves', ())), (0., 0., 0.))
        completion = renderer.parallel_layer_rotations(axis, tuple(
            math.copysign(90., angle) if math.isfinite(angle) and abs(angle) > TURN_HANDOFF_MIN_DEGREES
            and support and not offset else 0.
            for angle, support, offset in zip(angles, step['layer_support'], offsets, strict=True)))
        moves = tuple(renderer.cube_rotation_to_notation(r) for r in completion)
        return completion_cue(moves) if moves else None
    if len(rotations) != 1 or step.get('layer_inlier_count', 0) <= 0:
        return None
    rotation = rotations[0]
    angle = rotation['angle_degrees']
    if not math.isfinite(angle) or abs(angle) <= TURN_HANDOFF_MIN_DEGREES:
        return None
    move = renderer.cube_rotation_to_notation(renderer.CubeLayerRotation(
        rotation['axis'], rotation['side'], math.copysign(90., angle), rotation['layer_count']))
    if move is None or any(committed.rstrip("'2") == move.rstrip("'2")
                           for committed in step.get('state_moves', ())):
        return None
    return move


def completion_cue(moves):
    return moves[0] if len(moves) == 1 else moves


def completion_moves(cue):
    return (cue,) if isinstance(cue, str) else cue


def unsupported_state_update(step):
    """Distinguish a partial turn with no inliers from a flat fit."""
    return (bool(step.get("state_moves")) and step.get("layer_inlier_count") == 0
            and ("layer_rotations" not in step or bool(step["layer_rotations"])))


def score_handoff(scorer, frame, state, cue):
    """Commit the previous layer, then fit the current observation from that state."""
    moves = completion_moves(cue)
    after = scorer.apply_moves(state, moves)
    hold = scorer.score_hold(frame, after)
    if hold is None or 'geometry_option_index' not in hold.payload:
        return None
    step = dict(hold.payload, state_moves=moves, handoff_commit=True, is_baseline=False,
                label=f"commit {' '.join(moves)} then {hold.payload['label']}")
    return replay.Transition(after, hold.score, hold.rank, step, None)


def update_mismatch_window(window, step, weight):
    observed = step.get('color_match_count', 0)
    if observed or window:
        sample = (step['color_forbidden_count'] if observed else 0, observed)
        window = (*window[-(COLOR_MISMATCH_WINDOW_FRAMES-1):], sample)
    misses, total = map(sum, zip(*window)) if window else (0, 0)
    rate = misses / total if total else None
    penalty = weight * rate if len(window) == COLOR_MISMATCH_WINDOW_FRAMES and total else 0.
    step.update(window_mismatch_rate=rate, window_mismatch_penalty=penalty,
                window_mismatch_count=misses, window_color_observations=total)
    return window


def branch_stage_budget(solving_frame_count, override=None):
    if override is None:
        return max(1, solving_frame_count) * BRANCH_STAGE_EXPANSIONS_PER_FRAME
    if override < 1:
        raise ValueError("Branch stage expansion limit must be positive")
    return override


def frame_full_solve(initial_state, frames, transitions, solved, state_key, *,
                     queue=None, serialize_state=pickle.dumps, deserialize_state=pickle.loads,
                     progress=None, progress_seconds=10., max_explored_states=None,
                     bottom_quartile_weight=0., turn_handoff_bonus=None, ignore_queue_geometry=False,
                     mismatch_window_weight=0., queue_geometry_weight=.5,
                     branch_widths=(), branch_stage_expansions=None, resume_from=None):
    """Expand the lowest queue score, keeping one entry per frame and state."""
    frames = tuple(frames)
    if not frames or any(b <= a for a, b in zip(frames, frames[1:])):
        raise ValueError("Search frames must be strictly increasing")
    if not math.isfinite(progress_seconds) or progress_seconds <= 0:
        raise ValueError("Progress interval must be positive and finite")
    if max_explored_states is not None and (not isinstance(max_explored_states, int) or max_explored_states < 1):
        raise ValueError("State limit must be positive")
    if not math.isfinite(bottom_quartile_weight) or bottom_quartile_weight < 0:
        raise ValueError("Bottom-quartile weight must be finite and nonnegative")
    if not math.isfinite(mismatch_window_weight) or mismatch_window_weight < 0:
        raise ValueError("Mismatch-window weight must be finite and nonnegative")
    if not math.isfinite(queue_geometry_weight) or queue_geometry_weight < 0:
        raise ValueError("Queue geometry weight must be finite and nonnegative")
    branch_widths = tuple(branch_widths)
    if any(width < 1 for width in branch_widths) or any(b <= a for a, b in zip(branch_widths, branch_widths[1:])):
        raise ValueError("Branch widths must be positive and strictly increasing")
    static_end = getattr(transitions, 'turning_start_frame', None)
    static_count = bisect_right(frames, static_end) if static_end is not None else 0
    branch_stage_expansions = branch_stage_budget(len(frames) - static_count, branch_stage_expansions)
    if ignore_queue_geometry:
        queue_geometry_weight = 0.
    handoffs = turn_handoff_bonus is not None
    if handoffs and (not math.isfinite(turn_handoff_bonus) or turn_handoff_bonus < 0):
        raise ValueError("Turn handoff bonus must be finite and nonnegative")
    track_priority = bool(bottom_quartile_weight or mismatch_window_weight) or handoffs or queue_geometry_weight != .5
    if queue is None:
        if resume_from is not None:
            raise ValueError("Resume requires the saved search queue")
        queue = storage.BufferedMeanQueue()
        try:
            return frame_full_solve(initial_state, frames, transitions, solved, state_key,
                queue=queue, serialize_state=serialize_state, deserialize_state=deserialize_state,
                progress=progress, progress_seconds=progress_seconds, max_explored_states=max_explored_states,
                bottom_quartile_weight=bottom_quartile_weight, turn_handoff_bonus=turn_handoff_bonus,
                queue_geometry_weight=queue_geometry_weight, mismatch_window_weight=mismatch_window_weight,
                branch_widths=branch_widths, branch_stage_expansions=branch_stage_expansions)
        finally:
            queue.close()
    if resume_from is None and (queue.context_count or queue.deferred_count):
        raise ValueError("Start in a fresh search database")
    if resume_from is not None:
        stage = resume_from.get('branch_stage')
        if ((stage is None) != (not branch_widths)
                or stage is not None and (not 0 <= stage < len(branch_widths)
                                          or branch_widths[stage] != resume_from.get('branch_width'))
                or not (queue.pending_count or queue.deferred_count)
                or queue.deferred_count != resume_from.get('deferred_pending', 0)
                or queue.context_count != resume_from['stored_contexts']
                or queue.pending_count != resume_from['queue_pending']):
            raise ValueError("Saved queue does not match the search checkpoint")

    stats = dict(evaluated_states=0, attempted_transitions=0, invalid_pruned=0,
                 dominated_pruned=0, reopened_states=0, improved_pending=0,
                 unsolved_endpoints=0, peak_pending=0, furthest_frames=0,
                 furthest_frame_index=None, current_frames=0, current_frame_index=None,
                 current_frame_score=None)
    if branch_widths:
        stats.update(branch_stage=0, branch_stage_start_expansions=static_count, deferred_promoted=0)
    started = last_heartbeat = time.monotonic()
    if resume_from is not None:
        stats.update((key, resume_from[key]) for key in stats)
        started -= resume_from['elapsed_seconds']

    def report(event, **extra):
        item = dict(event=event, elapsed_seconds=time.monotonic() - started, **stats,
                    queue_pending=queue.pending_count, stored_contexts=queue.context_count)
        if hasattr(queue, 'cache_stats'):
            item.update(queue.cache_stats())
        if branch_widths:
            item.update(branch_width=branch_widths[stats['branch_stage']], deferred_pending=queue.deferred_count)
        if progress is not None:
            pending = queue.minimum()
            fit = queue.minimum_frame_score() if track_priority else pending
            progress(dict(item, best_pending_frame_score=fit, best_pending_queue_score=pending, **extra))
        return item

    def outcome(status, node=None):
        return dict(report(status), status=status, steps=queue.steps(node.path_id) if node else [],
                    solved_frame_index=frames[node.depth - 1] if node and node.depth else None)

    def context(depth, state, completion=None):
        key = ('latest-frame', depth, state_key(state))
        # Identical states can have different next-frame handoff candidates.
        return pickle.dumps((*key, completion) if handoffs else key, protocol=5)

    def sibling_order(choice):
        return choice.score, unsupported_state_update(choice.payload), choice.rank[1:]

    def offer(key, child, step=None):
        status = queue.offer(key, child, serialize_state(child.state), step)
        if status is None:
            stats['dominated_pruned'] += 1
        elif status == 'reopened':
            stats['reopened_states'] += 1
        elif status == 'improved_pending':
            stats['improved_pending'] += 1
        return status is not None

    empty_update = (None, None, 0)
    counter_dtype = np.min_scalar_type(len(frames))
    if resume_from is None:
        if solved(initial_state):
            return outcome('already_solved')
        empty_history = np.zeros((2, 54), dtype=counter_dtype).tobytes() if bottom_quartile_weight else ()
        if mismatch_window_weight:
            empty_history = (empty_history, ())
        root = storage.MeanNode(0, initial_state, 0., empty_update, recent=empty_history, priority=0.)
        queue.offer(context(0, initial_state), root, serialize_state(initial_state))
        stats['peak_pending'] = 1
        report('started')
    else:
        report('resumed')
    promote_ready = (branch_widths and resume_from is not None and queue.deferred_count
                     and queue.minimum_deferred() <= branch_widths[stats['branch_stage']])
    while queue.pending_count or queue.deferred_count:
        if max_explored_states is not None and stats['evaluated_states'] >= max_explored_states:
            return outcome('state_limit')
        now = time.monotonic()
        if now - last_heartbeat >= progress_seconds:
            last_heartbeat = now
            report('progress')
        if branch_widths:
            stage = stats['branch_stage']
            stage_expanded = stats['evaluated_states'] - stats['branch_stage_start_expansions']
            if (not promote_ready and stage + 1 < len(branch_widths)
                    and (not queue.pending_count or stage_expanded >= branch_stage_expansions)):
                stats.update(branch_stage=stage + 1, branch_stage_start_expansions=stats['evaluated_states'])
                promote_ready = True
                report('widened')
            width = branch_widths[stats['branch_stage']]
            if promote_ready:
                while queue.deferred_count and queue.minimum_deferred() <= width:
                    _, child = queue.pop_deferred(deserialize_state)
                    step = queue.last_step(child.path_id)
                    cue = supported_completion(transitions.m, step) if handoffs else None
                    if offer(context(child.depth, child.state, cue), child):
                        stats['deferred_promoted'] += 1
                    if time.monotonic() - last_heartbeat >= progress_seconds:
                        last_heartbeat = time.monotonic()
                        report('progress')
                promote_ready = False
                stats['peak_pending'] = max(stats['peak_pending'], queue.pending_count)
            if not queue.pending_count:
                continue
        node = queue.pop(deserialize_state)
        frame = frames[node.depth - 1] if node.depth else None
        incoming = queue.last_step(node.path_id) if track_priority else {}
        stats.update(current_frames=node.depth, current_frame_index=frame,
                     current_frame_score=(incoming.get('state_score') if track_priority else node.priority)
                     if node.depth else None)
        if track_priority:
            stats['current_queue_score'] = node.priority if node.depth else None
        if mismatch_window_weight:
            stats['current_window_mismatch_rate'] = incoming.get('window_mismatch_rate')
            stats['current_window_mismatch_penalty'] = incoming.get('window_mismatch_penalty')
        if node.depth > stats['furthest_frames']:
            stats.update(furthest_frames=node.depth, furthest_frame_index=frame)
            moves = [dict(frame_index=step['frame_index'], state_moves=step['state_moves'])
                     for step in queue.steps(node.path_id) if step.get('state_moves')]
            report('furthest', last_score=node.priority, frontier_moves=moves)
        if solved(node.state):
            return outcome('solved', node)
        if node.depth == len(frames):
            stats['unsolved_endpoints'] += 1
            continue

        stats['evaluated_states'] += 1
        frame = frames[node.depth]
        depth = node.depth + 1
        # A hold wins an equal-score tie with an unsupported state update.
        choices = list(transitions(frame, node.state))
        completion = supported_completion(transitions.m, incoming) if handoffs else None
        if completion is not None:
            handoff = score_handoff(transitions, frame, node.state, completion)
            if handoff is not None:
                choices.append(handoff)
        if branch_widths:
            choices.sort(key=sibling_order)
            siblings = set()
            previous_order = None
            admission_rank = 0
        else:
            choices.sort(key=lambda choice: unsupported_state_update(choice.payload))
        for choice in choices:
            stats['attempted_transitions'] += 1
            if not math.isfinite(choice.score):
                stats['invalid_pruned'] += 1
                continue
            step = dict(choice.payload, frame_index=frame, state_score=choice.score)
            next_completion = supported_completion(transitions.m, step) if handoffs else None
            key = context(depth, choice.state, next_completion)
            if branch_widths:
                if key in siblings:
                    continue
                siblings.add(key)
                rank = len(siblings)
                order = sibling_order(choice)
                if order != previous_order:
                    admission_rank = rank
                    previous_order = order
                if admission_rank > branch_widths[-1]:
                    break
                step['sibling_rank'] = rank
            recent = node.recent
            if mismatch_window_weight:
                recent, window = recent
            priority = choice.score
            if queue_geometry_weight == 0.:
                priority = .5 * (step.get('color_error') or 0.)
            elif queue_geometry_weight != .5 and 'score_components' in step:
                components = step['score_components']
                priority = queue_geometry_weight * components['geometry'] + .5 * components['color']
            if bottom_quartile_weight:
                rate = incoming.get('bottom_quartile_match_rate')
                if step.get('color_match_count', 0) and step.get('color_error') is not None:
                    recent, rate = update_bottom_quartile(recent, step, counter_dtype)
                penalty = bottom_quartile_weight * (1. - rate) if rate is not None else 0.
                priority += penalty
                step.update(bottom_quartile_match_rate=rate, bottom_quartile_penalty=penalty)
            if mismatch_window_weight:
                window = update_mismatch_window(window, step, mismatch_window_weight)
                priority += step['window_mismatch_penalty']
                recent = (recent, window)
            if completion is not None and tuple(step.get('state_moves', ())) == completion_moves(completion):
                priority -= turn_handoff_bonus
                step['turn_handoff_bonus'] = turn_handoff_bonus
            step['queue_priority'] = priority
            child = storage.MeanNode(depth, choice.state, priority, empty_update,
                                     node.path_id, recent=recent, priority=priority)
            if branch_widths and admission_rank > width:
                # The same state can become eligible at different widths.
                deferred_key = pickle.dumps((admission_rank, key), protocol=5)
                queue.defer(deferred_key, admission_rank, child, serialize_state(child.state), step)
            else:
                offer(key, child, step)
        stats['peak_pending'] = max(stats['peak_pending'], queue.pending_count)
    return outcome('exhausted')


def frame_passes(records, frames, parity, input_fps, *, already_sampled=False):
    """Select parity on the original source clock, including older split logs."""
    if parity == 'all':
        return [('all', frames, input_fps)]
    passes = []
    for name, remainder in (('even', 0), ('odd', 1)):
        if parity not in ('both', name):
            continue
        selected = tuple(frame for frame in frames
                         if records[frame].get('original_source_frame', frame) % 2 == remainder)
        if selected:
            passes.append((name, selected, input_fps if already_sampled else input_fps / 2.))
    return passes


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    replay.add_nine_class_input_arguments(parser)
    parser.add_argument("--initial-scramble", required=True)
    parser.add_argument("--frame-parity", choices=("both", "even", "odd", "all"), default="both",
                        help="Search both parities independently, one parity, or all input frames (default: both).")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, help="Last input frame, inclusive.")
    parser.add_argument("--fps", type=float, help="Override the input frame rate before parity selection.")
    parser.add_argument("--video-frame-count", type=int)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--scorer", choices=("auto", "prepared", "legacy"), default="auto")
    parser.add_argument("--score-cache-size", type=int, default=4096)
    parser.add_argument("--frontier-cache-size", type=int, default=4096)
    parser.add_argument("--cost-cache-size", type=int, default=131072)
    parser.add_argument("--sqlite-cache-mib", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-seconds", type=float, default=10.)
    parser.add_argument("--console-timezone", default="America/Chicago")
    parser.add_argument("--search-seconds", type=float, help="Search-only time limit per pass.")
    parser.add_argument("--max-explored-states", type=int, help="Maximum expanded states per pass.")
    parser.add_argument("--branch-widths", type=int, nargs="+", default=(),
                        help="Per-parent child cutoffs including ties, e.g. 3 5 8.")
    parser.add_argument("--branch-stage-expansions", type=int,
                        help=f"Override the stage budget of {BRANCH_STAGE_EXPANSIONS_PER_FRAME} expansions per solving frame.")
    queue_geometry = parser.add_mutually_exclusive_group()
    queue_geometry.add_argument("--queue-geometry-weight", type=float, default=.5,
                                help="Geometry coefficient in queue priority (default: 0.5).")
    queue_geometry.add_argument("--ignore-queue-geometry", action="store_true",
                                help="Keep geometry in candidate fitting and omit it from queue priority.")
    parser.add_argument("--bottom-quartile-weight", type=float, default=0.,
                        help="Add this weight times the mismatch rate of the lowest quartile of sticker histories.")
    parser.add_argument("--mismatch-window-weight", type=float, default=0.,
                        help="Weight for the pooled out-of-table mismatch rate over eight sampled frames.")
    parser.add_argument("--turn-handoff-bonus", type=float, nargs="?", const=.02,
                        help="Enable next-frame handoffs from supported partial turns over 12 degrees. "
                             "Subtract the given bonus from matching commits (default when enabled: 0.02).")
    parser.add_argument("--save-final-video", type=Path, help="Render the first solved pass.")
    parser.add_argument("--display", action="store_true", help="Display the first solved pass.")
    parser.add_argument("--render-source", type=Path, help="Override the original video for rendering.")
    parser.add_argument("--show-keypoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-orientation-axes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-virtual-cube", action=argparse.BooleanOptionalAction, default=True)
    return parser


def run_pass(args, directory, frames, scorer, initial, colors, config, *, resume_from=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    queue = storage.BufferedMeanQueue(directory / 'search.sqlite3',
        frontier_cache_size=args.frontier_cache_size, cost_cache_size=args.cost_cache_size,
        sqlite_cache_mib=args.sqlite_cache_mib)
    latest = {}
    start_calls, start_hits = scorer.scoring_calls, scorer.cache_hits
    if resume_from is not None:
        start_calls -= resume_from.get('scoring_calls', 0)
        start_hits -= resume_from.get('score_cache_hits', 0)
    with (directory / 'progress.jsonl').open('w', buffering=1) as log:
        def save_progress(item):
            log.write(json.dumps(item, allow_nan=False) + '\n')
            temporary = directory / 'status.tmp'
            temporary.write_text(json.dumps(item, indent=2, allow_nan=False))
            temporary.replace(directory / 'status.json')

        def progress(item):
            now = datetime.now(timezone.utc)
            furthest = item['furthest_frame_index']
            item.update(timestamp_utc=now.isoformat(), total_video_frames=config['video_frame_count'],
                        furthest_video_frame=furthest + 1 if furthest is not None else 0,
                        frame_parity=config['frame_parity'], scoring_calls=scorer.scoring_calls - start_calls,
                        score_cache_hits=scorer.cache_hits - start_hits)
            latest.clear()
            latest.update(item)
            save_progress(item)
            if item['event'] != 'furthest':
                print(storage.format_console_progress(item, args.console_timezone, now), flush=True)
            if item['event'] == 'widened':
                print(f"Widening to {item['branch_width']} children per parent.", flush=True)
            if (args.search_seconds is not None and item['elapsed_seconds'] >= args.search_seconds
                    and item['event'] not in {'solved', 'already_solved', 'exhausted', 'state_limit'}):
                raise SearchTimeLimit
            if ((directory / 'STOP').exists() or (args.output_dir / 'STOP').exists()) and (
                    resume_from is None or item['event'] in {'resumed', 'progress'}):
                raise KeyboardInterrupt

        try:
            result = frame_full_solve(initial, frames, scorer, scorer.solved_predicate(colors), scorer.key,
                queue=queue, serialize_state=scorer.key, deserialize_state=scorer.state_from_key,
                progress=progress, progress_seconds=args.progress_seconds,
                max_explored_states=args.max_explored_states,
                bottom_quartile_weight=args.bottom_quartile_weight, turn_handoff_bonus=args.turn_handoff_bonus,
                queue_geometry_weight=config['queue_score_weights']['geometry'],
                mismatch_window_weight=args.mismatch_window_weight,
                branch_widths=config.get('branch_widths', ()),
                branch_stage_expansions=config.get('branch_stage_expansions'),
                resume_from=resume_from)
        except (KeyboardInterrupt, SearchTimeLimit) as error:
            status = 'time_limit' if isinstance(error, SearchTimeLimit) else 'interrupted'
            result = dict(latest, event=status, status=status, **queue.cache_stats())
            result.update(queue_pending=queue.pending_count, stored_contexts=queue.context_count,
                          steps=[], solved_frame_index=None)
            save_progress(result)
        finally:
            queue.close()
    result['config'] = config
    (directory / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"{config['frame_parity']} pass: {result['status']}", flush=True)
    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    if (args.start_frame < 0 or args.end_frame is not None and args.end_frame < args.start_frame
            or args.turning_start_frame is not None and args.turning_start_frame < 0
            or args.frontier_cache_size < 1 or args.cost_cache_size < 0 or args.score_cache_size < 0
            or args.sqlite_cache_mib < 1 or not math.isfinite(args.progress_seconds) or args.progress_seconds <= 0
            or args.search_seconds is not None and (not math.isfinite(args.search_seconds) or args.search_seconds <= 0)
            or args.max_explored_states is not None and args.max_explored_states < 1):
        parser.error("Invalid frame interval, cache size, or search limit")
    if not math.isfinite(args.bottom_quartile_weight) or args.bottom_quartile_weight < 0:
        parser.error("--bottom-quartile-weight must be finite and nonnegative")
    if not math.isfinite(args.mismatch_window_weight) or args.mismatch_window_weight < 0:
        parser.error("--mismatch-window-weight must be finite and nonnegative")
    if not math.isfinite(args.queue_geometry_weight) or args.queue_geometry_weight < 0:
        parser.error("--queue-geometry-weight must be finite and nonnegative")
    if (any(width < 1 for width in args.branch_widths)
            or any(b <= a for a, b in zip(args.branch_widths, args.branch_widths[1:]))):
        parser.error("--branch-widths must be positive and strictly increasing")
    if args.branch_stage_expansions is not None and args.branch_stage_expansions < 1:
        parser.error("--branch-stage-expansions must be positive")
    queue_geometry_weight = 0. if args.ignore_queue_geometry else args.queue_geometry_weight
    if args.turn_handoff_bonus is not None and (not math.isfinite(args.turn_handoff_bonus)
                                                 or args.turn_handoff_bonus < 0):
        parser.error("--turn-handoff-bonus must be finite and nonnegative")
    try:
        ZoneInfo(args.console_timezone)
    except (KeyError, ValueError):
        parser.error("Unknown console timezone")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in
           ('config.json', 'result.json', 'progress.jsonl', 'search.sqlite3', 'observations.jsonl', 'even', 'odd')):
        parser.error("Use a fresh output directory; previous inputs and searches are preserved")

    import torch
    torch.set_num_threads(2)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; use --device cpu")
    renderer = replay.load_renderer()
    try:
        records, metadata = replay.load_nine_class_records(args, renderer)
        if args.turning_start_frame is None:
            args.turning_start_frame = metadata.get('turning_start_frame')
        end = max(records) if args.end_frame is None else args.end_frame
        frames = tuple(range(args.start_frame, end + 1))
        if not frames or any(frame not in records for frame in frames):
            raise ValueError("Source observations do not cover the requested consecutive frame interval")
        video_count = args.video_frame_count if args.video_frame_count is not None else max(records) + 1
        if video_count < max(records) + 1 or not math.isfinite(args.fps) or args.fps <= 0:
            raise ValueError("Invalid video frame count or FPS")
        sampled = any('original_source_frame' in record for record in records.values())
        passes = frame_passes(records, frames, args.frame_parity, args.fps, already_sampled=sampled)
        if not passes:
            raise ValueError("No frames match the requested parity")
        render_source = args.render_source or args.source or metadata.get('source')
        if (args.save_final_video or args.display) and not render_source:
            raise ValueError("Rendering needs the original video; specify --render-source")
        if args.save_final_video and Path(render_source).resolve() == args.save_final_video.resolve():
            raise ValueError("Output video must differ from the source video")
        colors = replay.initial_sticker_colors(renderer, args.initial_scramble)
        mode = ('prepared' if args.device == 'cpu' else 'legacy') if args.scorer == 'auto' else args.scorer
        cls = replay.PreparedReplayScorer if mode == 'prepared' else replay.ReplayScorer
        checkpoint = metadata.get('pose_model')
        use_26m_confusions = checkpoint and Path(checkpoint).resolve() == renderer.DEFAULT_POSE_MODEL.resolve()
        confusion_table = renderer.POSE_26M_COLOR_CONFUSIONS if use_26m_confusions else renderer.COLOR_CLASS_CONFUSIONS
        scorer_options = {}
        if mode == 'prepared':
            scorer_options['parallel_layers'] = bool(args.branch_widths)
        if use_26m_confusions:
            scorer_options.update(color_confusion_scoring=True, tracked_color_confusions=confusion_table)
        if args.mismatch_window_weight:
            if mode != 'prepared':
                parser.error("--mismatch-window-weight requires the prepared CPU scorer")
            scorer_options['track_color_confusions'] = True
        scorer = cls(renderer, records, args.device, cache_size=args.score_cache_size,
                     initial_colors=colors, turning_start_frame=args.turning_start_frame,
                     balance_geometry_color=True,
                     track_sticker_observations=bool(args.bottom_quartile_weight), **scorer_options)
        initial = scorer.state_from_key(scorer.key(renderer.CubeGeometryState.identity()))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))

    common_config = dict(algorithm='best_first_latest_frame', log=str(args.log.resolve()),
        log_sha256=hashlib.sha256(args.log.read_bytes()).hexdigest(), observation_input=metadata,
        initial_scramble=args.initial_scramble, turning_start_frame=args.turning_start_frame,
        priority_statistic='latest_frame_geometric_plus_color_cost',
        priority_ties='greater depth, then insertion order',
        color_scoring='direct_match_fraction', score_weights=dict(geometry=.5, color=.5),
        queue_score_weights=dict(geometry=queue_geometry_weight, color=.5),
        state_limit=args.max_explored_states, time_budget=args.search_seconds,
        input_fps=args.fps, video_frame_count=video_count, min_state_update_gap_frames=1,
        scorer=mode, device=args.device,
        merge_key='frame,physical_state', reference_solution_used=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.branch_widths:
        common_config['branch_widths'] = args.branch_widths
    if use_26m_confusions:
        common_config.update(color_scoring='directional_confusion_cost',
            color_confusion_cost=scorer.color_confusion_cost,
            color_confusions={color: sorted(allowed) for color, allowed in confusion_table.items() if allowed})
    if args.bottom_quartile_weight:
        common_config.update(priority_statistic='latest_frame_fit_plus_bottom_quartile_penalty',
            bottom_quartile_weight=args.bottom_quartile_weight,
            bottom_quartile_history='all scored frames, lowest ceil(observed sticker identities / 4) lifetime rates',
            dominance_policy='keep the lowest priority path and its complete history for each frame and state')
    if queue_geometry_weight == 0.:
        common_config['priority_statistic'] = 'latest_frame_color_cost'
        if args.bottom_quartile_weight:
            common_config['priority_statistic'] += '_plus_bottom_quartile_penalty'
    if args.mismatch_window_weight:
        common_config.update(mismatch_window_weight=args.mismatch_window_weight,
            mismatch_window_frames=COLOR_MISMATCH_WINDOW_FRAMES,
            mismatch_window_confusions={color: sorted(allowed) for color, allowed in confusion_table.items() if allowed},
            mismatch_window_statistic='pooled out-of-table mismatches / observed stickers',
            mismatch_window_activation='after eight sampled frames, starting with the first scored observation',
            dominance_policy='keep the lowest priority path and its history for each merge key')
        common_config['priority_statistic'] += '_plus_mismatch_window_penalty'
    if args.turn_handoff_bonus is not None:
        common_config.update(turn_handoff_bonus=args.turn_handoff_bonus,
            turn_handoff_min_degrees=TURN_HANDOFF_MIN_DEGREES,
            merge_key='frame,physical_state,next_turn_completion')
        common_config['priority_statistic'] += '_minus_turn_handoff_bonus'
    summaries = {}
    winner = None
    for name, selected, fps in passes:
        directory = args.output_dir / name if len(passes) > 1 else args.output_dir
        config = dict(common_config, frame_parity=name, sampled_frame_indices=selected,
                      start_frame=selected[0], end_frame=selected[-1], fps=fps)
        if args.branch_widths:
            static_count = bisect_right(selected, args.turning_start_frame) if args.turning_start_frame is not None else 0
            config['branch_stage_expansions'] = branch_stage_budget(
                len(selected) - static_count, args.branch_stage_expansions)
        print(f"Preparing {name} pass: {len(selected)} frames at {fps:.3f} fps", flush=True)
        try:
            for frame in selected:
                scorer.prepare(frame)
        except (ValueError, OSError, KeyError) as error:
            parser.error(str(error))
        result = run_pass(args, directory, selected, scorer, initial, colors, config)
        summaries[name] = dict(status=result['status'], result=str((directory / 'result.json').resolve()),
                               solved_frame_index=result['solved_frame_index'])
        if winner is None and result['status'] in {'solved', 'already_solved'}:
            winner = directory, result
        if result['status'] == 'interrupted':
            break
    if len(passes) > 1:
        summary = dict(passes=summaries, first_solved_parity=winner[1]['config']['frame_parity'] if winner else None)
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if winner and (args.save_final_video or args.display):
        directory, result = winner
        spec = importlib.util.spec_from_file_location(
            "full_search_video", ROOT / "scripts/render_independent_keypoint_search.py")
        video = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = video
        spec.loader.exec_module(video)
        report = video.render_result(renderer, records, result, render_source,
            output_path=args.save_final_video, display=args.display, show_keypoints=args.show_keypoints,
            show_orientation_axes=args.show_orientation_axes, show_virtual_cube=args.show_virtual_cube,
            turn_log=directory / 'rendered_turn_candidate_scores.jsonl')
        (directory / 'render.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == "__main__":
    main()
