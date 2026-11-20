#!/usr/bin/env python3
"""Full-solve best-first search ordered by a uniform-mean greedy forecast.

Each candidate's priority averages its current-frame cost and the next 19
greedy costs by default. Only the candidate's first step enters the actual
path. Forecast failures receive infinite priority, never prune a candidate.
There is no error gate, beam, move penalty, or work budget. All finite legal
choices remain searchable, subject to exact-context cost dominance.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
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
    "forecast_search_queue", ROOT / "scripts/search_independent_keypoint_mean.py")
storage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = storage
SPEC.loader.exec_module(storage)
replay = storage.replay


def canonical_update(update, next_frame, gap):
    """Forget move history as soon as it cannot constrain any future update."""
    last, _key, _streak = update
    if gap <= 1 or last is None or next_frame is None or next_frame-last >= gap:
        return None, None, 0
    return update


def advance_update(update, choice, frame, gap):
    if not choice.payload.get("state_moves"):
        return update
    last, key, streak = update
    consecutive = last is not None and frame-last == 1
    continuation = consecutive and streak < 2 and choice.update_key is not None and choice.update_key == key
    if last is not None and frame-last < gap and not continuation:
        return None
    return frame, choice.update_key, streak+1 if consecutive and choice.update_key == key else 1


def legal_choices(transitions, frame, state, update, gap):
    choices, skipped = replay.transitions_for_update_context(transitions, frame, state, update, gap)
    result = []
    invalid = 0
    for choice in choices:
        if not math.isfinite(choice.score):
            invalid += 1
            continue
        after = advance_update(update, choice, frame, gap)
        if after is None:
            skipped += 1
        else:
            result.append((choice, after))
    result.sort(key=lambda item: item[0].rank)
    return result, skipped, invalid


@dataclass(frozen=True)
class Forecast:
    total: float = 0.
    count: int = 0
    complete: bool = True


class GreedyForecaster:
    def __init__(self, frames, transitions, state_key, *, gap=2, cache_size=65536,
                 heartbeat=lambda: None):
        if gap < 1 or cache_size < 0:
            raise ValueError("Invalid forecast gap or cache size")
        self.frames, self.transitions, self.state_key = frames, transitions, state_key
        self.gap, self.cache_size, self.heartbeat = gap, cache_size, heartbeat
        self.cache = OrderedDict()
        self.calls = self.cache_hits = self.state_evaluations = self.dead_ends = 0

    def rollout(self, depth, state, update, count):
        self.calls += 1
        remaining = min(count, len(self.frames)-depth)
        trail = []
        result = Forecast()
        while remaining > 0:
            self.heartbeat()
            frame = self.frames[depth]
            update = canonical_update(update, frame, self.gap)
            key = depth, self.state_key(state), update, remaining
            if key in self.cache:
                self.cache_hits += 1
                self.cache.move_to_end(key)
                result = self.cache[key]
                break
            self.state_evaluations += 1
            choices, _skipped, _invalid = legal_choices(
                self.transitions, frame, state, update, self.gap)
            if not choices:
                self.dead_ends += 1
                result = Forecast(complete=False)
                trail.append((key, None))
                break
            choice, update = choices[0]
            trail.append((key, choice.score))
            state = choice.state
            depth += 1
            remaining -= 1
        for key, score in reversed(trail):
            if score is not None:
                result = Forecast(result.total+score, result.count+1, result.complete)
            if self.cache_size:
                self.cache[key] = result
                self.cache.move_to_end(key)
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        return result

    def candidate(self, depth, choice, update, horizon):
        tail = self.rollout(depth+1, choice.state, update, horizon-1)
        result = Forecast(choice.score+tail.total, tail.count+1, tail.complete)
        return (result.total/result.count if result.complete else float("inf")), result

    def metrics(self):
        return dict(forecast_calls=self.calls, forecast_cache_hits=self.cache_hits,
                    forecast_state_evaluations=self.state_evaluations,
                    forecast_dead_ends=self.dead_ends, forecast_cache_entries=len(self.cache))


def finite_or_none(value):
    return value if value is not None and math.isfinite(value) else None


def forecast_full_solve(initial_state, frames, transitions, solved, state_key, *,
                        horizon=20, min_update_gap=2, forecast_cache_size=65536,
                        queue=None, serialize_state=pickle.dumps, deserialize_state=pickle.loads,
                        progress=None, progress_seconds=10.):
    if (not frames or any(b != a+1 for a, b in zip(frames, frames[1:]))
            or horizon < 1 or min_update_gap < 1 or forecast_cache_size < 0
            or not math.isfinite(progress_seconds) or progress_seconds <= 0):
        raise ValueError("Require consecutive frames, positive horizon/gap/interval and nonnegative cache")
    if queue is None:
        queue = storage.BufferedMeanQueue()
        try:
            return forecast_full_solve(initial_state, frames, transitions, solved, state_key,
                horizon=horizon, min_update_gap=min_update_gap, forecast_cache_size=forecast_cache_size,
                queue=queue, serialize_state=serialize_state, deserialize_state=deserialize_state,
                progress=progress, progress_seconds=progress_seconds)
        finally:
            queue.close()
    if queue.context_count or queue.deferred_count:
        raise ValueError("Start in a fresh search database")
    stats = dict(evaluated_states=0, attempted_transitions=0, invalid_pruned=0, spacing_pruned=0,
        dominated_pruned=0, reopened_states=0, improved_pending=0, unsolved_endpoints=0,
        furthest_frames=0, furthest_frame_index=None, current_frames=0, current_frame_index=None,
        current_path_mean=0., current_forecast_mean=None, peak_pending=0, error_gate_enabled=False)
    started = last_heartbeat = time.monotonic()

    def report(event, **extra):
        if progress:
            best = queue.minimum()
            progress(dict(event=event, elapsed_seconds=time.monotonic()-started, **stats,
                queue_pending=queue.pending_count, stored_contexts=queue.context_count,
                best_pending_forecast_mean=finite_or_none(best),
                best_pending_forecast_incomplete=best is not None and not math.isfinite(best),
                **(queue.cache_stats() if hasattr(queue, "cache_stats") else {}),
                **forecaster.metrics(), **extra))

    def heartbeat():
        nonlocal last_heartbeat
        if time.monotonic()-last_heartbeat >= progress_seconds:
            last_heartbeat = time.monotonic()
            report("progress")

    forecaster = GreedyForecaster(frames, transitions, state_key, gap=min_update_gap,
        cache_size=forecast_cache_size, heartbeat=heartbeat)

    def context(node):
        next_frame = frames[node.depth] if node.depth < len(frames) else None
        return pickle.dumps(("greedy-forecast-v1", node.depth, state_key(node.state),
                             canonical_update(node.update, next_frame, min_update_gap)), protocol=5)

    def outcome(status, node=None):
        report(status)
        return dict(status=status, elapsed_seconds=time.monotonic()-started, **stats,
            **forecaster.metrics(), queue_pending=queue.pending_count, stored_contexts=queue.context_count,
            steps=queue.steps(node.path_id) if node else [],
            solved_frame_index=frames[node.depth-1] if node and node.depth else None)

    if solved(initial_state):
        return outcome("already_solved")
    root = storage.MeanNode(0, initial_state, 0., (None, None, 0), priority=0.)
    queue.offer(context(root), root, serialize_state(initial_state))
    stats["peak_pending"] = 1
    report("started")
    while queue.pending_count:
        heartbeat()
        node = queue.pop(deserialize_state)
        frame = frames[node.depth-1] if node.depth else None
        stats.update(current_frames=node.depth, current_frame_index=frame,
                     current_path_mean=node.mean, current_forecast_mean=finite_or_none(node.priority))
        if node.depth > stats["furthest_frames"]:
            stats.update(furthest_frames=node.depth, furthest_frame_index=frame)
            steps = queue.steps(node.path_id)
            report("furthest", last_score=steps[-1]["state_score"], furthest_path_mean=node.mean,
                   furthest_forecast_mean=finite_or_none(node.priority),
                   frontier_moves=[dict(frame_index=s["frame_index"], state_moves=s["state_moves"])
                                   for s in steps if s["state_moves"]])
        if solved(node.state):
            return outcome("solved", node)
        if node.depth == len(frames):
            stats["unsolved_endpoints"] += 1
            continue
        stats["evaluated_states"] += 1
        frame = frames[node.depth]
        choices, skipped, invalid = legal_choices(transitions, frame, node.state, node.update, min_update_gap)
        stats["spacing_pruned"] += skipped
        stats["invalid_pruned"] += invalid
        for choice, update in choices:
            heartbeat()
            stats["attempted_transitions"] += 1
            total = node.total_cost+choice.score
            if not math.isfinite(total):
                stats["invalid_pruned"] += 1
                continue
            priority, forecast = forecaster.candidate(node.depth, choice, update, horizon)
            child = storage.MeanNode(node.depth+1, choice.state, total, update, node.path_id,
                                     priority=priority)
            keep = ("trial_id", "state_moves", "color_error", "keypoint_mean_error",
                    "heatmap_mean_error", "geometry_option_index")
            step = {key: choice.payload[key] for key in keep if key in choice.payload}
            step.update(frame_index=frame, state_score=choice.score, path_total_cost=total,
                whole_path_mean=child.mean, forecast_mean=finite_or_none(priority),
                forecast_frames=forecast.count, forecast_complete=forecast.complete)
            status = queue.offer(context(child), child, serialize_state(choice.state), step)
            if status is None:
                stats["dominated_pruned"] += 1
            elif status == "reopened":
                stats["reopened_states"] += 1
            elif status == "improved_pending":
                stats["improved_pending"] += 1
            stats["peak_pending"] = max(stats["peak_pending"], queue.pending_count)
    return outcome("exhausted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--initial-scramble", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--video-frame-count", type=int)
    parser.add_argument("--fps", type=float, default=60.)
    parser.add_argument("--min-state-update-gap", type=int)
    parser.add_argument("--forecast-frames", type=int, default=20)
    parser.add_argument("--forecast-cache-size", type=int, default=65536)
    parser.add_argument("--score-cache-size", type=int, default=65536)
    parser.add_argument("--frontier-cache-size", type=int, default=4096)
    parser.add_argument("--cost-cache-size", type=int, default=131072)
    parser.add_argument("--sqlite-cache-mib", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-seconds", type=float, default=10.)
    parser.add_argument("--console-timezone", default="America/Chicago")
    args = parser.parse_args()
    if (args.forecast_frames < 1 or min(args.forecast_cache_size, args.score_cache_size, args.cost_cache_size) < 0
            or args.frontier_cache_size < 1 or args.sqlite_cache_mib < 1
            or args.end_frame < args.start_frame or not math.isfinite(args.fps) or args.fps <= 0
            or not math.isfinite(args.progress_seconds) or args.progress_seconds <= 0
            or args.min_state_update_gap is not None and args.min_state_update_gap < 1):
        parser.error("Invalid frame interval, forecast/cache size, FPS, spacing or progress interval")
    try:
        ZoneInfo(args.console_timezone)
    except (KeyError, ValueError):
        parser.error("Unknown console timezone")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir/name).exists() for name in ("config.json", "progress.jsonl", "search.sqlite3")):
        parser.error("Use a new output directory; previous searches are preserved")
    m = replay.load_renderer()
    import torch
    torch.set_num_threads(2)
    records = {r["frame_index"]: r for r in map(json.loads, args.log.open())}
    if not records:
        parser.error("Source log contains no frames")
    frames = tuple(range(args.start_frame, args.end_frame+1))
    if any(frame not in records for frame in frames):
        parser.error("Source log does not cover the requested interval")
    video_count = args.video_frame_count if args.video_frame_count is not None else max(records)+1
    if video_count < max(records)+1:
        parser.error("Whole-video length cannot be smaller than source-log extent")
    scorer = replay.PreparedReplayScorer(m, records, "cpu", cache_size=args.score_cache_size)
    for frame in frames:
        scorer.prepare(frame)
    initial = m.CubeGeometryState.identity()
    scrambled = m.CubeState.from_scramble(args.initial_scramble)
    colors = {label: scrambled.colors[position] for position, label in initial.facelets.items()}
    initial = scorer.key(initial)
    gap = m.turn_state_update_gap_frames(args.fps) if args.min_state_update_gap is None else args.min_state_update_gap
    config = dict(algorithm="best_first_greedy_forecast", log=str(args.log.resolve()),
        initial_scramble=args.initial_scramble, start_frame=args.start_frame, end_frame=args.end_frame,
        video_frame_count=video_count, fps=args.fps, min_state_update_gap_frames=gap,
        forecast_frames=args.forecast_frames, forecast_includes_candidate_frame=True,
        forecast_policy="lowest combined-error legal transition at each future frame; no forecast backtracking",
        priority_statistic="uniform mean of candidate cost and greedy future costs; no past-path term",
        priority_ties="greater depth, then insertion order", forecast_failure_policy="infinite priority; retain branch",
        end_of_video_policy="average available frames only", error_gate=None, threshold=None, move_penalty=0.,
        frame_or_state_budget=None, time_budget=None, beam_width=None, state_change_limit=None,
        duplicate_detection=True, merge_key="frame,physical_cube_state,unexpired_move_spacing_context",
        dominance_policy="lowest actual cumulative cost; reopen on improvement",
        forecast_cache_size=args.forecast_cache_size, score_cache_size=args.score_cache_size,
        queue_backend="buffered", frontier_cache_size=args.frontier_cache_size,
        cost_cache_size=args.cost_cache_size, sqlite_cache_mib=args.sqlite_cache_mib,
        console_interval_seconds=args.progress_seconds, console_timezone=args.console_timezone,
        goal="physical cube colors solved in any whole-cube orientation", reference_solution_used=False,
        log_sha256=hashlib.sha256(args.log.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        queue_script_sha256=hashlib.sha256(Path(storage.__file__).read_bytes()).hexdigest(),
        scorer_script_sha256=hashlib.sha256(Path(replay.__file__).read_bytes()).hexdigest(),
        renderer_script_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest())
    (args.output_dir/"config.json").write_text(json.dumps(config, indent=2))
    queue = storage.BufferedMeanQueue(args.output_dir/"search.sqlite3",
        frontier_cache_size=args.frontier_cache_size, cost_cache_size=args.cost_cache_size,
        sqlite_cache_mib=args.sqlite_cache_mib)
    latest = {}
    with (args.output_dir/"progress.jsonl").open("w", buffering=1) as log:
        def progress(item):
            now = datetime.now(timezone.utc)
            item.update(timestamp_utc=now.isoformat(), total_video_frames=video_count,
                furthest_video_frame=item["furthest_frame_index"]+1 if item["furthest_frame_index"] is not None else 0,
                scoring_calls=scorer.scoring_calls, score_cache_hits=scorer.cache_hits,
                color_configurations_scored=scorer.color_configurations_scored,
                child_states_built=scorer.child_states_built)
            latest.clear()
            latest.update(item)
            log.write(json.dumps(item, allow_nan=False)+"\n")
            temporary = args.output_dir/"status.tmp"
            temporary.write_text(json.dumps(item, indent=2, allow_nan=False))
            temporary.replace(args.output_dir/"status.json")
            if item["event"] in {"started", "progress", "solved", "already_solved", "exhausted"}:
                print(storage.format_console_progress(item, args.console_timezone, now), flush=True)
            if (args.output_dir/"STOP").exists():
                raise KeyboardInterrupt
        try:
            result = forecast_full_solve(initial, frames, scorer, scorer.solved_predicate(colors), scorer.key,
                horizon=args.forecast_frames, min_update_gap=gap, forecast_cache_size=args.forecast_cache_size,
                queue=queue, serialize_state=scorer.key, deserialize_state=scorer.state_from_key,
                progress=progress, progress_seconds=args.progress_seconds)
            (args.output_dir/"result.json").write_text(json.dumps(dict(config=config, **result), indent=2, allow_nan=False))
        except KeyboardInterrupt:
            item = dict(latest, event="interrupted", status="interrupted")
            log.write(json.dumps(item, allow_nan=False)+"\n")
            (args.output_dir/"status.json").write_text(json.dumps(item, indent=2, allow_nan=False))
            (args.output_dir/"result.json").write_text(json.dumps(dict(config=config, **item), indent=2, allow_nan=False))
            print(storage.format_console_progress(item, args.console_timezone)+" | stopped", flush=True)
        finally:
            queue.close()


if __name__ == "__main__":
    main()
