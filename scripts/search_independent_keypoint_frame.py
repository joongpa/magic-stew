#!/usr/bin/env python3
"""Full-solve best-first search prioritized only by the latest frame's fit.

Every queued branch competes by its latest geometric + categorical color cost.
Branches below 2/3 color accuracy are pruned. There is no path average,
forecast, combined-score gate, beam, or expansion budget.
Geometry and nine-class observations stay fixed during the state search.
"""
from __future__ import annotations

import argparse
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


def frame_full_solve(initial_state, frames, transitions, solved, state_key, *,
                     min_update_gap=2, queue=None, serialize_state=pickle.dumps,
                     deserialize_state=pickle.loads, progress=None, progress_seconds=10.,
                     minimum_color_accuracy=2./3.):
    """Expand the globally cheapest latest-frame branch, testing goals on pop.

    The shared queue's dominance cost is also the latest frame score. Past path
    costs cannot discard a better-priority branch at an equivalent future context.
    Only frame, physical state, and unexpired move-spacing context affect merging.
    """
    if (not frames or any(b != a + 1 for a, b in zip(frames, frames[1:]))
            or min_update_gap < 1 or not math.isfinite(progress_seconds) or progress_seconds <= 0
            or not math.isfinite(minimum_color_accuracy) or not 0. <= minimum_color_accuracy <= 1.):
        raise ValueError("Require consecutive frames, a positive move gap and progress interval")
    if queue is None:
        queue = storage.BufferedMeanQueue()
        try:
            return frame_full_solve(initial_state, frames, transitions, solved, state_key,
                min_update_gap=min_update_gap, queue=queue, serialize_state=serialize_state,
                deserialize_state=deserialize_state, progress=progress, progress_seconds=progress_seconds,
                minimum_color_accuracy=minimum_color_accuracy)
        finally:
            queue.close()
    if queue.context_count or queue.deferred_count:
        raise ValueError("Start in a fresh search database")
    stats = dict(evaluated_states=0, attempted_transitions=0, invalid_pruned=0, color_pruned=0,
        spacing_pruned=0, dominated_pruned=0, reopened_states=0, improved_pending=0,
        unsolved_endpoints=0, furthest_frames=0, furthest_frame_index=None,
        current_frames=0, current_frame_index=None, current_frame_score=None,
        peak_pending=0, error_gate_enabled=False, minimum_color_accuracy=minimum_color_accuracy)
    started = last_heartbeat = time.monotonic()

    def report(event, **extra):
        if progress:
            progress(dict(event=event, elapsed_seconds=time.monotonic() - started, **stats,
                queue_pending=queue.pending_count, stored_contexts=queue.context_count,
                best_pending_frame_score=queue.minimum(),
                **(queue.cache_stats() if hasattr(queue, "cache_stats") else {}), **extra))

    def heartbeat():
        nonlocal last_heartbeat
        if time.monotonic() - last_heartbeat >= progress_seconds:
            last_heartbeat = time.monotonic()
            report("progress")

    def context(node):
        next_frame = frames[node.depth] if node.depth < len(frames) else None
        return pickle.dumps(("latest-frame-v1", node.depth, state_key(node.state),
            canonical_update(node.update, next_frame, min_update_gap)), protocol=5)

    def outcome(status, node=None):
        report(status)
        return dict(status=status, elapsed_seconds=time.monotonic() - started, **stats,
            queue_pending=queue.pending_count, stored_contexts=queue.context_count,
            steps=queue.steps(node.path_id) if node else [],
            solved_frame_index=frames[node.depth - 1] if node and node.depth else None)

    if solved(initial_state):
        return outcome("already_solved")
    root = storage.MeanNode(0, initial_state, 0., (None, None, 0), priority=0.)
    queue.offer(context(root), root, serialize_state(initial_state))
    stats["peak_pending"] = 1
    report("started")
    while queue.pending_count:
        heartbeat()
        node = queue.pop(deserialize_state)
        frame = frames[node.depth - 1] if node.depth else None
        stats.update(current_frames=node.depth, current_frame_index=frame,
                     current_frame_score=node.priority if node.depth else None)
        if node.depth > stats["furthest_frames"]:
            stats.update(furthest_frames=node.depth, furthest_frame_index=frame)
            steps = queue.steps(node.path_id)
            report("furthest", last_score=node.priority,
                   frontier_moves=[dict(frame_index=s["frame_index"], state_moves=s["state_moves"])
                                   for s in steps if s.get("state_moves")])
        # A newly generated solved branch must compete with the entire queue.
        if solved(node.state):
            return outcome("solved", node)
        if node.depth == len(frames):
            stats["unsolved_endpoints"] += 1
            continue
        stats["evaluated_states"] += 1
        frame = frames[node.depth]
        choices, skipped = replay.transitions_for_update_context(
            transitions, frame, node.state, node.update, min_update_gap)
        stats["spacing_pruned"] += skipped
        for choice in choices:
            heartbeat()
            stats["attempted_transitions"] += 1
            if not math.isfinite(choice.score):
                stats["invalid_pruned"] += 1
                continue
            color_error = choice.payload.get("color_error")
            if color_error is not None:
                if not math.isfinite(float(color_error)):
                    stats["invalid_pruned"] += 1
                    continue
                if float(color_error) > 1. - minimum_color_accuracy:
                    stats["color_pruned"] += 1
                    continue
            update = advance_update(node.update, choice, frame, min_update_gap)
            if update is None:
                stats["spacing_pruned"] += 1
                continue
            # The queue uses this value for both priority and same-context
            # dominance. It is deliberately not a cumulative path cost.
            child = storage.MeanNode(node.depth + 1, choice.state, choice.score,
                                     update, node.path_id, priority=choice.score)
            step = dict(choice.payload, frame_index=frame, state_score=choice.score,
                        queue_priority=choice.score)
            status = queue.offer(context(child), child, serialize_state(choice.state), step)
            if status is None:
                stats["dominated_pruned"] += 1
            elif status == "reopened":
                stats["reopened_states"] += 1
            elif status == "improved_pending":
                stats["improved_pending"] += 1
            stats["peak_pending"] = max(stats["peak_pending"], queue.pending_count)
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
            or args.sqlite_cache_mib < 1 or not math.isfinite(args.progress_seconds) or args.progress_seconds <= 0):
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
                     balance_geometry_color=True, minimum_color_accuracy=2./3.)
        for frame in frames:
            scorer.prepare(frame)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    initial = scorer.state_from_key(scorer.key(renderer.CubeGeometryState.identity()))
    gap = (renderer.turn_state_update_gap_frames(args.fps)
           if args.min_state_update_gap is None else args.min_state_update_gap)
    config = dict(algorithm="best_first_latest_frame", log=str(args.log.resolve()),
        log_sha256=hashlib.sha256(args.log.read_bytes()).hexdigest(), observation_input=metadata,
        initial_scramble=args.initial_scramble, start_frame=args.start_frame, end_frame=end,
        turning_start_frame=args.turning_start_frame,
        priority_statistic="latest_frame_geometric_plus_color_cost", priority_ties="greater depth, then insertion order",
        color_scoring="direct_match_fraction", sticker_keypoint_min_confidence=.25,
        score_weights=dict(geometry=.5, color=.5),
        geometric_component_weights=dict(landmarks=.5, sticker_positions=.5),
        geometry_normalization=dict(landmarks=renderer.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION,
                                    sticker_positions=renderer.PNP_TURN_CENTER_PENALTY_SCALE, range=[0., 1.]),
        minimum_color_accuracy=2./3., color_pruning="after orientation rescue and before partial-angle/state collapse",
        error_gate=None, threshold=None, forecast_frames=0, past_path_priority_weight=0.,
        frame_or_state_budget=None, time_budget=None, beam_width=None, state_change_limit=None,
        video_frame_count=video_count, fps=args.fps, min_state_update_gap_frames=gap,
        scorer=mode, device=args.device,
        geometry_scope="all retained source geometry and discrete offsets; no top-k cutoff",
        orientation_policy="24-way state-color rescue below 2/3 accuracy before move search",
        orientation_rescue_minimum_color_accuracy=2./3.,
        merge_key="frame,physical_state,unexpired_move_spacing_context",
        dominance_policy="lowest latest-frame score; past path costs are not used",
        goal="physical cube colors solved in any whole-cube orientation", reference_solution_used=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
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
                scoring_calls=scorer.scoring_calls, score_cache_hits=scorer.cache_hits)
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
                deserialize_state=scorer.state_from_key, progress=progress, progress_seconds=args.progress_seconds)
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
