"""Run the standalone full-solve search on the renderer's cached poses."""
from __future__ import annotations

from dataclasses import replace
import json
from math import isfinite

from . import render_independent_keypoint_search as playback
from . import search_independent_keypoint_bidirectional as bidirectional


frame_search = bidirectional.frame_search
replay = frame_search.replay


def find_solved_anchor(renderer, records, frames, goals, scorer_options, source_fps):
    """Find the start of the last cluster of corroborating flat solved views."""
    max_gap = max(2, round(.15 * source_fps))
    cluster = []
    for frame in reversed(frames):
        if cluster and cluster[-1] - frame > max_gap:
            if len(cluster) >= 2:
                return cluster[-1]
            cluster.clear()
        if bidirectional.flat_solved_anchor(renderer, records[frame], goals, **scorer_options) is not None:
            cluster.append(frame)
    return cluster[-1] if len(cluster) >= 2 else None


def search_cached_records(renderer, records, args, colors, source_fps):
    """Use the same expansion, scoring, and widening code as the standalone tools."""
    scorer_options = bidirectional.widening_scorer_options(
        renderer, colors, args.turning_start_frame, args.model,
    )
    scorer = replay.PreparedReplayScorer(renderer, records, "cpu", **scorer_options)
    initial = scorer.key(renderer.CubeGeometryState.identity())
    solved = scorer.solved_predicate(colors)
    if solved(initial):
        return dict(status="already_solved", steps=[], solved_frame_index=None)
    frames = tuple(index for index in sorted(records)
                   if not renderer._frame_forces_static_hold(index, args.turning_start_frame))
    if not frames:
        return dict(status="exhausted", steps=[], solved_frame_index=None)

    goals = bidirectional.solved_seeds(scorer, args.initial_scramble)
    parity = args.frame_parity
    if parity == "auto":
        parity = "both" if source_fps >= 50. else "all"
    passes = frame_search.frame_passes(records, frames, parity, source_fps)
    requested_anchor = args.solved_anchor_frame
    if requested_anchor is not None:
        passes = [item for item in passes if requested_anchor in item[1]]
        if not passes or bidirectional.flat_solved_anchor(
                renderer, records[requested_anchor], goals, **scorer_options) is None:
            raise ValueError("--solved-anchor-frame needs a sampled, flat solved frame after the static holds")

    def progress(item):
        event = item["event"]
        if event == "widened":
            direction = f" {item['direction']}" if "direction" in item else ""
            print(f"Widening{direction} search to {item['branch_width']} children per parent.", flush=True)
        if event != "furthest":
            print(f"Full search {event}: {item['evaluated_states']:,} expanded "
                  f"in {item['elapsed_seconds']:.1f}s", flush=True)

    def forward_progress(item):
        progress(item)
        if (item["elapsed_seconds"] >= args.search_seconds
                and item["event"] in {"started", "progress", "furthest", "widened"}):
            raise frame_search.SearchTimeLimit

    result = dict(status="exhausted", steps=[], solved_frame_index=None)
    schedule = " → ".join(map(str, args.branch_widths))
    for name, selected, fps in passes:
        anchor = requested_anchor
        if anchor is None:
            print(f"Checking {name} frames for a flat solved anchor...", flush=True)
            anchor = find_solved_anchor(renderer, records, selected, goals, scorer_options, source_fps)
        search_frames = tuple(frame for frame in selected if anchor is None or frame <= anchor)
        mode = f"bidirectional, solved anchor {anchor}" if anchor is not None else "forward"
        print(f"Full search {name} pass ({mode}, {len(search_frames)} frames at {fps:.3f} fps, "
              f"widths {schedule})", flush=True)
        for index, frame in enumerate(search_frames, start=1):
            scorer.prepare(frame)
            if not args.display:
                renderer.print_headless_progress("compiling search observations", index, len(search_frames))
        if not args.display:
            print()
        if anchor is not None:
            result = bidirectional.bidirectional_search(
                scorer, search_frames, initial, goals, width=16, retries=0, overlap=0,
                time_limit=args.search_seconds, state_limit=args.turn_search_state_budget,
                handoff_bonus=bidirectional.WIDENING_HANDOFF_BONUS, progress=progress,
                branch_widths=args.branch_widths, branch_stage_expansions=args.branch_stage_expansions,
                mismatch_window_weight=bidirectional.WIDENING_MISMATCH_WEIGHT,
            )
        else:
            try:
                result = frame_search.frame_full_solve(
                    initial, search_frames, scorer, solved, scorer.key,
                    serialize_state=scorer.key, deserialize_state=scorer.state_from_key,
                    progress=forward_progress, max_explored_states=args.turn_search_state_budget,
                    turn_handoff_bonus=bidirectional.WIDENING_HANDOFF_BONUS,
                    ignore_queue_geometry=True, mismatch_window_weight=bidirectional.WIDENING_MISMATCH_WEIGHT,
                    branch_widths=args.branch_widths, branch_stage_expansions=args.branch_stage_expansions,
                )
            except frame_search.SearchTimeLimit:
                result = dict(status="time_limit", steps=[], solved_frame_index=None)
        if result["status"] == "solved":
            break
    return result


def apply_search_result(renderer, cache, records, result, colors, turning_start_frame, stream):
    """Materialize the accepted search path on the original, full-rate frames."""
    state = renderer.CubeGeometryState.identity()
    solved = result["status"] in {"solved", "already_solved"}
    steps = {step["frame_index"]: step for step in result["steps"]} if solved else {}
    if solved:
        moves = tuple(move for frame in sorted(steps) for move in steps[frame].get("state_moves", ()))
        final_state = state.apply_moves(moves)
        if not final_state.is_solved(colors):
            raise RuntimeError("The accepted full-search path did not solve the supplied scramble")
    goal = result["solved_frame_index"]
    if result["status"] == "already_solved":
        goal = cache[0].frame_index
    for record in cache:
        frame = record.frame_index
        step = steps.get(frame)
        source = records[frame]
        before = record.pose_candidate
        moves = tuple(step.get("state_moves", ())) if step else ()
        diagnostics = dict(source.get("search") or {})
        observation_status = diagnostics.get("status", "unavailable_geometry")
        diagnostics.update(status="full_search_winning_path" if step else result["status"],
                           observation_status=observation_status,
                           turning_start_frame=turning_start_frame)
        candidate = before
        if before is not None:
            candidate = replace(before, cube_geometry_state=state)
            if step and "geometry_option_index" in step:
                candidate = playback.candidate_for_step(renderer, source, step, state, colors)
                diagnostics.update(handoff_commit=bool(step.get("handoff_commit")),
                                   sibling_rank=step.get("sibling_rank"), queue_priority=step["queue_priority"])
                candidate = replace(candidate, turn_fit_evaluated=True, turn_candidate_diagnostics=(dict(step),))
            elif goal is not None and frame > goal:
                candidate = renderer.match_orientation_to_colors(candidate, colors)
            if renderer._frame_forces_static_hold(frame, turning_start_frame):
                diagnostics["status"] = "forced_static"
            if goal is not None and frame >= goal:
                if frame > goal:
                    diagnostics["status"] = "solved_hold"
                diagnostics.update(stopped=True, stop_reason="cube_solved", solved_frame_index=goal)
            # The options are only needed by the scorer and log, not the cached overlays.
            candidate = replace(candidate, pose_continuity_diagnostics=before.pose_continuity_diagnostics,
                                turn_search_diagnostics={
                                    key: value for key, value in diagnostics.items() if key != "geometry"
                                })
        if moves and candidate is None:
            raise RuntimeError(f"Full search returned a move without a pose at frame {frame}")
        record.pose_candidate = candidate
        record.pose_candidates = (candidate,) if candidate is not None else ()
        record.pose_ready = True
        if stream is not None:
            logged = replace(candidate, cube_geometry_state=state) if candidate is not None else None
            payload = renderer.turn_candidate_log_record(logged, frame)
            payload.update(search_pose=source["search_pose"], search=diagnostics,
                           turn_color_matches=source["turn_color_matches"],
                           turn_recovery_mode="full_search")
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        state = state.apply_moves(moves)


def run_cached_full_search(renderer, cache, args, colors, source_fps):
    if not cache:
        return
    import torch

    torch.set_num_threads(2)
    source_fps = source_fps if isfinite(source_fps) and source_fps > 0 else 30.
    device = torch.device("cpu")
    records = {}
    already_solved = renderer.CubeGeometryState.identity().is_solved(colors)
    for index, record in enumerate(cache, start=1):
        if already_solved:
            observation = renderer.turn_candidate_log_record(record.pose_candidate, record.frame_index)
        else:
            observation = replay.prepare_search_record(
                renderer, record, colors, args.turning_start_frame, device,
            )
            if not args.display:
                renderer.print_headless_progress("preparing search observations", index, len(cache))
        records[record.frame_index] = observation
    if not already_solved and not args.display:
        print()
    result = search_cached_records(renderer, records, args, colors, source_fps)
    if args.turn_candidate_log is None:
        apply_search_result(renderer, cache, records, result, colors, args.turning_start_frame, None)
    else:
        with args.turn_candidate_log.open("a", encoding="utf-8") as stream:
            apply_search_result(renderer, cache, records, result, colors, args.turning_start_frame, stream)
    goal = result["solved_frame_index"]
    outcome = f"solved at frame {goal}" if goal is not None else result["status"].replace("_", " ")
    print(f"Full search result: {outcome}.", flush=True)
    return result
