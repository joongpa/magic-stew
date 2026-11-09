#!/usr/bin/env python3
"""Replay cached cube observations with two bounded beams and deferred batches."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import heapq
from itertools import count
import json
import math
from pathlib import Path
import time

import numpy as np

if __package__:
    from . import search_independent_keypoint_frame as frame_search
    from .independent_keypoint_window_forecast import WindowForecaster
else:
    import search_independent_keypoint_frame as frame_search
    from independent_keypoint_window_forecast import WindowForecaster

replay = frame_search.replay
WIDENING_HANDOFF_BONUS = .02
WIDENING_MISMATCH_WEIGHT = 1.


def inverse(move):
    if move.endswith(("2", "2'")):
        return move.rstrip("'")
    return move[:-1] if move.endswith("'") else move + "'"


@dataclass(frozen=True, slots=True)
class Link:
    state: bytes
    cue: str | None = None
    link: Link | None = None
    step: dict | None = None
    rank: tuple = ()
    window: tuple = ()


class BudgetReached(Exception):
    pass


def solved_seeds(scorer, scramble):
    identity = scorer.key(scorer.m.CubeGeometryState.identity())
    moves = scorer.m.parse_scramble_moves(scramble) if isinstance(scramble, str) else scramble
    solved = scorer.apply_moves(identity, tuple(map(inverse, reversed(moves))))
    encoded = np.frombuffer(solved, dtype=np.uint8)
    return tuple(dict.fromkeys(encoded[sources].tobytes() for sources in scorer.orientation_sources))


def flat_solved_anchor(renderer, record, goals, **scorer_options):
    """Validate the solved boundary against the cached zero-angle geometry."""
    search = record.get("search") or {}
    geometry = search.get("geometry") or {}
    flat_options = [option for option in geometry.get("options", ()) if not option["layer_rotations"]]
    if not flat_options:
        return None
    frame = record["frame_index"]
    flat_record = dict(record, search=dict(search, geometry=dict(geometry, options=flat_options)))
    scorer = replay.PreparedReplayScorer(renderer, {frame: flat_record}, "cpu", **scorer_options)
    candidates = (scorer.score_hold(frame, state) for state in goals)
    return min((hold for hold in candidates if hold is not None
                and hold.payload.get("color_match_count", 0) >= 12
                and hold.payload.get("color_error", 1.) <= .1),
               key=lambda hold: hold.rank, default=None)


def widening_scorer_options(renderer, colors, static_end, checkpoint):
    use_26m = checkpoint is not None and Path(checkpoint).resolve() == renderer.DEFAULT_POSE_MODEL.resolve()
    table = renderer.POSE_26M_COLOR_CONFUSIONS if use_26m else renderer.COLOR_CLASS_CONFUSIONS
    return dict(initial_colors=colors, turning_start_frame=static_end, balance_geometry_color=True,
                joint_orientation_rescue=True, track_color_confusions=True,
                color_confusion_scoring=use_26m, tracked_color_confusions=table)


class WideningFrontier:
    """One direction's best-first queue and deferred siblings."""

    def __init__(self, direction, width_schedule):
        self.direction = direction
        self.width_schedule = width_schedule
        self.stage = 0
        self.stage_start = 0
        self.pending = []
        self.deferred = [[] for _ in range(width_schedule[-1] + 1)]
        self.seen = {}
        self.serials = count()
        self.seconds = 0.

    @property
    def width(self):
        return self.width_schedule[self.stage]

    def offer(self, depth, node):
        contexts = self.seen.setdefault((depth, node.state), {})
        prior = contexts.get(node.cue)
        priority = node.step['queue_priority'] if node.step else 0.
        if prior is not None and (prior.step['queue_priority'] if prior.step else 0.) <= priority:
            return
        contexts[node.cue] = node
        depth_tiebreak = -depth if self.direction == 0 else depth
        heapq.heappush(self.pending, (priority, depth_tiebreak, next(self.serials), depth, node))

    def peek(self):
        while self.pending:
            *_, depth, node = self.pending[0]
            if self.seen[(depth, node.state)][node.cue] is node:
                return True
            heapq.heappop(self.pending)
        return False

    def widen(self, expanded, stage_budget):
        if self.stage + 1 == len(self.width_schedule):
            return False
        if self.peek():
            if expanded - self.stage_start < stage_budget:
                return False
        elif not any(self.deferred):
            return False
        old_width = self.width
        self.stage += 1
        self.stage_start = expanded
        for rank in range(old_width + 1, self.width + 1):
            for depth, child in self.deferred[rank]:
                self.offer(depth, child)
            self.deferred[rank].clear()
        return True


def widening_join(forward, backward, initial, goals, length, *, widths, stage_budget,
                  mismatch_weight, expanded, reached, time_limit_check, state_limit, report):
    def sibling_order(edge):
        return edge.step['state_score'], frame_search.unsupported_state_update(edge.step), edge.rank[-1]

    sides = [WideningFrontier(direction, widths) for direction in range(2)]
    sides[0].offer(0, Link(initial))
    for goal in goals:
        sides[1].offer(length, Link(goal))
    last_report = time.monotonic()
    while True:
        time_limit_check()
        available = []
        for direction, side in enumerate(sides):
            if side.widen(expanded[direction], stage_budget):
                report('widened', direction=('forward', 'backward')[direction], branch_width=side.width,
                       forward_width=sides[0].width, backward_width=sides[1].width)
            if side.peek() or any(side.deferred):
                available.append(direction)
        if not available:
            return None
        if time.monotonic() - last_report >= 3.:
            last_report = time.monotonic()
            report('progress', forward_depth=reached[0], backward_depth=reached[1],
                   forward_width=sides[0].width, backward_width=sides[1].width)
        direction = min(available, key=lambda i: sides[i].seconds)
        side, opposite = sides[direction], sides[1-direction]
        if not side.peek():
            continue
        *_, depth, node = heapq.heappop(side.pending)
        matches = opposite.seen.get((depth, node.state), {})
        if direction == 0:
            reached[0] = max(reached[0], depth)
            match = matches.get(None) or matches.get(node.cue)
        else:
            reached[1] = min(reached[1], depth)
            match = matches.get(node.cue) if node.cue else next(iter(matches.values()), None)
        if match is not None:
            return (depth, node, match) if direction == 0 else (depth, match, node)
        if depth == (length if direction == 0 else 0):
            continue
        if state_limit is not None and sum(expanded) >= state_limit:
            raise BudgetReached('state_limit')
        start = time.monotonic()
        expanded[direction] += 1
        child_depth = depth + (1 if direction == 0 else -1)
        children = (forward if direction == 0 else backward)(node, depth)
        seen = set()
        previous_order = None
        admission_rank = 0
        for edge in sorted(children, key=sibling_order):
            key = edge.state, edge.cue
            if key in seen:
                continue
            seen.add(key)
            rank = len(seen)
            order = sibling_order(edge)
            if order != previous_order:
                admission_rank = rank
                previous_order = order
            if admission_rank > widths[-1]:
                break
            step = dict(edge.step, sibling_rank=rank)
            window = frame_search.update_mismatch_window(node.window, step, mismatch_weight)
            step['queue_priority'] += step['window_mismatch_penalty']
            child = Link(edge.state, edge.cue, node, step, window=window)
            if admission_rank <= side.width:
                side.offer(child_depth, child)
            else:
                side.deferred[admission_rank].append((child_depth, child))
        side.seconds += time.monotonic() - start


def bidirectional_search(scorer, frames, initial, goals, *, width, retries, overlap,
                         time_limit, state_limit, handoff_bonus, progress, physical_key=None,
                         lookahead_frames=0, lookahead_budget=384, branch_widths=(),
                         branch_stage_expansions=None, mismatch_window_weight=1.):
    """Retain a beam and a deferred batch at each frame, then retry from the stack."""
    if not frames or not goals or width < 1 or min(retries, overlap) < 0:
        raise ValueError("Search requires frames, solved seeds and a positive beam width")
    if (any(value < 1 for value in branch_widths)
            or any(b <= a for a, b in zip(branch_widths, branch_widths[1:]))
            or not math.isfinite(mismatch_window_weight)
            or mismatch_window_weight < 0 or branch_widths and lookahead_frames):
        raise ValueError("Invalid branch widening options")
    branch_stage_expansions = frame_search.branch_stage_budget(len(frames), branch_stage_expansions)
    start = time.monotonic()
    expanded = [0, 0]
    layers = [{0: {(initial, None): Link(initial)}},
              {len(frames): {(state, None): Link(state) for state in goals}}]
    stacks = [[], []]
    reverse_moves = {}
    for axis in range(3):
        for side in (-1, 1):
            for count in (1, 2):
                for angle in (-90., 90.):
                    rotation = scorer.m.CubeLayerRotation(axis, side, angle, count)
                    move = scorer.m.cube_rotation_to_notation(rotation)
                    if move:
                        reverse_moves[move] = ("layer", axis, side, count, (-1 if move.endswith("'") else 1,))
    midpoint = len(frames) // 2
    bounds = (min(len(frames), midpoint + overlap), max(0, midpoint - overlap))
    attempted_retries = 0

    def evaluations():
        return sum(expanded) + (forecaster.expanded if forecaster else 0)

    def forecast_metrics():
        return forecaster.metrics() if forecaster else {}

    def report(event, **data):
        progress(dict(event=event, elapsed_seconds=time.monotonic() - start,
                      forward_expanded=expanded[0], backward_expanded=expanded[1],
                      evaluated_states=evaluations(), **forecast_metrics(), **data))

    def check_budget():
        if time.monotonic() - start >= time_limit:
            raise BudgetReached("time_limit")

    def check_forecast_budget(forecast):
        check_budget()
        if forecast and evaluations() >= state_limit:
            raise BudgetReached("state_limit")

    def priority(choice):
        return .5 * (choice.payload.get("color_error") or 0.)

    def step_for(choice, frame, cue):
        bonus = handoff_bonus if cue and tuple(choice.payload["state_moves"]) == (cue,) else 0.
        step = dict(choice.payload, frame_index=frame, state_score=choice.score,
                    queue_priority=priority(choice)-bonus)
        if bonus:
            step["turn_handoff_bonus"] = bonus
        return step

    def forward_options(depth, state, previous_cue):
        frame = frames[depth]
        choices = list(scorer(frame, state))
        if previous_cue:
            handoff = frame_search.score_handoff(scorer, frame, state, previous_cue)
            if handoff is not None:
                choices.append(handoff)
        for choice in choices:
            if math.isfinite(choice.score):
                step = step_for(choice, frame, previous_cue)
                cue = frame_search.supported_completion(scorer.m, step)
                unsupported = frame_search.unsupported_state_update(step)
                yield Link(choice.state, cue, step=step, rank=(step["queue_priority"], unsupported, choice.rank))

    forecaster = (WindowForecaster(forward_options, scorer.m.move_lookahead_weights, len(frames),
                                   lookahead_frames, lookahead_budget, check_forecast_budget)
                  if lookahead_frames else None)

    def forward(node, depth):
        expand = forecaster.edges if forecaster else forward_options
        for edge in expand(depth, node.state, node.cue):
            yield Link(edge.state, edge.cue, node, edge.step, edge.rank)

    def backward(node, depth):
        frame = frames[depth-1]
        hold = scorer.score_hold(frame, node.state)
        possibilities = [(None, None, node.state)]
        if scorer.hold_reason(frame) is None:
            possibilities.extend((move, direction, scorer.apply_moves(node.state, (inverse(move),)))
                                 for move, direction in reverse_moves.items())
        for move, direction, before in possibilities:
            check_budget()
            action = (move,) if move else ()
            if move:
                ordinary = scorer(frame, before) if branch_widths else scorer._evaluate(frame, before, direction)
            else:
                ordinary = (hold,) if hold is not None else ()
            choices = [(choice, None) for choice in ordinary
                       if choice.state == node.state and tuple(choice.payload["state_moves"]) == action]
            can_handoff = move and hold is not None and "geometry_option_index" in hold.payload and depth > 1
            if not choices and not can_handoff:
                continue
            # Reverse alternatives can explain the same post-turn stickers. Use
            # the preceding frame's fit of the predecessor to rank those ties.
            earlier = (scorer.score_hold(frames[depth-2], before)
                       if depth > 1 and (not branch_widths or can_handoff) else None)
            earlier_cost = priority(earlier) if earlier is not None else 0.5
            # A speculative handoff can fill the beam with cheap predecessors
            # that have no way to supply the required cue on the previous frame.
            if can_handoff and earlier and frame_search.supported_completion(scorer.m, earlier.payload) == move:
                step = dict(hold.payload, state_moves=action, handoff_commit=True, is_baseline=False,
                            label=f"commit {move} then {hold.payload['label']}")
                choices.append((replay.Transition(node.state, hold.score, hold.rank, step, None), move))
            for choice, need in choices:
                if not math.isfinite(choice.score) or (node.cue is not None and
                        frame_search.supported_completion(scorer.m, choice.payload) != node.cue):
                    continue
                step = step_for(choice, frame, need)
                yield Link(before, need, node, step, (earlier_cost, step["queue_priority"], choice.rank))

    def meeting(depth, direction, nodes):
        opposite = layers[1-direction].get(depth)
        if not opposite:
            return None
        if direction == 0:
            for node in nodes:
                match = opposite.get((node.state, None)) or opposite.get((node.state, node.cue))
                if match is not None:
                    return node, match
        else:
            for node in nodes:
                if node.cue:
                    match = opposite.get((node.state, node.cue))
                else:
                    match = next((entry for (state, _), entry in opposite.items() if state == node.state), None)
                if match is not None:
                    return match, node
        return None

    def remember(depth, direction, nodes, retry):
        bank = layers[direction].setdefault(depth, {})
        for node in nodes:
            bank.setdefault((node.state, node.cue), node)
        report("layer", direction=("forward", "backward")[direction], depth=depth, retry=retry,
               states=[(node.state.hex(), node.cue) for node in nodes])
        match = meeting(depth, direction, nodes)
        return (depth, *match) if match else None

    def diverse_order(ordered):
        if physical_key is None:
            return ordered
        # Prefer distinct cube arrangements, with two contexts per
        # arrangement before filling spare slots.
        counts = Counter()
        diverse, extra = [], []
        for node in ordered:
            key = physical_key(node.state)
            counts[key] += 1
            (diverse if counts[key] <= 2 else extra).append(node)
        return diverse + extra

    def sweep(direction, depth, beam, retry):
        delta = 1 if direction == 0 else -1
        while beam:
            joined = remember(depth, direction, beam, retry)
            if joined or depth == bounds[direction]:
                return joined
            children = {}
            for parent in beam:
                check_budget()
                if evaluations() >= state_limit:
                    raise BudgetReached("state_limit")
                expanded[direction] += 1
                successors = forward(parent, depth) if direction == 0 else backward(parent, depth)
                for child in successors:
                    key = child.state, child.cue
                    if key not in children or child.rank < children[key].rank:
                        children[key] = child
            depth += delta
            ordered = diverse_order(sorted(children.values(), key=lambda node: node.rank))
            if direction == 0 and forecaster:
                ordered = diverse_order(forecaster.order(depth, ordered, width))
            beam = ordered[:width]
            if not retry and len(ordered) > width:
                stacks[direction].append((depth, ordered[width:2*width]))
        return None

    def replay_join(joined):
        depth, left, right = joined
        prefix, suffix = [], []
        while left.link is not None:
            prefix.append((left.step, left.state))
            left = left.link
        while right.link is not None:
            suffix.append((right.step, right.link.state))
            right = right.link
        path = prefix[::-1]+suffix
        if len(path) != len(frames):
            raise RuntimeError("Joined paths do not cover the same frame boundary")
        state, cue, window = initial, None, ()
        verified = []
        trial_key = ("state_moves", "geometry_option_index", "color_offset", "orientation_symmetry_index")
        for frame, (step, after) in zip(frames, path, strict=True):
            check_budget()
            if step["frame_index"] != frame:
                raise RuntimeError("Joined paths disagree about frame order")
            if step.get("handoff_commit"):
                if not cue or tuple(step["state_moves"]) != (cue,):
                    raise RuntimeError("Joined path is missing a required handoff")
                choices = (frame_search.score_handoff(scorer, frame, state, cue),)
            else:
                choices = scorer(frame, state)
            match = next((choice for choice in choices if choice is not None and choice.state == after
                          and all(choice.payload.get(k) == step.get(k) for k in trial_key)), None)
            if match is None:
                raise RuntimeError(f"Joined transition failed forward replay at frame {frame}")
            verified_step = step_for(match, frame, cue)
            if branch_widths:
                window = frame_search.update_mismatch_window(window, verified_step, mismatch_window_weight)
                verified_step['queue_priority'] += verified_step['window_mismatch_penalty']
                verified_step['sibling_rank'] = step['sibling_rank']
            verified.append(verified_step)
            state, cue = after, frame_search.supported_completion(scorer.m, verified[-1])
        if state not in goals:
            raise RuntimeError("Joined path failed to reach the solved anchor")
        return depth, verified

    status, steps, join_depth = "beam_exhausted", [], None
    reached = [0, len(frames)]
    try:
        if branch_widths:
            status = 'exhausted'
            joined = widening_join(forward, backward, initial, goals, len(frames), widths=branch_widths,
                stage_budget=branch_stage_expansions, mismatch_weight=mismatch_window_weight,
                expanded=expanded, reached=reached, time_limit_check=check_budget, state_limit=state_limit,
                report=report)
        else:
            joined = sweep(0, 0, tuple(layers[0][0].values()), 0)
            if not joined:
                joined = sweep(1, len(frames), tuple(layers[1][len(frames)].values()), 0)
            for retry in range(retries):
                if joined or not any(stacks):
                    break
                direction = retry % 2
                if not stacks[direction]:
                    direction = 1-direction
                depth, deferred = stacks[direction].pop()
                attempted_retries += 1
                joined = sweep(direction, depth, deferred, attempted_retries)
        if joined:
            join_depth, steps = replay_join(joined)
            status = "solved"
    except BudgetReached as error:
        status = str(error)
    if not branch_widths:
        reached = [max(layers[0]), min(layers[1])]
    report(status, join_depth=join_depth, retries=attempted_retries)
    return dict(status=status, elapsed_seconds=time.monotonic()-start, evaluated_states=evaluations(),
                forward_expanded=expanded[0], backward_expanded=expanded[1], retries=attempted_retries,
                join_depth=join_depth, steps=steps, solved_frame_index=frames[-1] if steps else None,
                forward_depth=reached[0], backward_depth=reached[1], **forecast_metrics())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--initial-scramble", required=True)
    parser.add_argument("--end-frame", required=True, type=int, help="Solved anchor index in the observation log.")
    parser.add_argument("--frame-parity", choices=("all", "even", "odd"), default="all",
                        help="Use all source frames or one parity for both search directions.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--overlap", type=int, default=8, help="Frames each beam crosses past the middle.")
    parser.add_argument("--search-seconds", type=float, default=300.)
    parser.add_argument("--max-explored-states", type=int)
    parser.add_argument("--branch-widths", type=int, nargs="+", default=(),
                        help="Per-parent child cutoffs including ties for the best-first queues, e.g. 3 5 8.")
    parser.add_argument("--branch-stage-expansions", type=int,
                        help=f"Override the per-direction stage budget of "
                             f"{frame_search.BRANCH_STAGE_EXPANSIONS_PER_FRAME} expansions per solving frame.")
    parser.add_argument("--mismatch-window-weight", type=float, default=WIDENING_MISMATCH_WEIGHT,
                        help="Weight of the eight-frame mismatch history in the widening queues.")
    parser.add_argument("--lookahead-frames", type=int, default=0, help="Future sampled frames for forward ordering.")
    parser.add_argument("--lookahead-budget", type=int, default=384, help="New forecast expansions per forward layer.")
    args = parser.parse_args()
    if (args.beam_width < 1 or args.max_explored_states is not None and args.max_explored_states < 1
            or min(args.retries, args.overlap, args.lookahead_frames) < 0 or args.lookahead_budget < 1
            or not math.isfinite(args.search_seconds) or args.search_seconds <= 0):
        parser.error("Invalid beam or search budget")
    if (any(value < 1 for value in args.branch_widths)
            or any(b <= a for a, b in zip(args.branch_widths, args.branch_widths[1:]))
            or args.branch_stage_expansions is not None and args.branch_stage_expansions < 1
            or not math.isfinite(args.mismatch_window_weight)
            or args.mismatch_window_weight < 0 or args.branch_widths and args.lookahead_frames):
        parser.error("Invalid branch widening options")
    if args.lookahead_frames and args.lookahead_budget < 2*args.beam_width*args.lookahead_frames:
        parser.error("Lookahead budget must allow a complete greedy forecast for twice the beam width")
    if not args.branch_widths and args.max_explored_states is None:
        args.max_explored_states = 20000
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(2)
    m = replay.load_renderer()
    args.source = args.fps = args.video_frame_count = None
    records, metadata = replay.load_nine_class_records(args, m)
    static_end = metadata.get("turning_start_frame")
    if static_end is None or not static_end < args.end_frame or args.end_frame not in records:
        parser.error("The log needs a static prefix and a solved anchor after it")
    frames = tuple(range(static_end+1, args.end_frame+1))
    if any(index not in records for index in frames):
        parser.error("The log must cover every sampled frame through the solved anchor")
    already_sampled = any("original_source_frame" in record for record in records.values())
    passes = frame_search.frame_passes(records, frames, args.frame_parity, args.fps,
                                      already_sampled=already_sampled)
    if not passes or passes[0][1][-1] != args.end_frame:
        parser.error("The solved anchor must match the requested source-frame parity")
    _, frames, search_fps = passes[0]
    stage_budget = frame_search.branch_stage_budget(len(frames), args.branch_stage_expansions)
    colors = replay.initial_sticker_colors(m, args.initial_scramble)
    scorer_options = dict(initial_colors=colors, turning_start_frame=static_end,
                          balance_geometry_color=True, joint_orientation_rescue=True)
    if args.branch_widths:
        scorer_options = widening_scorer_options(m, colors, static_end, metadata.get("pose_model"))
        use_26m = scorer_options["color_confusion_scoring"]
        confusion_table = scorer_options["tracked_color_confusions"]
    scorer = replay.PreparedReplayScorer(m, records, "cpu", **scorer_options)
    initial = scorer.key(m.CubeGeometryState.identity())
    goals = solved_seeds(scorer, args.initial_scramble)
    solved = scorer.solved_predicate(colors)
    if not all(map(solved, goals)):
        parser.error("Scramble inversion did not produce solved states")
    for index in frames:
        scorer.prepare(index)
    anchor = flat_solved_anchor(m, records[frames[-1]], goals, **scorer_options)
    if anchor is None:
        parser.error("The anchor needs a reliable flat solved fit with at least 12 color observations")

    @lru_cache(maxsize=65536)
    def physical_key(state):
        return min(row.tobytes() for row in np.frombuffer(state, dtype=np.uint8)[scorer.orientation_sources])

    config = dict(algorithm="bidirectional_bounded_beam_stack", log=str(args.log.resolve()),
        observation_input=metadata, initial_scramble=args.initial_scramble, turning_start_frame=static_end,
        fps=search_fps, frame_parity=args.frame_parity, video_frame_count=args.video_frame_count,
        sampled_frame_indices=frames,
        solved_anchor_frame=args.end_frame, solved_anchor_source_frame=records[args.end_frame].get(
            "original_source_frame", args.end_frame), beam_width=args.beam_width, retries=args.retries,
        overlap=args.overlap, time_budget=args.search_seconds, state_limit=args.max_explored_states,
        score_weights=dict(geometry=.5,color=.5), queue_score_weights=dict(geometry=0.,color=.5),
        turn_handoff_bonus=WIDENING_HANDOFF_BONUS, reverse_handoff_requires_prior_hold=True,
        preferred_orientations_per_physical_state=2, reference_solution_used=False,
        anchor_colors=dict(observed=anchor.payload["color_match_count"], error=anchor.payload["color_error"]))
    if args.branch_widths:
        for key in ("beam_width", "retries", "overlap", "preferred_orientations_per_physical_state"):
            config.pop(key)
        config.update(algorithm="bidirectional_best_first_widening",
            branch_widths=args.branch_widths, branch_stage_expansions=stage_budget,
            mismatch_window_weight=args.mismatch_window_weight,
            mismatch_window_frames=frame_search.COLOR_MISMATCH_WINDOW_FRAMES,
            mismatch_window_confusions={color:sorted(allowed) for color, allowed in confusion_table.items() if allowed},
            color_confusion_cost=scorer.color_confusion_cost if use_26m else None,
            color_scoring="directional_confusion_cost" if use_26m else "direct_match_fraction",
            history_order="independent in each direction")
    if args.lookahead_frames:
        config["forward_lookahead"] = dict(future_frames=args.lookahead_frames,
            max_new_expansions_per_layer=args.lookahead_budget, shortlist=2*args.beam_width,
            weights="30 percent front half, 70 percent back half",
            seeds=["greedy", "shifted prior forecast", "hold"])
    (args.output_dir/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    with (args.output_dir/"progress.jsonl").open("w", buffering=1) as log:
        def progress(row):
            log.write(json.dumps(row, allow_nan=False)+"\n")
            if row["event"] != "layer" or row["depth"] % 10 == 0:
                print({k:v for k,v in row.items() if k != "states"}, flush=True)

        result = bidirectional_search(scorer, frames, initial, goals, width=args.beam_width,
            retries=args.retries, overlap=args.overlap, time_limit=args.search_seconds,
            state_limit=args.max_explored_states, handoff_bonus=WIDENING_HANDOFF_BONUS, progress=progress,
            physical_key=physical_key, lookahead_frames=args.lookahead_frames,
            lookahead_budget=args.lookahead_budget, branch_widths=args.branch_widths,
            branch_stage_expansions=stage_budget,
            mismatch_window_weight=args.mismatch_window_weight)
    if result["steps"]:
        prefix = [dict(scorer(index, initial)[0].payload, frame_index=index, state_score=0., queue_priority=0.)
                  for index in sorted(records) if index <= static_end]
        result["steps"] = prefix + result["steps"]
    result["config"] = config
    (args.output_dir/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
