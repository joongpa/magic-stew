#!/usr/bin/env python3
"""Depth-first, whole-solve search using fixed observations from a turn log.

No search budget, beam, per-parent top-k cutoff, or reference-solution seed.
Above-threshold continuations are deferred. When eligible DFS work runs out,
resume the least costly deferred continuation with exactly the allowance its
rolling score requires. The next frame returns to the normal threshold.
The rolling score is the mean of the lowest three costs in the last four frames.
The goal tests physical sticker colors, starting with the supplied scramble.
Poses, geometry options and categorical color matches stay fixed.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sqlite3
import sys
import time
from typing import Any, Callable
import zlib

try:
    from scripts.independent_keypoint_rotation_budget import RotationBudgetScorerMixin
    from scripts import independent_keypoint_search_fast as fast
except ModuleNotFoundError:
    from independent_keypoint_rotation_budget import RotationBudgetScorerMixin
    import independent_keypoint_search_fast as fast

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_SHORTLIST_SIZE = 5
GEOMETRY_TIE_TOLERANCE = .01


def second_best_four(scores: tuple[float, ...]) -> float | None:
    """Historical comparison statistic; the search uses best_three_mean_four."""
    if len(scores) < 4:
        return None
    return sorted(scores[-4:])[1]


def best_three_mean_four(scores: tuple[float, ...]) -> float | None:
    """Discard the largest of the last four costs, then average the other three."""
    if len(scores) < 4:
        return None
    return sum(sorted(scores[-4:])[:3]) / 3


@dataclass(frozen=True)
class Transition:
    state: Any
    score: float
    rank: tuple
    payload: dict
    update_key: tuple | None = None


def transitions_for_update_context(transitions, frame, state, update, min_update_gap):
    """Allow evaluators to discard move-spacing failures before scoring them."""
    filtered = getattr(transitions, "for_update_context", None)
    if filtered is not None:
        return filtered(frame, state, update, min_update_gap)
    return transitions(frame, state), 0


@dataclass
class Node:
    depth: int
    state: Any
    recent: tuple[float, ...]
    last_update_frame: int | None
    last_update_key: tuple | None
    update_streak: int
    incoming: dict | None = None
    choices: tuple[Transition, ...] | None = None
    next_choice: int = 0


class DiskMemo:
    """Exact exhausted-prefix memo; disk storage is not an exploration limit."""

    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS failed (key BLOB PRIMARY KEY) WITHOUT ROWID")
        self.pending = 0

    def __contains__(self, key):
        return self.connection.execute("SELECT 1 FROM failed WHERE key=?", (key,)).fetchone() is not None

    def add(self, key):
        self.connection.execute("INSERT OR IGNORE INTO failed VALUES (?)", (key,))
        self.pending += 1
        if self.pending >= 500:
            self.connection.commit()
            self.pending = 0

    def close(self):
        self.connection.commit()
        self.connection.close()


def full_solve_dfs(
    initial_state: Any,
    frames: tuple[int, ...],
    transitions: Callable[[int, Any], tuple[Transition, ...]],
    solved: Callable[[Any], bool],
    state_key: Callable[[Any], Any],
    *,
    threshold: float = 6.0,
    min_update_gap: int = 1,
    failed: Any = None,
    progress: Callable[[dict], None] | None = None,
    progress_seconds: float = 5.0,
) -> dict:
    """First solved path in ranked DFS order; finite end-of-video backtracks.

    Memoize only exhausted prefixes with identical frame, physical state,
    move-spacing context and the last three numeric scores. Numeric history
    matters: equal threshold flags can produce different trimmed means.
    """
    if not frames or any(b != a + 1 for a, b in zip(frames, frames[1:])):
        raise ValueError("Provide a nonempty, consecutive source-frame interval")
    if not math.isfinite(threshold) or min_update_gap < 1 or progress_seconds <= 0:
        raise ValueError("Threshold must be finite; gap and progress interval must be positive")
    failed = set() if failed is None else failed
    stats = dict(evaluated_states=0, attempted_transitions=0, threshold_pruned=0,
                 spacing_pruned_before_scoring=0,
                 invalid_pruned=0, spacing_pruned=0, memo_pruned=0,
                 backtracks=0, unsolved_endpoints=0, furthest_frames=0,
                 furthest_frame_index=None, current_frames=0, current_frame_index=None)
    started = time.monotonic()
    last_report = started
    stack = [Node(0, initial_state, (), None, None, 0)]

    def report(event, **extra):
        nonlocal last_report
        last_report = time.monotonic()
        if progress:
            progress(dict(event=event, elapsed_seconds=last_report-started, **stats, **extra))

    def memo_key(node):
        return pickle.dumps(("fixed-best-three-mean-v1", threshold, node.depth, state_key(node.state),
            node.recent[-3:],
            node.last_update_frame if min_update_gap > 1 else None,
            node.last_update_key if min_update_gap > 1 else None,
            node.update_streak if min_update_gap > 1 else 0), protocol=5)

    def result(status):
        return dict(status=status, **stats, elapsed_seconds=time.monotonic()-started,
                    steps=[node.incoming for node in stack[1:]],
                    solved_frame_index=frames[len(stack)-2] if status == "solved" and len(stack)>1 else None)

    report("started")
    if solved(initial_state):
        report("already_solved")
        return result("already_solved")
    while stack:
        node = stack[-1]
        stats["current_frames"] = node.depth
        stats["current_frame_index"] = frames[node.depth-1] if node.depth else None
        if time.monotonic() - last_report >= progress_seconds:
            report("progress")
        if node.choices is None:
            if memo_key(node) in failed:
                stats["memo_pruned"] += 1
                stack.pop()
                continue
            if node.depth == len(frames):
                stats["unsolved_endpoints"] += 1
                failed.add(memo_key(node))
                stats["backtracks"] += 1
                stack.pop()
                continue
            stats["evaluated_states"] += 1
            choices, skipped = transitions_for_update_context(transitions, frames[node.depth], node.state,
                (node.last_update_frame, node.last_update_key, node.update_streak), min_update_gap)
            stats["spacing_pruned"] += skipped
            stats["spacing_pruned_before_scoring"] += skipped
            node.choices = tuple(sorted(choices, key=lambda t: t.rank))
        if node.next_choice == len(node.choices):
            failed.add(memo_key(node))
            stats["backtracks"] += 1
            stack.pop()
            continue
        choice = node.choices[node.next_choice]
        node.next_choice += 1
        stats["attempted_transitions"] += 1
        if not math.isfinite(choice.score):
            stats["invalid_pruned"] += 1
            continue
        frame = frames[node.depth]
        moves = tuple(choice.payload.get("state_moves", ()))
        update_frame, update_key, streak = node.last_update_frame, node.last_update_key, node.update_streak
        if moves:
            consecutive = update_frame is not None and frame-update_frame == 1
            continuation = consecutive and streak < 2 and choice.update_key is not None and choice.update_key == update_key
            if update_frame is not None and frame-update_frame < min_update_gap and not continuation:
                stats["spacing_pruned"] += 1
                continue
            streak = streak+1 if consecutive and choice.update_key == update_key else 1
            update_frame, update_key = frame, choice.update_key
        recent = (*node.recent[-3:], choice.score)
        statistic = best_three_mean_four(recent)
        if statistic is not None and statistic > threshold:
            stats["threshold_pruned"] += 1
            continue
        incoming = dict(choice.payload, frame_index=frame, state_score=choice.score,
                        best_three_mean_four=statistic)
        child = Node(node.depth+1, choice.state, recent[-3:], update_frame, update_key, streak, incoming)
        stack.append(child)
        stats["current_frames"] = child.depth
        stats["current_frame_index"] = frame
        if child.depth > stats["furthest_frames"]:
            stats["furthest_frames"] = child.depth
            stats["furthest_frame_index"] = frame
            report("furthest", last_score=choice.score, best_three_mean_four=statistic,
                   frontier_moves=[dict(frame_index=n.incoming["frame_index"], state_moves=n.incoming["state_moves"])
                                   for n in stack[1:] if n.incoming["state_moves"]])
        if solved(choice.state):
            report("solved", solved_frame_index=frame)
            return result("solved")
    report("exhausted")
    return result("exhausted")


@dataclass
class PathLink:
    parent: Any
    step: dict
    stored_id: int | None = None


@dataclass
class AdaptiveNode(Node):
    path: PathLink | int = 0
    blocked: list | None = None


class DeferredBranches:
    """Disk-backed deferred choices with shared prefix storage and no cutoff.

    One queue record holds all blocked choices from one parent. Its cheapest
    choice determines priority. Resuming that choice requeues its siblings.
    This avoids writing a full video-length prefix for each blocked edge.
    """

    def __init__(self, path=":memory:"):
        self.connection = sqlite3.connect(path)
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS paths (
                id INTEGER PRIMARY KEY, parent INTEGER NOT NULL, step BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS pending (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                required REAL NOT NULL, context BLOB NOT NULL UNIQUE, payload BLOB NOT NULL);
            CREATE INDEX IF NOT EXISTS pending_priority ON pending(required, id);
        """)
        self.pending_groups = self.connection.execute("SELECT count(*) FROM pending").fetchone()[0]
        self.pending_choices = 0
        for (payload,) in self.connection.execute("SELECT payload FROM pending"):
            self.pending_choices += len(self.unpack(payload)["blocked"])
        self.writes = 0

    @staticmethod
    def pack(value):
        return zlib.compress(pickle.dumps(value, protocol=5), level=1)

    @staticmethod
    def unpack(value):
        return pickle.loads(zlib.decompress(value))

    def persist_path(self, path):
        links = []
        while isinstance(path, PathLink) and path.stored_id is None:
            links.append(path)
            path = path.parent
        parent_id = path.stored_id if isinstance(path, PathLink) else path
        for link in reversed(links):
            cursor = self.connection.execute("INSERT INTO paths(parent,step) VALUES (?,?)",
                                             (parent_id, self.pack(link.step)))
            parent_id = cursor.lastrowid
            link.stored_id = parent_id
        return parent_id

    def steps(self, path):
        reverse = []
        while isinstance(path, PathLink):
            reverse.append(path.step)
            path = path.parent
        while path:
            parent, value = self.connection.execute("SELECT parent,step FROM paths WHERE id=?", (path,)).fetchone()
            reverse.append(self.unpack(value))
            path = parent
        return list(reversed(reverse))

    def push(self, context, group):
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO pending(required,context,payload) VALUES (?,?,?)",
            (group["blocked"][0][1], context, self.pack(group)))
        if cursor.rowcount:
            self.pending_groups += 1
            self.pending_choices += len(group["blocked"])
        self.writes += 1
        if self.writes % 500 == 0:
            self.connection.commit()

    def pop(self):
        record = self.connection.execute(
            "SELECT id,context,payload FROM pending ORDER BY required,id LIMIT 1").fetchone()
        if record is None:
            return None
        row_id, context, payload = record
        group = self.unpack(payload)
        self.connection.execute("DELETE FROM pending WHERE id=?", (row_id,))
        self.pending_groups -= 1
        self.pending_choices -= len(group["blocked"])
        return context, group

    def minimum(self):
        row = self.connection.execute("SELECT required FROM pending ORDER BY required,id LIMIT 1").fetchone()
        return row[0] if row is not None else None

    def close(self):
        self.connection.commit()
        self.connection.close()


def adaptive_full_solve_dfs(initial_state, frames, transitions, solved, state_key, *,
                            threshold=2., min_update_gap=1, failed=None, deferred=None,
                            serialize_state=pickle.dumps, deserialize_state=pickle.loads,
                            progress=None, progress_seconds=5.):
    """Ranked DFS at a normal threshold, with exact one-window fallback.

    A resumed blocked transition is admitted at its mean-of-best-three-of-four
    score. Every subsequent frame is tested at the normal threshold again.
    Memoization uses numeric history because it also determines the priority
    of future deferred transitions, not just normal-threshold eligibility.
    """
    if deferred is None:
        queue = DeferredBranches()
        try:
            return adaptive_full_solve_dfs(initial_state, frames, transitions, solved, state_key,
                threshold=threshold, min_update_gap=min_update_gap, failed=failed, deferred=queue,
                serialize_state=serialize_state, deserialize_state=deserialize_state,
                progress=progress, progress_seconds=progress_seconds)
        finally:
            queue.close()
    if not frames or any(b != a+1 for a, b in zip(frames, frames[1:])):
        raise ValueError("Provide a nonempty, consecutive source-frame interval")
    if not math.isfinite(threshold) or min_update_gap < 1 or progress_seconds <= 0:
        raise ValueError("Threshold must be finite; gap and progress interval must be positive")
    failed = set() if failed is None else failed
    stats = dict(evaluated_states=0, attempted_transitions=0, threshold_pruned=0,
        spacing_pruned_before_scoring=0,
        threshold_deferred=0, deferred_resumed=0, invalid_pruned=0, spacing_pruned=0,
        memo_pruned=0, backtracks=0, unsolved_endpoints=0, furthest_frames=0,
        furthest_frame_index=None, current_frames=0, current_frame_index=None,
        normal_threshold=threshold, largest_temporary_threshold=threshold)
    started = time.monotonic()
    last_report = started
    stack = [AdaptiveNode(0, initial_state, (), None, None, 0)]

    def report(event, **extra):
        nonlocal last_report
        last_report = time.monotonic()
        stats["deferred_pending"] = deferred.pending_choices
        stats["deferred_groups"] = deferred.pending_groups
        stats["next_deferred_threshold"] = deferred.minimum()
        if progress:
            progress(dict(event=event, elapsed_seconds=last_report-started, **stats, **extra))

    def context(node):
        return pickle.dumps(("adaptive-best-three-mean-one-window-v1", threshold, node.depth, state_key(node.state),
            node.recent[-3:], node.last_update_frame if min_update_gap>1 else None,
            node.last_update_key if min_update_gap>1 else None,
            node.update_streak if min_update_gap>1 else 0), protocol=5)

    def update_context(node, choice):
        frame = frames[node.depth]
        previous, key, streak = node.last_update_frame, node.last_update_key, node.update_streak
        if choice.payload.get("state_moves"):
            consecutive = previous is not None and frame-previous == 1
            continuation = consecutive and streak<2 and choice.update_key is not None and choice.update_key==key
            if previous is not None and frame-previous<min_update_gap and not continuation:
                return None
            streak = streak+1 if consecutive and choice.update_key==key else 1
            previous, key = frame, choice.update_key
        return previous, key, streak

    def child_for(node, choice, allowance):
        recent = (*node.recent[-3:], choice.score)
        statistic = best_three_mean_four(recent)
        update = update_context(node, choice)
        if update is None:
            raise RuntimeError("Deferred choice no longer satisfies its recorded move-spacing context")
        keep = ("trial_id", "state_moves", "color_error", "keypoint_mean_error",
                "heatmap_mean_error", "geometry_option_index")
        incoming = {key: choice.payload[key] for key in keep if key in choice.payload}
        incoming.update(frame_index=frames[node.depth], state_score=choice.score,
            best_three_mean_four=statistic, normal_threshold=threshold, applied_threshold=allowance,
            temporary_allowance=allowance>threshold)
        return AdaptiveNode(node.depth+1, choice.state, recent[-3:], *update,
                            incoming=incoming, path=PathLink(node.path, incoming))

    def accepted(child):
        stack.append(child)
        stats["current_frames"] = child.depth
        stats["current_frame_index"] = frames[child.depth-1]
        if child.depth > stats["furthest_frames"]:
            stats["furthest_frames"] = child.depth
            stats["furthest_frame_index"] = frames[child.depth-1]
            report("furthest", last_score=child.incoming["state_score"],
                best_three_mean_four=child.incoming["best_three_mean_four"],
                frontier_moves=[dict(frame_index=s["frame_index"],state_moves=s.get("state_moves",()))
                    for s in deferred.steps(child.path) if s.get("state_moves")])
        return solved(child.state)

    def outcome(status, path=0):
        report(status)
        steps = deferred.steps(path)
        return dict(status=status, elapsed_seconds=time.monotonic()-started, **stats,
            steps=steps, solved_frame_index=steps[-1]["frame_index"] if status=="solved" else None)

    report("started")
    if solved(initial_state):
        return outcome("already_solved")
    while stack or deferred.pending_groups:
        if time.monotonic()-last_report >= progress_seconds:
            report("progress")
        if not stack:
            key, group = deferred.pop()
            index, required = group["blocked"][0]
            remaining = group["blocked"][1:]
            if remaining:
                deferred.push(key, dict(group, blocked=remaining))
            parent = AdaptiveNode(group["depth"], deserialize_state(group["state"]),
                group["recent"], group["last_update_frame"], group["last_update_key"],
                group["update_streak"], path=group["path_id"])
            # Values may have been evicted from the score cache; recompute from the same inputs.
            choices, _skipped = transitions_for_update_context(transitions, frames[parent.depth], parent.state,
                (parent.last_update_frame, parent.last_update_key, parent.update_streak), min_update_gap)
            choices = tuple(sorted(choices, key=lambda t:t.rank))
            choice = choices[index]
            statistic = best_three_mean_four((*parent.recent[-3:], choice.score))
            allowance = max(threshold, statistic)
            child = child_for(parent, choice, allowance)
            stats["deferred_resumed"] += 1
            stats["largest_temporary_threshold"] = max(stats["largest_temporary_threshold"], allowance)
            # Detailed fallback records are emitted for every resumed blocked edge.
            report("temporary_allowance", source_frame=frames[parent.depth],
                requested_threshold=required, temporary_threshold=allowance,
                return_threshold_next_frame=threshold)
            if accepted(child):
                return outcome("solved", child.path)
            continue
        node = stack[-1]
        stats["current_frames"] = node.depth
        stats["current_frame_index"] = frames[node.depth-1] if node.depth else None
        if node.choices is None:
            if context(node) in failed:
                stats["memo_pruned"] += 1
                stack.pop()
                continue
            if node.depth == len(frames):
                stats["unsolved_endpoints"] += 1
                failed.add(context(node))
                stats["backtracks"] += 1
                stack.pop()
                continue
            stats["evaluated_states"] += 1
            choices, skipped = transitions_for_update_context(transitions, frames[node.depth], node.state,
                (node.last_update_frame, node.last_update_key, node.update_streak), min_update_gap)
            stats["spacing_pruned"] += skipped
            stats["spacing_pruned_before_scoring"] += skipped
            node.choices = tuple(sorted(choices, key=lambda t:t.rank))
            node.blocked = []
        if node.next_choice == len(node.choices):
            if node.blocked:
                group = dict(depth=node.depth, state=serialize_state(node.state), recent=node.recent,
                    last_update_frame=node.last_update_frame, last_update_key=node.last_update_key,
                    update_streak=node.update_streak, path_id=deferred.persist_path(node.path),
                    blocked=tuple(sorted(node.blocked,key=lambda value:(value[1],value[0]))))
                deferred.push(context(node), group)
            # This context has been expanded; its blocked edges remain independently queued.
            failed.add(context(node))
            stats["backtracks"] += 1
            stack.pop()
            continue
        index = node.next_choice
        choice = node.choices[index]
        node.next_choice += 1
        stats["attempted_transitions"] += 1
        if not math.isfinite(choice.score):
            stats["invalid_pruned"] += 1
            continue
        if update_context(node, choice) is None:
            stats["spacing_pruned"] += 1
            continue
        statistic = best_three_mean_four((*node.recent[-3:], choice.score))
        if statistic is not None and statistic>threshold:
            node.blocked.append((index,statistic))
            stats["threshold_deferred"] += 1
            continue
        child = child_for(node, choice, threshold)
        if accepted(child):
            return outcome("solved", child.path)
    return outcome("exhausted")


def load_renderer():
    spec = importlib.util.spec_from_file_location("full_solve_renderer", ROOT / "scripts/render_yolo11n_independent_keypoints.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add_nine_class_input_arguments(parser):
    try:
        from scripts.independent_keypoint_staged_pose import add_pose_method_arguments
    except ModuleNotFoundError:
        from independent_keypoint_staged_pose import add_pose_method_arguments

    add_pose_method_arguments(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log", type=Path, help="A nine-class turn/observation JSONL log, or its directory.")
    source.add_argument("--source", type=Path, help="Video to process with the landmark/color pose model.")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "runs/pose/runs/26m-pose/weights/best.pt")
    parser.add_argument("--bbox-model", type=Path, default=ROOT / "runs/500ExamplesWeights.pt")
    parser.add_argument("--inference-device", default="cpu",
                        help="Source-video network device: cpu, mps, or CUDA index; separate from search --device.")
    parser.add_argument("--pose-class-conf", help="Per-class detection thresholds, e.g. red=0.35.")
    parser.add_argument("--keypoint-conf", type=float, default=.05,
                        help="Landmark keypoint presence threshold; sticker color keypoints keep the 0.25 floor.")
    parser.add_argument("--roi-padding", type=float, default=.4,
                        help="Padding ratio around the pose-model ROI (default: 0.4).")
    parser.add_argument("--turning-start-frame", type=int,
                        help="Last static source frame, inclusive; no turn search occurs through this frame.")


def initial_sticker_colors(renderer, scramble):
    cube = renderer.CubeState.from_scramble(scramble)
    return {label: cube.colors[position]
            for position, label in renderer.CubeGeometryState.identity().facelets.items()}


def prepare_search_record(renderer, record, colors, turning_start_frame, device):
    """Prepare the same fixed-pose observation for cached and standalone search."""
    candidate = record.pose_candidate
    if candidate is not None:
        if renderer._frame_forces_static_hold(record.frame_index, turning_start_frame):
            candidate = renderer._hold_before_turning_start(
                candidate, record.frame_index, turning_start_frame)
        else:
            limit = renderer.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * renderer.projected_cube_diameter(candidate)
            joint_options = getattr(candidate, 'joint_geometry_candidates', ())
            staged = (getattr(candidate, 'joint_pose_diagnostics', None) or {}).get('pose_method') == 'staged'
            if joint_options:
                geometry = dict(options=joint_options)
            else:
                geometry = None if staged else renderer._build_turn_geometry_frame(candidate, limit, device)
            diagnostics = dict(status="unavailable_geometry", evaluated_states=0,
                               turning_start_frame=turning_start_frame)
            if geometry is not None:
                diagnostics.update(status="prepared_full_search", geometry=dict(
                    scoring_version=renderer.PNP_TURN_GEOMETRY_SCORE_VERSION,
                    scope='per_geometry_middle_pose' if joint_options else 'whole_cube',
                    joint_refined=bool(joint_options) and not staged,
                    options=list(joint_options) if joint_options else
                        [renderer.turn_trial_log_payload(option) for option in geometry['options']]))
            candidate = replace(candidate,
                turn_color_matches=renderer.turn_color_match_matrix(candidate, expected_color_by_label=colors),
                turn_search_diagnostics=diagnostics,
                turn_fit_search_rvec=candidate.rvec.copy(), turn_fit_search_tvec=candidate.tvec.copy())
    return renderer.turn_candidate_log_record(candidate, record.frame_index)


def prepare_nine_class_video(args, renderer):
    """Cache nine-class observations and all geometric options, without local move rollout."""
    import torch

    if not args.source.is_file():
        raise ValueError(f"Source video not found: {args.source}")
    checkpoint = renderer.pose_checkpoint_path(args)
    if not checkpoint.is_file() or not args.bbox_model.is_file():
        raise ValueError("Both the nine-class pose checkpoint and bbox checkpoint must exist")
    if not 0 <= args.keypoint_conf <= 1:
        raise ValueError("--keypoint-conf must be in [0, 1]")
    if not math.isfinite(args.roi_padding) or args.roi_padding < 0:
        raise ValueError("--roi-padding must be finite and nonnegative")
    if args.turning_start_frame is not None and args.turning_start_frame < 0:
        raise ValueError("--turning-start-frame must be nonnegative")
    output = args.output_dir / "observations.jsonl"
    if output.exists():
        raise ValueError("Observation cache already exists; choose a fresh output directory")
    options = ["--source", str(args.source), "--initial-scramble", args.initial_scramble,
               "--model", str(args.model), "--bbox-model", str(args.bbox_model),
               "--keypoint-conf", str(args.keypoint_conf), "--roi-padding", str(args.roi_padding),
               "--no-display", "--no-show-fps"]
    options += ["--pose-method", getattr(args, "pose_method", "manual"),
                "--device", getattr(args, "inference_device", "cpu")]
    if getattr(args, "pose_method", "manual") == "staged":
        options += ["--staged-model", str(checkpoint)]
    if args.pose_class_conf:
        options += ["--pose-class-conf", args.pose_class_conf]
    render_args = renderer.build_parser().parse_args(options)
    model = renderer.load_pose_model(render_args)
    detector = renderer.YOLOCubeDetector(
        model_name=str(args.bbox_model), confidence_threshold=render_args.bbox_conf,
        image_size=render_args.bbox_imgsz, device=render_args.device)
    print(f"Preparing nine-class observations with {checkpoint} ({render_args.pose_method})", flush=True)
    cache, count, _completed, fps = renderer.run_neural_video_pass(
        str(args.source), detector, model, render_args)
    colors = initial_sticker_colors(renderer, args.initial_scramble)
    renderer.process_cached_poses(cache, render_args, colors, source_fps=fps)
    device = torch.device(args.device)
    with output.open("x", encoding="utf-8") as stream:
        for index, record in enumerate(cache, start=1):
            static = renderer._frame_forces_static_hold(record.frame_index, args.turning_start_frame)
            renderer.print_headless_progress("static holds" if static else "preparing search observations", index, count)
            observation = prepare_search_record(renderer, record, colors, args.turning_start_frame, device)
            stream.write(json.dumps(observation, separators=(",", ":")) + "\n")
    if cache:
        print()
    metadata = dict(source=str(args.source.resolve()), pose_model=str(checkpoint.resolve()),
        pose_model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        pose_method=render_args.pose_method, inference_device=render_args.device,
        class_names=list(renderer.CLASS_NAMES), class_confidences=render_args.pose_class_conf,
        landmark_keypoint_confidence=args.keypoint_conf,
        roi_padding=render_args.roi_padding,
        geometry_scoring_version=renderer.PNP_TURN_GEOMETRY_SCORE_VERSION,
        sticker_keypoint_min_confidence=renderer.STICKER_KEYPOINT_MIN_CONFIDENCE,
        initial_scramble=args.initial_scramble, fps=fps, video_frame_count=count,
        turning_start_frame=args.turning_start_frame, color_scoring="direct_match_fraction",
        segmentation_used=False, pixel_color_sampling_used=False, local_move_rollout_used=False,
        orientation_policy=f"angular continuity plus {renderer.POSE_CONTINUITY_COLOR_WEIGHT:g} times center color error")
    output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2))
    return output, metadata


def load_nine_class_records(args, renderer):
    if args.source is not None:
        args.log, metadata = prepare_nine_class_video(args, renderer)
    else:
        if args.log.is_dir():
            directory = args.log
            args.log = next((directory / name for name in ("observations.jsonl", "turn_candidate_scores.jsonl")
                             if (directory / name).is_file()), directory / "observations.jsonl")
        metadata_path = args.log.with_suffix(".meta.json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    records = {}
    for line_number, line in enumerate(args.log.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        frame = record["frame_index"]
        if frame in records:
            raise ValueError(f"Duplicate source frame {frame} on log line {line_number}")
        records[frame] = record
    if not records:
        raise ValueError("Source log contains no frames")
    if args.fps is None:
        args.fps = float(metadata.get("fps") or 60.)
    if args.video_frame_count is None and metadata.get("video_frame_count") is not None:
        args.video_frame_count = metadata["video_frame_count"]
    return records, metadata


class ReplayScorer:
    def __init__(self, renderer, records, device, cache_size=4096, *, initial_colors=None,
                 turning_start_frame=None, balance_geometry_color=False, minimum_color_accuracy=None,
                 track_sticker_observations=False,
                 color_confusion_pruning=False, color_confusion_cost=.5,
                 color_confusion_scoring=False, preserve_layer_activity=False,
                 tracked_color_confusions=None):
        import numpy as np
        self.np = np
        self.m = renderer
        self.records = records
        self.device = device
        self.initial_colors = initial_colors
        self.turning_start_frame = turning_start_frame
        self.balance_geometry_color = balance_geometry_color
        self.minimum_color_accuracy = minimum_color_accuracy
        self.track_sticker_observations = track_sticker_observations
        if not np.isfinite(color_confusion_cost) or not 0 < color_confusion_cost < 1:
            raise ValueError('Color confusion cost must be strictly between zero and one')
        self.color_confusion_pruning = color_confusion_pruning
        self.color_confusion_scoring = color_confusion_scoring or color_confusion_pruning
        self.preserve_layer_activity = preserve_layer_activity
        self.color_confusion_cost = color_confusion_cost
        self.tracked_color_confusions = tracked_color_confusions
        self.confusion_pruned_fits = 0
        self.confusion_dead_end_states = 0
        self.held_observations = {}
        self.cache_size = cache_size  # Eviction only; never prunes search branches.
        self.cache = OrderedDict()
        self.prepared = OrderedDict()
        self.positions = tuple(sorted(renderer.CubeGeometryState.identity().facelets))
        self.identity_indices = {label: index for index, label in enumerate(sorted(renderer.CubeGeometryState.identity().facelets.values()))}
        self.cache_hits = 0
        self.scoring_calls = 0

    def key(self, state):
        return bytes(self.identity_indices[state.facelets[position]] for position in self.positions)

    def state_from_key(self, value):
        labels = tuple(sorted(self.identity_indices, key=self.identity_indices.get))
        return self.m.CubeGeometryState({position: labels[index]
            for position,index in zip(self.positions,value,strict=True)})

    def apply_moves(self, state, moves):
        return state.apply_moves(moves)

    def score_hold(self, frame, state):
        return next((choice for choice in self(frame, state)
                     if not choice.payload["state_moves"]), None)

    def solved_predicate(self, sticker_colors):
        return lambda state: state.is_solved(sticker_colors)

    def hold_reason(self, frame):
        if frame not in self.records:
            raise ValueError(f"Source log is missing frame {frame}")
        record = self.records[frame]
        if self.m._frame_forces_static_hold(frame, self.turning_start_frame):
            return "turning_start_frame"
        search = record.get("search") or {}
        status = search.get("observation_status", search.get("status"))
        if status == "forced_static":
            return "turning_start_frame"
        if record.get("has_pose") is False or status == "unavailable_geometry":
            return "unavailable_geometry"
        return None

    def hold_transition(self, frame, state):
        observations = {}
        if self.track_sticker_observations:
            observations = self.held_sticker_observations(frame, state)
        return (Transition(state, 0., (0.,), dict(
            state_moves=(), state_score=0., observation_status=self.hold_reason(frame),
            color_observed_count=0, **observations), None),)

    def held_sticker_observations(self, frame, state):
        """Observe the fixed opening pose without move fitting or rescue."""
        key = frame, self.key(state)
        if key in self.held_observations:
            return self.held_observations[key]
        observed = correct = 0
        record = self.records[frame]
        pose = record.get('search_pose') or {}
        if pose.get('rvec') is None:
            pose = record.get('selected_pose') or {}
        if (self.hold_reason(frame) == 'turning_start_frame' and self.initial_colors
                and pose.get('rvec') is not None and pose.get('sticker_color_names')):
            np, m = self.np, self.m
            geometry_state = ReplayScorer.state_from_key(self, self.key(state))
            candidate = m.StickerPnPCandidate(sticker_type=pose['sticker_type'], sticker_average_confidence=1.,
                image_points=np.empty((0, 2)), object_points=np.empty((0, 3)),
                projected_sticker_points=np.empty((0, 2)), reprojection_error=0.,
                rvec=np.asarray(pose['rvec']), tvec=np.asarray(pose['tvec']),
                camera_matrix=np.asarray(pose['camera_matrix']), dist_coeffs=np.asarray(pose['dist_coeffs']),
                image_size=tuple(pose['image_size']), cube_geometry_state=geometry_state,
                sticker_mask_centers=tuple(np.asarray(v) for v in pose['sticker_mask_centers']),
                sticker_color_names=tuple(pose['sticker_color_names']))
            groups = m.projected_color_observations(candidate)
            for index, identity in enumerate(m.canonical_sticker_identities()):
                if identity.label in groups:
                    observed |= 1 << index
                    if self.initial_colors[identity.label] in groups[identity.label]:
                        correct |= 1 << index
        result = dict(sticker_observed_mask_hex=f'{observed:014x}',sticker_correct_mask_hex=f'{correct:014x}')
        self.held_observations[key] = result
        return result

    def prepare(self, frame):
        if frame in self.prepared:
            self.prepared.move_to_end(frame)
            return self.prepared[frame]
        if self.hold_reason(frame) is not None:
            return None
        r = self.records[frame]
        p = r.get("search_pose") or {}
        if p.get("rvec") is None:
            p = r.get("selected_pose") or {}
        options = ((r.get("search") or {}).get("geometry") or {}).get("options")
        if p.get("rvec") is None or not options:
            raise ValueError(f"Missing replay pose/geometry at frame {frame}; regenerate nine-class observations")
        if p.get("assignment_image_points") and any(
                o.get("keypoint_mean_error") == 0. and o.get("keypoint_inlier_count") == 0
                for o in options):
            raise ValueError(f"Frame {frame} contains cached zero-keypoint fallback geometry; "
                             "regenerate observations with whole-cube scoring")
        np, m = self.np, self.m
        candidate = m.StickerPnPCandidate(sticker_type=p["sticker_type"], sticker_average_confidence=1,
            image_points=np.zeros((4, 2)), object_points=np.zeros((4, 3)), projected_sticker_points=np.zeros((4, 2)),
            camera_matrix=np.array(p["camera_matrix"]), dist_coeffs=np.array(p["dist_coeffs"]),
            rvec=np.array(p["rvec"]), tvec=np.array(p["tvec"]), reprojection_error=0,
            sticker_mask_centers=tuple(np.array(x) for x in p["sticker_mask_centers"]),
            sticker_mask_center_uncertainties=tuple(p["sticker_mask_center_uncertainties"]),
            sticker_color_names=tuple(p.get("sticker_color_names") or r.get("turn_color_observation_names") or ()),
            image_size=tuple(p["image_size"]))
        matrix = (m.turn_color_match_matrix(candidate, expected_color_by_label=self.initial_colors)
                  if self.initial_colors and candidate.sticker_color_names else r.get("turn_color_matches"))
        if matrix is None:
            raise ValueError(f"Frame {frame} has no nine-class color matches; regenerate the old probability log")
        matrix = np.asarray(matrix, dtype=np.float64)
        if (matrix.shape != (54, len(candidate.sticker_mask_centers))
                or not np.isfinite(matrix).all() or not np.isin(matrix, (0., 1.)).all()):
            raise ValueError(f"Frame {frame} must contain a binary 54-by-observation color match matrix")
        def metric(value):
            # JSON represents nonfinite geometry as null. Keep it invalid
            # through replay instead of dropping it or interpreting it as zero.
            return float(value) if value is not None else float("inf")

        geometry = []
        for option in options:
            keypoint, sticker = metric(option["keypoint_mean_error"]), metric(option["heatmap_mean_error"])
            robust = option.get("global_robust_error", option.get("robust_error", keypoint+sticker))
            geometry_score = keypoint + sticker if m.turn_geometry_has_layer_support(option) else float("inf")
            geometry.append(dict(option, combined_mean_error=geometry_score,
                keypoint_error=keypoint, sticker_error=sticker, robust_error=metric(robust),
                p90_error=metric(option["keypoint_p90_error"]),
                layer_rotations=tuple(m.CubeLayerRotation(**rotation) for rotation in option["layer_rotations"]),
                assignment_projected_points=(np.asarray(option['assignment_projected_points'])
                    if option.get('joint_pose_fit') else None),
                assignment_object_points=(np.asarray(option['assignment_object_points'])
                    if option.get('joint_pose_fit') else None),
                assignment_projected_class_ids=(np.asarray(option['assignment_projected_class_ids'])
                    if option.get('joint_pose_fit') else None)))
        geometry = tuple(geometry)
        projections = {}
        if self.color_confusion_scoring:
            table = self.tracked_color_confusions
            projections['color_confusion_costs'] = m.turn_color_confusion_costs(
                candidate, self.initial_colors or {}, self.color_confusion_cost,
                confusion_table=m.COLOR_CLASS_CONFUSIONS if table is None else table)
        result = (candidate, geometry, matrix,
                  m.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION*m.projected_cube_diameter(candidate), projections)
        self.prepared[frame] = result
        if len(self.prepared) > 32:
            self.prepared.popitem(last=False)
        return result

    def __call__(self, frame, state):
        if self.hold_reason(frame) is not None:
            return self.hold_transition(frame, state)
        cache_key = (frame, self.key(state))
        if cache_key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        m = self.m
        c, geometry, color_matches, limit, projections = self.prepare(frame)
        self.scoring_calls += 1
        pruned_before = projections.get('confusion_pruned_fits', 0)
        batches = m._score_turn_color_offsets_for_states_on_gpu(c, geometry, limit, self.device,
            color_matches, (state,), projection_cache=projections,
            balance_geometry_color=self.balance_geometry_color, minimum_color_accuracy=self.minimum_color_accuracy,
            include_sticker_observations=self.track_sticker_observations,
            confusion_costs=projections.get('color_confusion_costs'), confusion_cost=self.color_confusion_cost,
            prune_color_confusions=self.color_confusion_pruning,
            preserve_layer_activity=self.preserve_layer_activity)
        rejected = projections.get('confusion_pruned_fits', 0)-pruned_before
        self.confusion_pruned_fits += rejected
        if batches is None:
            raise RuntimeError(f"Scoring failed at frame {frame}")
        unique = {}
        for trial in batches[0]:
            score = float(trial["state_score"])
            if not math.isfinite(score) or m.turn_trial_has_invalid_color(trial):
                continue
            after = state.apply_moves(tuple(trial["state_moves"]))
            direction = m.state_update_layer_direction(trial)
            key = (self.key(after), direction,
                   m.turn_trial_activity_layer(trial) if self.preserve_layer_activity else None)
            prior = unique.get(key)
            if prior is None or tuple(trial["rank"]) < prior.rank:
                payload = m.turn_trial_log_payload(trial)
                unique[key] = Transition(after, score, tuple(trial["rank"]), payload, direction)
        result = tuple(sorted(unique.values(), key=lambda t: t.rank))
        if not result and rejected:
            self.confusion_dead_end_states += 1
        self.cache[cache_key] = result
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result


class PreparedReplayScorer(RotationBudgetScorerMixin, ReplayScorer):
    """Compile fixed frame observations once; score states by identity gathers.

    Optional geometry shortlisting limits the fits and slab updates before
    per-state color scoring. Compact frame caches retain observation assignments
    and color matches. Floating point reductions can differ from the legacy CPU scorer
    at roundoff scale; no scores are rounded for search or dominance keys.
    Production search keeps states as 54-byte position-to-identity permutations.
    The update-context hook filters illegal moves before color work or children.
    """

    def __init__(self, renderer, records, device, cache_size=4096, *, initial_colors=None,
                 turning_start_frame=None, balance_geometry_color=False, minimum_color_accuracy=None,
                 track_sticker_observations=False,
                 color_confusion_pruning=False, color_confusion_cost=.5,
                 color_confusion_scoring=False, preserve_layer_activity=False, track_color_confusions=False,
                 tracked_color_confusions=None, parallel_layers=False, geometry_shortlist_size=None):
        if str(device) != "cpu":
            raise ValueError("Prepared replay scoring currently requires CPU")
        super().__init__(renderer, records, device, cache_size,
                         initial_colors=initial_colors, turning_start_frame=turning_start_frame,
                         balance_geometry_color=balance_geometry_color, minimum_color_accuracy=minimum_color_accuracy,
                         track_sticker_observations=track_sticker_observations,
                         color_confusion_pruning=color_confusion_pruning, color_confusion_cost=color_confusion_cost,
                         color_confusion_scoring=color_confusion_scoring,
                         preserve_layer_activity=preserve_layer_activity,
                         tracked_color_confusions=tracked_color_confusions)
        self.track_color_confusions = track_color_confusions
        self.parallel_layers = parallel_layers
        if geometry_shortlist_size is not None and (
                not isinstance(geometry_shortlist_size, int) or geometry_shortlist_size < 1):
            raise ValueError("Geometry shortlist size must be a positive integer")
        self.geometry_shortlist_size = geometry_shortlist_size
        self.compiled_frames = {}
        identities = renderer.canonical_sticker_identities()
        canonical_indices = renderer._turn_gpu_static_geometry()[-1]
        position_indices = {position: index for index, position in enumerate(self.positions)}
        self.geometry_positions = self.np.asarray([
            position_indices[(identities[int(i)].cubie_index,
                              identities[int(i)].axis, identities[int(i)].side)]
            for i in canonical_indices], dtype=self.np.intp)
        canonical_by_label = {identity.label: index for index, identity in enumerate(identities)}
        self.canonical_from_encoded = self.np.asarray([
            canonical_by_label[label] for label in sorted(self.identity_indices)], dtype=self.np.intp)
        orientation_sources = []
        for symmetry in renderer.CUBE_SYMMETRY_MATRICES:
            source_by_target = dict(renderer._geometry_symmetry_permutation(
                tuple(int(v) for v in symmetry.T.ravel())))
            orientation_sources.append([position_indices[source_by_target[position]] for position in self.positions])
        self.orientation_sources = self.np.asarray(orientation_sources, dtype=self.np.intp)
        self.move_permutations = {}
        self.face_groups = tuple(tuple(index for index, position in enumerate(self.positions)
                                      if position[1:] == (axis, side))
                                 for axis in range(3) for side in (-1, 1))
        self.color_configurations_scored = 0
        self.child_states_built = 0
        self.row_cache = OrderedDict()

    def key(self, state):
        # Search, queue storage and dominance keys share the same 54-byte value.
        return state if isinstance(state, bytes) else super().key(state)

    def state_from_key(self, value):
        return value

    def apply_moves(self, state, moves):
        if not isinstance(state, bytes):
            return state.apply_moves(moves)
        moves = tuple(moves)
        if not moves:
            return state
        return fast.permute(state, self.move_sources(moves))

    def move_sources(self, moves):
        """Cache the exact position permutation separately from state creation."""
        np = self.np
        if moves not in self.move_permutations:
            indices = {position: index for index, position in enumerate(self.positions)}
            sources = self.m.CubeGeometryState.identity().apply_moves(moves).facelets
            identity_sources = self.m.CubeGeometryState.identity().facelets
            source_by_label = {label: indices[position] for position, label in identity_sources.items()}
            self.move_permutations[moves] = np.asarray(
                [source_by_label[sources[target]] for target in self.positions], dtype=np.intp)
        return self.move_permutations[moves]

    def solved_predicate(self, sticker_colors):
        valid_colors = set(self.m.CubeState.solved().colors.values())
        labels = sorted(self.identity_indices)
        if not sticker_colors or any(sticker_colors.get(label) not in valid_colors for label in labels):
            return lambda state: False
        color_codes = {color: index for index, color in enumerate(sorted(valid_colors))}
        translation = bytes(color_codes[sticker_colors[label]] for label in labels) + bytes(202)
        groups = self.face_groups

        def solved(state):
            colors = self.key(state).translate(translation)
            return (all(colors[position] == colors[group[0]] for group in groups for position in group[1:])
                    and len({colors[group[0]] for group in groups}) == 6)

        return solved

    def _geometry_score(self, option):
        if option.get('depth_fallback'):
            return float(option['recovery_score'])
        score = float(option['combined_mean_error'])
        if self.balance_geometry_color and math.isfinite(score):
            score = float(self.m.balanced_geometry_cost(option['keypoint_error'], option['sticker_error']))
        return score

    def _geometry_shortlist(self, geometry):
        """Rank distinct misalignments, counting flat labels once and admitting cutoff ties."""
        if self.geometry_shortlist_size is None:
            return tuple(range(len(geometry)))
        if geometry and all(option.get('geometry_bank_policy') == 'sticker_assignments_plus_flat'
                            for option in geometry):
            # The joint fitter has already selected assignment-distinct rows.
            # Its flat fallback is reserved independently of geometry score.
            ordinary = tuple(i for i, option in enumerate(geometry)
                             if not option.get('flat_fallback') and math.isfinite(self._geometry_score(option)))
            flat = tuple(i for i, option in enumerate(geometry)
                         if option.get('flat_fallback') and math.isfinite(self._geometry_score(option)))
            return ordinary[:self.geometry_shortlist_size] + flat[:1]
        scores, best = [], {}
        for option in geometry:
            score = self._geometry_score(option)
            rotations = option.get('layer_rotations', ())
            angles = tuple(-45. if angle == 45. else angle
                           for angle in self.m.parallel_layer_angles(rotations))
            key = (int(option['layer_axis']), angles) if any(angles) else None
            scores.append(score)
            if math.isfinite(score):
                best[key] = min(best.get(key, float('inf')), score)
        ranked = sorted(best.values())
        if not ranked:
            return ()
        cutoff = ranked[min(self.geometry_shortlist_size, len(ranked)) - 1]
        return tuple(index for index, score in enumerate(scores)
                     if math.isfinite(score) and score < cutoff + GEOMETRY_TIE_TOLERANCE)

    def prepare(self, frame):
        if frame in self.compiled_frames:
            return self.compiled_frames[frame]
        if self.hold_reason(frame) is not None:
            return None
        np, m = self.np, self.m
        candidate, geometry, color_matches, limit, projections = super().prepare(frame)
        # A quarter turn permutes sticker positions without changing the surface.
        # Project only the base partial angles; offsets are state permutations.
        # Collapse the identical static poses across the twelve layer labels too.
        joint = bool(((self.records[frame].get('search') or {}).get('geometry') or {}).get('joint_refined'))
        if joint:
            projected_geometry = list(geometry)
            projection_indices = list(range(len(geometry)))
            nearest_base = np.asarray([o['cached_sticker_indices'] for o in geometry], dtype=np.intp)
            valid_base = np.asarray([o['cached_sticker_valid'] for o in geometry], dtype=bool)
            assignments = (nearest_base, valid_base)
        else:
            projected_geometry, projection_indices, projected_keys = [], [], {}
            for option in geometry:
                rotations = option.get("layer_rotations", ())
                angle = float(rotations[0].angle_degrees) if rotations else 0.
                key = ((int(option["layer_axis"]), int(option["layer_side"]),
                        int(option["layer_count"]), angle) if abs(angle) > 1e-6 else None)
                if key not in projected_keys:
                    projected_keys[key] = len(projected_geometry)
                    projected_geometry.append(option)
                projection_indices.append(projected_keys[key])
            batches = m._score_turn_color_offsets_for_states_on_gpu(
                candidate, tuple(projected_geometry), limit, self.device, color_matches,
                (m.CubeGeometryState.identity(),), projection_cache=projections, color_offsets=(0.,))
            if batches is None:
                raise RuntimeError(f"Scoring preparation failed at frame {frame}")
            assignments = projections.get("assignments", {}).get("value")
            if self.parallel_layers and assignments is not None:
                nearest_base, valid_base = (value.cpu().numpy() for value in assignments[:2])
                extra, extra_nearest, extra_valid = self._parallel_geometry(frame, candidate, limit)
                flat = next((i for i, option in enumerate(geometry) if option.get('is_baseline')), None)
                flat_row = projection_indices[flat] if flat is not None else None
                projection_indices.extend(range(len(projected_geometry), len(projected_geometry) + len(extra)))
                geometry += tuple(extra)
                projected_geometry.extend(extra)
                if extra:
                    nearest_base = np.concatenate((nearest_base, extra_nearest))
                    valid_base = np.concatenate((valid_base, extra_valid))
                if flat_row is not None:
                    geometry += tuple(dict(geometry[flat], parallel_layers=True, layer_axis=axis,
                        layer_side=1, layer_count=1, layer_support=(0, 0, 0), label='static') for axis in range(3))
                    projection_indices.extend((flat_row,) * 3)
            elif assignments is not None:
                nearest_base, valid_base = (value.cpu().numpy() for value in assignments[:2])
        group_ids, groups, action_specs = [], {}, []
        templates, option_rows, source_permutations, projection_rows, flat_commit_rows = [], [], [], [], []
        rotation_previous_rows, rotation_degrees = [], []
        rotation_shapes, rotation_plans, rotation_aux_rows = {}, {}, {}
        canonical_indices = m._turn_gpu_static_geometry()[-1]
        geometry_from_canonical = np.argsort(canonical_indices)
        permutations = {}
        selected_geometry = self._geometry_shortlist(geometry)
        if self.rotation_budget_enabled:
            selected_geometry, rotation_shapes, rotation_plans = self._rotation_plans(
                frame, geometry, selected_geometry, projection_indices,
                nearest_base if assignments is not None else None,
                valid_base if assignments is not None else None)
        flat_indices = tuple(index for index in selected_geometry
                             if not any(m.parallel_layer_angles(geometry[index].get('layer_rotations', ()))))
        flat_index = min(flat_indices, key=lambda index: self._geometry_score(geometry[index]), default=None)
        flat_turn_fallback = (not self.rotation_budget_enabled and self.geometry_shortlist_size is not None and flat_index is not None and
            self._geometry_score(geometry[flat_index]) <
            min(self._geometry_score(geometry[index]) for index in selected_geometry) + GEOMETRY_TIE_TOLERANCE)
        if flat_turn_fallback:
            # A complete turn can fall between observations. Reuse the same
            # static projection to consider commits on every axis.
            axes = {int(geometry[index]['layer_axis']) for index in flat_indices}
            for axis in range(3):
                if axis not in axes:
                    selected_geometry += (len(geometry),)
                    geometry += (dict(geometry[flat_index], layer_axis=axis, parallel_layers=True,
                                      layer_support=(0, 0, 0)),)
                    projection_indices.append(projection_indices[flat_index])
        selected = set(selected_geometry)
        geometry_scores, inliers, p90s, layer_inliers = [], [], [], []
        for option_index, option in enumerate(geometry):
            axis, side, width = (int(option[name]) for name in
                                 ("layer_axis", "layer_side", "layer_count"))
            active = tuple(bool(angle) for angle in m.parallel_layer_angles(option.get('layer_rotations', ())))
            if self.geometry_shortlist_size is not None and not option.get('parallel_layers'):
                # Older fits expose evidence for the fitted layer group. Carry
                # that support to its slabs when admitting subset commits.
                support = int(option.get('layer_inlier_count', 0))
                option = dict(option, parallel_layers=True,
                              layer_support=tuple(support if moving else 0 for moving in active))
            if option.get('parallel_layers'):
                actions = [(0., action, (axis, action)) for action, offsets in m.parallel_turn_actions(axis).items()
                           if self.geometry_shortlist_size is None or (flat_turn_fallback and not any(active)) or
                           all(not offset or moving for offset, moving in zip(offsets, active, strict=True))]
            else:
                actions = []
                for offset in (-90., 0., 90.):
                    notation = m.cube_rotation_to_notation(m.CubeLayerRotation(axis, side, offset, width))
                    actions.append((offset, (notation,) if notation is not None else (), (axis, side, width, offset)))
            trial = dict(option, geometry_option_index=option_index, layer_axis=axis, layer_side=side, layer_count=width,
                inlier_count=int(option.get('inlier_count', 0)), layer_inlier_count=int(option.get('layer_inlier_count', 0)),
                keypoint_inlier_count=int(option.get('keypoint_inlier_count', 0)),
                keypoint_layer_inlier_count=int(option.get('keypoint_layer_inlier_count', 0)),
                sticker_match_count=int(option.get('sticker_match_count', 0)))
            geometry_score = self._geometry_score(option)
            templates.append((m.turn_trial_log_payload(trial), float(option.get('robust_error', float('inf')))))
            if option_index not in selected:
                continue
            if self.rotation_budget_enabled:
                action_rows = [(0., moves, (option_index, moves), previous, cost)
                               for previous, moves, cost in rotation_plans[option_index]]
                rotation_aux_rows[option_index] = len(option_rows) + len(action_rows)
                action_rows.append((0., (), (option_index, ()), -2, 0.))
            else:
                action_rows = [(offset, moves, group, -2, 0.) for offset, moves, group in actions]
            for offset, moves, group, previous, degrees in action_rows:
                if group not in groups:
                    groups[group] = len(groups)
                    direction = None if self.rotation_budget_enabled else m.state_update_layer_direction(dict(trial, state_moves=moves))
                    action_specs.append((moves, offset, direction))
                group_ids.append(groups[group])
                option_rows.append(option_index)
                rotation_previous_rows.append(previous)
                rotation_degrees.append(degrees)
                flat_commit_rows.append(bool(flat_turn_fallback and moves and not any(active)))
                if moves not in permutations:
                    moved_indices = m._turn_gpu_identity_indices_for_state(
                        m.CubeGeometryState.identity().apply_moves(moves), canonical_indices)
                    permutations[moves] = geometry_from_canonical[moved_indices]
                source_permutations.append(permutations[moves])
                projection_rows.append(projection_indices[option_index])
                geometry_scores.append(geometry_score)
                inliers.append(trial["inlier_count"])
                p90s.append(trial["p90_error"])
                layer_inliers.append(m.turn_layer_inliers_for_ranking(trial))
        if assignments is None:
            nearest = weights = match_matrix = denominator = has_weight = None
            matches = np.zeros(len(option_rows), dtype=np.int32)
        else:
            rows = np.asarray(projection_rows, dtype=np.intp)
            sources = np.asarray(source_permutations, dtype=np.intp).reshape(-1, len(canonical_indices))
            nearest = np.take_along_axis(sources, nearest_base[rows], axis=1).astype(np.uint8)
            weights = valid_base[rows].copy()
            count = nearest.shape[1]
            match_matrix = np.asarray(color_matches[:, :count], dtype=np.float64)
            occupied = np.zeros((len(option_rows), 54), dtype=bool)
            np.logical_or.at(occupied, (np.arange(len(option_rows))[:, None], nearest), weights)
            matches = occupied.sum(axis=1).astype(np.int32)
            denominator = np.maximum(matches, 1)
            has_weight = matches > 0
        confusion_costs = projections.get('color_confusion_costs')
        if self.track_color_confusions and confusion_costs is None:
            table = self.tracked_color_confusions
            confusion_costs = m.turn_color_confusion_costs(
                candidate, self.initial_colors or {}, self.color_confusion_cost,
                confusion_table=m.COLOR_CLASS_CONFUSIONS if table is None else table)
        result = dict(nearest=nearest, weights=weights, match_matrix=match_matrix,
            confusion_costs=confusion_costs,
            denominator=denominator, has_weight=has_weight,
            observation_indices=None if nearest is None else np.arange(nearest.shape[1])[None, :],
            matches=matches, geometry_scores=np.asarray(geometry_scores),
            inliers=np.asarray(inliers), p90s=np.asarray(p90s),
            layer_inliers=np.asarray(layer_inliers), group_ids=np.asarray(group_ids, dtype=np.intp),
            templates=templates, option_rows=np.asarray(option_rows, dtype=np.intp), action_specs=action_specs,
            projection_count=len(projected_geometry),
            selected_geometry_indices=selected_geometry,
            flat_turn_fallback=flat_turn_fallback,
            flat_commit_rows=np.asarray(flat_commit_rows, dtype=bool),
            state_actions=frozenset(moves for moves, _offset, _direction in action_specs if moves),
            rotation_budget=self.rotation_budget_enabled, rotation_geometries=rotation_shapes,
            rotation_plans=rotation_plans, rotation_aux_rows=rotation_aux_rows,
            rotation_previous_rows=np.asarray(rotation_previous_rows, dtype=np.intp),
            rotation_degrees=np.asarray(rotation_degrees), rotation_reverse_cache={},
            rotation_allowed=np.ones(len(option_rows), dtype=bool),
            rotation_budgets=np.full(len(option_rows), self.rotation_limits[frame] if self.rotation_budget_enabled else 0.),
            rotation_limit=self.rotation_limits[frame] if self.rotation_budget_enabled else None,
            source_geometry_count=len(geometry), configuration_count=len(option_rows))
        self.compiled_frames[frame] = result
        result["selections"] = {}
        self.prepared.pop(frame, None)
        return result

    @staticmethod
    def _template(prepared, index):
        cached = prepared.setdefault('row_templates', {})
        if index in cached:
            return cached[index]
        option = int(prepared['option_rows'][index])
        payload, robust = prepared['templates'][option]
        moves, offset, direction = prepared['action_specs'][prepared['group_ids'][index]]
        action = ' '.join(moves) or 'hold'
        if payload.get('parallel_layers'):
            trial_id, label = action, f"{payload['label']} · {action}"
        else:
            trial_id, label = f'{offset:+g}', f"{payload['label']} offset {offset:+.0f}"
        result = dict(payload, trial_id=f'{option}:{trial_id}', label=label,
                    is_baseline=bool(payload.get('is_baseline') and not moves),
                    state_moves=moves, color_offset=offset)
        if prepared['rotation_budget']:
            previous = int(prepared['rotation_previous_rows'][index])
            result.update(rotation_budget=True, rotation_context=option,
                          previous_geometry_index=previous if previous >= 0 else None,
                          angular_travel_degrees=float(prepared['rotation_degrees'][index]),
                          rotation_budget_degrees=float(prepared['rotation_budgets'][index]))
        cached[index] = result, direction, robust
        return cached[index]

    def _parallel_geometry(self, frame, candidate, limit):
        """Deduplicate assignments, preserving distinct angles for shortlisting."""
        record = self.records[frame]
        pose = record.get('search_pose') or record.get('selected_pose') or {}
        image_points = pose.get('assignment_image_points')
        if image_points is None:
            return [], [], []
        m, np = self.m, self.np
        candidate = replace(candidate, assignment_image_points=np.asarray(image_points),
                            assignment_observed_class_ids=pose.get('assignment_observed_class_ids'))
        extra, nearest_rows, valid_rows = [], [], []
        for axis in range(3):
            options, nearest, valid = m.parallel_turn_frame_geometry(candidate, limit, axis)
            unique = {}
            for i, option in enumerate(options):
                if not m.turn_geometry_has_layer_support(option):
                    continue
                keypoint, sticker = option['keypoint_error'], option['sticker_error']
                geometry = m.balanced_geometry_cost(keypoint, sticker) if self.balance_geometry_color else keypoint + sticker
                if not math.isfinite(geometry):
                    continue
                rank = (geometry, -option['inlier_count'], option['p90_error'],
                        -m.turn_layer_inliers_for_ranking(option))
                key = np.where(valid[i], nearest[i], 255).tobytes()
                if self.geometry_shortlist_size is not None:
                    angles = tuple(-45. if angle == 45. else angle
                                   for angle in m.parallel_layer_angles(option['layer_rotations']))
                    key = angles, key
                prior = unique.get(key)
                if prior is None or rank < prior[0]:
                    unique[key] = rank, i
            rows = sorted(i for _, i in unique.values())
            extra.extend(options[i] for i in rows)
            nearest_rows.extend(nearest[rows])
            valid_rows.extend(valid[rows])
        return extra, nearest_rows, valid_rows

    def __call__(self, frame, state):
        return self._evaluate(frame, state, None)

    def score_hold(self, frame, state):
        return next(iter(self._evaluate(frame, state, False)), None)

    def score_action(self, frame, state, action):
        return self._evaluate(frame, state, ('action', action))

    def frame_actions(self, frame):
        if self.hold_reason(frame) is not None:
            return frozenset()
        return self.prepare(frame)['state_actions']

    def flat_turn_fallback(self, frame):
        return self.hold_reason(frame) is None and self.prepare(frame)['flat_turn_fallback']

    def without_flat_turns(self, frame, state):
        return self._evaluate(frame, state, ('no_flat_turns',))

    def for_update_context(self, frame, state, update, min_update_gap):
        if self.hold_reason(frame) is not None:
            return self.hold_transition(frame, state), 0
        last, direction, streak = update
        restriction = None
        if min_update_gap > 1 and last is not None and frame - last < min_update_gap:
            restriction = direction if frame-last == 1 and streak < 2 and direction is not None else False
        prepared = self.prepare(frame)
        selection = self._selection(prepared, restriction)
        return self._evaluate(frame, state, restriction), selection["skipped_moves"]

    def _selection(self, prepared, restriction):
        if restriction in prepared["selections"]:
            return prepared["selections"][restriction]
        np = self.np
        if restriction is None:
            rows = np.arange(prepared['configuration_count'])
            skipped = 0
        else:
            specs = prepared['action_specs']
            if restriction and restriction[0] in ('rotation', 'geometry_hold'):
                if restriction[0] == 'rotation':
                    rows = np.flatnonzero((prepared['rotation_previous_rows'] == restriction[1])
                                          & prepared['rotation_allowed'])
                else:
                    rows = np.asarray([prepared['rotation_aux_rows'][restriction[1]]], dtype=np.intp)
                skipped = 0
            elif restriction == ('no_flat_turns',):
                allowed = [True] * len(specs)
            elif restriction and restriction[0] == 'action':
                allowed = [moves == restriction[1] for moves, _, _ in specs]
            else:
                allowed = [not moves or direction == restriction for moves, _, direction in specs]
            if not (restriction and restriction[0] in ('rotation', 'geometry_hold')):
                rows = np.flatnonzero(np.asarray(allowed)[prepared['group_ids']])
                if restriction == ('no_flat_turns',):
                    rows = rows[~prepared['flat_commit_rows'][rows]]
                skipped = len({moves for moves, _, _ in specs}) - len({
                    moves for keep, (moves, _, _) in zip(allowed, specs, strict=True) if keep})
        result = {name: prepared[name] if restriction is None or prepared[name] is None else prepared[name][rows]
                  for name in ("nearest", "weights", "denominator", "has_weight", "matches",
                               "geometry_scores", "inliers", "p90s", "layer_inliers", "group_ids")}
        result.update(rows=rows, skipped_moves=skipped)
        if self.parallel_layers and result['nearest'] is not None:
            # Color depends only on which observations were assigned to which
            # positions, even if those assignments came from different angles.
            assignment = np.where(result['weights'], result['nearest'], 255)
            _, color_rows, color_inverse = np.unique(assignment, axis=0, return_index=True, return_inverse=True)
            result.update(color_rows=color_rows, color_inverse=color_inverse.reshape(-1))
        if self.preserve_layer_activity:
            templates = [self._template(prepared, index)[0] for index in rows]
            active_holds = np.asarray([not template['state_moves'] and
                self.m.turn_trial_activity_layer(template) is not None for template in templates])
            result['group_ids'] = 2*result['group_ids']+active_holds
        result['rank_groups'] = np.unique(result['group_ids'], return_inverse=True)[1].astype(np.intp)
        result['rank_inliers'] = np.asarray(result['inliers'], dtype=np.float64)
        result['rank_layer_inliers'] = np.asarray(result['layer_inliers'], dtype=np.float64)
        prepared["selections"][restriction] = result
        return result

    def _confusion_fields(self, prepared, index, identities):
        if prepared['confusion_costs'] is None:
            return {}
        if prepared['nearest'] is None:
            zero = self.np.zeros(1, dtype=int)
            metrics = dict(correct=zero, confused=zero, forbidden=zero)
        else:
            metrics = self.m.sticker_color_confusion_metrics(identities,
                prepared['nearest'][index:index+1], prepared['weights'][index:index+1],
                prepared['confusion_costs'], self.color_confusion_cost)
        if self.color_confusion_scoring:
            return self.m.color_confusion_trial_fields(metrics, 0, self.color_confusion_cost)
        return dict(color_forbidden_count=int(metrics['forbidden'][0]))

    def evaluate_many(self, frame, states, restriction=None):
        """Score a fixed batch without changing expansion order or admission rules."""
        np = self.np
        states = tuple(states)
        if not states or self.hold_reason(frame) is not None or not self.color_confusion_scoring:
            return tuple(self._evaluate(frame, state, restriction) for state in states)
        prepared = self.prepare(frame)
        selection = self._selection(prepared, restriction)
        if selection['nearest'] is None:
            return tuple(self._evaluate(frame, state, restriction) for state in states)
        rows = selection.get('color_rows', slice(None))
        indices = self.canonical_from_encoded[np.asarray([
            np.frombuffer(self.key(state), dtype=np.uint8)[self.geometry_positions] for state in states])]
        metrics = fast.color_metrics(self.m, indices, selection['nearest'][rows], selection['weights'][rows],
                                     prepared['confusion_costs'], self.color_confusion_cost)
        return tuple(self._evaluate(frame, state, restriction,
            _confusion={key: value[index] for key, value in metrics.items()})
            for index, state in enumerate(states))

    def _evaluate(self, frame, state, restriction, *, _confusion=None, _rows_only=False):
        if self.hold_reason(frame) is not None:
            return self.hold_transition(frame, state)
        np, m = self.np, self.m
        encoded = self.key(state)
        # The same physical state can have different legal next moves depending
        # on its update history. A filtered result cannot serve another history.
        cache_key = (frame, encoded, restriction, isinstance(state, bytes))
        cache = self.row_cache if _rows_only else self.cache
        if cache_key in cache:
            self.cache_hits += 1
            cache.move_to_end(cache_key)
            return cache[cache_key]
        prepared = self.prepare(frame)
        pruned_before = self.confusion_pruned_fits
        selection = self._selection(prepared, restriction)
        self.scoring_calls += 1
        color_rows = selection.get('color_rows', slice(None))
        color_inverse = selection.get('color_inverse', slice(None))
        self.color_configurations_scored += len(selection["rows"][color_rows])
        indices = None
        confusion = None
        if selection["nearest"] is None:
            errors = np.zeros(len(selection["rows"]))
        else:
            indices = self.canonical_from_encoded[
                np.frombuffer(encoded, dtype=np.uint8)[self.geometry_positions]]
            nearest = selection['nearest'][color_rows]
            weights = selection['weights'][color_rows]
            if self.color_confusion_scoring:
                confusion = (_confusion if _confusion is not None else fast.color_metrics(
                    m, indices, nearest, weights, prepared['confusion_costs'], self.color_confusion_cost))
                errors = confusion['error']
            else:
                observation_matches = weights * prepared["match_matrix"][
                    indices[nearest], prepared["observation_indices"]]
                face_matches = np.zeros((len(nearest), 54), dtype=np.float64)
                np.add.at(face_matches, (np.arange(len(nearest))[:, None], nearest), observation_matches)
                correct = np.minimum(face_matches, 1.0).sum(axis=1)
                errors = np.where(selection['has_weight'][color_rows],
                                  1.0 - correct / selection['denominator'][color_rows], float('inf'))
            errors = errors[color_inverse]
        costs = selection["geometry_scores"] + m.PNP_TURN_COLOR_WEIGHT * errors
        if self.balance_geometry_color:
            costs = .5 * selection["geometry_scores"] + .5 * errors
        if self.minimum_color_accuracy is not None:
            costs = np.where(errors <= 1. - self.minimum_color_accuracy, costs, float("inf"))
        if confusion is not None and self.color_confusion_pruning:
            forbidden = confusion['forbidden'][color_inverse] > 0
            self.confusion_pruned_fits += int(forbidden.sum())
            costs = np.where(forbidden, np.inf, costs)
        best = fast.best_rows(costs, selection)
        observation_masks = {}
        if self.track_sticker_observations:
            observed = correct = np.zeros(len(best), dtype=np.uint64)
            if selection['nearest'] is not None:
                observed, correct = m.sticker_color_observation_masks(indices,
                    selection['nearest'][best], selection['weights'][best], prepared['match_matrix'])
            observation_masks = {int(index):dict(sticker_observed_mask_hex=f'{int(o):014x}',
                                                sticker_correct_mask_hex=f'{int(c):014x}')
                                 for index,o,c in zip(best,observed,correct,strict=True)}
        data = dict(frame=frame, state=state, encoded=encoded, prepared=prepared, selection=selection,
                    costs=costs, errors=errors, best=best, confusion=confusion, indices=indices,
                    color_inverse=color_inverse, observation_masks=observation_masks)
        if _rows_only:
            cache[cache_key] = data
            if len(cache) > self.cache_size:
                cache.popitem(last=False)
            return data
        result = self._transitions_from_rows(data)
        if not result and self.confusion_pruned_fits > pruned_before:
            self.confusion_dead_end_states += 1
        cache[cache_key] = result
        if len(cache) > self.cache_size:
            cache.popitem(last=False)
        return result

    def _transitions_from_rows(self, data, selected=None):
        m = self.m
        state, encoded, prepared, selection = (data[key] for key in ('state', 'encoded', 'prepared', 'selection'))
        costs, errors, confusion, indices, color_inverse, observation_masks = (
            data[key] for key in ('costs', 'errors', 'confusion', 'indices', 'color_inverse', 'observation_masks'))
        best = data['best'] if selected is None else selected
        compact = self.rotation_budget_enabled and isinstance(state, bytes)
        unique, moved = {}, {(): (state, self.move_sources(()).tobytes() if compact else encoded)}
        for index in best:
            score, color_error = float(costs[index]), float(errors[index])
            if not math.isfinite(score) or not math.isfinite(color_error):
                continue
            template, direction, robust = self._template(prepared, selection['rows'][index])
            moves = template["state_moves"]
            if moves not in moved:
                sources = self.move_sources(moves)
                after = fast.DeferredState(state, sources) if compact else self.apply_moves(state, moves)
                moved[moves] = (after, sources.tobytes() if compact else self.key(after))
                self.child_states_built += 1
            after, after_key = moved[moves]
            rank = (score, -float(selection["inliers"][index]), float(selection["p90s"][index]),
                    -float(selection["layer_inliers"][index]))
            key = (after_key, direction,
                   template['geometry_option_index'] if self.rotation_budget_enabled else
                   m.turn_trial_activity_layer(template) if self.preserve_layer_activity else None)
            prior = unique.get(key)
            if prior is None or rank < prior.rank:
                if confusion is not None:
                    color_index = index if isinstance(color_inverse, slice) else int(color_inverse[index])
                    confusion_fields = m.color_confusion_trial_fields(confusion, color_index, self.color_confusion_cost)
                else:
                    confusion_fields = self._confusion_fields(prepared, int(selection['rows'][index]), indices)
                payload = fast.ScoredPayload(template, m, robust, self.balance_geometry_color,
                    float(selection['geometry_scores'][index]), state_score=score, global_mean_error=score,
                    color_error=color_error, color_match_count=int(selection['matches'][index]),
                    **observation_masks.get(int(index), {}), **confusion_fields)
                transition = fast.CompactTransition if compact else Transition
                if not compact:
                    payload = dict(payload)
                unique[key] = transition(after, score, rank, payload, direction)
        result = tuple(sorted(unique.values(), key=lambda t: t.rank))
        return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--initial-scramble", required=True)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame", required=True, type=int)
    parser.add_argument("--video-frame-count", type=int,
                        help="Whole video length; defaults to the full source log's last frame index plus one. Set explicitly for a partial log.")
    parser.add_argument("--threshold", type=float, default=2., help="Normal rolling threshold (default: 2)")
    parser.add_argument("--fixed-threshold", action="store_true",
                        help="Reject rather than defer blocked continuations (legacy experiment)")
    parser.add_argument("--fps", type=float, default=60.)
    parser.add_argument("--min-state-update-gap", type=int, default=None,
                        help="Override the FPS-scaled move spacing (default: ceil(fps/30), with the consecutive double-turn exception)")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--scorer", choices=("auto", "prepared", "legacy"), default="auto",
                        help="Auto uses prepared base-pose mappings on CPU; MPS uses legacy scoring")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-seconds", type=float, default=5.)
    return parser


def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "progress.jsonl").exists():
        raise SystemExit("Choose a new output directory; existing search logs are never overwritten")
    m = load_renderer()
    import torch
    torch.set_num_threads(2)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; use --device cpu")
    records = {r["frame_index"]: r for r in map(json.loads, args.log.open())}
    if not records:
        raise SystemExit("Source log contains no frames")
    logged_frame_count = max(records)+1
    video_frame_count = args.video_frame_count if args.video_frame_count is not None else logged_frame_count
    if video_frame_count < logged_frame_count:
        raise SystemExit("Whole video frame count cannot be less than the logged source-frame extent")
    frames = tuple(range(args.start_frame, args.end_frame+1))
    if not frames:
        raise SystemExit("End frame precedes start frame")
    scorer_mode = ("prepared" if args.device == "cpu" else "legacy") if args.scorer == "auto" else args.scorer
    if scorer_mode == "prepared" and args.device != "cpu":
        raise SystemExit("Prepared replay scoring requires --device cpu")
    scorer_class = PreparedReplayScorer if scorer_mode == "prepared" else ReplayScorer
    scorer = scorer_class(m, records, torch.device(args.device))
    # Validate the entire interval before an unbudgeted search starts.
    for frame in frames:
        scorer.prepare(frame)
    initial = m.CubeGeometryState.identity()
    scrambled = m.CubeState.from_scramble(args.initial_scramble)
    colors = {label: scrambled.colors[position] for position, label in initial.facelets.items()}
    initial = scorer.state_from_key(scorer.key(initial))
    solved = scorer.solved_predicate(colors)
    min_gap = (m.turn_state_update_gap_frames(args.fps)
               if args.min_state_update_gap is None else args.min_state_update_gap)
    config = dict(algorithm="ranked_depth_first_full_solve" if args.fixed_threshold else "adaptive_ranked_depth_first_full_solve", log=str(args.log.resolve()),
        log_sha256=hashlib.sha256(args.log.read_bytes()).hexdigest(),
        initial_scramble=args.initial_scramble, start_frame=args.start_frame, end_frame=args.end_frame,
        video_frame_count=video_frame_count,
        video_frame_count_source="explicit" if args.video_frame_count is not None else "source_log_extent",
        console_frame_numbering="one-based whole-video position; source frame indices in logs remain zero-based",
        pruning_statistic="mean_best_three_of_four_combined_observation_costs", threshold=args.threshold,
        frame_or_state_budget=None, time_budget=None, beam_width=None, state_change_limit=None,
        min_state_update_gap_frames=min_gap, fps=args.fps, device=args.device,
        scorer=scorer_mode, projection_offsets="state permutations" if scorer_mode == "prepared" else "explicit projections",
        state_representation="54-byte permutation" if scorer_mode == "prepared" else "facelet dictionary",
        move_spacing_filter="before scoring" if scorer_mode == "prepared" else "after scoring",
        scorer_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        renderer_script_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
        state_update_gap_policy="FPS-scaled" if args.min_state_update_gap is None else "explicit override",
        consecutive_update_exception="only the second quarter-turn of the same layer, width, and direction",
        geometry_scope="all retained options in the source log, all discrete color offsets",
        orientation_policy="supplied fixed pose",
        goal="physical cube colors solved in any whole-cube orientation",
        reference_solution_used=False,
        threshold_policy="fixed_rejection" if args.fixed_threshold else "defer_then_retry_minimum_required_one_window",
        threshold_returns_to_normal="next frame",
        merge_key="frame,state,last_three_numeric_scores,move_spacing_context")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    memo = DiskMemo(args.output_dir / "exhausted.sqlite3")
    queue = None if args.fixed_threshold else DeferredBranches(args.output_dir / "deferred.sqlite3")
    with (args.output_dir / "progress.jsonl").open("w", buffering=1) as log:
        latest = {}
        def progress(item):
            video_position = item["furthest_frame_index"]+1 if item["furthest_frame_index"] is not None else 0
            item.update(scoring_calls=scorer.scoring_calls, score_cache_hits=scorer.cache_hits,
                        furthest_video_frame=video_position, total_video_frames=video_frame_count)
            if scorer_mode == "prepared":
                item.update(color_configurations_scored=scorer.color_configurations_scored,
                            child_states_built=scorer.child_states_built)
            latest.clear()
            latest.update(item)
            log.write(json.dumps(item, allow_nan=False)+"\n")
            print(f"furthest branch: {video_position}/{video_frame_count}", flush=True)
            temporary = args.output_dir / "status.tmp"
            temporary.write_text(json.dumps(item, indent=2))
            temporary.replace(args.output_dir / "status.json")
            if (args.output_dir / "STOP").exists():
                raise KeyboardInterrupt
        try:
            search = full_solve_dfs if args.fixed_threshold else adaptive_full_solve_dfs
            extra = {} if args.fixed_threshold else dict(deferred=queue,
                serialize_state=scorer.key, deserialize_state=scorer.state_from_key)
            outcome = search(initial, frames, scorer, solved, scorer.key,
                threshold=args.threshold, min_update_gap=min_gap, failed=memo,
                progress=progress, progress_seconds=args.progress_seconds, **extra)
            (args.output_dir / "result.json").write_text(json.dumps(dict(config=config, **outcome), indent=2))
        except KeyboardInterrupt:
            item = dict(latest, event="interrupted", status="interrupted")
            log.write(json.dumps(item)+"\n")
            (args.output_dir / "status.json").write_text(json.dumps(item, indent=2))
            (args.output_dir / "result.json").write_text(json.dumps(dict(config=config, **item), indent=2))
            print("Search interrupted; progress and exhausted-prefix memo preserved.", flush=True)
        finally:
            memo.close()
            if queue is not None:
                queue.close()


if __name__ == "__main__":
    main()
