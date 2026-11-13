"""Precompute angularly feasible routes between two residual cube geometries."""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import math


@dataclass(frozen=True)
class RotationGeometry:
    axis: int | None
    angles: tuple[float, float, float]

    @classmethod
    def make(cls, axis, angles):
        angles = tuple(float(value) for value in angles)
        return cls(int(axis) if any(angles) else None, angles)


FLAT = RotationGeometry(None, (0., 0., 0.))


@dataclass(frozen=True)
class RotationRoute:
    # Each group is (axis, negative/middle/positive slab quarter-turn offsets).
    groups: tuple
    degrees: float
    budget: float = 0.


def _groups(groups):
    result = []
    for axis, offsets in groups:
        offsets = tuple(int(value) % 4 for value in offsets)
        if result and result[-1][0] == axis:
            _, previous = result.pop()
            offsets = tuple((a + b) % 4 for a, b in zip(previous, offsets, strict=True))
        offsets = tuple(value if value <= 2 else value - 4 for value in offsets)
        if any(offsets):
            result.append((axis, offsets))
    return tuple(result)


def _relative(angles):
    return angles[0]-angles[1], angles[2]-angles[1]


def _direct_cost(start, end):
    """Concurrent outer-slab motion, each measured relative to the middle."""
    return max(abs(b-a) for a, b in zip(_relative(start), _relative(end), strict=True))


def _quarter_offsets(start, end, budget):
    # Bound the two relative angles before enumerating discrete updates. A
    # previously stationary outer slab can complete a turn relative to the
    # middle, so bounding individual absolute slab angles would miss routes.
    start_relative = (start[0] - start[1], start[2] - start[1])
    end_relative = (end[0] - end[1], end[2] - end[1])
    ranges = [range(math.ceil((-budget - b + a - 1e-9) / 90.),
                    math.floor((budget - b + a + 1e-9) / 90.) + 1)
              for a, b in zip(start_relative, end_relative, strict=True)]
    for negative, positive in product(*ranges):
        # Keep face, wide and slice interpretations of the relative update,
        # without adding arbitrary common whole-cube quarter rotations.
        for middle in range(-max(0, negative, positive), -min(0, negative, positive) + 1):
            yield negative + middle, middle, positive + middle


def _boundary_routes(geometry, budget, exiting):
    if geometry.axis is None:
        return (((), 0., (0, 0, 0)),)
    angles = geometry.angles
    result = []
    start, end = (angles, FLAT.angles) if exiting else (FLAT.angles, angles)
    for offsets in _quarter_offsets(start, end, budget):
        boundary = tuple(90. * value for value in offsets)
        if exiting:
            cost = _direct_cost(angles, boundary)
        else:
            endpoint = tuple(a + b for a, b in zip(angles, boundary, strict=True))
            cost = _direct_cost((0., 0., 0.), endpoint)
        if cost <= budget + 1e-9:
            result.append((_groups(((geometry.axis, offsets),)), cost, offsets))
    return tuple(result)


WHOLE_QUARTERS = tuple(
    (axis, tuple(s if index in layers else 0 for index in range(3)))
    for axis in range(3) for layers in ((0,), (1,), (2,), (0, 1), (1, 2)) for s in (-1, 1))


@lru_cache(maxsize=64)
def _hidden_routes(count):
    if count == 0:
        return ((),)
    result = set()
    for group in WHOLE_QUARTERS:
        for previous in _hidden_routes(count - 1):
            if not previous or previous[-1][0] != group[0]:
                result.add(_groups(previous + (group,)))
    return tuple(sorted(result))


@lru_cache(maxsize=16384)
def routes_between(previous, current, budget, double_budget=None, double_support=frozenset(), optimistic_doubles=False):
    """Budget simultaneous slabs independently and sequential axes in time.

    double_support identifies components in a geometry-supported 180-degree
    span, including frames before its second quarter commit. Each component
    otherwise uses the single-turn rate, regardless of another slab's motion.
    """
    double_budget = budget if double_budget is None else double_budget
    if not all(math.isfinite(v) for v in (budget, double_budget)) or budget < 0 or double_budget < budget:
        raise ValueError('Rotation budgets must be finite, nonnegative and ordered')
    unique = {}

    def phase(axis, start, end, offsets, kind):
        delta = tuple(b-a for a, b in zip(_relative(start), _relative(end), strict=True))
        quarters = _relative(offsets)
        double = tuple(abs(q) >= 2 or (kind, component, q) in double_support
                       or (optimistic_doubles and kind != 'hidden')
                       for component, q in enumerate(quarters))
        return axis, delta, quarters, double

    def add(groups, phases):
        merged = []
        for axis, delta, quarters, double in phases:
            if merged and merged[-1][0] == axis:
                _, old_delta, old_quarters, old_double = merged.pop()
                delta = tuple(a+b for a, b in zip(old_delta, delta, strict=True))
                quarters = tuple(a+b for a, b in zip(old_quarters, quarters, strict=True))
                double = tuple(a or b or abs(q) >= 2 for a, b, q in
                               zip(old_double, double, quarters, strict=True))
            merged.append((axis, delta, quarters, double))
        degrees, fraction = 0., 0.
        for _axis, delta, _quarters, double in merged:
            degrees += max(map(abs, delta))
            fractions = [abs(d)/(double_budget if fast else budget) if (double_budget if fast else budget) > 0
                         else (0. if d == 0 else float('inf')) for d, fast in zip(delta, double, strict=True)]
            fraction += max(fractions)
        if fraction <= 1.+1e-9:
            groups = _groups(groups)
            route = RotationRoute(groups, degrees, degrees/fraction if fraction else budget)
            if groups not in unique or fraction < unique[groups][0]:
                unique[groups] = fraction, route

    if previous.axis is None or current.axis is None or previous.axis == current.axis:
        axes = range(3) if previous.axis is None and current.axis is None else (
            current.axis if previous.axis is None else previous.axis,)
        for axis in axes:
            start, end = previous.angles, current.angles
            for offsets in _quarter_offsets(start, end, double_budget):
                target = tuple(a+90.*q for a, q in zip(end, offsets, strict=True))
                add(((axis, offsets),), (phase(axis, start, target, offsets, 'direct'),))

    for exit_groups, exit_cost, exit_offsets in _boundary_routes(previous, double_budget, True):
        for entry_groups, entry_cost, entry_offsets in _boundary_routes(current, double_budget, False):
            if exit_cost+entry_cost > double_budget+1e-9:
                continue
            exiting = (() if previous.axis is None else (phase(previous.axis, previous.angles,
                tuple(90.*q for q in exit_offsets), exit_offsets, 'exit'),))
            entering = (() if current.axis is None else (phase(current.axis, FLAT.angles,
                tuple(a+90.*q for a, q in zip(current.angles, entry_offsets, strict=True)), entry_offsets, 'entry'),))
            add(exit_groups+entry_groups, exiting+entering)
            for count in range(1, math.floor((double_budget-exit_cost-entry_cost+1e-9)/90.)+1):
                if not exiting and not entering and count*90. > budget+1e-9:
                    continue  # Hidden different-axis singles never inherit a double's rate.
                for hidden in _hidden_routes(count):
                    phases = tuple(phase(axis, FLAT.angles, tuple(90.*q for q in offsets), offsets, 'hidden')
                                   for axis, offsets in hidden)
                    add(exit_groups+hidden+entry_groups, exiting+phases+entering)
    return tuple(route for _, route in sorted(unique.values(), key=lambda item: (item[1].degrees, item[1].groups)))


def _turn_span_edges(frames, geometries, limits, turns):
    """Find monotone turn-span witnesses using only retained geometries.

    Unwrapped quarter indices distinguish continuing a double from reversing
    across a residual seam. Forward/backward reachability marks its entire
    span; a fast ordinary single with no double continuation stays restricted.
    Keys are (frame, previous geometry, current geometry, component, quarters).
    A None endpoint denotes an aligned boundary or another axis.
    """
    supported = set()
    target = 90.*turns
    for axis, component, direction in product(range(3), range(2), (-1, 1)):
        layers, edges, reached = [], [], set()
        for t, frame in enumerate(frames):
            nodes = {}
            boundary = any(g.axis is None or g.axis != axis for g in geometries[frame].values())
            for index, geometry in geometries[frame].items():
                if geometry.axis != axis:
                    continue
                value = _relative(geometry.angles)[component]
                for quarter in range(-3, 4):
                    position = direction*(value+90.*quarter)
                    if -1e-8 <= position <= target+1e-8:
                        nodes[index, quarter] = min(target, max(0., position))
            links = []
            prior_nodes = layers[-1][0] if layers else {}
            prior_boundary = layers[-1][1] if layers else True
            new_reached = set()
            for node, position in nodes.items():
                if prior_boundary and position <= limits[frame]+1e-8:
                    new_reached.add(node)
                    links.append((None, node))
                for before, previous_position in prior_nodes.items():
                    if -1e-8 <= position-previous_position <= limits[frame]+1e-8:
                        links.append((before, node))
                        if before in reached:
                            new_reached.add(node)
            # End the double at an aligned boundary in this frame.
            if boundary:
                for before, position in prior_nodes.items():
                    if target-position <= limits[frame]+1e-8:
                        links.append((before, None))
            layers.append((nodes, boundary, new_reached));edges.append(links);reached=new_reached
        live = set()
        for t in range(len(frames)-1, -1, -1):
            frame = frames[t];previous_live = set()
            for before, after in edges[t]:
                if after is not None and after not in live:
                    continue
                before_reached = before is None or (t > 0 and before in layers[t-1][2])
                if before_reached:
                    old_index, old_quarter = before if before is not None else (None, 0)
                    new_index, new_quarter = after if after is not None else (None, turns*direction)
                    supported.add((frame, old_index, new_index, component, new_quarter-old_quarter))
                    if before is not None:
                        previous_live.add(before)
            live = previous_live
    return frozenset(supported)


def double_turn_edges(frames, geometries, limits):
    """Grant the higher rate only where an ordinary single cannot explain it.

    Merely having a possible double elsewhere in the geometry bank must not
    let a search branch spend the higher budget and finish as a single.
    """
    return (_turn_span_edges(frames, geometries, limits, 2)
            - _turn_span_edges(frames, geometries, limits, 1))


class RotationBudgetScorerMixin:
    """Compile geometry-constrained routes once and batch their sticker scoring."""

    rotation_budget_enabled = False

    def configure_rotation_budget(self, frames, source_fps, max_tps=40., max_double_tps=None):
        frames = tuple(frames)
        max_double_tps = max_tps if max_double_tps is None else max_double_tps
        if (not frames or any(b <= a for a, b in zip(frames, frames[1:]))
                or not math.isfinite(source_fps) or source_fps <= 0
                or not math.isfinite(max_tps) or max_tps <= 0
                or not math.isfinite(max_double_tps) or max_double_tps < max_tps):
            raise ValueError('Rotation budgeting requires ordered frames, positive FPS and positive TPS')
        parameters = frames, float(source_fps), float(max_tps), float(max_double_tps)
        if getattr(self, '_rotation_parameters', None) == parameters:
            return
        self._rotation_parameters = parameters
        self.rotation_budget_enabled = True
        self.max_tps = float(max_tps)
        self.max_double_tps = float(max_double_tps)
        self.rotation_source_fps = float(source_fps)
        self.rotation_previous = {}
        self.rotation_limits = {}
        self.rotation_double_limits = {}
        self._rotation_double_edges = None
        self._rotation_route_limits = {}
        self._rotation_finalized = max_double_tps == max_tps
        self._rotation_permutations = {}
        earlier = [index for index in self.records if index < frames[0]]
        baseline = (self.records[max(earlier)].get('original_source_frame', max(earlier)) if earlier else
                    self.records[frames[0]].get('original_source_frame', frames[0])-1)
        previous = None
        for frame in frames:
            source = self.records[frame].get('original_source_frame', frame)
            reason = self.hold_reason(frame)
            if reason is not None:
                if reason == 'turning_start_frame':
                    previous, baseline = None, source
                continue
            prior_source = self.records[previous].get('original_source_frame', previous) if previous is not None else baseline
            self.rotation_previous[frame] = previous
            self.rotation_limits[frame] = 90. * max_tps * (source-prior_source) / source_fps
            self.rotation_double_limits[frame] = 90. * max_double_tps * (source-prior_source) / source_fps
            previous = frame
        self.compiled_frames.clear()
        self.prepared.clear()
        self.cache.clear()

    def finalize_rotation_budgets(self):
        """Classify multi-frame doubles and filter compiled routes once."""
        if self._rotation_finalized:
            return
        frames = tuple(self.rotation_previous)
        for frame in frames:
            self.prepare(frame)
        geometries = {frame: self.compiled_frames[frame]['rotation_geometries'] for frame in frames}
        self._rotation_double_edges = double_turn_edges(frames, geometries, self.rotation_double_limits)
        for frame in frames:
            prepared = self.compiled_frames[frame]
            plans = self._make_rotation_plans(frame, geometries[frame])
            prepared['rotation_plans'] = plans
            costs = {(after, before, moves): degrees for after, routes in plans.items()
                     for before, moves, degrees in routes}
            allowed = self.np.zeros(prepared['configuration_count'], dtype=bool)
            for row, (before, after) in enumerate(zip(prepared['rotation_previous_rows'], prepared['option_rows'], strict=True)):
                if before == -2:  # Endpoint-only row for reverse scoring.
                    allowed[row] = True
                    continue
                moves = prepared['action_specs'][prepared['group_ids'][row]][0]
                degrees = costs.get((after, before, moves))
                if degrees is not None:
                    allowed[row] = True
                    prepared['rotation_degrees'][row] = degrees
                    prepared['rotation_budgets'][row] = self._rotation_route_limits[frame, before, after, moves]
            prepared['rotation_allowed'] = allowed
            prepared['selections'].clear()
        self.cache.clear()
        self._rotation_finalized = True

    def _rotation_plans(self, frame, geometry, selected, projection_indices, nearest, valid):
        # Identical flat labels and assignment-identical duplicate fits do not
        # create separate temporal contexts. ±45 endpoints retain their sign.
        best = {}
        for index in selected:
            option = geometry[index]
            shape = RotationGeometry.make(option['layer_axis'], self.m.parallel_layer_angles(option.get('layer_rotations', ())))
            row = projection_indices[index]
            assignment = b'' if nearest is None else self.np.where(valid[row], nearest[row], 255).tobytes()
            key = shape, assignment
            rank = self._geometry_score(option), -option.get('inlier_count', 0), option.get('p90_error', float('inf'))
            if key not in best or rank < best[key][0]:
                best[key] = rank, index, shape
        shapes = {index: shape for _, index, shape in best.values()}
        return tuple(sorted(shapes)), shapes, self._make_rotation_plans(frame, shapes)

    def _make_rotation_plans(self, frame, shapes):
        previous_frame = self.rotation_previous[frame]
        previous = self.prepare(previous_frame)['rotation_geometries'] if previous_frame is not None else {-1: FLAT}
        plans = {index: {} for index in shapes}
        for before_index, before in previous.items():
            for after_index, after in shapes.items():
                support = set()
                if self._rotation_double_edges is not None:
                    for component in (0, 1):
                        for quarter in range(-3, 4):
                            for kind, old, new in (
                                    ('direct', before_index if before.axis is not None else None,
                                     after_index if after.axis is not None else None),
                                    ('exit', before_index, None), ('entry', None, after_index)):
                                if (frame, old, new, component, quarter) in self._rotation_double_edges:
                                    support.add((kind, component, quarter))
                optimistic = self._rotation_double_edges is None and not self._rotation_finalized
                for route in routes_between(before, after, self.rotation_limits[frame],
                        self.rotation_double_limits[frame], frozenset(support), optimistic):
                    moves = tuple(self.m.cube_rotation_to_notation(rotation)
                                  for axis, offsets in route.groups
                                  for rotation in self.m.parallel_layer_rotations(axis, tuple(90.*q for q in offsets)))
                    key = before_index, moves
                    plans[after_index][key] = min(plans[after_index].get(key, float('inf')), route.degrees)
                    self._rotation_route_limits[frame, before_index, after_index, moves] = route.budget
        return {
            after: tuple((before, moves, cost) for (before, moves), cost in entries.items())
            for after, entries in plans.items()}

    @staticmethod
    def rotation_context(payload, previous=None):
        return payload.get('rotation_context', previous)

    def rotation_choices(self, frame, state, previous=None):
        self.finalize_rotation_budgets()
        if self.hold_reason(frame) is not None:
            return tuple(type(choice)(choice.state, choice.score, choice.rank,
                dict(choice.payload, rotation_budget=True, rotation_context=previous, angular_travel_degrees=0.), None)
                for choice in self.hold_transition(frame, state))
        return self._evaluate(frame, state, ('rotation', -1 if previous is None else previous))

    def score_geometry_hold(self, frame, state, geometry_index):
        return next(iter(self._evaluate(frame, state, ('geometry_hold', geometry_index))), None)

    def _rotation_inverse_permutation(self, moves):
        if moves not in self._rotation_permutations:
            undo = tuple(move.rstrip("'") if move.endswith('2') else
                         move[:-1] if move.endswith("'") else move+"'" for move in reversed(moves))
            canonical = self.m._turn_gpu_static_geometry()[-1]
            moved = self.m._turn_gpu_identity_indices_for_state(self.m.CubeGeometryState.identity().apply_moves(undo), canonical)
            self._rotation_permutations[moves] = self.np.argsort(canonical)[moved]
        return self._rotation_permutations[moves]

    def _rotation_prior_errors(self, frame, state, geometry_index):
        np = self.np
        prepared = self.prepare(frame)
        routes = prepared['rotation_plans'][geometry_index]
        previous_frame = self.rotation_previous[frame]
        if previous_frame is None:
            return np.zeros(len(routes))
        previous = self.prepare(previous_frame)
        if geometry_index not in prepared['rotation_reverse_cache']:
            rows = np.asarray([previous['rotation_aux_rows'][before] for before, _, _ in routes], dtype=np.intp)
            nearest = previous['nearest']
            if nearest is not None:
                permutations = np.asarray([self._rotation_inverse_permutation(moves) for _, moves, _ in routes])
                nearest = np.take_along_axis(permutations, nearest[rows], axis=1)
            prepared['rotation_reverse_cache'][geometry_index] = rows, nearest
        rows, nearest = prepared['rotation_reverse_cache'][geometry_index]
        if nearest is None:
            return np.zeros(len(routes))
        indices = self.canonical_from_encoded[np.frombuffer(self.key(state), dtype=np.uint8)[self.geometry_positions]]
        weights = previous['weights'][rows]
        self.color_configurations_scored += len(rows)
        if self.color_confusion_scoring:
            return self.m.sticker_color_confusion_metrics(indices, nearest, weights,
                previous['confusion_costs'], self.color_confusion_cost)['error']
        matches = weights * previous['match_matrix'][indices[nearest], previous['observation_indices']]
        faces = np.zeros((len(rows), 54))
        np.add.at(faces, (np.arange(len(rows))[:, None], nearest), matches)
        correct = np.minimum(faces, 1.).sum(axis=1)
        return np.where(previous['has_weight'][rows], 1.-correct/previous['denominator'][rows],
                        0. if nearest.shape[1] == 0 else float('inf'))

    def rotation_predecessors(self, frame, after, geometry_index=None):
        self.finalize_rotation_budgets()
        if self.hold_reason(frame) is not None:
            for choice in self.rotation_choices(frame, after, geometry_index):
                yield after, geometry_index, choice, 0.
            return
        prepared = self.prepare(frame)
        geometries = prepared['rotation_geometries'] if geometry_index is None else (geometry_index,)
        for current in geometries:
            endpoint = self.score_geometry_hold(frame, after, current)
            if endpoint is None:
                continue
            routes = prepared['rotation_plans'][current]
            if not routes:
                continue
            previous_errors = self._rotation_prior_errors(frame, after, current)
            for (before_index, moves, degrees), prior_error in zip(routes, previous_errors, strict=True):
                if not math.isfinite(float(prior_error)):
                    continue
                undo = tuple(move.rstrip("'") if move.endswith('2') else
                             move[:-1] if move.endswith("'") else move+"'" for move in reversed(moves))
                before = self.apply_moves(after, undo)
                if before_index == -1 and before != self.key(self.m.CubeGeometryState.identity()):
                    continue
                context = None if before_index == -1 else before_index
                payload = dict(endpoint.payload, state_moves=moves, color_offset=0.,
                    previous_geometry_index=context, rotation_context=current,
                    angular_travel_degrees=degrees,
                    rotation_budget_degrees=self._rotation_route_limits[frame, before_index, current, moves],
                    label=f"{' '.join(moves) or 'hold'} · {endpoint.payload['label']}",
                    is_baseline=bool(endpoint.payload.get('is_baseline') and not moves))
                yield before, context, type(endpoint)(after, endpoint.score, endpoint.rank, payload, None), float(prior_error)
