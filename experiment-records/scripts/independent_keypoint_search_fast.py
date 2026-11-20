"""Mechanical search acceleration, with identical candidates and rank keys."""
from collections.abc import Mapping
import os
import numpy as np

try:
    if os.environ.get('CUBE_SEARCH_PYTHON_KERNELS') == '1':
        raise ImportError
    from live_recognition import _search_kernels as native
except ImportError:
    native = None

_MISSING = object()


class DeferredState:
    __slots__ = ('source', 'permutation', 'value')

    def __init__(self, source, permutation):
        self.source, self.permutation, self.value = source, permutation, None

    def resolve(self):
        if self.value is None:
            self.value = permute(self.source, self.permutation)
        return self.value


class StateBatch:
    __slots__ = ('source', 'sources', 'values')

    def __init__(self, source, sources):
        self.source, self.sources, self.values = source, sources, None

    def resolve(self, index):
        if self.values is None:
            self.values = permute_many(self.source, self.sources)
        return self.values[index]


class BatchedState(DeferredState):
    __slots__ = ('batch', 'index')

    def __init__(self, batch, index):
        self.batch, self.index = batch, index

    def resolve(self):
        return self.batch.resolve(self.index)


class CompactTransition:
    __slots__ = ('_state', 'score', 'rank', 'payload', 'update_key')

    def __init__(self, state, score, rank, payload, update_key=None):
        self._state, self.score, self.rank = state, score, rank
        self.payload, self.update_key = payload, update_key

    @property
    def state(self):
        return self._state.resolve() if isinstance(self._state, DeferredState) else self._state


class CompactEdge:
    __slots__ = ('value', 'cue', 'step', 'rank')

    def __init__(self, value, cue, step, rank):
        self.value, self.cue, self.step, self.rank = value, cue, step, rank

    @property
    def state(self):
        if isinstance(self.value, bytes):
            return self.value
        return self.value.resolve() if isinstance(self.value, DeferredState) else self.value.state


class ExpansionBatch:
    """Rank primitive rows before constructing the already-admitted candidates."""
    __slots__ = ('records', 'materialize', 'order')

    def __init__(self, records, materialize):
        self.records, self.materialize = records, materialize
        self.order = sorted(range(len(records)), key=lambda index: records[index][0])

    def __iter__(self):
        # Exhaustive iteration remains available for validation and other callers.
        return (self.materialize(index) for index in self.order)

    def select(self, width, sticker_threshold):
        unrestricted = (sticker_threshold is not None and bool(self.records) and
                        all(record[3] <= sticker_threshold for record in self.records))
        moves_seen, states_seen, selected = set(), set(), []
        previous_order, admission_rank = None, 0
        for index in self.order:
            order, moves, state_key, _ = self.records[index]
            if moves in moves_seen or state_key in states_seen:
                continue
            moves_seen.add(moves); states_seen.add(state_key)
            rank = len(states_seen)
            if order != previous_order:
                admission_rank, previous_order = rank, order
            if not unrestricted and admission_rank > width:
                break
            selected.append(self.materialize(index))
        return selected, unrestricted


def forward_expansion(scorer, frame, state, context, step_for):
    scorer.finalize_rotation_budgets()
    if scorer.hold_reason(frame) is not None:
        return None
    data = scorer._evaluate(frame, state, ('rotation', -1 if context is None else context), _rows_only=True)
    prepared, selection = data['prepared'], data['selection']
    unique = {}
    for index in data['best']:
        index = int(index)
        score = float(data['costs'][index])
        if not np.isfinite(score) or not np.isfinite(data['errors'][index]):
            continue
        template = scorer._template(prepared, int(selection['rows'][index]))[0]
        moves = template['state_moves']
        rank = (score, -float(selection['inliers'][index]), float(selection['p90s'][index]),
                -float(selection['layer_inliers'][index]))
        key = scorer.move_sources(moves).tobytes(), template['geometry_option_index']
        if key not in unique or rank < unique[key][0]:
            unique[key] = rank, index, moves, key
    rows = list(unique.values())
    records = [((rank[0], False, rank), moves, key, int(selection['matches'][index]))
               for rank, index, moves, key in rows]

    def materialize(position):
        _, index, _, _ = rows[position]
        choice = scorer._transitions_from_rows(data, [index])[0]
        step = step_for(choice, frame, context)
        return CompactEdge(choice, choice.payload['rotation_context'], step,
                           (step['queue_priority'], False, choice.rank))
    return ExpansionBatch(records, materialize)


def backward_expansion(scorer, frame, state, context, step_for):
    scorer.finalize_rotation_budgets()
    if scorer.hold_reason(frame) is not None:
        return None
    prepared = scorer.prepare(frame)
    entries, records = [], []
    geometries = prepared['rotation_geometries'] if context is None else (context,)
    for current in geometries:
        endpoint = scorer.score_geometry_hold(frame, state, current)
        if endpoint is None:
            continue
        routes = prepared['rotation_plans'][current]
        if not routes:
            continue
        errors = scorer._rotation_prior_errors(frame, state, current)
        sources, route_rows = scorer._rotation_state_specs(prepared, current, routes)
        batch = StateBatch(state, sources)
        static = prepared.setdefault('rotation_expansion_specs', {})
        if current not in static:
            specs = []
            for (before, moves, degrees), row in zip(routes, route_rows, strict=True):
                previous = None if before == -1 else before
                updates = dict(state_moves=moves, color_offset=0., previous_geometry_index=previous,
                    rotation_context=current, angular_travel_degrees=degrees,
                    rotation_budget_degrees=scorer._rotation_route_limits[frame, before, current, moves],
                    label=f"{' '.join(moves) or 'hold'} · {endpoint.payload['label']}",
                    is_baseline=bool(endpoint.payload.get('is_baseline') and not moves))
                specs.append((previous, moves, sources[row].tobytes(), updates))
            static[current] = specs
        for prior, row, spec in zip(errors, route_rows, static[current], strict=True):
            prior = float(prior)
            if not np.isfinite(prior):
                continue
            previous, moves, permutation, updates = spec
            if previous is None and batch.resolve(row) != scorer.key(scorer.m.CubeGeometryState.identity()):
                continue
            rank = endpoint.score, False, (prior, endpoint.rank)
            records.append((rank, moves, (permutation, previous), endpoint.payload.get('color_match_count', 0)))
            entries.append((endpoint, batch, row, previous, updates, prior))

    def materialize(index):
        endpoint, batch, row, previous, updates, prior = entries[index]
        choice = CompactTransition(state, endpoint.score, endpoint.rank, PayloadOverlay(endpoint.payload, **updates))
        step = step_for(choice, frame, None)
        return CompactEdge(BatchedState(batch, row), previous, step,
                           (step['queue_priority'], False, (prior, endpoint.rank)))
    return ExpansionBatch(records, materialize)


class PayloadOverlay(Mapping):
    """Immutable candidate fields; copy the full diagnostic record only on admission."""
    __slots__ = ('base', 'updates')

    def __init__(self, base, **updates):
        self.base, self.updates = base, updates

    def __getitem__(self, key):
        value = self.updates.get(key, _MISSING)
        return self.base[key] if value is _MISSING else value

    def get(self, key, default=None):
        value = self.updates.get(key, _MISSING)
        return self.base.get(key, default) if value is _MISSING else value

    def __iter__(self):
        yield from self.updates
        yield from (key for key in self.base if key not in self.updates)

    def __len__(self):
        return len(self.updates) + sum(key not in self.updates for key in self.base)


class ScoredPayload(PayloadOverlay):
    """Diagnostic-only normalization is evaluated when exported, never for ranking."""
    __slots__ = ('renderer', 'robust', 'balanced', 'geometry')

    def __init__(self, base, renderer, robust, balanced, geometry, **updates):
        super().__init__(base, **updates)
        self.renderer, self.robust, self.balanced, self.geometry = renderer, robust, balanced, geometry

    def __getitem__(self, key):
        if key not in self.updates:
            if key == 'global_robust_error':
                color = self.updates['color_error']
                self.updates[key] = (float(self.renderer.balanced_robust_cost(self.robust, color))
                                     if self.balanced else self.robust + self.renderer.PNP_TURN_COLOR_WEIGHT*color)
            elif key == 'score_components' and self.balanced:
                self.updates[key] = dict(geometry=self.geometry, color=self.updates['color_error'],
                                         geometry_weight=.5, color_weight=.5)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == 'global_robust_error' or key == 'score_components' and self.balanced:
            return self[key]
        return super().get(key, default)

    def __iter__(self):
        # Materializing this mapping also computes exactly the original diagnostics.
        self['global_robust_error']
        if self.balanced:
            self['score_components']
        yield from super().__iter__()

    def __len__(self):
        deferred = ('global_robust_error', 'score_components') if self.balanced else ('global_robust_error',)
        return super().__len__() + sum(key not in self.base and key not in self.updates for key in deferred)


def permute(state, sources):
    if native is not None:
        return native.permute(state, sources)
    return np.frombuffer(state, dtype=np.uint8)[sources].tobytes()


def permute_many(state, sources):
    """Build multiple independently permuted states in one native call."""
    if native is not None:
        return native.permute_many(state, sources)
    return [row.tobytes() for row in np.frombuffer(state, dtype=np.uint8)[sources]]


def best_rows(costs, selection):
    if native is not None:
        return native.best_rows(costs, selection['rank_groups'], selection['rank_inliers'],
                                selection['p90s'], selection['rank_layer_inliers'])
    eligible = np.flatnonzero(np.isfinite(costs))
    order = eligible[np.lexsort((-selection['layer_inliers'][eligible], selection['p90s'][eligible],
                                 -selection['inliers'][eligible], costs[eligible], selection['group_ids'][eligible]))]
    groups = selection['group_ids'][order]
    return np.sort(order[np.r_[True, groups[1:] != groups[:-1]]]) if len(order) else np.empty(0, dtype=np.intp)


def color_metrics(renderer, identities, nearest, valid, costs, confusion_cost):
    """Exact duplicate-sticker semantics, batched over both states and fits."""
    if native is None:
        return renderer.sticker_color_confusion_metrics(identities, nearest, valid, costs, confusion_cost)
    identities = np.asarray(identities)
    single = identities.ndim == 1
    indices = np.ascontiguousarray(identities[None, :] if single else identities, dtype=np.uint8)
    shape = (len(indices), len(nearest))
    values = native.color_metrics(indices, np.ascontiguousarray(nearest, dtype=np.uint8),
        np.ascontiguousarray(valid, dtype=bool), np.ascontiguousarray(costs, dtype=np.float64), confusion_cost)
    result = {name: np.frombuffer(value, dtype=np.float64 if index == 0 else np.int32).reshape(shape)
              for index, (name, value) in enumerate(zip(
                  ('error', 'correct', 'confused', 'forbidden', 'total'), values, strict=True))}
    return {name: value[0] for name, value in result.items()} if single else result
