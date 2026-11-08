"""Bounded receding-horizon ordering for a single layer of a cube search."""
from collections import OrderedDict


class WindowBudgetReached(Exception):
    pass


class WindowForecaster:
    def __init__(self, expand, weights, frame_count, horizon, budget, check_budget, cache_size=2048):
        self.expand = expand
        self.weights = weights
        self.frame_count = frame_count
        self.horizon = horizon
        self.budget = budget
        self.check_budget = check_budget
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.warm = {}
        self.cache_hits = self.cross_window_hits = self.warm_starts = self.expanded = 0
        self.windows = self.completed_forecasts = 0

    def edges(self, depth, state, cue, *, window_start=None):
        key = depth, state, cue
        if key in self.cache:
            options, window = self.cache[key]
            self.cache.move_to_end(key)
            self.cache_hits += 1
            if window != self.windows:
                self.cross_window_hits += 1
            return options
        self.check_budget(window_start is not None)
        if window_start is not None:
            if self.expanded-window_start >= self.budget:
                raise WindowBudgetReached
            self.expanded += 1
        options = tuple(sorted(self.expand(depth, state, cue), key=lambda item: item.rank))
        self.cache[key] = options, self.windows
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return options

    def order(self, depth, nodes, width):
        """Rank a pooled shortlist by complete windows, then shift the cached tail."""
        horizon = min(self.horizon, self.frame_count-depth)
        if horizon <= 0 or not nodes:
            return nodes
        self.windows += 1
        started = self.expanded
        shortlist_size = min(2*width, self.budget//horizon, len(nodes))
        shortlist = nodes[:shortlist_size]
        weights = self.weights(horizon+1)
        best = {}

        def try_forecast(index, seed=(), *, hold=False):
            node = shortlist[index]
            path = list(seed[:horizon])
            current = path[-1] if path else node
            for offset in range(len(path), horizon):
                choices = self.edges(depth+offset, current.state, current.cue, window_start=started)
                current = next((edge for edge in choices if not hold or not edge.step["state_moves"]), None)
                if current is None:
                    return
                path.append(current)
            score = sum(weight*edge.step["queue_priority"]
                        for weight, edge in zip(weights, (node, *path), strict=True))
            self.completed_forecasts += 1
            if index not in best or score < best[index][0]:
                best[index] = score, tuple(path)

        warm_indices = []
        for i, node in enumerate(shortlist):
            seed = self.warm.get((depth, node.state, node.cue), ())
            if seed:
                self.warm_starts += 1
                warm_indices.append(i)
            try_forecast(i, seed)
        # Establish a comparable full window per candidate before spending
        # the rest of the shared budget on challenger forecasts.
        try:
            for i in warm_indices:
                try_forecast(i)
            for i in range(shortlist_size):
                try_forecast(i, hold=True)
        except WindowBudgetReached:
            pass

        self.warm = {
            (depth+1, path[0].state, path[0].cue): path[1:]
            for _, path in best.values() if len(path) > 1
        }
        ranked = sorted(best, key=lambda i: (best[i][0], shortlist[i].rank))
        ranked.extend(i for i in range(shortlist_size) if i not in best)
        return [shortlist[i] for i in ranked] + nodes[shortlist_size:]

    def metrics(self):
        return dict(forecast_windows=self.windows, forecast_expanded=self.expanded,
                    forecast_cache_hits=self.cache_hits, forecast_cross_window_hits=self.cross_window_hits,
                    forecast_warm_starts=self.warm_starts, completed_forecasts=self.completed_forecasts)
