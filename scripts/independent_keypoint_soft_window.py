"""Soft preference for extending supported runs over making distant new forks."""

import math


class SoftMoveWindow:
    def __init__(
        self,
        state_key,
        *,
        apply_moves=None,
        moves=4,
        fork_penalty=0.5,
        max_geometry=0.5,
        minimum_color_accuracy=2.0 / 3.0,
    ):
        if (
            moves < 1
            or not math.isfinite(fork_penalty)
            or fork_penalty < 0
            or not math.isfinite(max_geometry)
            or max_geometry < 0
            or not 0 <= minimum_color_accuracy <= 1
        ):
            raise ValueError("Invalid soft move window settings")
        self.state_key = state_key
        self.apply_moves = apply_moves or (lambda state, moves: state.apply_moves(moves))
        self.moves, self.fork_penalty = moves, fork_penalty
        self.max_geometry, self.minimum_color_accuracy = max_geometry, minimum_color_accuracy
        self.protected_frame = None
        self.stats = dict(
            soft_window_advances=0, soft_window_protected_frame=None, late_forks_penalized=0
        )

    def initial(self, stickers, state):
        # Keep the entire winning path's sticker history and policy together.
        return stickers, ((self.state_key(state), None, False),), None

    def trail_after(self, trail, state, frame, moves, *, supported=False):
        if not moves:
            return trail
        key = self.state_key(state)
        for i, (previous, _, _) in enumerate(trail):
            if previous == key:
                # Loop erasure: inverse pairs and cube rotations earn no
                # additional support. The production key ignores rotations.
                return trail[: i + 1]
        return (*trail, (key, frame, supported))

    def prefix_trail(self, trail, state, steps):
        for step in steps:
            moves = step.get("state_moves", ())
            state = self.apply_moves(state, moves)
            trail = self.trail_after(trail, state, step["frame_index"], moves)
        return trail

    def is_supported(self, payload, previous_history_penalty, history_penalty, *, reliable=True):
        geometry = payload.get("score_components", {}).get("geometry")
        error = payload.get("color_error")
        return bool(
            reliable
            and payload.get("state_moves")
            and geometry is not None
            and math.isfinite(geometry)
            and geometry <= self.max_geometry
            and error is not None
            and math.isfinite(error)
            and error <= 1.0 - self.minimum_color_accuracy
            and history_penalty <= previous_history_penalty
        )

    def establish(self, recent):
        supported = [frame for _, frame, good in recent[1] if good]
        if len(supported) > self.moves:
            boundary = supported[-self.moves - 1]
            if self.protected_frame is None or boundary > self.protected_frame:
                self.protected_frame = boundary
                self.stats["soft_window_advances"] += 1
                self.stats["soft_window_protected_frame"] = boundary

    def fork_cost(self, frame, inherited_until, *, new_fork=False):
        until = inherited_until
        if new_fork and self.protected_frame is not None and frame <= self.protected_frame:
            until = max(until if until is not None else self.protected_frame, self.protected_frame)
            self.stats["late_forks_penalized"] += 1
        if until is not None and frame > until:
            until = None
        return (self.fork_penalty if until is not None else 0.0), until

    def describe(self, recent, penalty):
        return dict(
            soft_window_penalty=penalty,
            soft_window_penalty_until_frame=recent[2],
            supported_move_count=sum(good for _, _, good in recent[1]),
            soft_window_protected_frame=self.protected_frame,
        )
