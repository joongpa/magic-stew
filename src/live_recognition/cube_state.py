"""Reusable Rubik's Cube facelet-state and scramble utilities.

The coordinate frame is the renderer's fixed standard frame: U is white/top,
F is green/front, and R is red/right.  A facelet key is
``(cubie_index, normal_axis, normal_side)`` where ``cubie_index`` contains
integer coordinates in ``0..2`` and the normal is one of the six signed axes.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations, product
from typing import Callable

import numpy as np

FaceletKey = tuple[tuple[int, int, int], int, int]

FACE_COLORS = {
    "U": "white",
    "D": "yellow",
    "F": "green",
    "B": "blue",
    "R": "red",
    "L": "orange",
}
FACE_AXES = {
    "R": ((1, 0, 0), 1),
    "L": ((-1, 0, 0), 1),
    "U": ((0, 0, 1), 1),
    "D": ((0, 0, -1), 1),
    "F": ((0, -1, 0), 1),
    "B": ((0, 1, 0), 1),
    "M": ((-1, 0, 0), 0),
    "E": ((0, 0, -1), 0),
    "S": ((0, -1, 0), 0),
    "x": ((1, 0, 0), None),
    "y": ((0, 0, 1), None),
    "z": ((0, -1, 0), None),
}
SCRAMBLE_MOVE_RE = re.compile(r"(?:[RLUDFB](?:w)?|[rludfbMESxyz])(?:2|')?")


def parse_scramble_moves(value: str) -> tuple[str, ...]:
    """Parse whitespace-separated Singmaster moves."""

    normalized = value.replace("’", "'").strip()
    if not normalized:
        return ()
    moves = tuple(normalized.split())
    invalid = [move for move in moves if SCRAMBLE_MOVE_RE.fullmatch(move) is None]
    if invalid:
        raise argparse.ArgumentTypeError(
            "invalid scramble move(s) "
            + ", ".join(repr(move) for move in invalid)
            + "; expected moves such as R U2 F' Rw or r'"
        )
    return moves


def _quarter_turn_matrix(axis: tuple[int, int, int], quarter_turns: int) -> np.ndarray:
    """Return an exact integer rotation matrix for a quarter-turn axis."""

    unit_axis = np.asarray(axis, dtype=np.float64)
    direction = 1.0 if quarter_turns > 0 else -1.0
    turns = abs(int(quarter_turns)) % 4
    result = np.eye(3, dtype=np.int32)
    for _ in range(turns):
        cross = np.asarray(
            (
                (0.0, -unit_axis[2], unit_axis[1]),
                (unit_axis[2], 0.0, -unit_axis[0]),
                (-unit_axis[1], unit_axis[0], 0.0),
            ),
            dtype=np.float64,
        )
        one_turn = np.outer(unit_axis, unit_axis) + direction * cross
        result = np.rint(one_turn @ result).astype(np.int32)
    return result


def _facelet_keys() -> tuple[FaceletKey, ...]:
    keys: list[FaceletKey] = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                position = (x, y, z)
                for axis, coordinate in enumerate(position):
                    if coordinate == 0:
                        keys.append((position, axis, -1))
                    elif coordinate == 2:
                        keys.append((position, axis, 1))
    return tuple(keys)


FACELET_KEYS = _facelet_keys()
FACELET_KEYS_BY_GEOMETRY = {key: key for key in FACELET_KEYS}
FACE_BY_NORMAL = {
    (0, 1): "R",
    (0, -1): "L",
    (1, -1): "F",
    (1, 1): "B",
    (2, 1): "U",
    (2, -1): "D",
}
KOCIEMBA_FACE_ORDER = ("U", "R", "F", "D", "L", "B")
COLOR_TO_FACE = {color: face for face, color in FACE_COLORS.items()}

# Kociemba uses the face order URFDLB, with each face read row-by-row as if
# viewed directly from outside the face.  These coordinate grids use the same
# fixed frame as CubeState: F is y=0, B is y=2, U is z=2, and R is x=2.
_KOCIEMBA_FACE_COORDINATES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "U": tuple((x, y, 2) for y in (2, 1, 0) for x in (0, 1, 2)),
    "R": tuple((2, y, z) for z in (2, 1, 0) for y in (0, 1, 2)),
    "F": tuple((x, 0, z) for z in (2, 1, 0) for x in (0, 1, 2)),
    "D": tuple((x, y, 0) for y in (0, 1, 2) for x in (0, 1, 2)),
    "L": tuple((0, y, z) for z in (2, 1, 0) for y in (2, 1, 0)),
    "B": tuple((x, 2, z) for z in (2, 1, 0) for x in (2, 1, 0)),
}
_FACE_NORMALS = {
    "U": (2, 1),
    "R": (0, 1),
    "F": (1, -1),
    "D": (2, -1),
    "L": (0, -1),
    "B": (1, 1),
}


def _proper_cube_rotations() -> tuple[np.ndarray, ...]:
    """Return the 24 orientation-preserving signed axis permutations."""

    rotations: list[np.ndarray] = []
    for permutation in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.int32)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if round(float(np.linalg.det(matrix))) == 1:
                rotations.append(matrix)
    return tuple(rotations)


PROPER_CUBE_ROTATIONS = _proper_cube_rotations()


def solved_facelet_colors() -> dict[FaceletKey, str]:
    """Return the solved color at every facelet key."""

    return {
        key: FACE_COLORS[FACE_BY_NORMAL[(key[1], key[2])]]
        for key in FACELET_KEYS
    }


@dataclass
class CubeState:
    """Mutable facelet-color state for a 3x3x3 cube."""

    colors: dict[FaceletKey, str]

    @classmethod
    def solved(cls) -> "CubeState":
        return cls(solved_facelet_colors())

    @classmethod
    def from_face_grids(
        cls,
        faces: Mapping[str, Sequence[Sequence[str]]],
    ) -> "CubeState":
        """Build a state from six named 3x3 face grids.

        Each grid is ordered top-to-bottom and left-to-right as viewed from
        outside that face, with faces named ``U``, ``R``, ``F``, ``D``, ``L``,
        and ``B``.  This is also the layout expected by Kociemba.
        """

        colors: dict[FaceletKey, str] = {}
        for face in KOCIEMBA_FACE_ORDER:
            if face not in faces:
                raise ValueError(f"Missing face grid {face!r}")
            grid = faces[face]
            if len(grid) != 3 or any(len(row) != 3 for row in grid):
                raise ValueError(f"Face {face!r} must be a 3x3 color grid")
            normal_axis, normal_side = _FACE_NORMALS[face]
            for row in range(3):
                for column, grid_position in enumerate(
                    _KOCIEMBA_FACE_COORDINATES[face][row * 3 : row * 3 + 3]
                ):
                    colors[(grid_position, normal_axis, normal_side)] = grid[row][column]
        return cls(colors)

    @classmethod
    def from_net_cells(cls, cells: Mapping[tuple[str, int, int], object]) -> "CubeState":
        """Build a state from ``CubeNetState.cells``-style entries.

        Values may be color strings or objects exposing a ``color`` attribute,
        such as ``live_recognition.cube_net.NetCell``.
        """

        faces: dict[str, list[list[str]]] = {}
        for face in KOCIEMBA_FACE_ORDER:
            grid: list[list[str]] = []
            for row in range(3):
                grid_row: list[str] = []
                for column in range(3):
                    try:
                        value = cells[(face, row, column)]
                    except KeyError as exc:
                        raise ValueError(f"Missing recognized color at {face}{row}{column}") from exc
                    color = getattr(value, "color", value)
                    if not isinstance(color, str):
                        raise ValueError(f"Recognized color at {face}{row}{column} is not a string")
                    grid_row.append(color)
                grid.append(grid_row)
            faces[face] = grid
        return cls.from_face_grids(faces)

    @classmethod
    def from_scramble(cls, scramble: str | tuple[str, ...] | list[str]) -> "CubeState":
        state = cls.solved()
        moves = parse_scramble_moves(scramble) if isinstance(scramble, str) else tuple(scramble)
        for move in moves:
            state.apply_move(move)
        return state

    def apply_move(self, move: str) -> None:
        """Apply a face, wide, slice, or whole-cube turn."""

        if SCRAMBLE_MOVE_RE.fullmatch(move) is None:
            raise ValueError(f"Invalid scramble move {move!r}")
        base_move = move[0]
        is_wide = base_move in "rludfb" or (len(move) > 1 and move[1] == "w")
        if base_move in "rludfb":
            face_move = base_move.upper()
            suffix = move[1:]
        elif len(move) > 1 and move[1] == "w":
            face_move = base_move
            suffix = move[2:]
        else:
            face_move = base_move
            suffix = move[1:]
        axis, layer_side = FACE_AXES[face_move]
        quarter_turns = 1 if suffix == "'" else -2 if suffix == "2" else -1
        # The standard-frame generator uses -90 degrees for a normal
        # Singmaster move; the prime modifier reverses that direction.
        rotation = _quarter_turn_matrix(axis, quarter_turns)
        updated = dict(self.colors)
        for key, color in self.colors.items():
            position, normal_axis, normal_side = key
            center = np.asarray(position, dtype=np.int32) - 1
            if layer_side is not None:
                layer_normal = np.asarray(axis, dtype=np.int32)
                layer_coordinate = int(center @ layer_normal)
                if is_wide:
                    if layer_coordinate not in (layer_side, 0):
                        continue
                elif layer_coordinate != layer_side:
                    continue
            target_center = rotation @ center
            normal = np.zeros(3, dtype=np.int32)
            normal[normal_axis] = normal_side
            target_normal = rotation @ normal
            target_axis = int(np.argmax(np.abs(target_normal)))
            target_side = int(np.sign(target_normal[target_axis]))
            target_key = (
                tuple((target_center + 1).tolist()),
                target_axis,
                target_side,
            )
            if target_key not in FACELET_KEYS_BY_GEOMETRY:
                raise ValueError(f"Move {move!r} mapped facelet {key!r} off the cube")
            updated[target_key] = color
        self.colors = updated

    def color_at(self, key: FaceletKey) -> str:
        return self.colors[key]

    def is_solved(self) -> bool:
        """Return whether all six complete faces are uniform, in any orientation."""

        if self.colors.keys() != FACELET_KEYS_BY_GEOMETRY.keys():
            return False
        face_colors: dict[tuple[int, int], str] = {}
        for (_position, axis, side), color in self.colors.items():
            if color not in FACE_COLORS.values():
                return False
            normal = (axis, side)
            if face_colors.setdefault(normal, color) != color:
                return False
        return len(set(face_colors.values())) == 6

    def reoriented_to_standard_centers(self) -> "CubeState":
        """Rotate the state so its center colors use the standard face frame."""

        center_colors = {}
        for face in KOCIEMBA_FACE_ORDER:
            normal_axis, normal_side = _FACE_NORMALS[face]
            center_position = _KOCIEMBA_FACE_COORDINATES[face][4]
            center_colors[face] = self.colors[
                (center_position, normal_axis, normal_side)
            ]

        standard_face_by_color = {
            color: face for face, color in FACE_COLORS.items()
        }
        for rotation in PROPER_CUBE_ROTATIONS:
            matches = True
            for current_face, color in center_colors.items():
                expected_face = standard_face_by_color.get(color)
                if expected_face is None:
                    matches = False
                    break
                current_axis, current_side = _FACE_NORMALS[current_face]
                expected_axis, expected_side = _FACE_NORMALS[expected_face]
                current_vector = np.zeros(3, dtype=np.int32)
                expected_vector = np.zeros(3, dtype=np.int32)
                current_vector[current_axis] = current_side
                expected_vector[expected_axis] = expected_side
                if not np.array_equal(rotation @ current_vector, expected_vector):
                    matches = False
                    break
            if not matches:
                continue

            reoriented: dict[FaceletKey, str] = {}
            for key, color in self.colors.items():
                position, normal_axis, normal_side = key
                center = np.asarray(position, dtype=np.int32) - 1
                target_center = rotation @ center
                normal = np.zeros(3, dtype=np.int32)
                normal[normal_axis] = normal_side
                target_normal = rotation @ normal
                target_axis = int(np.argmax(np.abs(target_normal)))
                target_side = int(np.sign(target_normal[target_axis]))
                target_key = (
                    tuple((target_center + 1).tolist()),
                    target_axis,
                    target_side,
                )
                if target_key not in FACELET_KEYS_BY_GEOMETRY:
                    raise ValueError(f"Rotation mapped facelet {key!r} off the cube")
                reoriented[target_key] = color
            return type(self)(reoriented)

        raise ValueError(
            "Cube center colors do not form a legal whole-cube rotation of "
            "the configured U/R/F/D/L/B scheme"
        )

    def to_kociemba_facelets(self) -> str:
        """Return this state in Kociemba's 54-character URFDLB format."""

        facelets: list[str] = []
        for face in KOCIEMBA_FACE_ORDER:
            normal_axis, normal_side = _FACE_NORMALS[face]
            for position in _KOCIEMBA_FACE_COORDINATES[face]:
                key = (position, normal_axis, normal_side)
                try:
                    color = self.colors[key].strip().lower()
                except KeyError as exc:
                    raise ValueError(f"Missing facelet at {face}{len(facelets) % 9 + 1}") from exc
                try:
                    facelets.append(COLOR_TO_FACE[color])
                except KeyError as exc:
                    valid = ", ".join(sorted(COLOR_TO_FACE))
                    raise ValueError(
                        f"Unknown cube color {self.colors[key]!r}; expected one of {valid}"
                    ) from exc

        counts = {face: facelets.count(face) for face in KOCIEMBA_FACE_ORDER}
        if any(count != 9 for count in counts.values()):
            raise ValueError(f"Cube colors must contain nine of each color; got {counts}")
        if any(facelets[index * 9 + 4] != face for index, face in enumerate(KOCIEMBA_FACE_ORDER)):
            raise ValueError("Cube center colors do not match the configured U/R/F/D/L/B frame")
        return "".join(facelets)

    def scramble_from_solved(
        self,
        *,
        solve: Callable[[str], str] | None = None,
    ) -> str:
        """Return moves that reproduce this state from the solved cube.

        Kociemba returns a solution from this state back to solved.  Since cube
        moves are reversible, reversing that solution and inverting each move
        gives a scramble from solved to this state.

        ``solve`` is injectable for tests or for callers that already manage a
        Kociemba solver instance.  When omitted, the ``kociemba`` package is
        imported lazily.
        """

        if solve is None:
            try:
                import kociemba
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Scramble generation requires kociemba; install it with `pip install kociemba`"
                ) from exc
            solve = kociemba.solve

        solution = solve(self.to_kociemba_facelets())
        return invert_scramble(solution)


def invert_scramble(value: str | tuple[str, ...] | list[str]) -> str:
    """Reverse and invert a Singmaster move sequence."""

    moves = parse_scramble_moves(value) if isinstance(value, str) else tuple(value)
    inverted: list[str] = []
    for move in reversed(moves):
        if move.endswith("'"):
            inverted.append(move[:-1])
        elif move.endswith("2"):
            inverted.append(move)
        else:
            inverted.append(f"{move}'")
    return " ".join(inverted)

