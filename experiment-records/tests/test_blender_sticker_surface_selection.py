"""Exercise the actual geometry/policy functions without importing Blender."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class Vector(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values, dtype=float).view(cls)

    @property
    def length(self):
        return float(np.linalg.norm(self))

    def normalized(self):
        return self / self.length


@pytest.fixture
def generator():
    path = Path(__file__).resolve().parents[1] / "scripts/apply_blender_cube_moves.py"
    functions = {"physical_sticker_face_candidates", "select_sticker_face_spec",
                 "sticker_point_visibility", "point_faces_camera"}
    tree = ast.parse(path.read_text())
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             *(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions)],
                        type_ignores=[])
    namespace = {"math": math, "FACE_VISIBILITY_MIN_PLANE_ANGLE_DEGREES": 10,
                 "STICKER_KEYPOINT_MIN_VISIBLE_AREA_RATIO": .1,
                 "POINT_VISIBILITY_UNLABELED": 0, "POINT_VISIBILITY_LABELED_OCCLUDED": 1,
                 "POINT_VISIBILITY_VISIBLE": 2,
                 "projected_sticker_area_mask": lambda candidate, width, height: candidate["mask"]}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def face(normal, point, mask):
    return {"normal": Vector(normal), "point": Vector(point), "mask": np.asarray(mask)}


def test_outward_sticker_wins_even_when_wrong_back_face_overlaps_more_pixels(generator):
    wrong = face((0, -1, 0), (1, -.5, 0), np.ones((4, 4)))
    physical = face((1, 0, 0), (1.5, 0, 0), np.eye(4))
    spec = {"name": "sticker", "mask_key": "cubie:Yellow", "surface_normal_hint": Vector((1, 0, 0)),
            "face_candidates": [wrong, physical]}
    selected = generator.select_sticker_face_spec(spec, np.ones((4, 4)), 4, 4)
    np.testing.assert_array_equal(selected["normal"], physical["normal"])
    np.testing.assert_array_equal(selected["point"], physical["point"])
    # Same camera: the erroneous surface fails the unchanged angle rule,
    # while the actual sticker's outward face passes it.
    camera = physical["point"] + Vector((.5, math.sqrt(.75), 0)) * 10
    assert generator.point_faces_camera(selected["point"], [selected["normal"]], camera, 5)
    assert not generator.point_faces_camera(wrong["point"], [wrong["normal"]], camera, 5)
    assert generator.sticker_point_visibility(True, .2) == 1


def test_detached_sticker_uses_front_plane_when_back_normal_was_flipped_outward(generator):
    back = face((1, 0, 0), (.498, 0, 0), np.ones((2, 2)))
    front = face((1, 0, 0), (.503, 0, 0), np.eye(2))
    spec = {"name": "detached", "mask_key": "detached", "surface_normal_hint": Vector((1, 0, 0)),
            "face_candidates": [back, front]}
    selected = generator.select_sticker_face_spec(spec, np.ones((2, 2)), 2, 2)
    np.testing.assert_array_equal(selected["point"], front["point"])


def test_physical_surface_is_selected_even_when_no_visible_mask_is_available(generator):
    wrong = face((-1, 0, 0), (.5, 0, 0), np.ones((2, 2)))
    physical = face((1, 0, 0), (1.5, 0, 0), np.eye(2))
    spec = {"name": "center", "mask_key": "center", "surface_normal_hint": Vector((1, 0, 0)),
            "face_candidates": [wrong, physical]}
    selected = generator.select_sticker_face_spec(spec, None, 2, 2)
    np.testing.assert_array_equal(selected["point"], physical["point"])


@pytest.mark.parametrize("coverage, expected", [(.099, 0), (.1, 1), (.101, 1), (.75, 1)])
def test_occluded_sticker_requires_at_least_ten_percent_coverage(generator, coverage, expected):
    assert generator.sticker_point_visibility(True, coverage) == expected


def test_directly_visible_point_still_uses_visibility_two(generator):
    assert generator.sticker_point_visibility(False, .75) == 2


@pytest.mark.parametrize("angle, expected", [(4, False), (6, True), (-20, False)])
def test_angle_rule_is_preserved_for_the_physical_sticker_surface(generator, angle, expected):
    normal = Vector((1, 0, 0))
    camera = Vector((math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0))
    assert generator.point_faces_camera(Vector((0, 0, 0)), [normal], camera, 5) == expected
