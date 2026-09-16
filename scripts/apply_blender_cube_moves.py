"""Apply Rubik's Cube face turns and scene randomization for synthetic data.

Run inside Blender, for example:

    Blender /path/to/cube.blend --background --python scripts/apply_blender_cube_moves.py -- --count 100 --output-dir ./synthetic_cube

Assumptions:
- The cube is modeled as separate cubie objects, one object per movable cubie.
- Object origins are centered on their cubies.
- The cube core is at the world origin.
- Cubies are 1 meter wide, so cubie origins sit on the -1/0/1 grid.
- Cubie objects use quaternion rotation mode.
- Cubie object pass_index values identify physical cubies.
- Sticker material pass_index values identify sticker colors; material pass_index 0 is
  treated as non-sticker cubie plastic and is not labeled.
- Blender axes are X=right, Z=up, and Y=front/back. Change FRONT_AXIS if needed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import site
import sys
import tempfile
import time
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector

PROJECT = Path("/Users/jeffpark/Documents/LiveRecognition")
VENV = PROJECT / ".venv"
site_packages = next((VENV / "lib").glob("python*/site-packages"))
site.addsitedir(str(site_packages))


# Common Blender convention: X right, Z up, and negative Y faces the camera/front.
# Use "+Y" instead if your modeled front face is on positive Y.
FRONT_AXIS = "-Y"

# Random scene defaults for synthetic data generation.
DEFAULT_SCRAMBLE_LENGTH = 30
CAMERA_TARGET_JITTER = 0.25
CAMERA_VIEWPOINT_WEIGHTS = (
    ("front", 0.24),
    ("front_diagonal", 0.28),
    ("over_shoulder", 0.18),
    ("behind_diagonal", 0.18),
    ("overhead", 0.12),
)
CAMERA_ROI_LENS_RANGE = (42.0, 49.0)
CAMERA_ROI_SHIFT_X_RANGE = (-0.09, 0.09)
CAMERA_ROI_SHIFT_Y_RANGE = (-0.07, 0.07)
# The existing framing is calibrated around four feet from the cube. Varying
# distance and focal length together produces different perspective compression
# without making the cube substantially larger or smaller in the image.
CAMERA_SCENE_UNITS_PER_FOOT = 3.0
CAMERA_DISTANCE_BUCKETS_FEET = (
    ("near", 0.35, (3.0, 6.0)),
    ("medium", 0.40, (6.0, 10.0)),
    ("far", 0.25, (10.0, 15.0)),
)

SYNTHETIC_LIGHT_PREFIX = "SyntheticDataLight"
SYNTHETIC_WINDOW_BLOCKER_PREFIX = "SyntheticWindowBlocker"
LIGHTING_SETUP_WEIGHTS = (
    ("ceiling_lights", 0.76),
    ("desk_lamp", 0.24),
)
SUNLIGHT_WINDOW_PROBABILITY = 0.42
LIGHT_TEMPERATURE_PROFILES = (
    # name, probability, indoor sources, window daylight
    ("warm_yellow", 0.30, (2700.0, 3600.0), (3500.0, 4800.0)),
    ("neutral_white", 0.50, (4000.0, 5600.0), (5000.0, 6500.0)),
    ("cool_blue", 0.20, (6500.0, 8500.0), (7000.0, 10000.0)),
)
LIGHT_TARGET_JITTER_RANGE = (-0.35, 0.35)
LIGHT_TEMPERATURE_KELVIN_RANGE = (2700.0, 7500.0)
WORLD_AMBIENT_LEVEL_RANGE = (0.045, 0.18)
BACKGROUND_NAME = "SyntheticDataBackground"
BACKGROUND_MATERIAL_NAME = "SyntheticDataBackgroundMaterial"
BACKGROUND_DISTANCE = 4.0
BACKGROUND_SIZE = 12.0
BACKGROUND_SCENE_COLLECTION_NAME = "SyntheticBackgroundScene"
BACKGROUND_SCENE_PREFIX = "SyntheticBackgroundObject"
BACKGROUND_SCENE_MATERIAL_PREFIX = "SyntheticBackgroundMaterial"
BACKGROUND_ASSET_COLLECTION_PREFIX = "SyntheticBackgroundAsset"
BACKGROUND_ASSET_ROOT_SUFFIX = "_ROOT"
BACKGROUND_ASSET_BLEND_CANDIDATES = (
    PROJECT / "syntheticAssets" / "BackgroundAssets.blend",
    PROJECT / "config" / "synthetic_assets" / "BackgroundAssets.blend",
)
SOLVE_STATION_ASSET_BLEND_CANDIDATES = (
    PROJECT / "syntheticAssets" / "SolveStationAssets.blend",
    PROJECT / "config" / "synthetic_assets" / "SolveStationAssets.blend",
)
SOLVE_STATION_COLLECTION_NAME = "SolveStation_Complete"
SOLVE_STATION_SCALE = 1.85
SOLVE_STATION_LAPTOP_PROBABILITY = 0.48
SOLVE_STATION_PHONE_PROBABILITY = 0.72
SOLVE_STATION_OTHER_CUBE_PROBABILITY = 0.58
SOLVE_STATION_OTHER_CUBE_SOLVED_PROBABILITY = 0.68
SOLVE_STATION_YAW_JITTER_DEGREES = (-7.0, 7.0)
SOLVE_STATION_CLUTTER_PADDING = 0.22
SOLVE_STATION_DESK_TOP_LOCAL_Z = -1.51
BACKGROUND_TABLE_Z = -1.5
BACKGROUND_TABLE_SIZE_RANGE = (18.0, 26.0)
BACKGROUND_WALL_DISTANCE_RANGE = (5.5, 8.5)
BACKGROUND_WALL_SIZE_RANGE = (18.0, 28.0)
BACKGROUND_OBJECT_DEPTH_RANGE = (1.35, 5.5)
BACKGROUND_OBJECT_RIGHT_RANGE = (-4.8, 4.8)
BACKGROUND_OBJECT_SCALE_RANGE = (0.24, 1.35)
BACKGROUND_KEEP_OUT_RIGHT = 1.35
BACKGROUND_KEEP_OUT_DEPTH = 1.2
DEFAULT_BACKGROUND_OBJECT_COUNT = 5
BACKGROUND_OBJECT_COUNT_JITTER = (0.45, 1.15)
BACKGROUND_REALISTIC_PROP_PROBABILITY = 1.0
BACKGROUND_BLEND_ASSET_PROBABILITY = 1.0
BACKGROUND_BLEND_ASSET_SCALE_RANGE = (1.25, 2.25)
BACKGROUND_BLEND_ASSET_VISIBLE_DEPTH_RANGE = (1.45, 3.4)
BACKGROUND_BLEND_ASSET_VISIBLE_RIGHT_RANGE = (-3.1, 3.1)
BACKGROUND_BLEND_ASSET_VISIBLE_KEEP_OUT_RIGHT = 1.15
BACKGROUND_BLEND_ASSET_COLLECTION_PREFIXES = (
    "Asset_",
)
BACKGROUND_BLEND_ASSET_EXCLUDED_PREFIXES = (
    "Asset_PrimitiveClutter",
)
BACKGROUND_BLEND_ASSET_FAMILY_WEIGHTS = {
    "Asset_Laptop": 1.0,
    "Asset_BookStack": 1.0,
    "Asset_CardboardBox": 1.0,
    "Asset_FabricPanel": 1.0,
    "Asset_WallScreen": 1.0,
    "Asset_Poster": 1.0,
    "Asset_RubiksCube": 1.0,
    "Asset_Mug": 1.4,
    "Asset_DeskLamp": 1.2,
    "Asset_Pens": 1.1,
    "Asset_NotebookPhone": 1.0,
}
STICKER_COLOR_MATERIAL_NAMES = ("White", "Yellow", "Red", "Orange", "Blue", "Green")
STICKER_ROUGHNESS_RANGE = (0.18, 0.38)
STICKER_SPECULAR_RANGE = (0.45, 0.75)
STICKER_COAT_WEIGHT_RANGE = (0.08, 0.24)
STICKER_COAT_ROUGHNESS_RANGE = (0.12, 0.32)
CORE_FILL_OBJECT_NAMES = ("StickerlessBlackCoreFill", "StickerlessBlackCoreSphereFill")
LAYER_RENDER_STATE_WEIGHTS = (
    ("none", 0.40),
    ("mid_turn", 0.40),
    ("corner_cut", 0.20),
)
MID_TURN_DEGREES_RANGE = (5.0, 45.0)
CORNER_CUT_SMALL_DEGREES_RANGE = (0.0, 10.0)
CORNER_CUT_ADJACENT_DEGREES_RANGE = (5.0, 45.0)
CUBE_FLOAT_PROBABILITY = 0.55
CUBE_TABLE_CLEARANCE_RANGE = (0.02, 0.18)
CUBE_FLOAT_CLEARANCE_RANGE = (0.12, 0.55)
CUBE_TABLE_TILT_DEGREES = (-14.0, 14.0)
CUBE_FLOAT_TILT_DEGREES = (-32.0, 32.0)
CUBE_POSE_XY_JITTER_RANGE = (-0.28, 0.28)
DEFAULT_CUBE_ROTATIONAL_BLUR_PROBABILITY = 0.0
DEFAULT_MAX_CUBE_ROTATIONAL_BLUR_DEGREES = 8.0
DEFAULT_CUBE_ROTATIONAL_BLUR_SAMPLES = 7
DEFAULT_DEGRADE_PROBABILITY = 0.85
DEGRADE_GAUSSIAN_BLUR_PROBABILITY = 0.45
DEGRADE_MOTION_BLUR_PROBABILITY = 0.35
DEGRADE_NOISE_PROBABILITY = 0.7
DEGRADE_COMPRESSION_PROBABILITY = 0.8
DEGRADE_RESCALE_PROBABILITY = 0.35
HAND_NAMES = ("left-hand", "right-hand")
HAND_VISIBILITY_WEIGHTS = (
    # ("none", 0.3),
    # ("left", 0.1),
    # ("right", 0.1),
    # ("both", 0.5),
    ("none", 1),
    ("left", 0),
    ("right", 0),
    ("both", 0),
)
HAND_X_OFFSET_RANGE = (-0.2, 0.2)
HAND_Z_OFFSET_RANGE = (-1.0, 1.0)
HAND_X_ROTATION_DEGREES = (-90.0, 0.0, 90.0)
HAND_SKIN_MATERIAL_NAME = "SyntheticHandSkinMaterial"
DEFAULT_MIN_VISIBLE_KEYPOINTS_WITH_HANDS = 8
DEFAULT_MAX_HAND_PLACEMENT_ATTEMPTS = 12
HAND_SKIN_TONE_RANGES = (
    ((0.96, 0.76, 0.62), (1.00, 0.86, 0.74)),
    ((0.84, 0.58, 0.42), (0.96, 0.72, 0.55)),
    ((0.66, 0.40, 0.27), (0.82, 0.55, 0.38)),
    ((0.46, 0.27, 0.18), (0.64, 0.40, 0.28)),
    ((0.26, 0.14, 0.09), (0.44, 0.26, 0.17)),
    ((0.12, 0.07, 0.045), (0.28, 0.16, 0.10)),
)
ACTIVE_RANDOM_SEED: int | None = None

SOLVED_LOCATION_PROP = "synthetic_solved_location"
SOLVED_ROTATION_PROP = "synthetic_solved_rotation"
DEFAULT_OUTPUT_DIR = "data/synthetic_cube_dataset"
DEFAULT_IMAGE_SIZE = 512
CLASS_NAMES = ["corner", "corner-edge", "corner-edge-center", "center"]
CORNER_CLASS_ID = 0
CORNER_EDGE_CLASS_ID = 1
CORNER_EDGE_CENTER_CLASS_ID = 2
CENTER_CLASS_ID = 3
CUBE_KEYPOINT_NAMES = (
    "UBL", "UBR", "UFR", "UFL",
    "DBL", "DBR", "DFR", "DFL",
    "UBl", "UbR", "UFr", "UfL",
    "UBr", "UfR", "UFl", "UbL",
    "DBl", "DbR", "DFr", "DfL",
    "DBr", "DfR", "DFl", "DbL",
    "uBL", "uBR", "uFR", "uFL",
    "dBL", "dBR", "dFR", "dFL",
    "Ubl", "Ubr", "Ufr", "Ufl",
    "Dbl", "Dbr", "Dfr", "Dfl",
    "uBl", "ubR", "uFr", "ufL",
    "dBl", "dbR", "dFr", "dfL",
    "uBr", "ufR", "uFl", "ubL",
    "dBr", "dfR", "dFl", "dbL",
)
KEYPOINT_COUNT = len(CUBE_KEYPOINT_NAMES)
POSE_KEYPOINT_COUNT = 1
KEYPOINT_BBOX_RADIUS_RATIO = 0.018
MIN_KEYPOINT_BBOX_RADIUS_PX = 4
POINT_VISIBILITY_UNLABELED = 0
POINT_VISIBILITY_LABELED_OCCLUDED = 1
POINT_VISIBILITY_VISIBLE = 2
FACE_VISIBILITY_MIN_PLANE_ANGLE_DEGREES = 10.0


def parse_axis(axis_name: str) -> Vector:
    sign = -1.0 if axis_name.startswith("-") else 1.0
    axis = axis_name[-1].upper()
    if axis == "X":
        return Vector((sign, 0.0, 0.0))
    if axis == "Y":
        return Vector((0.0, sign, 0.0))
    if axis == "Z":
        return Vector((0.0, 0.0, sign))
    raise ValueError(f"Unsupported axis: {axis_name}")


def face_specs() -> dict[str, Vector]:
    front = parse_axis(FRONT_AXIS)
    return {
        "R": Vector((1.0, 0.0, 0.0)),
        "L": Vector((-1.0, 0.0, 0.0)),
        "U": Vector((0.0, 0.0, 1.0)),
        "D": Vector((0.0, 0.0, -1.0)),
        "F": front,
        "B": -front,
    }


def is_cubie(obj: bpy.types.Object) -> bool:
    return obj.name == "cubie" or obj.name.startswith("cubie.")


def cube_objects() -> list[bpy.types.Object]:
    objects = [obj for obj in bpy.data.objects if obj.type == "MESH" and is_cubie(obj)]

    if not objects:
        raise ValueError("No mesh objects named cubie or cubie.xxx found.")
    return objects


def reset_cube(objects: list[bpy.types.Object]) -> None:
    for obj in objects:
        obj.rotation_mode = "QUATERNION"

        if SOLVED_LOCATION_PROP not in obj:
            obj[SOLVED_LOCATION_PROP] = list(obj.location)
        if SOLVED_ROTATION_PROP not in obj:
            obj[SOLVED_ROTATION_PROP] = list(obj.rotation_quaternion)

        obj.location = Vector(obj[SOLVED_LOCATION_PROP])
        obj.rotation_quaternion = Quaternion(obj[SOLVED_ROTATION_PROP])

    bpy.context.view_layer.update()


def core_fill_objects() -> list[bpy.types.Object]:
    return [
        obj
        for obj in bpy.data.objects
        if any(obj.name == name or obj.name.startswith(name + ".") for name in CORE_FILL_OBJECT_NAMES)
    ]


def reset_core_fill_objects(objects: list[bpy.types.Object]) -> None:
    for obj in objects:
        obj.location = Vector((0.0, 0.0, 0.0))
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
    bpy.context.view_layer.update()


def random_cube_pose_rotation(floating: bool) -> tuple[Quaternion, dict[str, float]]:
    tilt_range = CUBE_FLOAT_TILT_DEGREES if floating else CUBE_TABLE_TILT_DEGREES
    yaw_degrees = random.uniform(0.0, 360.0)
    pitch_degrees = random.uniform(*tilt_range)
    roll_degrees = random.uniform(*tilt_range)
    rotation = (
        Quaternion(Vector((0.0, 0.0, 1.0)), math.radians(yaw_degrees))
        @ Quaternion(Vector((1.0, 0.0, 0.0)), math.radians(pitch_degrees))
        @ Quaternion(Vector((0.0, 1.0, 0.0)), math.radians(roll_degrees))
    )
    return rotation, {
        "yaw_degrees": yaw_degrees,
        "pitch_degrees": pitch_degrees,
        "roll_degrees": roll_degrees,
    }


def transformed_min_z(objects: list[bpy.types.Object], rotation: Quaternion) -> float:
    rotation_matrix = rotation.to_matrix().to_4x4()
    min_z: float | None = None
    for obj in objects:
        if obj.type != "MESH":
            continue
        matrix = rotation_matrix @ obj.matrix_world
        for vertex in obj.data.vertices:
            z = (matrix @ vertex.co).z
            min_z = z if min_z is None else min(min_z, z)
    if min_z is None:
        raise ValueError("Cannot randomize cube pose without mesh vertices.")
    return min_z


def matrix_rows(matrix: Matrix) -> list[list[float]]:
    return [[matrix[row][column] for column in range(4)] for row in range(4)]


def matrix_from_rows(rows: list[list[float]]) -> Matrix:
    return Matrix(rows)


def apply_global_pose(objects: list[bpy.types.Object], rotation: Quaternion, translation: Vector) -> Matrix:
    transform = Matrix.Translation(translation) @ rotation.to_matrix().to_4x4()
    for obj in objects:
        obj.matrix_world = transform @ obj.matrix_world
        obj.rotation_mode = "QUATERNION"
    bpy.context.view_layer.update()
    return transform


def randomize_cube_global_pose(
    objects: list[bpy.types.Object],
    bounds_objects: list[bpy.types.Object] | None = None,
) -> dict[str, object]:
    floating = random.random() < CUBE_FLOAT_PROBABILITY
    rotation, rotation_metadata = random_cube_pose_rotation(floating)
    clearance_range = CUBE_FLOAT_CLEARANCE_RANGE if floating else CUBE_TABLE_CLEARANCE_RANGE
    clearance = random.uniform(*clearance_range)
    min_z = transformed_min_z(bounds_objects or objects, rotation)
    translation = Vector((
        random.uniform(*CUBE_POSE_XY_JITTER_RANGE),
        random.uniform(*CUBE_POSE_XY_JITTER_RANGE),
        BACKGROUND_TABLE_Z + clearance - min_z,
    ))
    transform = apply_global_pose(objects, rotation, translation)
    return {
        "mode": "floating" if floating else "near_table",
        "clearance": clearance,
        "table_z": BACKGROUND_TABLE_Z,
        "translation": [translation.x, translation.y, translation.z],
        "transform": matrix_rows(transform),
        **rotation_metadata,
    }


def objects_on_face(objects: list[bpy.types.Object], axis: Vector) -> list[bpy.types.Object]:
    return [obj for obj in objects if round(obj.location.dot(axis)) == 1]


def snap_to_cube_grid(location: Vector) -> Vector:
    return Vector((round(location.x), round(location.y), round(location.z)))


def rotate_face(objects: list[bpy.types.Object], move: str) -> None:
    if move not in "RLUDBF":
        raise ValueError(f"Unsupported move {move!r}; only R L U D B F are supported.")

    axis = face_specs()[move]
    moving = objects_on_face(objects, axis)
    if not moving:
        raise ValueError(f"Move {move} matched no cubie objects.")

    turn = Quaternion(axis, -math.pi / 2.0)

    for obj in moving:
        obj.rotation_mode = "QUATERNION"
        obj.location = snap_to_cube_grid(turn @ obj.location)
        obj.rotation_quaternion = turn @ obj.rotation_quaternion

    bpy.context.view_layer.update()


def adjacent_faces(face: str) -> list[str]:
    axes = face_specs()
    axis = axes[face]
    return [candidate for candidate, candidate_axis in axes.items() if abs(axis.dot(candidate_axis)) < 1e-6]


def random_turn_direction() -> int:
    return random.choice((-1, 1))


def rotate_face_continuous(
    objects: list[bpy.types.Object],
    move: str,
    angle_degrees: float,
    direction: int,
) -> list[str]:
    if move not in "RLUDBF":
        raise ValueError(f"Unsupported move {move!r}; only R L U D B F are supported.")

    axis = face_specs()[move]
    moving = objects_on_face(objects, axis)
    if not moving:
        raise ValueError(f"Move {move} matched no cubie objects.")

    turn = Quaternion(axis, direction * math.radians(angle_degrees))
    for obj in moving:
        obj.rotation_mode = "QUATERNION"
        obj.location = turn @ obj.location
        obj.rotation_quaternion = turn @ obj.rotation_quaternion

    bpy.context.view_layer.update()
    return [obj.name for obj in moving]


def randomize_layer_render_state(objects: list[bpy.types.Object]) -> dict[str, object]:
    state = random.choices(
        [mode for mode, _weight in LAYER_RENDER_STATE_WEIGHTS],
        weights=[weight for _mode, weight in LAYER_RENDER_STATE_WEIGHTS],
        k=1,
    )[0]
    if state == "none":
        return {"mode": "none"}

    face = random.choice(list("RLUDBF"))
    if state == "mid_turn":
        angle_degrees = random.uniform(*MID_TURN_DEGREES_RANGE)
        direction = random_turn_direction()
        moving = rotate_face_continuous(objects, face, angle_degrees, direction)
        return {
            "mode": "mid_turn",
            "face": face,
            "angle_degrees": angle_degrees,
            "direction": direction,
            "objects": moving,
        }

    small_angle_degrees = random.uniform(*CORNER_CUT_SMALL_DEGREES_RANGE)
    small_direction = random_turn_direction()
    small_objects = rotate_face_continuous(objects, face, small_angle_degrees, small_direction)

    adjacent_face = random.choice(adjacent_faces(face))
    adjacent_angle_degrees = random.uniform(*CORNER_CUT_ADJACENT_DEGREES_RANGE)
    adjacent_direction = random_turn_direction()
    adjacent_objects = rotate_face_continuous(
        objects,
        adjacent_face,
        adjacent_angle_degrees,
        adjacent_direction,
    )
    return {
        "mode": "corner_cut",
        "face": face,
        "angle_degrees": small_angle_degrees,
        "direction": small_direction,
        "objects": small_objects,
        "adjacent_face": adjacent_face,
        "adjacent_angle_degrees": adjacent_angle_degrees,
        "adjacent_direction": adjacent_direction,
        "adjacent_objects": adjacent_objects,
        "method": "naive_small_layer_then_adjacent_layer",
    }


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("-Z", "Y")


def random_point_on_unit_sphere() -> Vector:
    z = random.uniform(-1.0, 1.0)
    angle = random.uniform(0.0, math.tau)
    radius_xy = math.sqrt(max(0.0, 1.0 - z * z))
    return Vector((radius_xy * math.cos(angle), radius_xy * math.sin(angle), z))


def random_target() -> Vector:
    jitter = CAMERA_TARGET_JITTER
    return Vector((
        random.uniform(-jitter, jitter),
        random.uniform(-jitter, jitter),
        random.uniform(-jitter, jitter),
    ))


def get_camera() -> bpy.types.Object:
    camera = bpy.context.scene.camera
    if camera is None:
        raise ValueError("Scene has no active camera.")
    return camera


def random_camera_distance() -> tuple[str, float, float]:
    bucket_name, _weight, distance_range = random.choices(
        CAMERA_DISTANCE_BUCKETS_FEET,
        weights=[weight for _name, weight, _range in CAMERA_DISTANCE_BUCKETS_FEET],
        k=1,
    )[0]
    distance_feet = random.uniform(*distance_range)
    return bucket_name, distance_feet, distance_feet * CAMERA_SCENE_UNITS_PER_FOOT


def randomize_camera() -> dict[str, object]:
    camera = get_camera()
    viewpoint = random.choices(
        [name for name, _weight in CAMERA_VIEWPOINT_WEIGHTS],
        weights=[weight for _name, weight in CAMERA_VIEWPOINT_WEIGHTS],
        k=1,
    )[0]
    if viewpoint == "front":
        azimuth_degrees = -90.0 + random.uniform(-12.0, 12.0)
        elevation_degrees = random.uniform(18.0, 36.0)
        reference_distance = 11.75
        roll_degrees = random.uniform(-6.0, 6.0)
    elif viewpoint == "front_diagonal":
        azimuth_degrees = random.choice((-45.0, -135.0)) + random.uniform(-9.0, 9.0)
        elevation_degrees = random.uniform(20.0, 43.0)
        reference_distance = 12.25
        roll_degrees = random.uniform(-8.0, 8.0)
    elif viewpoint == "over_shoulder":
        azimuth_degrees = random.choice((58.0, 122.0)) + random.uniform(-6.0, 6.0)
        elevation_degrees = random.uniform(57.0, 68.0)
        reference_distance = 13.75
        roll_degrees = random.uniform(-7.0, 7.0)
    elif viewpoint == "behind_diagonal":
        azimuth_degrees = random.choice((42.0, 138.0)) + random.uniform(-8.0, 8.0)
        elevation_degrees = random.uniform(27.0, 48.0)
        reference_distance = 12.5
        roll_degrees = random.uniform(-8.0, 8.0)
    else:
        azimuth_degrees = random.choice((-135.0, -90.0, -45.0, 45.0, 90.0, 135.0))
        azimuth_degrees += random.uniform(-10.0, 10.0)
        elevation_degrees = random.uniform(64.0, 80.0)
        reference_distance = 13.5
        roll_degrees = random.uniform(-18.0, 18.0)

    distance_bucket, distance_feet, distance = random_camera_distance()
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    radius_xy = distance * math.cos(elevation)
    camera.location = Vector((
        radius_xy * math.cos(azimuth),
        radius_xy * math.sin(azimuth),
        distance * math.sin(elevation),
    ))
    camera.data.type = "PERSP"
    framing_lens = random.uniform(*CAMERA_ROI_LENS_RANGE)
    camera.data.lens = framing_lens * distance / reference_distance
    camera.data.shift_x = random.uniform(*CAMERA_ROI_SHIFT_X_RANGE)
    camera.data.shift_y = random.uniform(*CAMERA_ROI_SHIFT_Y_RANGE)
    target = random_target()
    look_at(camera, target)
    camera.rotation_quaternion = camera.rotation_quaternion @ Quaternion(
        Vector((0.0, 0.0, -1.0)),
        math.radians(roll_degrees),
    )
    return {
        "viewpoint": viewpoint,
        "location": [camera.location.x, camera.location.y, camera.location.z],
        "target": [target.x, target.y, target.z],
        "azimuth_degrees": azimuth_degrees,
        "elevation_degrees": elevation_degrees,
        "distance": distance,
        "distance_feet": distance_feet,
        "distance_bucket": distance_bucket,
        "reference_distance": reference_distance,
        "roll_degrees": roll_degrees,
        "lens_mm": camera.data.lens,
        "framing_lens_mm": framing_lens,
        "roi_crop": {
            "method": "camera_lens_and_sensor_shift",
            "shift_x": camera.data.shift_x,
            "shift_y": camera.data.shift_y,
        },
    }


def kelvin_to_rgb(temperature_kelvin: float) -> tuple[float, float, float]:
    temperature = max(1000.0, min(40000.0, temperature_kelvin)) / 100.0
    if temperature <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(temperature) - 161.1195681661
        blue = 0.0 if temperature <= 19.0 else 138.5177312231 * math.log(temperature - 10.0) - 305.0447927307
    else:
        red = 329.698727446 * ((temperature - 60.0) ** -0.1332047592)
        green = 288.1221695283 * ((temperature - 60.0) ** -0.0755148492)
        blue = 255.0
    return tuple(max(0.0, min(channel / 255.0, 1.0)) for channel in (red, green, blue))


def clear_scene_lights() -> None:
    """Remove authored and previously generated lights so each rig is self-contained."""
    for obj in list(bpy.data.objects):
        if obj.name.startswith(SYNTHETIC_WINDOW_BLOCKER_PREFIX):
            mesh_data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh_data.users == 0:
                bpy.data.meshes.remove(mesh_data)
            continue
        if obj.type != "LIGHT":
            continue
        light_data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if light_data.users == 0:
            bpy.data.lights.remove(light_data)


def lighting_target(z_range: tuple[float, float] = (-0.35, 0.45)) -> Vector:
    return Vector((
        random.uniform(*LIGHT_TARGET_JITTER_RANGE),
        random.uniform(*LIGHT_TARGET_JITTER_RANGE),
        random.uniform(*z_range),
    ))


def add_light_source(
    role: str,
    light_type: str,
    location: Vector,
    target: Vector,
    energy: float,
    temperature_kelvin: float,
    *,
    size: float | None = None,
    size_y: float | None = None,
    sun_angle_degrees: float | None = None,
) -> dict[str, object]:
    """Create one intentionally placed source and return reproducibility metadata."""
    name = f"{SYNTHETIC_LIGHT_PREFIX}.{role}"
    data = bpy.data.lights.new(name, type=light_type)
    light = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(light)
    light.location = location
    look_at(light, target)

    color = kelvin_to_rgb(temperature_kelvin)
    data.energy = energy
    data.color = color
    data.use_shadow = True
    if light_type == "AREA":
        data.shape = "RECTANGLE"
        data.size = size if size is not None else 1.0
        data.size_y = size_y if size_y is not None else data.size
    elif light_type == "SUN":
        data.angle = math.radians(sun_angle_degrees if sun_angle_degrees is not None else 1.0)

    return {
        "name": light.name,
        "role": role,
        "type": data.type,
        "location": list(location),
        "target": list(target),
        "energy": energy,
        "temperature_kelvin": temperature_kelvin,
        "color": list(color),
        "size": getattr(data, "size", None) if light_type == "AREA" else None,
        "size_y": getattr(data, "size_y", None) if light_type == "AREA" else None,
        "sun_angle_degrees": sun_angle_degrees if light_type == "SUN" else None,
    }


def add_window_aperture(
    source_location: Vector,
    patch_target: Vector,
    opening_width: float,
    opening_height: float,
) -> dict[str, object]:
    """Add camera-invisible shadow casters that form a rectangular window opening."""
    direction = (patch_target - source_location).normalized()
    rotation = direction.to_track_quat("Z", "Y")
    aperture_fraction = random.uniform(0.28, 0.38)
    aperture_center = source_location.lerp(patch_target, aperture_fraction)
    surround = 10.0
    thickness = 0.08
    panels = (
        (-(opening_width + surround) / 2.0, 0.0, surround, opening_height + 2.0 * surround),
        ((opening_width + surround) / 2.0, 0.0, surround, opening_height + 2.0 * surround),
        (0.0, -(opening_height + surround) / 2.0, opening_width, surround),
        (0.0, (opening_height + surround) / 2.0, opening_width, surround),
    )
    panel_names = []
    for index, (offset_x, offset_y, width, height) in enumerate(panels, start=1):
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        panel = bpy.context.object
        panel.name = f"{SYNTHETIC_WINDOW_BLOCKER_PREFIX}.{index}"
        panel.location = aperture_center + rotation @ Vector((offset_x, offset_y, 0.0))
        panel.rotation_mode = "QUATERNION"
        panel.rotation_quaternion = rotation
        panel.scale = Vector((width, height, thickness))
        panel.visible_camera = False
        panel_names.append(panel.name)

    return {
        "opening_width": opening_width,
        "opening_height": opening_height,
        "center": list(aperture_center),
        "patch_target": list(patch_target),
        "approximate_projected_width": opening_width / aperture_fraction,
        "approximate_projected_height": opening_height / aperture_fraction,
        "blockers": panel_names,
    }


def configure_world_lighting(
    strength_range: tuple[float, float],
    temperature_range: tuple[float, float],
) -> dict[str, object]:
    world = bpy.context.scene.world or bpy.data.worlds.new("SyntheticDataWorld")
    bpy.context.scene.world = world
    strength = random.uniform(*strength_range)
    temperature_kelvin = random.uniform(*temperature_range)
    tint = kelvin_to_rgb(temperature_kelvin)
    color = tuple(channel * random.uniform(0.92, 1.08) for channel in tint)
    world.color = tuple(strength * channel for channel in color)

    # Cycles and Eevee use the node background when nodes are enabled. Set both
    # representations so this behaves consistently across source .blend files.
    world.use_nodes = True
    background_node = world.node_tree.nodes.get("Background")
    if background_node is not None:
        background_node.inputs["Color"].default_value = (*color, 1.0)
        background_node.inputs["Strength"].default_value = strength
    return {
        "strength": strength,
        "temperature_kelvin": temperature_kelvin,
        "color": list(color),
        "display_color": list(world.color),
    }


def randomize_light() -> dict[str, object]:
    """Build broad room lighting with an optional physical window aperture."""
    clear_scene_lights()
    setup = random.choices(
        [name for name, _weight in LIGHTING_SETUP_WEIGHTS],
        weights=[weight for _name, weight in LIGHTING_SETUP_WEIGHTS],
        k=1,
    )[0]
    temperature_profile, _profile_weight, room_temperature_range, window_temperature_range = random.choices(
        LIGHT_TEMPERATURE_PROFILES,
        weights=[weight for _name, weight, _room_range, _window_range in LIGHT_TEMPERATURE_PROFILES],
        k=1,
    )[0]
    base_temperature = random.uniform(*room_temperature_range)
    target = lighting_target()
    sources: list[dict[str, object]] = []

    if setup == "ceiling_lights":
        fixture_count = random.choice((2, 3, 3, 4))
        axis = random.choice((0, 1))
        for index in range(fixture_count):
            along = (index - (fixture_count - 1) / 2.0) * random.uniform(2.2, 3.2)
            across = random.uniform(-1.8, 1.8)
            location = Vector((along, across, random.uniform(5.0, 7.5)))
            if axis == 1:
                location.x, location.y = location.y, location.x
            downward_target = Vector((location.x, location.y, BACKGROUND_TABLE_Z))
            sources.append(add_light_source(
                f"ceiling_{index + 1}", "AREA", location, downward_target,
                random.uniform(260.0, 560.0), base_temperature + random.uniform(-180.0, 180.0),
                size=random.uniform(1.8, 3.0), size_y=random.uniform(3.5, 5.5),
            ))
        world_ranges = ((0.12, 0.25), room_temperature_range)
        exposure_range = (-0.20, 0.18)

    else:  # desk_lamp
        table_forward, table_right = camera_table_basis()
        side = random.choice((-1.0, 1.0))
        location = target + table_right * side * random.uniform(2.0, 4.0)
        location -= table_forward * random.uniform(-0.5, 2.2)
        location.z = random.uniform(2.0, 4.2)
        lamp_target = Vector((
            target.x + random.uniform(-0.8, 0.8),
            target.y + random.uniform(-0.8, 0.8),
            random.uniform(BACKGROUND_TABLE_Z, 0.0),
        ))
        sources.append(add_light_source(
            "lamp", "AREA", location, lamp_target,
            random.uniform(260.0, 580.0), base_temperature + random.uniform(-180.0, 180.0),
            size=random.uniform(1.2, 2.2), size_y=random.uniform(1.2, 2.2),
        ))
        sources.append(add_light_source(
            "room_fill", "AREA",
            Vector((random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5), random.uniform(4.5, 6.5))),
            Vector((0.0, 0.0, BACKGROUND_TABLE_Z)),
            random.uniform(140.0, 280.0), base_temperature + random.uniform(-300.0, 300.0),
            size=random.uniform(5.0, 8.0), size_y=random.uniform(5.0, 8.0),
        ))
        world_ranges = ((0.10, 0.22), room_temperature_range)
        exposure_range = (-0.05, 0.28)

    window_metadata = None
    if random.random() < SUNLIGHT_WINDOW_PROBABILITY:
        side_angle = random.choice((0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0))
        side_angle += math.radians(random.uniform(-10.0, 10.0))
        patch_target = Vector((
            random.uniform(-0.55, 0.55),
            random.uniform(-0.55, 0.55),
            BACKGROUND_TABLE_Z,
        ))
        distance = random.uniform(3.5, 4.5)
        window_location = Vector((
            patch_target.x + distance * math.cos(side_angle),
            patch_target.y + distance * math.sin(side_angle),
            random.uniform(2.5, 4.0),
        ))
        opening_width = random.uniform(2.8, 4.2)
        opening_height = opening_width * random.uniform(0.92, 1.08)
        aperture = add_window_aperture(
            window_location,
            patch_target,
            opening_width,
            opening_height,
        )
        sources.append(add_light_source(
            "window_sun", "AREA", window_location, patch_target,
            random.uniform(900.0, 1700.0), random.uniform(*window_temperature_range),
            size=random.uniform(0.22, 0.42), size_y=random.uniform(0.22, 0.42),
        ))
        window_metadata = {
            "enabled": True,
            "probability": SUNLIGHT_WINDOW_PROBABILITY,
            "aperture": aperture,
            "effect": "rectangular_sun_patch",
        }
        world_ranges = ((0.14, 0.28), room_temperature_range)
        exposure_range = (exposure_range[0] - 0.16, exposure_range[1] - 0.08)

    world = configure_world_lighting(*world_ranges)

    exposure = random.uniform(*exposure_range)
    bpy.context.scene.view_settings.exposure = exposure
    bpy.context.view_layer.update()
    primary = sources[0]
    return {
        "setup": setup,
        "source_count": len(sources),
        "sources": sources,
        "conditions": [setup] + (["sunlight_window"] if window_metadata else []),
        "temperature_profile": {
            "name": temperature_profile,
            "base_kelvin": base_temperature,
            "room_range_kelvin": list(room_temperature_range),
            "window_range_kelvin": list(window_temperature_range),
        },
        "sunlight_window": window_metadata or {
            "enabled": False,
            "probability": SUNLIGHT_WINDOW_PROBABILITY,
        },
        "world": world,
        "exposure": exposure,
        # Preserve the former single-light fields for downstream metadata readers.
        "type": primary["type"],
        "location": primary["location"],
        "target": primary["target"],
        "energy": primary["energy"],
        "temperature_kelvin": primary["temperature_kelvin"],
        "color": primary["color"],
        "size": primary["size"],
        "size_y": primary["size_y"],
    }


def set_principled_input(material: bpy.types.Material, input_name: str, value: object) -> None:
    bsdf = material.node_tree.nodes.get("Principled BSDF") if material.use_nodes else None
    if bsdf is not None and input_name in bsdf.inputs:
        bsdf.inputs[input_name].default_value = value


def randomize_sticker_material_gloss() -> dict[str, dict[str, float]]:
    """Give Rubik's cube stickers mild glossy highlights while preserving labels."""
    material_settings = {}
    for name in STICKER_COLOR_MATERIAL_NAMES:
        material = bpy.data.materials.get(name)
        if material is None:
            continue

        material.use_nodes = True
        roughness = random.uniform(*STICKER_ROUGHNESS_RANGE)
        specular = random.uniform(*STICKER_SPECULAR_RANGE)
        coat_weight = random.uniform(*STICKER_COAT_WEIGHT_RANGE)
        coat_roughness = random.uniform(*STICKER_COAT_ROUGHNESS_RANGE)

        set_principled_input(material, "Roughness", roughness)
        set_principled_input(material, "Specular IOR Level", specular)
        set_principled_input(material, "Specular", specular)
        set_principled_input(material, "Coat Weight", coat_weight)
        set_principled_input(material, "Clearcoat", coat_weight)
        set_principled_input(material, "Coat Roughness", coat_roughness)
        set_principled_input(material, "Clearcoat Roughness", coat_roughness)

        material_settings[name] = {
            "roughness": roughness,
            "specular": specular,
            "coat_weight": coat_weight,
            "coat_roughness": coat_roughness,
        }
    return material_settings


def runtime_random_seed() -> int:
    try:
        entropy = int.from_bytes(os.urandom(8), "big")
    except OSError:
        entropy = 0
    return entropy ^ time.time_ns() ^ os.getpid() ^ id(object())


def random_color_from_ranges(
    ranges: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...],
) -> tuple[float, float, float, float]:
    low, high = random.choice(ranges)
    return (
        random.uniform(low[0], high[0]),
        random.uniform(low[1], high[1]),
        random.uniform(low[2], high[2]),
        1.0,
    )


def random_background_color() -> tuple[float, float, float, float]:
    palettes = (
        ((0.45, 0.48, 0.46), (0.72, 0.70, 0.64)),
        ((0.30, 0.34, 0.38), (0.75, 0.78, 0.80)),
        ((0.36, 0.30, 0.28), (0.72, 0.62, 0.55)),
        ((0.28, 0.36, 0.34), (0.68, 0.76, 0.70)),
        ((0.20, 0.18, 0.34), (0.68, 0.52, 0.86)),
        ((0.16, 0.36, 0.52), (0.62, 0.82, 0.95)),
        ((0.50, 0.20, 0.16), (0.95, 0.58, 0.40)),
        ((0.50, 0.42, 0.08), (0.96, 0.84, 0.36)),
        ((0.12, 0.42, 0.24), (0.58, 0.9, 0.46)),
    )
    low, high = random.choice(palettes)
    return (
        random.uniform(low[0], high[0]),
        random.uniform(low[1], high[1]),
        random.uniform(low[2], high[2]),
        1.0,
    )


def random_table_color() -> tuple[float, float, float, float]:
    palettes = (
        # Dark surfaces: black desks, dark wood, slate, and dim fabric.
        ((0.015, 0.015, 0.015), (0.12, 0.12, 0.12)),
        ((0.05, 0.035, 0.025), (0.20, 0.12, 0.08)),
        ((0.025, 0.045, 0.035), (0.12, 0.18, 0.13)),
        ((0.035, 0.035, 0.065), (0.16, 0.16, 0.24)),
        # Low-contrast cube-like colors, including colors close to red/green/blue/yellow/white.
        ((0.00, 0.05, 0.00), (0.12, 0.42, 0.12)),
        ((0.18, 0.00, 0.00), (0.55, 0.08, 0.08)),
        ((0.00, 0.015, 0.16), (0.08, 0.16, 0.55)),
        ((0.35, 0.30, 0.02), (0.82, 0.72, 0.18)),
        ((0.58, 0.56, 0.50), (0.93, 0.90, 0.82)),
        # Keep some ordinary mid-tone tables in the mix.
        ((0.28, 0.22, 0.18), (0.62, 0.48, 0.36)),
        ((0.25, 0.25, 0.25), (0.58, 0.58, 0.58)),
    )
    low, high = random.choice(palettes)
    return (
        random.uniform(low[0], high[0]),
        random.uniform(low[1], high[1]),
        random.uniform(low[2], high[2]),
        1.0,
    )


def random_hand_skin_color() -> tuple[float, float, float, float]:
    return random_color_from_ranges(HAND_SKIN_TONE_RANGES)


def random_bright_object_color() -> tuple[float, float, float, float]:
    colors = (
        (1.0, 0.02, 0.02),
        (0.02, 0.16, 1.0),
        (0.02, 0.82, 0.12),
        (1.0, 0.9, 0.02),
        (1.0, 0.38, 0.0),
        (0.72, 0.02, 1.0),
        (0.0, 0.92, 1.0),
        (1.0, 0.02, 0.45),
        (0.05, 0.05, 0.05),
        (0.96, 0.96, 0.96),
    )
    base = random.choice(colors)
    jitter = random.uniform(0.86, 1.0)
    return (
        min(1.0, base[0] * jitter + random.uniform(0.0, 0.06)),
        min(1.0, base[1] * jitter + random.uniform(0.0, 0.06)),
        min(1.0, base[2] * jitter + random.uniform(0.0, 0.06)),
        1.0,
    )


def noisy_material(
    name: str,
    base_color: tuple[float, float, float, float] | None = None,
    *,
    textured: bool = True,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    base_color = base_color or random_background_color()
    set_principled_input(material, "Base Color", base_color)
    set_principled_input(material, "Roughness", random.uniform(0.55, 1.0))
    set_principled_input(material, "Metallic", random.choice((0.0, 0.0, 0.0, random.uniform(0.0, 0.35))))

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if textured and bsdf is not None:
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = random.uniform(8.0, 38.0)
        noise.inputs["Detail"].default_value = random.uniform(3.0, 11.0)
        noise.inputs["Roughness"].default_value = random.uniform(0.45, 0.75)

        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = random.uniform(0.18, 0.42)
        ramp.color_ramp.elements[1].position = random.uniform(0.58, 0.9)
        ramp.color_ramp.elements[0].color = tuple(max(0.0, channel * 0.55) for channel in base_color[:3]) + (1.0,)
        ramp.color_ramp.elements[1].color = tuple(min(1.0, channel * 1.35) for channel in base_color[:3]) + (1.0,)
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = random.uniform(0.015, 0.08)
        bump.inputs["Distance"].default_value = random.uniform(0.04, 0.18)
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return material


def glossy_material(
    name: str,
    base_color: tuple[float, float, float, float],
    *,
    roughness: float,
    specular: float,
    metallic: float = 0.0,
) -> bpy.types.Material:
    material = noisy_material(name, base_color, textured=False)
    set_principled_input(material, "Roughness", roughness)
    set_principled_input(material, "Specular IOR Level", specular)
    set_principled_input(material, "Specular", specular)
    set_principled_input(material, "Metallic", metallic)
    return material


def emissive_material(
    name: str,
    base_color: tuple[float, float, float, float],
    *,
    strength: float,
) -> bpy.types.Material:
    material = glossy_material(
        name,
        base_color,
        roughness=random.uniform(0.12, 0.32),
        specular=random.uniform(0.35, 0.75),
    )
    set_principled_input(material, "Emission Color", base_color)
    set_principled_input(material, "Emission", base_color)
    set_principled_input(material, "Emission Strength", strength)
    return material


def background_object_material(name: str) -> bpy.types.Material:
    if random.random() < 0.72:
        return noisy_material(
            name,
            random_bright_object_color(),
            textured=random.random() < 0.25,
        )
    return noisy_material(name)


def clear_background_scene() -> None:
    for obj in list(bpy.data.objects):
        if obj.name.startswith(BACKGROUND_SCENE_PREFIX) or obj.name.startswith(BACKGROUND_ASSET_COLLECTION_PREFIX):
            bpy.data.objects.remove(obj, do_unlink=True)
    for collection in list(bpy.data.collections):
        if collection.name.startswith(BACKGROUND_ASSET_COLLECTION_PREFIX):
            bpy.data.collections.remove(collection)
    for material in list(bpy.data.materials):
        if material.name.startswith(BACKGROUND_SCENE_MATERIAL_PREFIX) or material.name.startswith("SolveStation.Material."):
            bpy.data.materials.remove(material, do_unlink=True)


def background_scene_collection() -> bpy.types.Collection:
    collection = bpy.data.collections.get(BACKGROUND_SCENE_COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(BACKGROUND_SCENE_COLLECTION_NAME)
        bpy.context.scene.collection.children.link(collection)
    return collection


def link_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    if obj.name not in collection.objects.keys():
        collection.objects.link(obj)
    for existing_collection in list(obj.users_collection):
        if existing_collection != collection:
            existing_collection.objects.unlink(obj)


def active_created_object() -> bpy.types.Object:
    obj = getattr(bpy.context, "object", None) or bpy.context.view_layer.objects.active
    if obj is None:
        raise RuntimeError("Blender did not expose the object created by the previous operator.")
    return obj


def camera_basis() -> tuple[Vector, Vector, Vector]:
    camera = get_camera()
    camera_rotation = camera.rotation_quaternion
    return (
        camera_rotation @ Vector((0.0, 0.0, -1.0)),
        camera_rotation @ Vector((1.0, 0.0, 0.0)),
        camera_rotation @ Vector((0.0, 1.0, 0.0)),
    )


def camera_table_basis() -> tuple[Vector, Vector]:
    camera_direction, camera_right, _camera_up = camera_basis()
    table_forward = Vector((camera_direction.x, camera_direction.y, 0.0))
    if table_forward.length <= 1e-6:
        table_forward = Vector((0.0, -1.0, 0.0))
    table_forward.normalize()

    table_right = Vector((camera_right.x, camera_right.y, 0.0))
    if table_right.length <= 1e-6:
        table_right = Vector((1.0, 0.0, 0.0))
    table_right.normalize()
    return table_forward, table_right


def configure_world_background() -> tuple[float, float, float]:
    world = bpy.context.scene.world or bpy.data.worlds.new("SyntheticDataWorld")
    bpy.context.scene.world = world
    level = random.uniform(*WORLD_AMBIENT_LEVEL_RANGE)
    tint = kelvin_to_rgb(random.uniform(*LIGHT_TEMPERATURE_KELVIN_RANGE))
    color = tuple(level * random.uniform(0.75, 1.25) * channel for channel in tint)
    world.color = color
    return color


def hide_flat_background() -> None:
    background = bpy.data.objects.get(BACKGROUND_NAME)
    if background is not None:
        background.hide_render = True
        background.hide_viewport = True
        background.hide_set(True)


def add_background_plane(
    collection: bpy.types.Collection,
    name: str,
    location: Vector,
    rotation: Quaternion | None,
    size: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    if rotation is None:
        bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    else:
        bpy.ops.mesh.primitive_plane_add(size=size, location=location, rotation=rotation.to_euler())
    plane = active_created_object()
    plane.name = name
    plane.data.materials.append(material)
    link_to_collection(plane, collection)
    return plane


def background_object_offsets(clearance: float) -> tuple[float, float]:
    min_right = BACKGROUND_KEEP_OUT_RIGHT + clearance
    min_depth = BACKGROUND_KEEP_OUT_DEPTH + clearance
    for _attempt in range(12):
        right = random.uniform(*BACKGROUND_OBJECT_RIGHT_RANGE)
        depth = random.uniform(*BACKGROUND_OBJECT_DEPTH_RANGE)
        if abs(right) >= min_right or depth >= min_depth:
            return right, depth

    if random.random() < 0.5:
        right = random.choice((-1.0, 1.0)) * random.uniform(
            min_right,
            max(min_right, BACKGROUND_OBJECT_RIGHT_RANGE[1]),
        )
        depth = random.uniform(*BACKGROUND_OBJECT_DEPTH_RANGE)
    else:
        right = random.uniform(*BACKGROUND_OBJECT_RIGHT_RANGE)
        depth = random.uniform(min_depth, max(min_depth, BACKGROUND_OBJECT_DEPTH_RANGE[1]))
    return right, depth


def add_background_box(
    collection: bpy.types.Collection,
    name: str,
    location: Vector,
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    rotation: Quaternion | None = None,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = active_created_object()
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = rotation or Quaternion((1.0, 0.0, 0.0, 0.0))
    obj.scale = scale
    obj.data.materials.append(material)
    link_to_collection(obj, collection)
    return obj


def prop_location(table_forward: Vector, table_right: Vector, right: float, depth: float, height: float) -> Vector:
    return (table_forward * depth) + (table_right * right) + Vector((0.0, 0.0, BACKGROUND_TABLE_Z + height * 0.5))


def random_table_yaw() -> Quaternion:
    return Quaternion(Vector((0.0, 0.0, 1.0)), random.uniform(0.0, math.tau))


def background_asset_blend_path() -> Path | None:
    return next((path for path in BACKGROUND_ASSET_BLEND_CANDIDATES if path.exists()), None)


def background_asset_collection_names(path: Path) -> list[str]:
    with bpy.data.libraries.load(str(path), link=False) as (data_from, _data_to):
        return [
            name
            for name in data_from.collections
            if any(
                name.startswith(prefix)
                for prefix in BACKGROUND_BLEND_ASSET_COLLECTION_PREFIXES
            )
            and not any(
                name.startswith(prefix)
                for prefix in BACKGROUND_BLEND_ASSET_EXCLUDED_PREFIXES
            )
        ]


def append_background_asset_collection(path: Path, source_name: str, instance_name: str) -> bpy.types.Collection | None:
    with bpy.data.libraries.load(str(path), link=False) as (data_from, data_to):
        if source_name not in data_from.collections:
            return None
        data_to.collections = [source_name]

    imported = data_to.collections[0] if data_to.collections else None
    if imported is None:
        return None
    imported.name = instance_name
    return imported


def background_asset_collection_weight(name: str) -> float:
    return next(
        (
            weight
            for prefix, weight in BACKGROUND_BLEND_ASSET_FAMILY_WEIGHTS.items()
            if name.startswith(prefix)
        ),
        0.7,
    )


def collection_asset_objects(collection: bpy.types.Collection) -> list[bpy.types.Object]:
    objects = getattr(collection, "all_objects", collection.objects)
    return [obj for obj in objects if obj.type in {"MESH", "LIGHT"}]


def collection_asset_root(collection: bpy.types.Collection) -> bpy.types.Object | None:
    direct_objects = list(collection.objects)
    root_name = f"{collection.name}{BACKGROUND_ASSET_ROOT_SUFFIX}"
    for obj in direct_objects:
        if obj.type == "EMPTY" and obj.name == root_name:
            return obj
    for obj in direct_objects:
        if obj.type == "EMPTY" and obj.name.endswith(BACKGROUND_ASSET_ROOT_SUFFIX):
            return obj
    return None


def linked_object_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector] | None:
    points = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not points:
        return None
    min_point = Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
    max_point = Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
    return min_point, max_point


def link_collection_to_collection(
    child: bpy.types.Collection,
    parent: bpy.types.Collection,
) -> None:
    if child.name not in parent.children.keys():
        parent.children.link(child)
    for scene_child in list(bpy.context.scene.collection.children):
        if scene_child == child and scene_child != parent:
            bpy.context.scene.collection.children.unlink(child)


def place_background_asset_collection(
    asset_collection: bpy.types.Collection,
    parent_collection: bpy.types.Collection,
    name_prefix: str,
    location: Vector,
    rotation: Quaternion,
    scale: float,
) -> dict[str, object] | None:
    asset_objects = collection_asset_objects(asset_collection)
    mesh_objects = [obj for obj in asset_objects if obj.type == "MESH"]
    if not mesh_objects:
        return None

    link_collection_to_collection(asset_collection, parent_collection)
    root = collection_asset_root(asset_collection)
    for obj in asset_objects:
        obj.name = f"{name_prefix}.{obj.name}"
    if root is not None:
        root.name = f"{name_prefix}.{root.name}"

    bounds = linked_object_bounds(mesh_objects)
    if bounds is None:
        return None
    min_point, max_point = bounds
    center = (min_point + max_point) * 0.5
    min_z = min_point.z
    transform = (
        Matrix.Translation(location)
        @ rotation.to_matrix().to_4x4()
        @ Matrix.Diagonal((scale, scale, scale, 1.0))
        @ Matrix.Translation(Vector((-center.x, -center.y, -min_z)))
    )
    if root is not None:
        root.matrix_world = transform @ root.matrix_world
    else:
        for obj in asset_objects:
            obj.matrix_world = transform @ obj.matrix_world
            if hasattr(obj, "rotation_mode"):
                obj.rotation_mode = "QUATERNION"

    bpy.context.view_layer.update()
    bounds_after = linked_object_bounds(mesh_objects)
    if bounds_after is None:
        dimensions = [0.0, 0.0, 0.0]
    else:
        placed_min, placed_max = bounds_after
        dimensions = [
            placed_max.x - placed_min.x,
            placed_max.y - placed_min.y,
            placed_max.z - placed_min.z,
        ]
    return {
        "family": "blend_asset",
        "collection": asset_collection.name,
        "root": root.name if root is not None else None,
        "names": [obj.name for obj in asset_objects],
        "mesh_count": len(mesh_objects),
        "light_count": len([obj for obj in asset_objects if obj.type == "LIGHT"]),
        "scale": scale,
        "dimensions": dimensions,
    }


def add_blend_asset_prop(
    collection: bpy.types.Collection,
    index: int,
    table_forward: Vector,
    table_right: Vector,
    used_asset_sources: set[str] | None = None,
) -> dict[str, object] | None:
    path = background_asset_blend_path()
    if path is None:
        return None

    asset_names = background_asset_collection_names(path)
    if not asset_names:
        return None

    available_collections = [
        name
        for name in asset_names
        if used_asset_sources is None or name not in used_asset_sources
    ]
    if not available_collections:
        available_collections = asset_names

    source_name = random.choices(
        available_collections,
        weights=[background_asset_collection_weight(name) for name in available_collections],
        k=1,
    )[0]
    instance_name = f"{BACKGROUND_ASSET_COLLECTION_PREFIX}.{index:03d}.{source_name}"
    asset_collection = append_background_asset_collection(path, source_name, instance_name)
    if asset_collection is None:
        return None

    camera_direction, camera_right, _camera_up = camera_basis()
    right = random.uniform(*BACKGROUND_BLEND_ASSET_VISIBLE_RIGHT_RANGE)
    if abs(right) < BACKGROUND_BLEND_ASSET_VISIBLE_KEEP_OUT_RIGHT:
        right = random.choice((-1.0, 1.0)) * random.uniform(
            BACKGROUND_BLEND_ASSET_VISIBLE_KEEP_OUT_RIGHT,
            max(BACKGROUND_BLEND_ASSET_VISIBLE_KEEP_OUT_RIGHT, BACKGROUND_BLEND_ASSET_VISIBLE_RIGHT_RANGE[1]),
        )
    depth = random.uniform(*BACKGROUND_BLEND_ASSET_VISIBLE_DEPTH_RANGE)
    location = (camera_direction * depth) + (camera_right * right) + Vector((0.0, 0.0, BACKGROUND_TABLE_Z))
    rotation = random_table_yaw()
    scale = random.uniform(*BACKGROUND_BLEND_ASSET_SCALE_RANGE)
    placed = place_background_asset_collection(
        asset_collection,
        collection,
        f"{BACKGROUND_SCENE_PREFIX}.blend_asset.{index:03d}",
        location,
        rotation,
        scale,
    )
    if placed is None:
        bpy.data.collections.remove(asset_collection)
        return None

    return {
        **placed,
        "source_file": str(path),
        "source_name": source_name,
        "camera_offsets": {"right": right, "depth": depth},
    }


def add_laptop_prop(
    collection: bpy.types.Collection,
    index: int,
    table_forward: Vector,
    table_right: Vector,
) -> dict[str, object]:
    right, depth = background_object_offsets(0.45)
    yaw = random_table_yaw()
    local_right = yaw @ Vector((1.0, 0.0, 0.0))
    local_forward = yaw @ Vector((0.0, 1.0, 0.0))
    width = random.uniform(1.45, 2.55)
    base_depth = random.uniform(1.0, 1.75)
    base_height = random.uniform(0.05, 0.11)
    screen_height = random.uniform(0.8, 1.55)
    base_location = prop_location(table_forward, table_right, right, depth, base_height)

    body_mat = glossy_material(
        f"{BACKGROUND_SCENE_MATERIAL_PREFIX}.laptop_body.{index:03d}",
        random.choice(((0.02, 0.022, 0.026, 1.0), (0.45, 0.46, 0.47, 1.0), (0.08, 0.08, 0.09, 1.0))),
        roughness=random.uniform(0.22, 0.52),
        specular=random.uniform(0.28, 0.62),
        metallic=random.choice((0.0, 0.0, random.uniform(0.15, 0.45))),
    )
    screen_mat = emissive_material(
        f"{BACKGROUND_SCENE_MATERIAL_PREFIX}.laptop_screen.{index:03d}",
        random.choice(((0.75, 0.9, 1.0, 1.0), (0.95, 0.96, 1.0, 1.0), (0.25, 0.52, 1.0, 1.0))),
        strength=random.uniform(0.25, 1.6),
    )
    key_mat = glossy_material(
        f"{BACKGROUND_SCENE_MATERIAL_PREFIX}.laptop_keys.{index:03d}",
        random.choice(((0.015, 0.015, 0.016, 1.0), (0.82, 0.82, 0.78, 1.0))),
        roughness=random.uniform(0.34, 0.72),
        specular=random.uniform(0.12, 0.38),
    )

    base = add_background_box(
        collection,
        f"{BACKGROUND_SCENE_PREFIX}.laptop_base.{index:03d}",
        base_location,
        (width, base_depth, base_height),
        body_mat,
        yaw,
    )
    screen_location = base_location + (local_forward * (base_depth * 0.48)) + Vector((0.0, 0.0, screen_height * 0.52))
    screen = add_background_box(
        collection,
        f"{BACKGROUND_SCENE_PREFIX}.laptop_screen.{index:03d}",
        screen_location,
        (width * 0.94, 0.035, screen_height),
        screen_mat,
        yaw,
    )

    key_count = 0
    rows = random.randint(3, 5)
    cols = random.randint(7, 11)
    key_w = width * 0.72 / cols
    key_d = base_depth * 0.42 / rows
    start_right = -width * 0.36 + key_w * 0.5
    start_forward = -base_depth * 0.28 + key_d * 0.5
    for row in range(rows):
        for col in range(cols):
            if random.random() < 0.08:
                continue
            key_location = (
                base_location
                + (local_right * (start_right + col * key_w))
                + (local_forward * (start_forward + row * key_d))
                + Vector((0.0, 0.0, base_height * 0.62))
            )
            add_background_box(
                collection,
                f"{BACKGROUND_SCENE_PREFIX}.laptop_key.{index:03d}.{row:02d}.{col:02d}",
                key_location,
                (key_w * 0.72, key_d * 0.68, 0.012),
                key_mat,
                yaw,
            )
            key_count += 1

    return {
        "family": "laptop",
        "name": base.name,
        "screen": screen.name,
        "key_count": key_count,
        "table_offsets": {"right": right, "depth": depth},
        "size": {"width": width, "depth": base_depth, "screen_height": screen_height},
    }


def add_book_or_box_prop(
    collection: bpy.types.Collection,
    index: int,
    table_forward: Vector,
    table_right: Vector,
) -> dict[str, object]:
    right, depth = background_object_offsets(0.4)
    yaw = random_table_yaw()
    count = random.randint(1, 4)
    placed = []
    z_cursor = 0.0
    width = random.uniform(0.9, 1.9)
    depth_size = random.uniform(0.7, 1.55)
    for layer in range(count):
        height = random.uniform(0.12, 0.45)
        color = random.choice((random_bright_object_color(), random_background_color(), random_table_color()))
        material = noisy_material(f"{BACKGROUND_SCENE_MATERIAL_PREFIX}.book_box.{index:03d}.{layer:02d}", color, textured=True)
        location = prop_location(table_forward, table_right, right, depth, height) + Vector((0.0, 0.0, z_cursor))
        obj = add_background_box(
            collection,
            f"{BACKGROUND_SCENE_PREFIX}.book_box.{index:03d}.{layer:02d}",
            location,
            (width * random.uniform(0.85, 1.08), depth_size * random.uniform(0.78, 1.1), height),
            material,
            yaw @ Quaternion(Vector((0.0, 0.0, 1.0)), random.uniform(-0.15, 0.15)),
        )
        placed.append(obj.name)
        z_cursor += height

    return {
        "family": "books_boxes",
        "names": placed,
        "table_offsets": {"right": right, "depth": depth},
        "layers": count,
    }


def add_fabric_panel_prop(
    collection: bpy.types.Collection,
    index: int,
    table_forward: Vector,
    table_right: Vector,
) -> dict[str, object]:
    right, depth = background_object_offsets(0.45)
    yaw = random_table_yaw()
    width = random.uniform(0.8, 1.9)
    height = random.uniform(1.35, 3.2)
    thickness = random.uniform(0.035, 0.11)
    base_color = random.choice((random_bright_object_color(), random_background_color()))
    material = noisy_material(f"{BACKGROUND_SCENE_MATERIAL_PREFIX}.fabric.{index:03d}", base_color, textured=True)
    location = prop_location(table_forward, table_right, right, depth, height)
    panel = add_background_box(collection, f"{BACKGROUND_SCENE_PREFIX}.fabric.{index:03d}", location, (width, thickness, height), material, yaw)

    stripe_count = random.randint(2, 6)
    stripe_mat = noisy_material(f"{BACKGROUND_SCENE_MATERIAL_PREFIX}.fabric_stripe.{index:03d}", random_bright_object_color(), textured=False)
    local_right = yaw @ Vector((1.0, 0.0, 0.0))
    local_forward = yaw @ Vector((0.0, -1.0, 0.0))
    for stripe in range(stripe_count):
        stripe_x = random.uniform(-width * 0.42, width * 0.42)
        stripe_location = location + (local_right * stripe_x) + (local_forward * (thickness * 0.65))
        add_background_box(
            collection,
            f"{BACKGROUND_SCENE_PREFIX}.fabric_stripe.{index:03d}.{stripe:02d}",
            stripe_location,
            (random.uniform(0.025, 0.09), 0.012, height * random.uniform(0.82, 1.0)),
            stripe_mat,
            yaw,
        )

    return {
        "family": "fabric_panel",
        "name": panel.name,
        "stripe_count": stripe_count,
        "table_offsets": {"right": right, "depth": depth},
        "size": {"width": width, "height": height},
    }


def add_screen_or_poster_prop(
    collection: bpy.types.Collection,
    index: int,
    table_forward: Vector,
    table_right: Vector,
    wall_distance: float,
) -> dict[str, object]:
    right = random.uniform(*BACKGROUND_OBJECT_RIGHT_RANGE)
    width = random.uniform(1.2, 3.3)
    height = random.uniform(0.8, 2.25)
    z = BACKGROUND_TABLE_Z + random.uniform(0.9, 3.0)
    location = (table_forward * (wall_distance - 0.035)) + (table_right * right) + Vector((0.0, 0.0, z))
    wall_normal = -table_forward
    rotation = wall_normal.to_track_quat("Y", "Z")
    is_screen = random.random() < 0.55
    material = (
        emissive_material(
            f"{BACKGROUND_SCENE_MATERIAL_PREFIX}.wall_screen.{index:03d}",
            random_bright_object_color(),
            strength=random.uniform(0.2, 1.4),
        )
        if is_screen
        else noisy_material(f"{BACKGROUND_SCENE_MATERIAL_PREFIX}.poster.{index:03d}", random_bright_object_color(), textured=True)
    )
    panel = add_background_box(
        collection,
        f"{BACKGROUND_SCENE_PREFIX}.{'wall_screen' if is_screen else 'poster'}.{index:03d}",
        location,
        (width, 0.018, height),
        material,
        rotation,
    )

    block_count = random.randint(2, 7)
    for block in range(block_count):
        block_mat = glossy_material(
            f"{BACKGROUND_SCENE_MATERIAL_PREFIX}.poster_block.{index:03d}.{block:02d}",
            random.choice((random_bright_object_color(), (0.04, 0.04, 0.045, 1.0), (0.96, 0.96, 0.92, 1.0))),
            roughness=random.uniform(0.18, 0.62),
            specular=random.uniform(0.12, 0.55),
        )
        local_right = rotation @ Vector((1.0, 0.0, 0.0))
        local_up = rotation @ Vector((0.0, 0.0, 1.0))
        local_forward = rotation @ Vector((0.0, -1.0, 0.0))
        block_location = (
            location
            + local_right * random.uniform(-width * 0.38, width * 0.38)
            + local_up * random.uniform(-height * 0.32, height * 0.32)
            + local_forward * 0.016
        )
        add_background_box(
            collection,
            f"{BACKGROUND_SCENE_PREFIX}.poster_block.{index:03d}.{block:02d}",
            block_location,
            (random.uniform(width * 0.08, width * 0.32), 0.012, random.uniform(height * 0.04, height * 0.18)),
            block_mat,
            rotation,
        )

    return {
        "family": "wall_screen" if is_screen else "poster",
        "name": panel.name,
        "block_count": block_count,
        "wall_offset": {"right": right, "distance": wall_distance},
        "size": {"width": width, "height": height},
    }


def add_primitive_clutter_prop(
    collection: bpy.types.Collection,
    index: int,
    table_forward: Vector,
    table_right: Vector,
) -> dict[str, object]:
    scale = random.uniform(*BACKGROUND_OBJECT_SCALE_RANGE)
    right, depth = background_object_offsets(scale * 1.2)
    z_scale = scale * random.uniform(0.35, 1.8)
    location = prop_location(table_forward, table_right, right, depth, z_scale)
    shape = random.choice(("cube", "cylinder", "uv_sphere"))
    if shape == "cube":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    elif shape == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=random.choice((12, 16, 24)), radius=0.5, depth=1.0, location=location)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.5, location=location)

    obj = active_created_object()
    obj.name = f"{BACKGROUND_SCENE_PREFIX}.primitive.{index:03d}"
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = random_table_yaw()
    obj.scale = (
        scale * random.uniform(0.45, 2.4),
        scale * random.uniform(0.45, 2.4),
        z_scale,
    )
    obj.data.materials.append(background_object_material(f"{BACKGROUND_SCENE_MATERIAL_PREFIX}.primitive.{index:03d}"))
    link_to_collection(obj, collection)
    return {
        "family": "primitive",
        "name": obj.name,
        "shape": shape,
        "location": [obj.location.x, obj.location.y, obj.location.z],
        "table_offsets": {"right": right, "depth": depth},
        "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
    }


def add_realistic_background_prop(
    collection: bpy.types.Collection,
    index: int,
    table_forward: Vector,
    table_right: Vector,
    wall_distance: float,
    used_asset_sources: set[str] | None = None,
) -> dict[str, object]:
    if random.random() < BACKGROUND_BLEND_ASSET_PROBABILITY:
        blend_asset = add_blend_asset_prop(
            collection,
            index,
            table_forward,
            table_right,
            used_asset_sources,
        )
        if blend_asset is not None:
            source_name = blend_asset.get("source_name")
            if used_asset_sources is not None and isinstance(source_name, str):
                used_asset_sources.add(source_name)
            return blend_asset
        if background_asset_blend_path() is not None:
            return {
                "family": "blend_asset_failed",
                "source_file": str(background_asset_blend_path()),
                "reason": "append_failed",
            }

    family = random.choices(
        ("laptop", "books_boxes", "fabric_panel", "screen_poster", "primitive"),
        weights=(0.22, 0.26, 0.18, 0.18, 0.16),
        k=1,
    )[0]
    if family == "laptop":
        return add_laptop_prop(collection, index, table_forward, table_right)
    if family == "books_boxes":
        return add_book_or_box_prop(collection, index, table_forward, table_right)
    if family == "fabric_panel":
        return add_fabric_panel_prop(collection, index, table_forward, table_right)
    if family == "screen_poster":
        return add_screen_or_poster_prop(collection, index, table_forward, table_right, wall_distance)
    return add_primitive_clutter_prop(collection, index, table_forward, table_right)


def solve_station_asset_blend_path() -> Path:
    path = next((candidate for candidate in SOLVE_STATION_ASSET_BLEND_CANDIDATES if candidate.exists()), None)
    if path is None:
        candidates = ", ".join(str(candidate) for candidate in SOLVE_STATION_ASSET_BLEND_CANDIDATES)
        raise FileNotFoundError(
            "The solve-station asset library has not been built. "
            f"Expected one of: {candidates}. Run scripts/build_solve_station_assets.py in Blender."
        )
    return path


def scaled_color(
    color: tuple[float, float, float, float],
    factor: float,
) -> tuple[float, float, float, float]:
    return tuple(min(1.0, max(0.0, channel * factor)) for channel in color[:3]) + (1.0,)


def configure_station_color_pattern(
    material: bpy.types.Material,
    base_color: tuple[float, float, float, float],
    accent_color: tuple[float, float, float, float],
    pattern: str,
) -> None:
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return
    for link in list(bsdf.inputs["Base Color"].links):
        links.remove(link)
    if pattern == "solid":
        bsdf.inputs["Base Color"].default_value = base_color
        return

    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.name = f"SolveStation.Pattern.{pattern}.Coordinates"
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.name = f"SolveStation.Pattern.{pattern}.Colors"
    ramp.color_ramp.elements[0].color = base_color
    ramp.color_ramp.elements[1].color = accent_color

    if pattern == "stripes":
        texture = nodes.new("ShaderNodeTexWave")
        texture.name = "SolveStation.Pattern.Stripes"
        texture.wave_type = "BANDS"
        texture.bands_direction = random.choice(("X", "Y", "Z"))
        texture.inputs["Scale"].default_value = random.uniform(2.2, 6.5)
        texture.inputs["Distortion"].default_value = random.uniform(0.0, 1.2)
        ramp.color_ramp.interpolation = "CONSTANT"
        ramp.color_ramp.elements[0].position = random.uniform(0.38, 0.48)
        ramp.color_ramp.elements[1].position = random.uniform(0.52, 0.62)
        links.new(texcoord.outputs["Generated"], texture.inputs["Vector"])
        links.new(texture.outputs["Fac"], ramp.inputs["Fac"])
    else:
        texture = nodes.new("ShaderNodeTexNoise")
        texture.name = f"SolveStation.Pattern.{pattern.title()}"
        if pattern == "heather":
            texture.inputs["Scale"].default_value = random.uniform(70.0, 145.0)
            texture.inputs["Detail"].default_value = random.uniform(2.0, 5.0)
            texture.inputs["Roughness"].default_value = random.uniform(0.55, 0.8)
            ramp.color_ramp.elements[0].color = scaled_color(base_color, 0.7)
            ramp.color_ramp.elements[1].color = scaled_color(base_color, 1.3)
        else:
            texture.inputs["Scale"].default_value = random.uniform(3.5, 9.0)
            texture.inputs["Detail"].default_value = random.uniform(4.0, 9.0)
            texture.inputs["Roughness"].default_value = random.uniform(0.62, 0.88)
        links.new(texcoord.outputs["Generated"], texture.inputs["Vector"])
        links.new(texture.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])


def randomize_solve_station_materials(asset_collection: bpy.types.Collection) -> dict[str, object]:
    shirt_color = random.choice((
        (0.012, 0.015, 0.022, 1.0),
        (0.055, 0.065, 0.08, 1.0),
        (0.018, 0.055, 0.12, 1.0),
        (0.32, 0.018, 0.015, 1.0),
        (0.05, 0.22, 0.12, 1.0),
        (0.34, 0.12, 0.38, 1.0),
        (0.55, 0.53, 0.48, 1.0),
    ))
    shirt_accent_color = random.choice((
        (0.015, 0.018, 0.024, 1.0),
        (0.82, 0.78, 0.68, 1.0),
        (0.03, 0.18, 0.36, 1.0),
        (0.48, 0.035, 0.025, 1.0),
        (0.03, 0.31, 0.16, 1.0),
        (0.34, 0.08, 0.46, 1.0),
    ))
    shirt_pattern = random.choices(
        ("solid", "heather", "stripes", "mottled"),
        weights=(0.3, 0.28, 0.27, 0.15),
        k=1,
    )[0]
    shirt_seam_color = tuple(max(0.0, channel * 0.72) for channel in shirt_color[:3]) + (1.0,)
    skin_color = random_hand_skin_color()
    wall_color = random.choice((
        (0.22, 0.23, 0.24, 1.0),
        (0.38, 0.36, 0.32, 1.0),
        (0.28, 0.34, 0.36, 1.0),
        (0.48, 0.47, 0.43, 1.0),
    ))
    mat_color, mat_print_color = random.choice((
        ((0.018, 0.028, 0.045, 1.0), (0.02, 0.25, 0.48, 1.0)),
        ((0.02, 0.11, 0.12, 1.0), (0.02, 0.48, 0.43, 1.0)),
        ((0.095, 0.018, 0.025, 1.0), (0.58, 0.035, 0.025, 1.0)),
        ((0.075, 0.065, 0.055, 1.0), (0.56, 0.34, 0.08, 1.0)),
        ((0.055, 0.025, 0.09, 1.0), (0.36, 0.08, 0.58, 1.0)),
        ((0.055, 0.06, 0.065, 1.0), (0.3, 0.34, 0.38, 1.0)),
    ))
    mat_pattern = random.choices(
        ("solid", "heather", "stripes", "mottled"),
        weights=(0.3, 0.2, 0.28, 0.22),
        k=1,
    )[0]
    timer_color, timer_trim_color, sensor_color = random.choice((
        (
            (0.01, 0.012, 0.016, 1.0),
            (0.025, 0.16, 0.31, 1.0),
            (0.04, 0.22, 0.42, 1.0),
        ),
        (
            (0.035, 0.038, 0.044, 1.0),
            (0.32, 0.055, 0.035, 1.0),
            (0.48, 0.085, 0.045, 1.0),
        ),
        (
            (0.018, 0.022, 0.028, 1.0),
            (0.06, 0.31, 0.16, 1.0),
            (0.08, 0.46, 0.22, 1.0),
        ),
        (
            (0.025, 0.022, 0.035, 1.0),
            (0.28, 0.08, 0.4, 1.0),
            (0.4, 0.13, 0.57, 1.0),
        ),
    ))
    phone_color = random.choice((
        (0.018, 0.021, 0.026, 1.0),
        (0.24, 0.25, 0.27, 1.0),
        (0.035, 0.075, 0.13, 1.0),
        (0.22, 0.055, 0.045, 1.0),
        (0.13, 0.055, 0.2, 1.0),
    ))
    phone_screen_color = random.choice((
        (0.025, 0.08, 0.16, 1.0),
        (0.12, 0.025, 0.18, 1.0),
        (0.018, 0.14, 0.12, 1.0),
        (0.18, 0.055, 0.018, 1.0),
    ))
    laptop_color = random.choice((
        (0.12, 0.13, 0.15, 1.0),
        (0.24, 0.25, 0.27, 1.0),
        (0.055, 0.065, 0.08, 1.0),
        (0.24, 0.21, 0.18, 1.0),
    ))
    laptop_screen_color = random.choice((
        (0.02, 0.07, 0.14, 1.0),
        (0.08, 0.025, 0.13, 1.0),
        (0.015, 0.12, 0.15, 1.0),
    ))
    desk_color = random.choice((
        (0.14, 0.045, 0.018, 1.0),
        (0.24, 0.075, 0.025, 1.0),
        (0.34, 0.16, 0.055, 1.0),
        (0.12, 0.09, 0.065, 1.0),
    ))
    color_by_material = {
        "SolveStation.Material.Shirt": shirt_color,
        "SolveStation.Material.ShirtSeam": shirt_seam_color,
        "SolveStation.Material.Skin": skin_color,
        "SolveStation.Material.Wall": wall_color,
        "SolveStation.Material.MatFabric": mat_color,
        "SolveStation.Material.MatPrint": mat_print_color,
        "SolveStation.Material.TimerBody": timer_color,
        "SolveStation.Material.TimerTrim": timer_trim_color,
        "SolveStation.Material.Sensor": sensor_color,
        "SolveStation.Material.PhoneBody": phone_color,
        "SolveStation.Material.PhoneScreen": phone_screen_color,
        "SolveStation.Material.Metal": laptop_color,
        "SolveStation.Material.LaptopScreen": laptop_screen_color,
        "SolveStation.Material.DeskWood": desk_color,
    }
    emissive_materials = {
        "SolveStation.Material.PhoneScreen",
        "SolveStation.Material.LaptopScreen",
    }
    changed_materials: set[str] = set()
    randomized_materials: dict[str, bpy.types.Material] = {}
    for obj in asset_collection.all_objects:
        if obj.type != "MESH":
            continue
        for mat in obj.data.materials:
            if mat is None or mat.name in changed_materials:
                continue
            source_name = next(
                (
                    name
                    for name in sorted(color_by_material, key=len, reverse=True)
                    if mat.name.startswith(name)
                ),
                None,
            )
            if source_name is None:
                continue
            color = color_by_material[source_name]
            mat.diffuse_color = color
            set_principled_input(mat, "Base Color", color)
            if source_name in emissive_materials:
                set_principled_input(mat, "Emission Color", color)
                set_principled_input(mat, "Emission", color)
            changed_materials.add(mat.name)
            randomized_materials[source_name] = mat
    shirt_material = randomized_materials.get("SolveStation.Material.Shirt")
    if shirt_material is not None:
        configure_station_color_pattern(
            shirt_material,
            shirt_color,
            shirt_accent_color,
            shirt_pattern,
        )
    mat_material = randomized_materials.get("SolveStation.Material.MatFabric")
    if mat_material is not None:
        configure_station_color_pattern(
            mat_material,
            mat_color,
            mat_print_color,
            mat_pattern,
        )
    return {
        "shirt_color": list(shirt_color),
        "shirt_accent_color": list(shirt_accent_color),
        "shirt_pattern": shirt_pattern,
        "skin_color": list(skin_color),
        "wall_color": list(wall_color),
        "mat_color": list(mat_color),
        "mat_print_color": list(mat_print_color),
        "mat_pattern": mat_pattern,
        "timer_color": list(timer_color),
        "timer_trim_color": list(timer_trim_color),
        "phone_color": list(phone_color),
        "phone_screen_color": list(phone_screen_color),
        "laptop_color": list(laptop_color),
        "laptop_screen_color": list(laptop_screen_color),
        "desk_color": list(desk_color),
    }


def solve_station_family_objects(
    asset_collection: bpy.types.Collection,
    families: tuple[str, ...],
) -> list[bpy.types.Object]:
    markers = tuple(f"SolveStation.{family}." for family in families)
    return [
        obj
        for obj in asset_collection.all_objects
        if obj.type == "MESH" and any(marker in obj.name for marker in markers)
    ]


def object_group_local_xy_bounds(
    objects: list[bpy.types.Object],
) -> tuple[float, float, float, float]:
    parent = objects[0].parent if objects else None
    parent_inverse = (
        parent.matrix_world.inverted_safe()
        if parent is not None
        else Matrix.Identity(4)
    )
    points = [
        parent_inverse @ obj.matrix_world @ Vector(corner)
        for obj in objects
        for corner in obj.bound_box
    ]
    if not points:
        raise ValueError("Cannot calculate bounds for an empty solve-station family.")
    return (
        min(point.x for point in points),
        max(point.x for point in points),
        min(point.y for point in points),
        max(point.y for point in points),
    )


def xy_bounds_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    padding: float = 0.0,
) -> bool:
    return not (
        first[1] + padding <= second[0]
        or second[1] + padding <= first[0]
        or first[3] + padding <= second[2]
        or second[3] + padding <= first[2]
    )


def set_object_group_local_pose(
    objects: list[bpy.types.Object],
    base_matrices: dict[str, Matrix],
    target_x: float,
    target_y: float,
    yaw_degrees: float,
) -> tuple[float, float, float, float]:
    for obj in objects:
        obj.matrix_world = base_matrices[obj.name].copy()
    bpy.context.view_layer.update()
    bounds = object_group_local_xy_bounds(objects)
    center = Vector(((bounds[0] + bounds[1]) * 0.5, (bounds[2] + bounds[3]) * 0.5, 0.0))
    target = Vector((target_x, target_y, 0.0))
    rotation = Matrix.Rotation(math.radians(yaw_degrees), 4, "Z")
    local_transform = Matrix.Translation(target) @ rotation @ Matrix.Translation(-center)
    parent = objects[0].parent if objects else None
    if parent is None:
        world_transform = local_transform
    else:
        world_transform = (
            parent.matrix_world
            @ local_transform
            @ parent.matrix_world.inverted_safe()
        )
    for obj in objects:
        obj.matrix_world = world_transform @ base_matrices[obj.name]
    bpy.context.view_layer.update()
    return object_group_local_xy_bounds(objects)


def randomize_solve_station_layout(
    asset_collection: bpy.types.Collection,
) -> dict[str, object]:
    """Jitter prop groups while keeping their footprints separated on the desk."""
    placements: dict[str, object] = {}
    occupied: list[tuple[str, tuple[float, float, float, float]]] = [
        ("target_cube_keepout", (-1.3, 1.3, -1.3, 1.3)),
    ]

    mat_objects = solve_station_family_objects(asset_collection, ("Mat",))
    if mat_objects:
        base = {obj.name: obj.matrix_world.copy() for obj in mat_objects}
        bounds = object_group_local_xy_bounds(mat_objects)
        target_x = (bounds[0] + bounds[1]) * 0.5 + random.uniform(-0.24, 0.24)
        target_y = (bounds[2] + bounds[3]) * 0.5 + random.uniform(-0.18, 0.18)
        yaw = random.uniform(-2.5, 2.5)
        set_object_group_local_pose(mat_objects, base, target_x, target_y, yaw)
        placements["mat"] = {
            "center": [target_x, target_y],
            "yaw_degrees": yaw,
        }

    solver_objects = solve_station_family_objects(asset_collection, ("Shirt", "Arm"))
    if solver_objects:
        base = {obj.name: obj.matrix_world.copy() for obj in solver_objects}
        bounds = object_group_local_xy_bounds(solver_objects)
        target_x = (bounds[0] + bounds[1]) * 0.5 + random.uniform(-0.38, 0.38)
        target_y = (bounds[2] + bounds[3]) * 0.5 + random.uniform(-0.16, 0.28)
        yaw = random.uniform(-3.5, 3.5)
        set_object_group_local_pose(solver_objects, base, target_x, target_y, yaw)
        placements["solver"] = {
            "center": [target_x, target_y],
            "yaw_degrees": yaw,
        }

    timer_objects = solve_station_family_objects(asset_collection, ("Timer",))
    if timer_objects:
        base = {obj.name: obj.matrix_world.copy() for obj in timer_objects}
        target_x = random.uniform(-0.28, 0.28)
        target_y = random.uniform(2.55, 2.85)
        yaw = random.uniform(-2.8, 2.8)
        bounds = set_object_group_local_pose(
            timer_objects,
            base,
            target_x,
            target_y,
            yaw,
        )
        occupied.append(("timer", bounds))
        placements["timer"] = {
            "center": [target_x, target_y],
            "yaw_degrees": yaw,
            "bounds": list(bounds),
        }

    laptop_side = random.choice((-1.0, 1.0))
    laptop_objects = solve_station_family_objects(asset_collection, ("Laptop",))
    if laptop_objects:
        base = {obj.name: obj.matrix_world.copy() for obj in laptop_objects}
        accepted = False
        for _attempt in range(30):
            target_x = laptop_side * random.uniform(4.25, 4.75)
            target_y = random.uniform(-3.45, -3.15)
            yaw = random.uniform(-4.5, 4.5)
            bounds = set_object_group_local_pose(
                laptop_objects,
                base,
                target_x,
                target_y,
                yaw,
            )
            if not any(
                xy_bounds_overlap(bounds, other, SOLVE_STATION_CLUTTER_PADDING)
                for _name, other in occupied
            ):
                accepted = True
                break
        if not accepted:
            target_x = laptop_side * 4.65
            target_y = -3.5
            yaw = 0.0
            bounds = set_object_group_local_pose(
                laptop_objects,
                base,
                target_x,
                target_y,
                yaw,
            )
        occupied.append(("laptop", bounds))
        placements["laptop"] = {
            "center": [target_x, target_y],
            "yaw_degrees": yaw,
            "bounds": list(bounds),
            "opposite_solver_side": True,
        }

    phone_objects = solve_station_family_objects(asset_collection, ("Phone",))
    if phone_objects:
        base = {obj.name: obj.matrix_world.copy() for obj in phone_objects}
        accepted = False
        preferred_side = -laptop_side if laptop_objects else random.choice((-1.0, 1.0))
        for attempt in range(60):
            side = preferred_side if attempt < 42 else -preferred_side
            target_x = side * random.uniform(4.55, 5.35)
            target_y = random.uniform(-0.25, 0.75)
            yaw = random.uniform(-28.0, 28.0)
            bounds = set_object_group_local_pose(
                phone_objects,
                base,
                target_x,
                target_y,
                yaw,
            )
            if not any(
                xy_bounds_overlap(bounds, other, SOLVE_STATION_CLUTTER_PADDING)
                for _name, other in occupied
            ):
                accepted = True
                break
        if not accepted:
            for obj in phone_objects:
                bpy.data.objects.remove(obj, do_unlink=True)
            placements["phone"] = {"visible": False, "reason": "no_clear_placement"}
        else:
            occupied.append(("phone", bounds))
            placements["phone"] = {
                "visible": True,
                "center": [target_x, target_y],
                "yaw_degrees": yaw,
                "bounds": list(bounds),
            }

    placements["occupied_bounds"] = {
        name: list(bounds) for name, bounds in occupied
    }
    return placements


def set_solve_station_family_visible(
    asset_collection: bpy.types.Collection,
    family: str,
    visible: bool,
) -> None:
    marker = f"SolveStation.{family}."
    for obj in list(asset_collection.all_objects):
        if marker not in obj.name:
            continue
        if not visible:
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        obj.hide_render = False
        obj.hide_viewport = False
        obj.hide_set(False)


def add_solve_station_background() -> dict[str, object]:
    path = solve_station_asset_blend_path()
    instance_name = f"{BACKGROUND_ASSET_COLLECTION_PREFIX}.SolveStation"
    asset_collection = append_background_asset_collection(
        path,
        SOLVE_STATION_COLLECTION_NAME,
        instance_name,
    )
    if asset_collection is None:
        raise RuntimeError(
            f"Could not append {SOLVE_STATION_COLLECTION_NAME!r} from {path}"
        )

    parent_collection = background_scene_collection()
    link_collection_to_collection(asset_collection, parent_collection)
    root = collection_asset_root(asset_collection)
    if root is None:
        raise RuntimeError(f"The solve-station collection in {path} has no root object")

    yaw_jitter = math.radians(random.uniform(*SOLVE_STATION_YAW_JITTER_DEGREES))
    rotation = Quaternion(Vector((0.0, 0.0, 1.0)), yaw_jitter)
    station_scale = SOLVE_STATION_SCALE
    table_anchor_offset = (
        BACKGROUND_TABLE_Z - station_scale * SOLVE_STATION_DESK_TOP_LOCAL_Z
    )
    root.matrix_world = (
        Matrix.Translation(Vector((0.0, 0.0, table_anchor_offset)))
        @ rotation.to_matrix().to_4x4()
        @ Matrix.Diagonal((station_scale, station_scale, station_scale, 1.0))
    )
    bpy.context.view_layer.update()

    laptop_visible = random.random() < SOLVE_STATION_LAPTOP_PROBABILITY
    phone_visible = random.random() < SOLVE_STATION_PHONE_PROBABILITY
    other_cube_visible = random.random() < SOLVE_STATION_OTHER_CUBE_PROBABILITY
    set_solve_station_family_visible(asset_collection, "Laptop", laptop_visible)
    set_solve_station_family_visible(asset_collection, "Phone", phone_visible)
    layout_metadata = randomize_solve_station_layout(asset_collection)
    phone_layout = layout_metadata.get("phone")
    if isinstance(phone_layout, dict) and phone_layout.get("visible") is False:
        phone_visible = False
    material_metadata = randomize_solve_station_materials(asset_collection)

    objects = list(asset_collection.all_objects)
    for obj in objects:
        if obj == root:
            continue
        obj.name = f"{BACKGROUND_SCENE_PREFIX}.solve_station.{obj.name}"
    root.name = f"{BACKGROUND_SCENE_PREFIX}.solve_station.ROOT"
    bpy.context.view_layer.update()

    return {
        "family": "solve_station",
        "source_file": str(path),
        "source_collection": SOLVE_STATION_COLLECTION_NAME,
        "collection": asset_collection.name,
        "root": root.name,
        "object_count": len([obj for obj in objects if obj.type == "MESH"]),
        "laptop_visible": laptop_visible,
        "phone_visible": phone_visible,
        "other_cube_visible": other_cube_visible,
        "other_cube": None,
        "scale": station_scale,
        "yaw_jitter_degrees": math.degrees(yaw_jitter),
        "layout": layout_metadata,
        "materials": material_metadata,
    }


def object_tree(source: bpy.types.Object) -> list[bpy.types.Object]:
    result = [source]
    for child in source.children:
        result.extend(object_tree(child))
    return result


def duplicate_object_hierarchies(
    roots: list[bpy.types.Object],
    collection: bpy.types.Collection,
    name_family: str,
) -> tuple[list[bpy.types.Object], list[bpy.types.Object]]:
    sources = []
    seen: set[int] = set()
    for root in roots:
        for source in object_tree(root):
            pointer = source.as_pointer()
            if pointer in seen:
                continue
            seen.add(pointer)
            sources.append(source)

    duplicates: dict[bpy.types.Object, bpy.types.Object] = {}
    for index, source in enumerate(sources):
        duplicate = source.copy()
        if source.data is not None:
            duplicate.data = source.data
        duplicate.name = (
            f"{BACKGROUND_SCENE_PREFIX}.secondary_cube.{name_family}.{index:03d}"
        )
        duplicate.pass_index = 0
        collection.objects.link(duplicate)
        duplicates[source] = duplicate

    for source, duplicate in duplicates.items():
        if source.parent in duplicates:
            duplicate.parent = duplicates[source.parent]
            duplicate.matrix_parent_inverse = source.matrix_parent_inverse.copy()
            duplicate.matrix_basis = source.matrix_basis.copy()
        else:
            duplicate.parent = None
            duplicate.matrix_world = source.matrix_world.copy()

    duplicate_roots = [duplicates[root] for root in roots]
    return duplicate_roots, list(duplicates.values())


def transformed_object_group_bounds(
    objects: list[bpy.types.Object],
    transform: Matrix,
) -> tuple[float, float, float, float, float, float]:
    points = [
        transform @ obj.matrix_world @ Vector(corner)
        for obj in objects
        if obj.type == "MESH"
        for corner in obj.bound_box
    ]
    if not points:
        raise ValueError("Cannot calculate bounds for an empty object group.")
    return (
        min(point.x for point in points),
        max(point.x for point in points),
        min(point.y for point in points),
        max(point.y for point in points),
        min(point.z for point in points),
        max(point.z for point in points),
    )


def projected_object_group_bounds(
    objects: list[bpy.types.Object],
    transform: Matrix,
) -> tuple[float, float, float, float] | None:
    scene = bpy.context.scene
    camera = get_camera()
    projected = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            point = transform @ obj.matrix_world @ Vector(corner)
            coordinate = world_to_camera_view(scene, camera, point)
            if coordinate.z <= 0.0:
                return None
            projected.append((coordinate.x, coordinate.y))
    if not projected:
        return None
    return (
        min(point[0] for point in projected),
        max(point[0] for point in projected),
        min(point[1] for point in projected),
        max(point[1] for point in projected),
    )


def duplicate_secondary_cube(
    objects: list[bpy.types.Object],
    core_objects: list[bpy.types.Object],
    background_randomization: dict[str, object],
) -> dict[str, object] | None:
    station_objects = background_randomization.get("objects", [])
    if not station_objects or not isinstance(station_objects[0], dict):
        return None
    station_metadata = station_objects[0]
    if not station_metadata.get("other_cube_visible"):
        return None

    collection = background_scene_collection()
    duplicate_objects, duplicate_cube_parts = duplicate_object_hierarchies(
        objects,
        collection,
        "cubie_part",
    )
    duplicate_core_objects, duplicate_core_parts = duplicate_object_hierarchies(
        core_objects,
        collection,
        "core_part",
    )

    reset_cube(duplicate_objects)
    reset_core_fill_objects(duplicate_core_objects)
    secondary_solved = (
        random.random() < SOLVE_STATION_OTHER_CUBE_SOLVED_PROBABILITY
    )
    faces = list("RLUDBF")
    secondary_moves = []
    secondary_move_count = 0 if secondary_solved else random.randint(8, 22)
    for _ in range(secondary_move_count):
        choices = [face for face in faces if not secondary_moves or face != secondary_moves[-1]]
        move = random.choice(choices)
        secondary_moves.append(move)
        rotate_face(duplicate_objects, move)

    table_forward, table_right = camera_table_basis()
    rotation = (
        Quaternion(Vector((0.0, 0.0, 1.0)), random.uniform(0.0, math.tau))
        @ Quaternion(Vector((1.0, 0.0, 0.0)), math.radians(random.uniform(-9.0, 9.0)))
        @ Quaternion(Vector((0.0, 1.0, 0.0)), math.radians(random.uniform(-9.0, 9.0)))
    )
    pose_roots = duplicate_objects + duplicate_core_objects
    render_parts = duplicate_cube_parts + duplicate_core_parts
    min_z = transformed_min_z(render_parts, rotation)
    preferred_side = random.choice((-1.0, 1.0))
    layout = station_metadata.get("layout")
    if isinstance(layout, dict):
        phone_layout = layout.get("phone")
        if isinstance(phone_layout, dict):
            phone_center = phone_layout.get("center")
            if isinstance(phone_center, list) and phone_center:
                preferred_side = -1.0 if phone_center[0] > 0.0 else 1.0
    side = preferred_side
    initial_right_distance = random.uniform(5.6, 6.4)
    depth_offset = random.uniform(0.0, 0.65)
    placement_zone = "mat_left" if side < 0.0 else "mat_right"
    clearance = random.uniform(0.02, 0.08)
    cropped_candidate = None
    for crop_attempt in range(48):
        right_distance = initial_right_distance + crop_attempt * 0.08
        right_offset = side * right_distance
        horizontal_location = (
            table_right * right_offset + table_forward * depth_offset
        )
        translation = Vector((
            horizontal_location.x,
            horizontal_location.y,
            BACKGROUND_TABLE_Z + clearance - min_z,
        ))
        candidate_transform = (
            Matrix.Translation(translation) @ rotation.to_matrix().to_4x4()
        )
        projected_bounds = projected_object_group_bounds(
            render_parts,
            candidate_transform,
        )
        if projected_bounds is None:
            continue
        fully_in_frame = (
            projected_bounds[0] >= 0.0
            and projected_bounds[1] <= 1.0
            and projected_bounds[2] >= 0.0
            and projected_bounds[3] <= 1.0
        )
        intersects_frame = (
            projected_bounds[1] > 0.0
            and projected_bounds[0] < 1.0
            and projected_bounds[3] > 0.0
            and projected_bounds[2] < 1.0
        )
        if intersects_frame and not fully_in_frame:
            cropped_candidate = (
                crop_attempt + 1,
                right_offset,
                translation,
                candidate_transform,
                projected_bounds,
            )
            break
    if cropped_candidate is None:
        for duplicate in render_parts:
            bpy.data.objects.remove(duplicate, do_unlink=True)
        station_metadata["other_cube_visible"] = False
        station_metadata["other_cube"] = {
            "model": "duplicate_of_main_cube",
            "placement": "rejected",
            "reason": "no_partial_frame_placement",
        }
        return None
    (
        crop_attempts,
        right_offset,
        translation,
        candidate_transform,
        projected_bounds,
    ) = cropped_candidate
    world_bounds = transformed_object_group_bounds(render_parts, candidate_transform)

    transform = apply_global_pose(
        pose_roots,
        rotation,
        translation,
    )
    actual_projected_bounds = projected_object_group_bounds(
        render_parts,
        Matrix.Identity(4),
    )
    metadata = {
        "model": "duplicate_of_main_cube",
        "object_count": len(duplicate_objects),
        "core_object_count": len(duplicate_core_objects),
        "mesh_part_count": len(render_parts),
        "child_mesh_part_count": len(render_parts) - len(pose_roots),
        "all_parts_in_frame": (
            actual_projected_bounds is not None
            and actual_projected_bounds[0] >= 0.0
            and actual_projected_bounds[1] <= 1.0
            and actual_projected_bounds[2] >= 0.0
            and actual_projected_bounds[3] <= 1.0
        ),
        "partially_in_frame": (
            actual_projected_bounds is not None
            and actual_projected_bounds[1] > 0.0
            and actual_projected_bounds[0] < 1.0
            and actual_projected_bounds[3] > 0.0
            and actual_projected_bounds[2] < 1.0
        ),
        "solved": secondary_solved,
        "moves": secondary_moves,
        "placement_zone": placement_zone,
        "crop_attempts": crop_attempts,
        "right_offset": right_offset,
        "depth_offset": depth_offset,
        "projected_bounds": (
            list(projected_bounds)
            if projected_bounds is not None
            else None
        ),
        "actual_projected_bounds": (
            list(actual_projected_bounds) if actual_projected_bounds is not None else None
        ),
        "world_bounds": list(world_bounds),
        "transform": matrix_rows(transform),
    }
    station_metadata["other_cube"] = metadata
    return metadata


def randomize_3d_background(object_count: int) -> dict[str, object]:
    clear_background_scene()
    hide_flat_background()
    world_color = configure_world_background()

    station = add_solve_station_background()

    bpy.context.view_layer.update()
    return {
        "mode": "solve_station",
        "world_color": list(world_color),
        "table": {"z": BACKGROUND_TABLE_Z, "source": "SolveStation.Desk.Top"},
        "requested_object_count": object_count,
        "object_count": station["object_count"],
        "blend_asset_count": 1,
        "blend_asset_sources": [station["source_collection"]],
        "families": [
            "room",
            "desk",
            "mat",
            "stackmat_timer",
            "shirt",
            "arms",
            "phone",
            "laptop",
            "other_cube",
        ],
        "objects": [station],
    }


def randomize_background(object_count: int) -> dict[str, object]:
    clear_background_scene()
    configure_world_background()
    return randomize_3d_background(object_count)


def cubie_world_center(objects: list[bpy.types.Object]) -> Vector:
    if not objects:
        return Vector((0.0, 0.0, 0.0))
    center = Vector((0.0, 0.0, 0.0))
    for obj in objects:
        center += obj.location
    return center / len(objects)


def hand_objects() -> dict[str, bpy.types.Object]:
    objects = {
        name: obj
        for name in HAND_NAMES
        if (obj := bpy.data.objects.get(name)) is not None
    }
    if objects and len(objects) != len(HAND_NAMES):
        missing = sorted(set(HAND_NAMES) - set(objects))
        raise ValueError(f"Expected both hand objects; missing: {', '.join(missing)}")
    return objects


def hand_object_states(
    objects: dict[str, bpy.types.Object],
) -> dict[str, dict[str, object]]:
    return {
        name: {
            "matrix_world": obj.matrix_world.copy(),
            "rotation_mode": obj.rotation_mode,
            "hide_render": obj.hide_render,
            "hide_viewport": obj.hide_viewport,
            "hide": obj.hide_get(),
            "materials": [material for material in obj.data.materials],
        }
        for name, obj in objects.items()
    }


def restore_hand_objects(
    objects: dict[str, bpy.types.Object],
    states: dict[str, dict[str, object]],
) -> None:
    for name, obj in objects.items():
        state = states[name]
        obj.matrix_world = state["matrix_world"]
        obj.rotation_mode = state["rotation_mode"]
        obj.hide_render = state["hide_render"]
        obj.hide_viewport = state["hide_viewport"]
        obj.hide_set(state["hide"])
        obj.data.materials.clear()
        for material in state["materials"]:
            obj.data.materials.append(material)
    bpy.context.view_layer.update()


def hand_skin_material(color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.get(HAND_SKIN_MATERIAL_NAME)
    if material is None:
        material = bpy.data.materials.new(HAND_SKIN_MATERIAL_NAME)
    material.use_nodes = True
    material.diffuse_color = color
    set_principled_input(material, "Base Color", color)
    set_principled_input(material, "Roughness", random.uniform(0.58, 0.88))
    set_principled_input(material, "Metallic", 0.0)
    set_principled_input(material, "Specular IOR Level", random.uniform(0.18, 0.36))
    set_principled_input(material, "Specular", random.uniform(0.18, 0.36))
    return material


def apply_hand_material(objects: dict[str, bpy.types.Object], material: bpy.types.Material) -> None:
    for obj in objects.values():
        slot_count = max(1, len(obj.data.materials))
        obj.data.materials.clear()
        for _index in range(slot_count):
            obj.data.materials.append(material)


def randomize_hands(
    objects: dict[str, bpy.types.Object],
    states: dict[str, dict[str, object]],
    cube_pose: dict[str, object] | None = None,
    forced_visibility: str | None = None,
) -> dict[str, object] | None:
    if not objects:
        return None

    visibility = forced_visibility or random.choices(
        [mode for mode, _weight in HAND_VISIBILITY_WEIGHTS],
        weights=[weight for _mode, weight in HAND_VISIBILITY_WEIGHTS],
        k=1,
    )[0]
    visible_hands = set()
    if visibility in ("left", "both"):
        visible_hands.add("left-hand")
    if visibility in ("right", "both"):
        visible_hands.add("right-hand")
    randomization: dict[str, object] = {
        "visibility": visibility,
        "objects": {},
    }
    skin_color = random_hand_skin_color()
    skin_material = hand_skin_material(skin_color)
    apply_hand_material(objects, skin_material)
    randomization["skin_color"] = list(skin_color)
    randomization["skin_material"] = skin_material.name
    cube_transform = None
    if cube_pose is not None and isinstance(cube_pose.get("transform"), list):
        cube_transform = matrix_from_rows(cube_pose["transform"])
        randomization["follows_cube_pose"] = True
    else:
        randomization["follows_cube_pose"] = False

    for name, obj in objects.items():
        base_matrix = states[name]["matrix_world"]
        x_offset = random.uniform(*HAND_X_OFFSET_RANGE)
        z_offset = random.uniform(*HAND_Z_OFFSET_RANGE)
        x_rotation_degrees = random.choice(HAND_X_ROTATION_DEGREES)
        hidden = name not in visible_hands
        location_offset = Vector((x_offset, 0.0, z_offset))
        global_rotation = Quaternion(
            Vector((1.0, 0.0, 0.0)),
            math.radians(x_rotation_degrees),
        ).to_matrix().to_4x4()

        local_matrix = global_rotation @ Matrix.Translation(location_offset) @ base_matrix
        obj.matrix_world = cube_transform @ local_matrix if cube_transform is not None else local_matrix
        obj.rotation_mode = "QUATERNION"
        obj.hide_render = hidden
        obj.hide_viewport = hidden
        obj.hide_set(hidden)

        randomization["objects"][name] = {
            "visible": not hidden,
            "x_offset": x_offset,
            "z_offset": z_offset,
            "x_rotation_degrees": x_rotation_degrees,
            "location": [obj.location.x, obj.location.y, obj.location.z],
        }

    bpy.context.view_layer.update()
    return randomization


def randomize_scene(
    scramble_length: int = DEFAULT_SCRAMBLE_LENGTH,
    background_object_count: int = DEFAULT_BACKGROUND_OBJECT_COUNT,
) -> tuple[
    list[str],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, dict[str, float]],
]:
    sticker_materials = randomize_sticker_material_gloss()
    objects = cube_objects()
    core_objects = core_fill_objects()
    reset_cube(objects)
    reset_core_fill_objects(core_objects)

    camera_randomization = randomize_camera()
    background_randomization = randomize_background(background_object_count)
    # Build the rig after appending the background so lights bundled in an asset
    # cannot accidentally leak into, or conflict with, the selected setup.
    light_randomization = randomize_light()
    background_randomization["world_color"] = light_randomization["world"]["display_color"]
    background_randomization["world_lighting"] = light_randomization["world"]

    faces = list("RLUDBF")
    moves = []
    for _ in range(scramble_length):
        choices = [face for face in faces if not moves or face != moves[-1]]
        moves.append(random.choice(choices))

    for move in moves:
        rotate_face(objects, move)

    duplicate_secondary_cube(objects, core_objects, background_randomization)
    layer_state = randomize_layer_render_state(objects)

    pose_objects = list(objects) + core_objects
    pose_bounds = []
    pose_bound_pointers: set[int] = set()
    for root in pose_objects:
        for part in object_tree(root):
            pointer = part.as_pointer()
            if pointer in pose_bound_pointers:
                continue
            pose_bound_pointers.add(pointer)
            pose_bounds.append(part)
    cube_pose = randomize_cube_global_pose(pose_objects, bounds_objects=pose_bounds)
    return (
        moves,
        layer_state,
        background_randomization,
        camera_randomization,
        light_randomization,
        cube_pose,
        sticker_materials,
    )


def configure_render(width: int, height: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False


def render_png(path: Path) -> None:
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def object_transforms(objects: list[bpy.types.Object]) -> dict[str, tuple[Vector, Quaternion]]:
    return {
        obj.name: (obj.location.copy(), obj.rotation_quaternion.copy())
        for obj in objects
    }


def restore_object_transforms(
    objects: list[bpy.types.Object],
    transforms: dict[str, tuple[Vector, Quaternion]],
) -> None:
    for obj in objects:
        location, rotation = transforms[obj.name]
        obj.location = location
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = rotation
    bpy.context.view_layer.update()


def rotate_objects_around_origin(
    objects: list[bpy.types.Object],
    base_transforms: dict[str, tuple[Vector, Quaternion]],
    axis: Vector,
    angle_radians: float,
) -> None:
    rotation = Quaternion(axis, angle_radians)
    for obj in objects:
        base_location, base_rotation = base_transforms[obj.name]
        obj.location = rotation @ base_location
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = rotation @ base_rotation
    bpy.context.view_layer.update()


def rotational_blur_sample_angles(magnitude_degrees: float, samples: int) -> list[float]:
    magnitude_radians = math.radians(magnitude_degrees)
    return [
        ((sample_index / (samples - 1)) - 0.5) * magnitude_radians
        for sample_index in range(samples)
    ]


def average_rendered_pngs(sample_paths: list[Path], output_path: Path) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "OpenCV and NumPy are required to write rotational blur renders. "
            "Install cv2 and numpy in Blender's Python environment."
        ) from None

    accumulator = None
    for sample_path in sample_paths:
        image = cv2.imread(str(sample_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not read rendered blur sample: {sample_path}")
        image_float = image.astype(np.float32)
        accumulator = image_float if accumulator is None else accumulator + image_float

    averaged = np.clip(accumulator / len(sample_paths), 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), averaged)


def render_cube_rotational_blur_png(
    path: Path,
    objects: list[bpy.types.Object],
    probability: float,
    max_degrees: float,
    samples: int,
) -> dict[str, object] | None:
    if probability <= 0.0 or max_degrees <= 0.0 or samples < 2 or random.random() >= probability:
        render_png(path)
        return None

    axis = random_point_on_unit_sphere()
    magnitude_degrees = random.uniform(0.0, max_degrees)
    base_transforms = object_transforms(objects)
    sample_paths = []

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for sample_index, angle in enumerate(rotational_blur_sample_angles(magnitude_degrees, samples)):
                rotate_objects_around_origin(objects, base_transforms, axis, angle)
                sample_path = temp_path / f"rotational_blur_{sample_index:03d}.png"
                render_png(sample_path)
                sample_paths.append(sample_path)

            average_rendered_pngs(sample_paths, path)
    finally:
        restore_object_transforms(objects, base_transforms)

    return {
        "axis": [axis.x, axis.y, axis.z],
        "magnitude_degrees": magnitude_degrees,
        "samples": samples,
    }


def odd_kernel(size: int) -> int:
    return max(1, int(size) | 1)


def motion_blur_kernel(size: int, angle_degrees: float):
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "OpenCV and NumPy are required to apply camera degradation. "
            "Install cv2 and numpy in Blender's Python environment."
        ) from None

    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((size / 2.0 - 0.5, size / 2.0 - 0.5), angle_degrees, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (size, size))
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0.0:
        kernel[size // 2, :] = 1.0
        kernel_sum = float(kernel.sum())
    return kernel / kernel_sum


def apply_camera_degradation(path: Path, probability: float) -> dict[str, object] | None:
    if probability <= 0.0 or random.random() >= probability:
        return None

    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "OpenCV and NumPy are required to apply camera degradation. "
            "Install cv2 and numpy in Blender's Python environment."
        ) from None

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read rendered image for degradation: {path}")

    metadata: dict[str, object] = {}
    alpha = image[:, :, 3] if image.ndim == 3 and image.shape[2] == 4 else None
    frame = image[:, :, :3] if alpha is not None else image

    if random.random() < DEGRADE_RESCALE_PROBABILITY:
        scale = random.uniform(0.55, 0.9)
        height, width = frame.shape[:2]
        small = cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        frame = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
        metadata["rescale"] = {"scale": scale}

    if random.random() < DEGRADE_GAUSSIAN_BLUR_PROBABILITY:
        kernel = odd_kernel(random.choice((3, 3, 5, 7)))
        sigma = random.uniform(0.35, 1.6)
        frame = cv2.GaussianBlur(frame, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
        metadata["gaussian_blur"] = {"kernel": kernel, "sigma": sigma}

    if random.random() < DEGRADE_MOTION_BLUR_PROBABILITY:
        kernel_size = odd_kernel(random.choice((3, 5, 7, 9, 11)))
        angle = random.uniform(0.0, 180.0)
        frame = cv2.filter2D(frame, -1, motion_blur_kernel(kernel_size, angle))
        metadata["motion_blur"] = {"kernel": kernel_size, "angle_degrees": angle}

    if random.random() < DEGRADE_NOISE_PROBABILITY:
        sigma = random.uniform(2.0, 13.0)
        noise = np.random.normal(0.0, sigma, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
        metadata["sensor_noise"] = {"sigma": sigma}

    if random.random() < DEGRADE_COMPRESSION_PROBABILITY:
        quality = random.randint(35, 92)
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise RuntimeError(f"Could not encode degraded JPEG buffer for: {path}")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError(f"Could not decode degraded JPEG buffer for: {path}")
        frame = decoded
        metadata["jpeg_compression"] = {"quality": quality}

    if alpha is not None:
        frame = np.dstack([frame, alpha])

    cv2.imwrite(str(path), frame)
    return metadata or None


def normalize_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return (
        max(0.0, min(x / width, 1.0)),
        max(0.0, min(y / height, 1.0)),
    )


def project_world_point(point: Vector, camera: bpy.types.Object, scene: bpy.types.Scene) -> tuple[float, float] | None:
    co = world_to_camera_view(scene, camera, point)
    if co.z < 0.0:
        return None
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    x = co.x * width
    y = (1.0 - co.y) * height
    if x < 0.0 or x >= width or y < 0.0 or y >= height:
        return None
    return x, y


def cube_keypoint_coordinate(letter: str) -> tuple[int, float, Vector | None]:
    mapping = {
        "R": (0, 1.5, Vector((1.0, 0.0, 0.0))),
        "r": (0, 0.5, None),
        "L": (0, -1.5, Vector((-1.0, 0.0, 0.0))),
        "l": (0, -0.5, None),
        "B": (1, 1.5, Vector((0.0, 1.0, 0.0))),
        "b": (1, 0.5, None),
        "F": (1, -1.5, Vector((0.0, -1.0, 0.0))),
        "f": (1, -0.5, None),
        "U": (2, 1.5, Vector((0.0, 0.0, 1.0))),
        "u": (2, 0.5, None),
        "D": (2, -1.5, Vector((0.0, 0.0, -1.0))),
        "d": (2, -0.5, None),
    }
    if letter not in mapping:
        raise ValueError(f"Unsupported cube keypoint letter: {letter!r}")
    return mapping[letter]


def cube_keypoint_spec(name: str) -> dict[str, object]:
    coords: list[float | None] = [None, None, None]
    normals = []
    for letter in name:
        axis_index, value, normal = cube_keypoint_coordinate(letter)
        if coords[axis_index] is not None:
            raise ValueError(f"Duplicate axis in cube keypoint name: {name}")
        coords[axis_index] = value
        if normal is not None:
            normals.append(normal)
    if any(value is None for value in coords):
        raise ValueError(f"Cube keypoint name does not cover all axes: {name}")
    return {
        "name": name,
        "point": Vector((float(coords[0]), float(coords[1]), float(coords[2]))),
        "normals": normals,
    }


def cube_keypoint_coord_key(name: str) -> tuple[float, float, float]:
    coords: list[float | None] = [None, None, None]
    for letter in name:
        axis_index, value, _normal = cube_keypoint_coordinate(letter)
        if coords[axis_index] is not None:
            raise ValueError(f"Duplicate axis in cube keypoint name: {name}")
        coords[axis_index] = value
    if any(value is None for value in coords):
        raise ValueError(f"Cube keypoint name does not cover all axes: {name}")
    return float(coords[0]), float(coords[1]), float(coords[2])


def canonical_cube_keypoint_name(name: str) -> str:
    names_by_coord = {
        cube_keypoint_coord_key(keypoint_name): keypoint_name
        for keypoint_name in CUBE_KEYPOINT_NAMES
    }
    return names_by_coord[cube_keypoint_coord_key(name)]


def cube_pose_keypoint_specs() -> list[dict[str, object]]:
    return [cube_keypoint_spec(name) for name in CUBE_KEYPOINT_NAMES]


def corner_cubie_objects() -> list[bpy.types.Object]:
    corners = []
    for obj in cube_objects():
        solved_location = Vector(obj.get(SOLVED_LOCATION_PROP, obj.location))
        if all(abs(round(solved_location[axis_index])) == 1 for axis_index in range(3)):
            corners.append(obj)
    return sorted(corners, key=lambda obj: obj.name)


def center_cubie_objects() -> list[bpy.types.Object]:
    centers = []
    for obj in cube_objects():
        solved_location = Vector(obj.get(SOLVED_LOCATION_PROP, obj.location))
        outer_axis_count = sum(
            1
            for axis_index in range(3)
            if abs(round(solved_location[axis_index])) == 1
        )
        if outer_axis_count == 1:
            centers.append(obj)
    return sorted(centers, key=lambda obj: obj.name)


def corner_point_class_id(relative_one_count: int) -> int:
    if relative_one_count == 3:
        return CORNER_CLASS_ID
    if relative_one_count == 2:
        return CORNER_EDGE_CLASS_ID
    if relative_one_count == 1:
        return CORNER_EDGE_CENTER_CLASS_ID
    raise ValueError(f"Unsupported keypoint relative-one count: {relative_one_count}")


def corner_relative_keypoint_specs() -> list[dict[str, object]]:
    specs = []
    corners = corner_cubie_objects()
    cube_center = cubie_world_center(cube_objects())
    for obj in corners:
        inner_offset = min(
            (
                Vector((x, y, z))
                for x in (-0.5, 0.5)
                for y in (-0.5, 0.5)
                for z in (-0.5, 0.5)
            ),
            key=lambda offset: (obj.matrix_world @ offset - cube_center).length,
        )
        for relative_one_axes in (
            (0, 1, 2),
            (0, 1),
            (0, 2),
            (1, 2),
            (0,),
            (1,),
            (2,),
        ):
            relative_one_axis_set = set(relative_one_axes)
            local_offset = Vector(
                tuple(
                    -inner_offset[axis_index]
                    if axis_index in relative_one_axis_set
                    else inner_offset[axis_index]
                    for axis_index in range(3)
                )
            )
            point = obj.matrix_world @ local_offset
            normals = [
                (obj.matrix_world.to_3x3() @ axis_vector(axis_index, -inner_offset[axis_index] * 2.0)).normalized()
                for axis_index in relative_one_axes
            ]
            class_id = corner_point_class_id(len(relative_one_axes))
            specs.append({
                "name": f"{obj.name}:{CLASS_NAMES[class_id]}:{''.join(str(axis) for axis in relative_one_axes)}",
                "class_id": class_id,
                "class": CLASS_NAMES[class_id],
                "source_object": obj.name,
                "source_piece_type": "corner",
                "relative_point": [
                    1 if axis_index in relative_one_axis_set else 0
                    for axis_index in range(3)
                ],
                "point": point,
                "normals": normals,
            })
    return specs


def center_keypoint_specs() -> list[dict[str, object]]:
    specs = []
    cube_center = cubie_world_center(cube_objects())
    for obj in center_cubie_objects():
        best_axis_index = 0
        best_sign = 1.0
        best_distance = -1.0
        for axis_index in range(3):
            for sign in (-1.0, 1.0):
                local_offset = axis_vector(axis_index, sign * 0.5)
                distance = (obj.matrix_world @ local_offset - cube_center).length
                if distance > best_distance:
                    best_axis_index = axis_index
                    best_sign = sign
                    best_distance = distance
        local_offset = axis_vector(best_axis_index, best_sign * 0.5)
        point = obj.matrix_world @ local_offset
        normal = (obj.matrix_world.to_3x3() @ axis_vector(best_axis_index, best_sign)).normalized()
        specs.append({
            "name": f"{obj.name}:{CLASS_NAMES[CENTER_CLASS_ID]}",
            "class_id": CENTER_CLASS_ID,
            "class": CLASS_NAMES[CENTER_CLASS_ID],
            "source_object": obj.name,
            "source_piece_type": "center",
            "relative_point": "center",
            "point": point,
            "normals": [normal],
        })
    return specs


def cube_keypoint_specs_from_corners_and_centers() -> list[dict[str, object]]:
    return [*corner_relative_keypoint_specs(), *center_keypoint_specs()]


def lowercase_letter(letter: str) -> str:
    return letter.lower()


def corner_piece_keypoint_names(corner_name: str) -> tuple[str, list[str], list[str]]:
    if len(corner_name) != 3 or not corner_name.isupper():
        raise ValueError(f"Corner keypoint name must contain three outer faces: {corner_name}")
    edge_facing = [
        canonical_cube_keypoint_name(
            corner_name[:index] + lowercase_letter(corner_name[index]) + corner_name[index + 1 :]
        )
        for index in range(3)
    ]
    center_facing = []
    for first_index, second_index in ((0, 1), (1, 2), (0, 2)):
        letters = list(corner_name)
        letters[first_index] = lowercase_letter(letters[first_index])
        letters[second_index] = lowercase_letter(letters[second_index])
        center_facing.append(canonical_cube_keypoint_name("".join(letters)))
    return canonical_cube_keypoint_name(corner_name), edge_facing, center_facing


def corner_keypoint_name_for_faces(corner_name: str, uppercase_faces: set[str]) -> str:
    letters = [
        face if face in uppercase_faces else lowercase_letter(face)
        for face in corner_name
    ]
    return canonical_cube_keypoint_name("".join(letters))


def edge_piece_keypoint_names(outer_faces: tuple[str, str], free_faces: tuple[str, str]) -> tuple[list[str], list[list[str]]]:
    first_outer, second_outer = outer_faces
    edge_line = [
        canonical_cube_keypoint_name(
            "".join(sorted((first_outer, second_outer, lowercase_letter(free_face)), key=axis_sort_key))
        )
        for free_face in free_faces
    ]
    center_sides = []
    for lowered_index in range(2):
        center_side = []
        for free_face in free_faces:
            letters = [first_outer, second_outer, lowercase_letter(free_face)]
            letters[lowered_index] = lowercase_letter(letters[lowered_index])
            center_side.append(canonical_cube_keypoint_name("".join(sorted(letters, key=axis_sort_key))))
        center_sides.append(center_side)
    return edge_line, center_sides


def axis_sort_key(letter: str) -> int:
    return {"R": 0, "r": 0, "L": 0, "l": 0, "B": 1, "b": 1, "F": 1, "f": 1, "U": 2, "u": 2, "D": 2, "d": 2}[letter]


def edge_piece_name_specs() -> list[dict[str, object]]:
    return [
        {
            "name": "".join(outer_faces),
            "edge_line": edge_line,
            "center_sides": center_sides,
        }
        for outer_faces, free_faces in (
            (("U", "B"), ("L", "R")),
            (("U", "R"), ("B", "F")),
            (("U", "F"), ("L", "R")),
            (("U", "L"), ("B", "F")),
            (("D", "B"), ("L", "R")),
            (("D", "R"), ("B", "F")),
            (("D", "F"), ("L", "R")),
            (("D", "L"), ("B", "F")),
            (("B", "L"), ("U", "D")),
            (("B", "R"), ("U", "D")),
            (("F", "R"), ("U", "D")),
            (("F", "L"), ("U", "D")),
        )
        for edge_line, center_sides in [edge_piece_keypoint_names(outer_faces, free_faces)]
    ]


def point_faces_camera(point: Vector, normals: list[Vector], camera_location: Vector) -> bool:
    toward_camera = camera_location - point
    if toward_camera.length <= 1e-6:
        return False
    min_dot = math.sin(math.radians(FACE_VISIBILITY_MIN_PLANE_ANGLE_DEGREES))
    view_direction = toward_camera.normalized()
    return any(normal.normalized().dot(view_direction) >= min_dot for normal in normals)


def ray_to_point(camera: bpy.types.Object, point: Vector) -> tuple[Vector, Vector, float]:
    origin = camera.matrix_world.translation
    offset = point - origin
    distance = offset.length
    return origin, offset.normalized(), distance


def object_occluding_point(
    point: Vector,
    depsgraph: bpy.types.Depsgraph,
    ignored_object_names: set[str] | None = None,
    distance_tolerance: float = 0.02,
) -> str | None:
    ignored_object_names = ignored_object_names or set()
    scene = bpy.context.scene
    camera = get_camera()
    origin, direction, point_distance = ray_to_point(camera, point)
    traveled = 0.0
    epsilon = 0.001
    while traveled < point_distance - distance_tolerance:
        hit, location, _normal, _polygon_index, hit_object, _matrix = scene.ray_cast(
            depsgraph,
            origin,
            direction,
            distance=(point_distance - traveled) + distance_tolerance,
        )
        if not hit or hit_object is None:
            return None

        hit_distance = (location - origin).length
        traveled += hit_distance
        if traveled >= point_distance - distance_tolerance:
            return None
        if (
            hit_object.name not in ignored_object_names
            and not hit_object.name.startswith(SYNTHETIC_WINDOW_BLOCKER_PREFIX)
        ):
            return str(hit_object.name)

        origin = location + (direction * epsilon)
        traveled += epsilon
    return None


def yolo_pose_line(
    class_id: int,
    x_center: float,
    y_center: float,
    bbox_width: float,
    bbox_height: float,
    keypoints: list[tuple[float, float, int]],
    width: int,
    height: int,
) -> str:
    normalized_x_center, normalized_y_center = normalize_point(x_center, y_center, width, height)
    values = [
        str(class_id),
        f"{normalized_x_center:.6f}",
        f"{normalized_y_center:.6f}",
        f"{bbox_width / width:.6f}",
        f"{bbox_height / height:.6f}",
    ]
    for keypoint_x, keypoint_y, visibility in keypoints:
        if visibility == POINT_VISIBILITY_UNLABELED:
            values.extend(["0.000000", "0.000000", str(visibility)])
            continue
        kpt_x, kpt_y = normalize_point(keypoint_x, keypoint_y, width, height)
        values.extend([f"{kpt_x:.6f}", f"{kpt_y:.6f}", str(visibility)])
    return " ".join(values)


def keypoint_bbox(
    projected_keypoints: list[tuple[float, float, int]],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    labeled_points = [
        (x, y)
        for x, y, visibility in projected_keypoints
        if visibility != POINT_VISIBILITY_UNLABELED
    ]
    if not labeled_points:
        return None

    radius = max(MIN_KEYPOINT_BBOX_RADIUS_PX, round(min(width, height) * KEYPOINT_BBOX_RADIUS_RATIO))
    xs = [point[0] for point in labeled_points]
    ys = [point[1] for point in labeled_points]
    x1 = max(0.0, min(xs) - radius)
    y1 = max(0.0, min(ys) - radius)
    x2 = min(float(width), max(xs) + radius)
    y2 = min(float(height), max(ys) + radius)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, x2 - x1), max(1.0, y2 - y1)


def left_to_right_key(point: tuple[float, float, int]) -> tuple[float, float]:
    return point[0], point[1]


def clockwise_keypoints(
    center: tuple[float, float, int],
    keypoints: list[tuple[str, tuple[float, float, int]]],
) -> list[tuple[str, tuple[float, float, int]]]:
    center_x, center_y, _visibility = center
    ordered = sorted(
        keypoints,
        key=lambda item: math.atan2(item[1][1] - center_y, item[1][0] - center_x),
    )
    if not ordered:
        return ordered
    start_index = min(range(len(ordered)), key=lambda index: left_to_right_key(ordered[index][1]))
    return ordered[start_index:] + ordered[:start_index]


def rotate_sequence_to_start(values: list[str], start_value: str) -> list[str]:
    if start_value not in values:
        return values
    start_index = values.index(start_value)
    return values[start_index:] + values[:start_index]


def unlabeled_pose_keypoint() -> tuple[float, float, int]:
    return 0.0, 0.0, POINT_VISIBILITY_UNLABELED


def projected_piece_keypoint(
    name: str,
    projected_by_name: dict[str, tuple[float, float, int]],
) -> tuple[float, float, int]:
    return projected_by_name.get(name, unlabeled_pose_keypoint())


def pose_object_from_keypoints(
    *,
    class_id: int,
    name: str,
    keypoint_names: list[str],
    projected_by_name: dict[str, tuple[float, float, int]],
    metadata_by_name: dict[str, dict[str, object]],
    width: int,
    height: int,
) -> tuple[str | None, dict[str, object] | None]:
    projected_keypoints = [
        projected_piece_keypoint(keypoint_name, projected_by_name)
        for keypoint_name in keypoint_names
    ]
    while len(projected_keypoints) < POSE_KEYPOINT_COUNT:
        projected_keypoints.append(unlabeled_pose_keypoint())

    bbox = keypoint_bbox(projected_keypoints, width, height)
    if bbox is None:
        return None, None

    x_center, y_center, bbox_width, bbox_height = bbox
    line = yolo_pose_line(
        class_id,
        x_center,
        y_center,
        bbox_width,
        bbox_height,
        projected_keypoints,
        width,
        height,
    )
    keypoint_metadata = []
    for index, keypoint_name in enumerate(keypoint_names):
        source = metadata_by_name.get(keypoint_name)
        if source is None:
            continue
        keypoint_metadata.append({
            **source,
            "index": index,
            "label": str(index + 1),
            "name": str(index + 1),
        })
    return line, {
        "class_id": class_id,
        "class": CLASS_NAMES[class_id],
        "name": name,
        "bbox_xyxy": [
            x_center - bbox_width / 2.0,
            y_center - bbox_height / 2.0,
            x_center + bbox_width / 2.0,
            y_center + bbox_height / 2.0,
        ],
        "keypoint_labels": [str(index + 1) for index in range(len(keypoint_names))],
        "keypoints": keypoint_metadata,
    }


def axis_vector(axis_index: int, sign: float = 1.0) -> Vector:
    values = [0.0, 0.0, 0.0]
    values[axis_index] = sign
    return Vector(values)


def nearest_axis_vector(vector: Vector, exclude: set[tuple[int, int]] | None = None) -> Vector:
    exclude = exclude or set()
    best_axis = 0
    best_sign = 1
    best_dot = -1.0
    for axis_index in range(3):
        for sign in (-1, 1):
            if (axis_index, sign) in exclude:
                continue
            axis = axis_vector(axis_index, float(sign))
            dot = vector.normalized().dot(axis) if vector.length > 1e-6 else -1.0
            if dot > best_dot:
                best_axis = axis_index
                best_sign = sign
                best_dot = dot
    return axis_vector(best_axis, float(best_sign))


def axis_key(vector: Vector) -> tuple[int, int]:
    axis_index = max(range(3), key=lambda index: abs(vector[index]))
    return axis_index, 1 if vector[axis_index] >= 0.0 else -1


def visible_world_normals() -> list[Vector]:
    normals = []
    for axis_index in range(3):
        for sign in (-1.0, 1.0):
            normal = axis_vector(axis_index, sign)
            if normal.dot(get_camera().location) > 0.0:
                normals.append(normal)
    return normals


def camera_relative_axes(visible_normals: list[Vector]) -> dict[str, Vector]:
    camera = get_camera()
    up_reference = Vector((0.0, 0.0, 1.0))
    u_axis = max(visible_normals, key=lambda normal: normal.dot(up_reference)) if visible_normals else up_reference

    side_normals = [normal for normal in visible_normals if axis_key(normal) != axis_key(u_axis)]
    if len(side_normals) >= 2:
        first, second = side_normals[:2]
        if (u_axis.cross(first)).dot(second) > 0.5:
            f_axis = first
            r_axis = second
        else:
            f_axis = second
            r_axis = first
    elif len(side_normals) == 1:
        f_axis = side_normals[0]
        r_axis = nearest_axis_vector(u_axis.cross(f_axis))
    else:
        camera_direction = camera.location - u_axis * camera.location.dot(u_axis)
        f_axis = nearest_axis_vector(camera_direction, {axis_key(u_axis), axis_key(-u_axis)})
        r_axis = nearest_axis_vector(u_axis.cross(f_axis), {axis_key(u_axis), axis_key(-u_axis), axis_key(f_axis), axis_key(-f_axis)})

    return {
        "U": u_axis,
        "D": -u_axis,
        "F": f_axis,
        "B": -f_axis,
        "R": r_axis,
        "L": -r_axis,
    }


def render_pose_keypoints(path: Path | None) -> tuple[list[str], list[dict[str, object]], dict[str, object]]:
    scene = bpy.context.scene
    camera = get_camera()
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    depsgraph = bpy.context.evaluated_depsgraph_get()
    visible_hand_names = {
        obj.name
        for obj in hand_objects().values()
        if not obj.hide_get() and not obj.hide_viewport and not obj.hide_render
    }
    lines = []
    pose_objects = []

    bpy.context.view_layer.update()
    projected_by_name: dict[str, tuple[float, float, int]] = {}
    metadata_by_name: dict[str, dict[str, object]] = {}
    keypoint_specs = cube_keypoint_specs_from_corners_and_centers()
    for index, spec in enumerate(keypoint_specs):
        point = spec["point"]
        if not isinstance(point, Vector):
            continue
        name = str(spec["name"])
        normals = spec["normals"]
        projected = project_world_point(point, camera, scene)
        hand_occluded = False
        occluded_by = None
        if projected is None:
            x, y = 0.0, 0.0
            visibility = POINT_VISIBILITY_UNLABELED
        elif not isinstance(normals, list) or not point_faces_camera(point, normals, camera.location):
            x, y = projected
            visibility = POINT_VISIBILITY_UNLABELED
        else:
            x, y = projected
            source_object = str(spec.get("source_object", ""))
            ignored_object_names = {source_object} if source_object else set()
            occluded_by = object_occluding_point(point, depsgraph, ignored_object_names)
            hand_occluded = occluded_by in visible_hand_names
            visibility = POINT_VISIBILITY_LABELED_OCCLUDED if occluded_by is not None else POINT_VISIBILITY_VISIBLE

        projected_by_name[name] = (x, y, visibility)
        metadata_by_name[name] = {
            "index": index,
            "name": name,
            "class": spec.get("class"),
            "class_id": spec.get("class_id"),
            "source_object": spec.get("source_object"),
            "source_piece_type": spec.get("source_piece_type"),
            "relative_point": spec.get("relative_point"),
            "world_point": [point.x, point.y, point.z],
            "pixel_point": [x, y],
            "normalized_point": list(normalize_point(x, y, width, height)),
            "visibility": visibility,
            "hand_occluded": hand_occluded,
            "occluded_by": occluded_by,
        }

    for point_index, spec in enumerate(keypoint_specs, start=1):
        name = str(spec["name"])
        line, pose_object = pose_object_from_keypoints(
            class_id=int(spec["class_id"]),
            name=f"{spec['class']}_{point_index}",
            keypoint_names=[name],
            projected_by_name=projected_by_name,
            metadata_by_name=metadata_by_name,
            width=width,
            height=height,
        )
        if line is not None and pose_object is not None:
            lines.append(line)
            pose_objects.append(pose_object)

    if path is not None:
        write_pose_keypoint_mask(path, pose_objects, width, height)

    return lines, pose_objects, {
        "keypoint_shape": [POSE_KEYPOINT_COUNT, 3],
        "source_keypoints": len(keypoint_specs),
        "visible_hands": sorted(visible_hand_names),
        "candidates": len(keypoint_specs),
        "piece_candidates": len(corner_cubie_objects()) + len(center_cubie_objects()),
    }


def pose_visibility_summary(pose_objects: list[dict[str, object]]) -> dict[str, int]:
    summary = {
        "unlabeled": 0,
        "occluded": 0,
        "visible": 0,
        "total": 0,
    }
    for pose_object in pose_objects:
        for keypoint in pose_object.get("keypoints", []):
            visibility = int(keypoint.get("visibility", POINT_VISIBILITY_UNLABELED))
            summary["total"] += 1
            if visibility == POINT_VISIBILITY_VISIBLE:
                summary["visible"] += 1
            elif visibility == POINT_VISIBILITY_LABELED_OCCLUDED:
                summary["occluded"] += 1
            else:
                summary["unlabeled"] += 1
    return summary


def hand_pose_is_acceptable(
    hand_randomization: dict[str, object] | None,
    pose_objects: list[dict[str, object]],
    min_visible_keypoints: int,
) -> tuple[bool, dict[str, int]]:
    summary = pose_visibility_summary(pose_objects)
    if hand_randomization is None:
        return True, summary
    hand_objects_metadata = hand_randomization.get("objects", {})
    visible_hand_count = sum(
        1
        for metadata in hand_objects_metadata.values()
        if isinstance(metadata, dict) and bool(metadata.get("visible"))
    )
    if visible_hand_count == 0:
        return True, summary
    return summary["visible"] >= max(0, min_visible_keypoints), summary


def write_pose_keypoint_mask(
    path: Path,
    pose_objects: list[dict[str, object]],
    width: int,
    height: int,
) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "OpenCV and NumPy are required to write keypoint visualization masks. "
            "Install cv2 and numpy in Blender's Python environment before generating training data."
        ) from None

    output = np.zeros((height, width, 3), dtype=np.uint8)
    fill_colors = {
        CORNER_CLASS_ID: (90, 90, 90),
        CORNER_EDGE_CLASS_ID: (0, 95, 95),
        CORNER_EDGE_CENTER_CLASS_ID: (95, 0, 95),
        CENTER_CLASS_ID: (95, 95, 0),
    }
    point_colors = {
        CORNER_CLASS_ID: (255, 255, 255),
        CORNER_EDGE_CLASS_ID: (0, 255, 255),
        CORNER_EDGE_CENTER_CLASS_ID: (255, 120, 255),
        CENTER_CLASS_ID: (255, 255, 0),
    }
    for pose_object in pose_objects:
        class_id = int(pose_object["class_id"])
        fill_color = fill_colors.get(class_id, (90, 90, 90))
        outline_color = point_colors.get(class_id, (255, 255, 255))
        labeled_points = [
            keypoint["pixel_point"]
            for keypoint in pose_object["keypoints"]
            if int(keypoint["visibility"]) != POINT_VISIBILITY_UNLABELED
        ]
        if len(labeled_points) >= 3:
            points = np.asarray(labeled_points, dtype=np.float32)
            hull = cv2.convexHull(np.round(points).astype(np.int32))
            cv2.fillConvexPoly(output, hull, fill_color, lineType=cv2.LINE_AA)
            cv2.polylines(output, [hull], isClosed=True, color=outline_color, thickness=1, lineType=cv2.LINE_AA)
        elif len(labeled_points) == 2:
            start = tuple(np.round(labeled_points[0]).astype(np.int32))
            end = tuple(np.round(labeled_points[1]).astype(np.int32))
            cv2.line(output, start, end, fill_color, thickness=9, lineType=cv2.LINE_AA)
            cv2.line(output, start, end, outline_color, thickness=2, lineType=cv2.LINE_AA)

        for keypoint in pose_object["keypoints"]:
            visibility = int(keypoint["visibility"])
            if visibility == POINT_VISIBILITY_UNLABELED:
                continue
            x, y = keypoint["pixel_point"]
            draw_color = outline_color
            if visibility == POINT_VISIBILITY_LABELED_OCCLUDED:
                draw_color = (0, 0, 255)
            center = (round(float(x)), round(float(y)))
            cv2.circle(output, center, 5, draw_color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(output, center, 7, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    cv2.imwrite(str(path), output)


def write_yolo_pose(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def generate_dataset(
    output_dir: Path,
    count: int,
    scramble_length: int,
    width: int,
    height: int,
    background_object_count: int,
    cube_rotational_blur_probability: float,
    max_cube_rotational_blur_degrees: float,
    cube_rotational_blur_samples: int,
    degrade_probability: float,
    render_masks: bool,
    min_visible_keypoints_with_hands: int,
    max_hand_placement_attempts: int,
) -> None:
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks" if render_masks else None
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    if masks_dir is not None:
        masks_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    configure_render(width, height)

    metadata = []
    for index in range(count):
        (
            moves,
            layer_state,
            background_randomization,
            camera_randomization,
            light_randomization,
            cube_pose,
            sticker_materials,
        ) = randomize_scene(
            scramble_length,
            background_object_count=background_object_count,
        )
        objects = cube_objects()
        render_objects = list(objects)

        stem = f"{(index):06d}"
        image_path = images_dir / f"{stem}.png"
        mask_path = masks_dir / f"{stem}.png" if masks_dir is not None else None
        label_path = labels_dir / f"{stem}.txt"

        hands = hand_objects()
        hand_states = hand_object_states(hands)
        hand_attempts = []
        hand_randomization = None
        preflight_pose_objects = []
        preflight_pose_geometry = {}
        accepted_hand_pose = False
        attempts = max(1, int(max_hand_placement_attempts))
        for attempt_index in range(1, attempts + 1):
            hand_randomization = randomize_hands(hands, hand_states, cube_pose)
            _preflight_lines, preflight_pose_objects, preflight_pose_geometry = render_pose_keypoints(None)
            acceptable, visibility_summary = hand_pose_is_acceptable(
                hand_randomization,
                preflight_pose_objects,
                min_visible_keypoints=min_visible_keypoints_with_hands,
            )
            hand_attempts.append({
                "attempt": attempt_index,
                "accepted": acceptable,
                "visibility": visibility_summary,
                "hands": hand_randomization,
            })
            if acceptable:
                accepted_hand_pose = True
                break
            restore_hand_objects(hands, hand_states)

        if not accepted_hand_pose:
            hand_randomization = randomize_hands(hands, hand_states, cube_pose, forced_visibility="none")
            _preflight_lines, preflight_pose_objects, preflight_pose_geometry = render_pose_keypoints(None)
            _acceptable, visibility_summary = hand_pose_is_acceptable(
                hand_randomization,
                preflight_pose_objects,
                min_visible_keypoints=min_visible_keypoints_with_hands,
            )
            hand_attempts.append({
                "attempt": attempts + 1,
                "accepted": True,
                "fallback": "hide_hands",
                "visibility": visibility_summary,
                "hands": hand_randomization,
            })

        try:
            rotational_blur = render_cube_rotational_blur_png(
                image_path,
                render_objects,
                probability=cube_rotational_blur_probability,
                max_degrees=max_cube_rotational_blur_degrees,
                samples=cube_rotational_blur_samples,
            )
            camera_degradation = apply_camera_degradation(
                image_path,
                probability=degrade_probability,
            )
            yolo_lines, pose_objects, pose_geometry = render_pose_keypoints(mask_path)
            write_yolo_pose(label_path, yolo_lines)
            pose_geometry["preflight_visibility"] = pose_visibility_summary(preflight_pose_objects)
            pose_geometry["hand_attempts"] = len(hand_attempts)
        finally:
            restore_hand_objects(hands, hand_states)

        metadata.append({
            "active_random_seed": ACTIVE_RANDOM_SEED,
            "image": str(image_path),
            "mask": str(mask_path) if mask_path is not None else None,
            "label": str(label_path),
            "moves": moves,
            "layer_state": layer_state,
            "cubie_labels": {obj.name: obj.pass_index for obj in objects},
            "camera": camera_randomization,
            "light": light_randomization,
            "background": background_randomization,
            "cube_pose": cube_pose,
            "sticker_materials": sticker_materials,
            "cube_rotational_blur": rotational_blur,
            "camera_degradation": camera_degradation,
            "hands": hand_randomization,
            "hand_attempts": hand_attempts,
            "pose_geometry": pose_geometry,
            "keypoints": pose_objects,
            "objects": len(yolo_lines),
        })
        mask_message = f", {mask_path}" if mask_path is not None else ""
        blend_asset_count = background_randomization.get("blend_asset_count", 0)
        blend_asset_sources = background_randomization.get("blend_asset_sources", [])
        print(
            f"Wrote {image_path}{mask_message}, and {label_path} "
            f"with {len(yolo_lines)} pose keypoint objects, "
            f"{blend_asset_count} blend background assets {blend_asset_sources}, "
            f"{'with' if rotational_blur else 'without'} rotational blur "
            f"and {'with' if camera_degradation else 'without'} camera degradation"
        )

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "path: " + str(output_dir.resolve()) + "\n"
        "train: images\n"
        "val: images\n"
        f"kpt_shape: [{POSE_KEYPOINT_COUNT}, 3]\n"
        "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )


def cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scramble-length", type=int, default=DEFAULT_SCRAMBLE_LENGTH)
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--background-object-count",
        type=int,
        default=DEFAULT_BACKGROUND_OBJECT_COUNT,
        help="Nominal number of sparse background props to place behind and around the cube.",
    )
    parser.add_argument(
        "--cube-rotational-blur-probability",
        type=float,
        default=DEFAULT_CUBE_ROTATIONAL_BLUR_PROBABILITY,
        help="Probability of rendering the cube image with rotational motion blur.",
    )
    parser.add_argument(
        "--max-cube-rotational-blur-degrees",
        type=float,
        default=DEFAULT_MAX_CUBE_ROTATIONAL_BLUR_DEGREES,
        help="Maximum random rotational blur arc in degrees.",
    )
    parser.add_argument(
        "--cube-rotational-blur-samples",
        type=int,
        default=DEFAULT_CUBE_ROTATIONAL_BLUR_SAMPLES,
        help="Number of sub-exposure renders averaged for rotational blur.",
    )
    parser.add_argument(
        "--degrade-probability",
        type=float,
        default=DEFAULT_DEGRADE_PROBABILITY,
        help="Probability of applying post-render blur, noise, rescale softness, and JPEG artifacts.",
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Skip writing visualization masks; segmentation labels are still written.",
    )
    parser.add_argument(
        "--min-visible-keypoints-with-hands",
        type=int,
        default=DEFAULT_MIN_VISIBLE_KEYPOINTS_WITH_HANDS,
        help=(
            "Minimum visible keypoint objects required when hands are visible. "
            "Hand poses below this threshold are retried."
        ),
    )
    parser.add_argument(
        "--max-hand-placement-attempts",
        type=int,
        default=DEFAULT_MAX_HAND_PLACEMENT_ATTEMPTS,
        help="Maximum hand randomization attempts before falling back to a no-hand frame.",
    )

    argv = sys.argv
    if "--" not in argv:
        return parser.parse_args([])
    return parser.parse_args(argv[argv.index("--") + 1 :])


def main() -> None:
    global ACTIVE_RANDOM_SEED
    args = cli_args()
    if args.seed is not None:
        ACTIVE_RANDOM_SEED = args.seed
    else:
        ACTIVE_RANDOM_SEED = runtime_random_seed()
    random.seed(ACTIVE_RANDOM_SEED)
    print(f"Using random seed {ACTIVE_RANDOM_SEED}")

    generate_dataset(
        output_dir=args.output_dir,
        count=args.count,
        scramble_length=args.scramble_length,
        width=args.width,
        height=args.height,
        background_object_count=args.background_object_count,
        cube_rotational_blur_probability=args.cube_rotational_blur_probability,
        max_cube_rotational_blur_degrees=args.max_cube_rotational_blur_degrees,
        cube_rotational_blur_samples=args.cube_rotational_blur_samples,
        degrade_probability=args.degrade_probability,
        render_masks=not args.no_masks,
        min_visible_keypoints_with_hands=args.min_visible_keypoints_with_hands,
        max_hand_placement_attempts=args.max_hand_placement_attempts,
    )


if __name__ == "__main__":
    main()
