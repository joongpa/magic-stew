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
from pathlib import Path
import random
import sys
import tempfile
import urllib.request

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector

import site

PROJECT = Path("/Users/jeffpark/Documents/LiveRecognition")
VENV = PROJECT / ".venv"
site_packages = next((VENV / "lib").glob("python*/site-packages"))
site.addsitedir(str(site_packages))


# Common Blender convention: X right, Z up, and negative Y faces the camera/front.
# Use "+Y" instead if your modeled front face is on positive Y.
FRONT_AXIS = "-Y"

# Random scene defaults for synthetic data generation.
DEFAULT_SCRAMBLE_LENGTH = 30
CAMERA_DISTANCE_RANGE = (7.2, 9.2)
CAMERA_ELEVATION_DEGREES = (12.0, 42.0)
CAMERA_TARGET_JITTER = 0.25
CAMERA_LENS_MM = 35.0
CAMERA_ROLL_DEGREES = (-45.0, 45.0)

LIGHT_DISTANCE_RANGE = (3.0, 6.0)
LIGHT_ELEVATION_DEGREES = (20.0, 70.0)
BACKGROUND_NAME = "SyntheticDataBackground"
BACKGROUND_MATERIAL_NAME = "SyntheticDataBackgroundMaterial"
BACKGROUND_DISTANCE = 4.0
BACKGROUND_SIZE = 12.0
BACKGROUND_IMAGE_URL = "https://picsum.photos/256/256"
BACKGROUND_IMAGE_PATH = f"{tempfile.gettempdir()}/synthetic_cube_background.jpg"
BACKGROUND_CUBIE_COLLECTION_NAME = "SyntheticBackgroundCubies"
BACKGROUND_CUBIE_PREFIX = "SyntheticBackgroundCubie"
BACKGROUND_CUBIE_DISTANCE_RANGE = (3.0, 3.8)
BACKGROUND_CUBIE_RIGHT_RANGE = (-8, 8)
BACKGROUND_CUBIE_UP_RANGE = (-6, 6)
BACKGROUND_CUBIE_SCALE_RANGE = (0.65, 1.05)
DEFAULT_BACKGROUND_CUBIE_PROBABILITY = 0.0
DEFAULT_MAX_BACKGROUND_CUBIES = 3
DEFAULT_CUBE_ROTATIONAL_BLUR_PROBABILITY = 0.0
DEFAULT_MAX_CUBE_ROTATIONAL_BLUR_DEGREES = 8.0
DEFAULT_CUBE_ROTATIONAL_BLUR_SAMPLES = 7
HAND_NAMES = ("left-hand", "right-hand")
HAND_VISIBILITY_WEIGHTS = (
    ("none", 0.3),
    ("left", 0.1),
    ("right", 0.1),
    ("both", 0.5),
    # ("none", 1),
    # ("left", 0),
    # ("right", 0),
    # ("both", 0),
)
HAND_X_OFFSET_RANGE = (-0.2, 0.2)
HAND_Z_OFFSET_RANGE = (-1.0, 1.0)
HAND_X_ROTATION_DEGREES = (-90.0, 0.0, 90.0)

SOLVED_LOCATION_PROP = "synthetic_solved_location"
SOLVED_ROTATION_PROP = "synthetic_solved_rotation"
DEFAULT_OUTPUT_DIR = "synthetic_cube_dataset"
DEFAULT_IMAGE_SIZE = 512
CUBE_HALF_EXTENT = 1.5
CUBE_GRID_LINES = (-1.5, -0.5, 0.5, 1.5)
CLASS_NAMES = ["visible_face", "center_sticker", "corner_sticker", "edge_sticker"]
FACE_CLASS_ID = 0
STICKER_CLASS_ID_BY_CUBIE_TYPE = {
    "center": CLASS_NAMES.index("center_sticker"),
    "corner": CLASS_NAMES.index("corner_sticker"),
    "edge": CLASS_NAMES.index("edge_sticker"),
}
INSTANCE_VALUES = {
    "center": [20, 40, 60, 80, 100, 120],
    "edge": [130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240],
    "corner": [248, 249, 250, 251, 252, 253, 254, 255],
}
CUBIE_PASS_INDEXES = [
    cubie_value
    for class_name in ("center", "edge", "corner")
    for cubie_value in INSTANCE_VALUES[class_name]
]
STICKER_COLOR_INDEX_BY_MATERIAL_PASS_INDEX = {
    material_pass_index: material_pass_index - 1 for material_pass_index in range(1, 7)
}
CUBIE_VALUE_TO_CLASS = {
    value: STICKER_CLASS_ID_BY_CUBIE_TYPE[class_name]
    for class_name in ("center", "edge", "corner")
    for value in INSTANCE_VALUES[class_name]
}
STICKER_TO_MASK_VALUE = {
    sticker_key: mask_value
    for mask_value, sticker_key in enumerate(
        (
            (sticker_color_index, cubie_value)
            for cubie_value in CUBIE_PASS_INDEXES
            for sticker_color_index in range(6)
        ),
        start=1,
    )
}
MASK_VALUE_TO_STICKER = {
    mask_value: sticker_key for sticker_key, mask_value in STICKER_TO_MASK_VALUE.items()
}
MASK_VALUE_TO_CLASS = {
    mask_value: CUBIE_VALUE_TO_CLASS[cubie_value]
    for mask_value, (_sticker_color_index, cubie_value) in MASK_VALUE_TO_STICKER.items()
}
MIN_STICKER_VISIBLE_FRACTION = 0.15
STICKER_VISIBLE_SAMPLE_GRID = 7
MIN_STICKER_VISIBLE_SAMPLES = 3
PROJECTION_BOUNDS_EPSILON_PX = 1.5
SURFACE_FRONT_TOLERANCE_METERS = 0.015
SURFACE_BACK_TOLERANCE_METERS = float('inf')
KEYPOINT_VISIBILITY_INSET_FRACTIONS = (0.015, 0.035, 0.07)
GRID_STICKER_MASK_VALUE_BY_CLASS = {
    "center_sticker": 80,
    "corner_sticker": 240,
    "edge_sticker": 180,
}


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


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("-Z", "Y")


def random_point_on_shell(
    distance_range: tuple[float, float],
    elevation_degrees: tuple[float, float],
) -> Vector:
    distance = random.uniform(*distance_range)
    azimuth = random.uniform(0.0, math.tau)
    elevation = math.radians(random.uniform(*elevation_degrees))
    radius_xy = distance * math.cos(elevation)
    return Vector((
        radius_xy * math.cos(azimuth),
        radius_xy * math.sin(azimuth),
        distance * math.sin(elevation),
    ))


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


def randomize_camera() -> dict[str, object]:
    camera = get_camera()
    camera.location = random_point_on_shell(CAMERA_DISTANCE_RANGE, CAMERA_ELEVATION_DEGREES)
    camera.data.lens = CAMERA_LENS_MM
    target = random_target()
    roll_degrees = random.uniform(*CAMERA_ROLL_DEGREES)
    look_at(camera, target)
    camera.rotation_quaternion = camera.rotation_quaternion @ Quaternion(
        Vector((0.0, 0.0, -1.0)),
        math.radians(roll_degrees),
    )
    return {
        "location": [camera.location.x, camera.location.y, camera.location.z],
        "target": [target.x, target.y, target.z],
        "roll_degrees": roll_degrees,
        "lens_mm": camera.data.lens,
    }


def get_light() -> bpy.types.Object:
    lights = [obj for obj in bpy.data.objects if obj.type == "LIGHT"]
    if lights:
        return lights[0]

    data = bpy.data.lights.new("SyntheticDataLight", type="AREA")
    light = bpy.data.objects.new("SyntheticDataLight", data)
    bpy.context.collection.objects.link(light)
    return light


def random_light_power() -> float:
    bucket = random.random()
    if bucket < 0.12:
        return random.uniform(60.0, 160.0)
    if bucket < 0.85:
        return random.uniform(300.0, 800.0)
    return random.uniform(900.0, 1500.0)


def randomize_light() -> None:
    light = get_light()
    light.location = random_point_on_shell(LIGHT_DISTANCE_RANGE, LIGHT_ELEVATION_DEGREES)
    light.data.energy = random_light_power()
    if hasattr(light.data, "size"):
        light.data.size = random.uniform(2.0, 5.0)


def download_background_image() -> str:
    url = f"{BACKGROUND_IMAGE_URL}?random={random.randrange(1_000_000_000)}"
    urllib.request.urlretrieve(url, BACKGROUND_IMAGE_PATH)
    return BACKGROUND_IMAGE_PATH


def randomize_background() -> None:
    camera = get_camera()
    world = bpy.context.scene.world or bpy.data.worlds.new("SyntheticDataWorld")
    bpy.context.scene.world = world
    world.color = (0.0, 0.0, 0.0)

    background = bpy.data.objects.get(BACKGROUND_NAME)
    if background is None:
        bpy.ops.mesh.primitive_plane_add(size=BACKGROUND_SIZE)
        background = bpy.context.object
        background.name = BACKGROUND_NAME

    material = bpy.data.materials.get(BACKGROUND_MATERIAL_NAME)
    if material is None:
        material = bpy.data.materials.new(BACKGROUND_MATERIAL_NAME)
        material.use_nodes = True

    image_path = download_background_image()
    image = bpy.data.images.load(image_path, check_existing=False)
    image.name = "SyntheticDataBackgroundImage"

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        texture = nodes.get("SyntheticDataBackgroundTexture")
        if texture is None:
            texture = nodes.new("ShaderNodeTexImage")
            texture.name = "SyntheticDataBackgroundTexture"
        texture.image = image
        if not bsdf.inputs["Base Color"].is_linked:
            links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 1.0

    background.data.materials.clear()
    background.data.materials.append(material)
    background.scale = (1.0, 1.0, 1.0)

    camera_direction = camera.rotation_quaternion @ Vector((0.0, 0.0, -1.0))
    background.location = camera_direction * BACKGROUND_DISTANCE
    background.rotation_mode = "QUATERNION"
    background.rotation_quaternion = camera.rotation_quaternion


def clear_background_cubies() -> None:
    for obj in list(bpy.data.objects):
        if obj.name.startswith(BACKGROUND_CUBIE_PREFIX):
            bpy.data.objects.remove(obj, do_unlink=True)


def background_cubie_collection() -> bpy.types.Collection:
    collection = bpy.data.collections.get(BACKGROUND_CUBIE_COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(BACKGROUND_CUBIE_COLLECTION_NAME)
        bpy.context.scene.collection.children.link(collection)
    return collection


def randomize_background_cubies(
    source_objects: list[bpy.types.Object],
    probability: float,
    max_count: int,
) -> int:
    clear_background_cubies()
    if probability <= 0.0 or max_count <= 0 or random.random() >= probability:
        return 0

    camera = get_camera()
    camera_rotation = camera.rotation_quaternion
    camera_direction = camera_rotation @ Vector((0.0, 0.0, -1.0))
    camera_right = camera_rotation @ Vector((1.0, 0.0, 0.0))
    camera_up = camera_rotation @ Vector((0.0, 1.0, 0.0))
    collection = background_cubie_collection()

    count = random.randint(1, max_count)
    for index in range(count):
        source = random.choice(source_objects)
        cubie = source.copy()
        cubie.data = source.data
        cubie.animation_data_clear()
        cubie.name = f"{BACKGROUND_CUBIE_PREFIX}.{index:03d}"

        depth = random.uniform(*BACKGROUND_CUBIE_DISTANCE_RANGE)
        right = random.uniform(*BACKGROUND_CUBIE_RIGHT_RANGE)
        up = random.uniform(*BACKGROUND_CUBIE_UP_RANGE)
        cubie.location = (camera_direction * depth) + (camera_right * right) + (camera_up * up)
        cubie.rotation_mode = "QUATERNION"
        cubie.rotation_quaternion = Quaternion(
            (
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0),
            )
        ).normalized()
        cubie.scale = (1.0, 1.0, 1.0)
        cubie.scale *= random.uniform(*BACKGROUND_CUBIE_SCALE_RANGE)

        collection.objects.link(cubie)

    bpy.context.view_layer.update()
    return count


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
    bpy.context.view_layer.update()


def randomize_hands(
    objects: dict[str, bpy.types.Object],
    states: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    if not objects:
        return None

    visibility = random.choices(
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

        obj.matrix_world = global_rotation @ Matrix.Translation(location_offset) @ base_matrix
        obj.rotation_mode = "QUATERNION"
        obj.hide_render = hidden
        obj.hide_viewport = hidden
        obj.hide_set(hidden)

        randomization["objects"][name] = {
            "visible": not hidden,
            "x_offset": x_offset,
            "z_offset": z_offset,
            "x_rotation_degrees": x_rotation_degrees,
        }

    bpy.context.view_layer.update()
    return randomization


def randomize_scene(
    scramble_length: int = DEFAULT_SCRAMBLE_LENGTH,
    background_cubie_probability: float = DEFAULT_BACKGROUND_CUBIE_PROBABILITY,
    max_background_cubies: int = DEFAULT_MAX_BACKGROUND_CUBIES,
) -> tuple[list[str], int, dict[str, object]]:
    clear_background_cubies()
    objects = cube_objects()
    reset_cube(objects)

    camera_randomization = randomize_camera()
    randomize_light()
    randomize_background()

    faces = list("RLUDBF")
    moves = []
    for _ in range(scramble_length):
        choices = [face for face in faces if not moves or face != moves[-1]]
        moves.append(random.choice(choices))

    for move in moves:
        rotate_face(objects, move)

    background_cubies = randomize_background_cubies(
        objects,
        probability=background_cubie_probability,
        max_count=max_background_cubies,
    )
    return moves, background_cubies, camera_randomization


def configure_render(width: int, height: int) -> None:
    scene = bpy.context.scene
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


def camera_frame_bounds(camera: bpy.types.Object, scene: bpy.types.Scene) -> tuple[float, float, float, float, float]:
    frame = camera.data.view_frame(scene=scene)
    xs = [corner.x for corner in frame]
    ys = [corner.y for corner in frame]
    zs = [corner.z for corner in frame]
    return min(xs), max(xs), min(ys), max(ys), sum(zs) / len(zs)


def camera_ray(camera: bpy.types.Object, scene: bpy.types.Scene, x: int, y: int) -> tuple[Vector, Vector]:
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    left, right, bottom, top, frame_z = camera_frame_bounds(camera, scene)
    u = (x + 0.5) / width
    v = 1.0 - ((y + 0.5) / height)
    frame_x = left + u * (right - left)
    frame_y = bottom + v * (top - bottom)
    camera_matrix = camera.matrix_world

    if camera.data.type == "ORTHO":
        origin = camera_matrix @ Vector((frame_x, frame_y, 0.0))
        direction = (camera_matrix.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
        return origin, direction

    origin = camera_matrix.translation
    direction = (camera_matrix.to_3x3() @ Vector((frame_x, frame_y, frame_z))).normalized()
    return origin, direction


def camera_rays(camera: bpy.types.Object, scene: bpy.types.Scene) -> list[tuple[Vector, Vector]]:
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    return [
        camera_ray(camera, scene, x, y)
        for y in range(height)
        for x in range(width)
    ]


def normalize_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return (
        max(0.0, min(x / width, 1.0)),
        max(0.0, min(y / height, 1.0)),
    )


def project_world_point(point: Vector, width: int, height: int) -> tuple[float, float]:
    projected = world_to_camera_view(bpy.context.scene, get_camera(), point)
    return projected.x * width, (1.0 - projected.y) * height


def point_visibility_for_projection(
    point: tuple[float, float],
    width: int,
    height: int,
    visible: bool,
) -> int:
    x, y = point
    if not (
        -PROJECTION_BOUNDS_EPSILON_PX <= x <= width + PROJECTION_BOUNDS_EPSILON_PX
        and -PROJECTION_BOUNDS_EPSILON_PX <= y <= height + PROJECTION_BOUNDS_EPSILON_PX
    ):
        return 0
    return 2 if visible else 1


def ray_reaches_cube_surface(point: Vector, cubie_names: set[str], depsgraph: bpy.types.Depsgraph) -> bool:
    camera = get_camera()
    direction = point - camera.location
    distance = direction.length
    if distance <= 1e-6:
        return False

    hit, location, _normal, _polygon_index, hit_object, _matrix = bpy.context.scene.ray_cast(
        depsgraph,
        camera.location,
        direction.normalized(),
        distance=distance + SURFACE_BACK_TOLERANCE_METERS,
    )
    if not hit or hit_object is None:
        return False
    hit_distance = (location - camera.location).length
    if hit_distance >= distance - SURFACE_FRONT_TOLERANCE_METERS:
        return True
    return hit_object.name in cubie_names


def keypoint_visibility_samples(corner: Vector, center: Vector) -> list[Vector]:
    return [corner.lerp(center, fraction) for fraction in KEYPOINT_VISIBILITY_INSET_FRACTIONS]


def any_sample_reaches_cube_surface(
    samples: list[Vector],
    cubie_names: set[str],
    depsgraph: bpy.types.Depsgraph,
) -> bool:
    return any(ray_reaches_cube_surface(sample, cubie_names, depsgraph) for sample in samples)


def cube_face_specs() -> list[tuple[str, Vector, list[Vector]]]:
    half = CUBE_HALF_EXTENT
    specs = []
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        for sign in (-1.0, 1.0):
            normal_values = [0.0, 0.0, 0.0]
            normal_values[axis_index] = sign
            normal = Vector(normal_values)
            corners = []
            free_indexes = [index for index in range(3) if index != axis_index]
            for first in (-half, half):
                for second in (-half, half):
                    values = [0.0, 0.0, 0.0]
                    values[axis_index] = sign * half
                    values[free_indexes[0]] = first
                    values[free_indexes[1]] = second
                    corners.append(Vector(values))
            specs.append((f"{axis_name}{'+' if sign > 0.0 else '-'}", normal, corners))
    return specs


def vector_projected_to_plane(vector: Vector, normal: Vector) -> Vector:
    return vector - normal * vector.dot(normal)


def fallback_face_down(normal: Vector) -> Vector:
    down = vector_projected_to_plane(Vector((0.0, 0.0, -1.0)), normal)
    if down.length <= 1e-6:
        down = vector_projected_to_plane(Vector((0.0, -1.0, 0.0)), normal)
    if down.length <= 1e-6:
        down = vector_projected_to_plane(Vector((1.0, 0.0, 0.0)), normal)
    down.normalize()
    return down


def face_down_direction(normal: Vector, visible_normals: list[Vector]) -> Vector:
    down = Vector((0.0, 0.0, 0.0))
    for visible_normal in visible_normals:
        if visible_normal.dot(normal) < 0.999:
            down += visible_normal * CUBE_HALF_EXTENT

    down = vector_projected_to_plane(down, normal)
    if down.length <= 1e-6:
        return fallback_face_down(normal)

    down.normalize()
    return down


def order_face_quad_items(
    world_points: list[Vector],
    projected_points: list[tuple[float, float]],
    visibilities: list[int],
    normal: Vector,
    down: Vector,
) -> tuple[list[tuple[float, float]], list[int]]:
    center = sum(world_points, Vector((0.0, 0.0, 0.0))) / len(world_points)
    right = normal.cross(down)
    if right.length <= 1e-6:
        right = vector_projected_to_plane(Vector((1.0, 0.0, 0.0)), normal)
    right.normalize()

    indexed = []
    for index, world_point in enumerate(world_points):
        offset = world_point - center
        x = offset.dot(right)
        y = offset.dot(down)
        indexed.append((index, x, y, math.atan2(y, x)))

    ordered = sorted(indexed, key=lambda item: item[3])
    start = min(range(len(ordered)), key=lambda index: (ordered[index][2], ordered[index][1]))
    ordered = ordered[start:] + ordered[:start]
    ordered_indexes = [index for index, _x, _y, _angle in ordered]
    return [projected_points[index] for index in ordered_indexes], [visibilities[index] for index in ordered_indexes]


def projected_cube_face_quads(
    objects: list[bpy.types.Object],
    width: int,
    height: int,
) -> list[dict[str, object]]:
    camera = get_camera()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cubie_names = {obj.name for obj in objects}
    face_candidates = []
    for face_name, normal, corners in cube_face_specs():
        face_center = normal * CUBE_HALF_EXTENT
        if (camera.location - face_center).dot(normal) <= 0.0:
            continue

        points = []
        visibilities = []
        for corner in corners:
            projected = project_world_point(corner, width, height)
            points.append(projected)
            visibilities.append(
                point_visibility_for_projection(
                    projected,
                    width,
                    height,
                    any_sample_reaches_cube_surface(
                        keypoint_visibility_samples(corner, face_center),
                        cubie_names,
                        depsgraph,
                    ),
                )
            )

        min_x = min(x for x, _y in points)
        max_x = max(x for x, _y in points)
        min_y = min(y for _x, y in points)
        max_y = max(y for _x, y in points)
        if max_x < 0.0 or min_x > width or max_y < 0.0 or min_y > height:
            continue

        face_candidates.append({
            "face": face_name,
            "normal": normal,
            "world_points": corners,
            "points": points,
            "visibilities": visibilities,
        })

    visible_normals = [candidate["normal"] for candidate in face_candidates]
    quads = []
    for candidate in face_candidates:
        down = face_down_direction(candidate["normal"], visible_normals)
        ordered_points, ordered_visibilities = order_face_quad_items(
            candidate["world_points"],
            candidate["points"],
            candidate["visibilities"],
            candidate["normal"],
            down,
        )
        quads.append({
            "face": candidate["face"],
            "points": ordered_points,
            "visibilities": ordered_visibilities,
            "keypoint_order": "toward_visible_face_intersection",
        })
    return quads


def yolo_pose_line_for_quad(
    class_id: int,
    points: list[tuple[float, float]],
    visibilities: list[int],
    width: int,
    height: int,
) -> str:
    min_x = max(0.0, min(x for x, _y in points))
    max_x = min(float(width), max(x for x, _y in points))
    min_y = max(0.0, min(y for _x, y in points))
    max_y = min(float(height), max(y for _x, y in points))

    center_x = ((min_x + max_x) / 2.0) / width
    center_y = ((min_y + max_y) / 2.0) / height
    box_width = (max_x - min_x) / width
    box_height = (max_y - min_y) / height
    values = [
        str(class_id),
        f"{center_x:.6f}",
        f"{center_y:.6f}",
        f"{box_width:.6f}",
        f"{box_height:.6f}",
    ]

    for (x, y), visibility in zip(points, visibilities):
        nx, ny = normalize_point(x, y, width, height)
        values.extend((f"{nx:.6f}", f"{ny:.6f}", str(visibility)))
    return " ".join(values)


def face_pose_labels(
    objects: list[bpy.types.Object],
    width: int,
    height: int,
) -> tuple[list[dict[str, object]], list[str]]:
    face_quads = projected_cube_face_quads(objects, width, height)
    lines = [
        yolo_pose_line_for_quad(
            FACE_CLASS_ID,
            face_quad["points"],
            face_quad["visibilities"],
            width,
            height,
        )
        for face_quad in face_quads
    ]
    return face_quads, lines


def projected_polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0

    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        * 0.5
    )


def bilinear_quad_point(points: list[Vector], u: float, v: float) -> Vector:
    top = points[0].lerp(points[1], u)
    bottom = points[3].lerp(points[2], u)
    return top.lerp(bottom, v)


def sticker_visible_sample_count(
    quad_world: list[Vector],
    cubie_names: set[str],
    depsgraph: bpy.types.Depsgraph,
) -> int:
    visible_samples = 0
    for row in range(STICKER_VISIBLE_SAMPLE_GRID):
        v = (row + 0.5) / STICKER_VISIBLE_SAMPLE_GRID
        for column in range(STICKER_VISIBLE_SAMPLE_GRID):
            u = (column + 0.5) / STICKER_VISIBLE_SAMPLE_GRID
            if ray_reaches_cube_surface(
                bilinear_quad_point(quad_world, u, v),
                cubie_names,
                depsgraph,
            ):
                visible_samples += 1
    return visible_samples


def cube_face_grid_specs() -> list[tuple[str, Vector, int, int, int]]:
    specs = []
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        free_indexes = [index for index in range(3) if index != axis_index]
        for sign in (-1.0, 1.0):
            normal_values = [0.0, 0.0, 0.0]
            normal_values[axis_index] = sign
            specs.append((
                f"{axis_name}{'+' if sign > 0.0 else '-'}",
                Vector(normal_values),
                axis_index,
                free_indexes[0],
                free_indexes[1],
            ))
    return specs


def grid_sticker_class_name(row: int, column: int) -> str:
    if row == 1 and column == 1:
        return "center_sticker"
    if row == 1 or column == 1:
        return "edge_sticker"
    return "corner_sticker"


def grid_cell_world_quad(
    axis_index: int,
    u_index: int,
    v_index: int,
    sign: float,
    row: int,
    column: int,
) -> list[Vector]:
    u0 = CUBE_GRID_LINES[column]
    u1 = CUBE_GRID_LINES[column + 1]
    v0 = CUBE_GRID_LINES[row]
    v1 = CUBE_GRID_LINES[row + 1]
    corners = []
    for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
        values = [0.0, 0.0, 0.0]
        values[axis_index] = sign * CUBE_HALF_EXTENT
        values[u_index] = u
        values[v_index] = v
        corners.append(Vector(values))
    return corners


def axis_vector(axis_index: int, sign: float = 1.0) -> Vector:
    values = [0.0, 0.0, 0.0]
    values[axis_index] = sign
    return Vector(values)


def order_quad_relative_to_face_center(
    quad_world: list[Vector],
    normal: Vector,
    v_index: int,
) -> list[Vector]:
    center = sum(quad_world, Vector((0.0, 0.0, 0.0))) / len(quad_world)
    face_center = normal * CUBE_HALF_EXTENT
    down = face_center - center
    down = down - normal * down.dot(normal)
    if down.length <= 1e-6:
        down = axis_vector(v_index)
        down = down - normal * down.dot(normal)
    down.normalize()

    right = normal.cross(down)
    if right.length <= 1e-6:
        right = axis_vector(0)
        right = right - normal * right.dot(normal)
    right.normalize()

    indexed = []
    for index, point in enumerate(quad_world):
        offset = point - center
        x = offset.dot(right)
        y = offset.dot(down)
        indexed.append((index, x, y, math.atan2(y, x)))

    ordered = sorted(indexed, key=lambda item: item[3])
    start = min(range(len(ordered)), key=lambda index: (ordered[index][2], ordered[index][1]))
    ordered = ordered[start:] + ordered[:start]
    return [quad_world[index] for index, _x, _y, _angle in ordered]


def grid_sticker_poses(
    objects: list[bpy.types.Object],
    width: int,
    height: int,
    visible_face_names: set[str],
) -> list[dict[str, object]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cubie_names = {obj.name for obj in objects}
    poses = []

    for face_name, normal, axis_index, u_index, v_index in cube_face_grid_specs():
        if face_name not in visible_face_names:
            continue

        sign = 1.0 if normal[axis_index] > 0.0 else -1.0
        for row in range(3):
            for column in range(3):
                quad_world = grid_cell_world_quad(axis_index, u_index, v_index, sign, row, column)
                quad_world = order_quad_relative_to_face_center(quad_world, normal, v_index)
                quad_points = [project_world_point(point, width, height) for point in quad_world]
                projected_area = projected_polygon_area(quad_points)
                if projected_area <= 0.0:
                    continue

                visible_samples = sticker_visible_sample_count(quad_world, cubie_names, depsgraph)
                visible_fraction = visible_samples / (STICKER_VISIBLE_SAMPLE_GRID * STICKER_VISIBLE_SAMPLE_GRID)
                if visible_samples < MIN_STICKER_VISIBLE_SAMPLES or visible_fraction < MIN_STICKER_VISIBLE_FRACTION:
                    continue

                center = sum(quad_world, Vector((0.0, 0.0, 0.0))) / len(quad_world)
                visibilities = [
                    point_visibility_for_projection(
                        projected_point,
                        width,
                        height,
                        any_sample_reaches_cube_surface(
                            keypoint_visibility_samples(world_point, center),
                            cubie_names,
                            depsgraph,
                        ),
                    )
                    for world_point, projected_point in zip(quad_world, quad_points)
                ]

                class_name = grid_sticker_class_name(row, column)
                class_id = CLASS_NAMES.index(class_name)
                poses.append({
                    "mask_value": GRID_STICKER_MASK_VALUE_BY_CLASS[class_name],
                    "class_id": class_id,
                    "class": class_name,
                    "face": face_name,
                    "row": row,
                    "column": column,
                    "keypoint_order": "relative_to_face_center",
                    "points": quad_points,
                    "visibilities": visibilities,
                    "projected_area": projected_area,
                    "visible_samples": visible_samples,
                    "visible_fraction": visible_fraction,
                })
    return poses


def sticker_pose_line(
    class_id: int,
    points: list[tuple[float, float]],
    visibilities: list[int],
    width: int,
    height: int,
) -> str:
    min_x = max(0.0, min(x for x, _y in points))
    max_x = min(float(width), max(x for x, _y in points))
    min_y = max(0.0, min(y for _x, y in points))
    max_y = min(float(height), max(y for _x, y in points))

    center_x = ((min_x + max_x) / 2.0) / width
    center_y = ((min_y + max_y) / 2.0) / height
    box_width = (max_x - min_x) / width
    box_height = (max_y - min_y) / height
    values = [
        str(class_id),
        f"{center_x:.6f}",
        f"{center_y:.6f}",
        f"{box_width:.6f}",
        f"{box_height:.6f}",
    ]

    for (x, y), visibility in zip(points, visibilities):
        nx, ny = normalize_point(x, y, width, height)
        values.extend((f"{nx:.6f}", f"{ny:.6f}", str(visibility)))
    return " ".join(values)


def sticker_pose_labels(poses: list[dict[str, object]], width: int, height: int) -> list[str]:
    return [
        sticker_pose_line(
            int(pose["class_id"]),
            pose["points"],
            pose["visibilities"],
            width,
            height,
        )
        for pose in poses
    ]


def write_pose_labels(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def draw_pose_corner_visibility(mask, points, visibilities, cv2) -> None:
    white = (255, 255, 255) if len(mask.shape) == 3 else 255
    black = (0, 0, 0) if len(mask.shape) == 3 else 0
    gray = (160, 160, 160) if len(mask.shape) == 3 else 160
    for point, visibility in zip(points.reshape(-1, 2), visibilities):
        if visibility == 0:
            continue

        x, y = (int(value) for value in point)
        if visibility == 2:
            cv2.circle(mask, (x, y), radius=3, color=white, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(mask, (x, y), radius=4, color=black, thickness=1, lineType=cv2.LINE_AA)
            continue

        cv2.circle(mask, (x, y), radius=4, color=gray, thickness=1, lineType=cv2.LINE_AA)
        cv2.line(mask, (x - 3, y - 3), (x + 3, y + 3), color=gray, thickness=1, lineType=cv2.LINE_AA)
        cv2.line(mask, (x - 3, y + 3), (x + 3, y - 3), color=gray, thickness=1, lineType=cv2.LINE_AA)


def draw_sticker_orientation_arrow(mask, points, pose: dict[str, object], cv2) -> None:
    class_name = pose.get("class")
    if class_name not in {"edge_sticker", "corner_sticker"}:
        return

    flat_points = points.reshape(-1, 2)
    if len(flat_points) < 4:
        return

    if class_name == "edge_sticker":
        start_point = (flat_points[0] + flat_points[1]) / 2.0
        end_point = (flat_points[2] + flat_points[3]) / 2.0
    else:
        start_point = flat_points[0]
        end_point = flat_points[2]

    start = tuple(int(round(value)) for value in start_point)
    end = tuple(int(round(value)) for value in end_point)
    cv2.arrowedLine(mask, start, end, color=(0, 0, 255), thickness=1, line_type=cv2.LINE_AA, tipLength=0.25)


def draw_pose_point_zero_marker(mask, points, cv2) -> None:
    flat_points = points.reshape(-1, 2)
    if len(flat_points) == 0:
        return

    x, y = (int(value) for value in flat_points[0])
    cv2.circle(mask, (x, y), radius=6, color=(255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(mask, (x, y), radius=7, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


def write_sticker_pose_mask(path: Path, poses: list[dict[str, object]], width: int, height: int) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "OpenCV and NumPy are required to draw sticker pose masks. "
            "Install cv2 in Blender's Python environment before generating training data."
        ) from None

    mask = np.zeros((height, width, 3), dtype=np.uint8)
    for pose in poses:
        points = np.asarray(pose["points"], dtype=np.float32)
        points = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
        mask_value = int(pose["mask_value"])
        cv2.fillConvexPoly(mask, points, (mask_value, mask_value, mask_value))
        cv2.polylines(mask, [points], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        draw_sticker_orientation_arrow(mask, points, pose, cv2)
        draw_pose_corner_visibility(mask, points, pose["visibilities"], cv2)

    cv2.imwrite(str(path), mask)


def draw_face_quads_on_mask(path: Path, face_quads: list[dict[str, object]]) -> None:
    if not face_quads:
        return

    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "OpenCV and NumPy are required to draw face pose masks. "
            "Install cv2 in Blender's Python environment before generating training data."
        ) from None

    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read mask for face pose visualization: {path}")
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for face_quad in face_quads:
        points = np.asarray(face_quad["points"], dtype=np.float32)
        points = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(mask, [points], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        draw_pose_corner_visibility(mask, points, face_quad["visibilities"], cv2)
        draw_pose_point_zero_marker(mask, points, cv2)

    cv2.imwrite(str(path), mask)


def generate_dataset(
    output_dir: Path,
    count: int,
    scramble_length: int,
    width: int,
    height: int,
    background_cubie_probability: float,
    max_background_cubies: int,
    cube_rotational_blur_probability: float,
    max_cube_rotational_blur_degrees: float,
    cube_rotational_blur_samples: int,
    render_masks: bool,
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
        moves, background_cubies, camera_randomization = randomize_scene(
            scramble_length,
            background_cubie_probability=background_cubie_probability,
            max_background_cubies=max_background_cubies,
        )
        objects = cube_objects()

        stem = f"{(index):06d}"
        image_path = images_dir / f"{stem}.png"
        mask_path = masks_dir / f"{stem}.png" if masks_dir is not None else None
        label_path = labels_dir / f"{stem}.txt"

        hands = hand_objects()
        hand_states = hand_object_states(hands)
        hand_randomization = randomize_hands(hands, hand_states)
        try:
            rotational_blur = render_cube_rotational_blur_png(
                image_path,
                objects,
                probability=cube_rotational_blur_probability,
                max_degrees=max_cube_rotational_blur_degrees,
                samples=cube_rotational_blur_samples,
            )
            face_quads, face_lines = face_pose_labels(objects, width, height)
            visible_face_names = {str(face_quad["face"]) for face_quad in face_quads}
            sticker_poses = grid_sticker_poses(
                objects,
                width,
                height,
                visible_face_names,
            )
            sticker_lines = sticker_pose_labels(sticker_poses, width, height)
            yolo_lines = face_lines + sticker_lines
            write_pose_labels(label_path, yolo_lines)
            if mask_path is not None:
                write_sticker_pose_mask(mask_path, sticker_poses, width, height)
                draw_face_quads_on_mask(mask_path, face_quads)
        finally:
            restore_hand_objects(hands, hand_states)

        metadata.append({
            "image": str(image_path),
            "mask": str(mask_path) if mask_path is not None else None,
            "label": str(label_path),
            "moves": moves,
            "cubie_labels": {obj.name: obj.pass_index for obj in objects},
            "camera": camera_randomization,
            "background_cubies": background_cubies,
            "cube_rotational_blur": rotational_blur,
            "hands": hand_randomization,
            "visible_faces": face_quads,
            "sticker_poses": sticker_poses,
            "sticker_mask_values": {
                str(mask_value): {
                    "sticker_color_index": sticker_color_index,
                    "cubie_pass_index": cubie_value,
                    "material_pass_index": sticker_color_index + 1,
                    "class": CLASS_NAMES[MASK_VALUE_TO_CLASS[mask_value]],
                }
                for mask_value, (sticker_color_index, cubie_value) in MASK_VALUE_TO_STICKER.items()
            },
            "poses": len(yolo_lines),
        })
        mask_message = f", {mask_path}" if mask_path is not None else ""
        print(
            f"Wrote {image_path}{mask_message}, and {label_path} "
            f"with {len(face_lines)} face poses, {len(sticker_lines)} sticker poses, "
            f"{background_cubies} background cubies "
            f"and {'rotational blur' if rotational_blur else 'no rotational blur'}"
        )

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "path: " + str(output_dir.resolve()) + "\n"
        "train: images\n"
        "val: images\n"
        "kpt_shape: [4, 3]\n"
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
        "--background-cubie-probability",
        type=float,
        default=DEFAULT_BACKGROUND_CUBIE_PROBABILITY,
        help="Probability of adding unlabeled individual cubie distractors behind the cube.",
    )
    parser.add_argument(
        "--max-background-cubies",
        type=int,
        default=DEFAULT_MAX_BACKGROUND_CUBIES,
        help="Maximum number of unlabeled background cubie distractors when enabled.",
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
        "--no-masks",
        action="store_true",
        help="Skip writing visualization masks; pose labels still use the internal raycast mask.",
    )

    argv = sys.argv
    if "--" not in argv:
        return parser.parse_args([])
    return parser.parse_args(argv[argv.index("--") + 1 :])


def main() -> None:
    args = cli_args()
    if args.seed is not None:
        random.seed(args.seed)

    generate_dataset(
        output_dir=args.output_dir,
        count=args.count,
        scramble_length=args.scramble_length,
        width=args.width,
        height=args.height,
        background_cubie_probability=args.background_cubie_probability,
        max_background_cubies=args.max_background_cubies,
        cube_rotational_blur_probability=args.cube_rotational_blur_probability,
        max_cube_rotational_blur_degrees=args.max_cube_rotational_blur_degrees,
        cube_rotational_blur_samples=args.cube_rotational_blur_samples,
        render_masks=not args.no_masks,
    )


if __name__ == "__main__":
    main()
