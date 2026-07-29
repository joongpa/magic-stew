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
FALSE_CUBIE_COLLECTION_NAME = "SyntheticFalseCubies"
FALSE_CUBIE_PREFIX = "SyntheticFalseCubie"
FALSE_CUBIE_COLOR_MATERIAL_NAMES = ("White", "Yellow", "Red", "Orange", "Blue", "Green")
DEFAULT_FALSE_CUBIE_PROBABILITY = 1.0
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
DEFAULT_OUTPUT_DIR = "data/synthetic_cube_dataset"
DEFAULT_IMAGE_SIZE = 512
CLASS_NAMES = ["corner", "edge"]
CORNER_CLASS_ID = 0
EDGE_CLASS_ID = 1
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
POSE_KEYPOINT_COUNT = 7
KEYPOINT_BBOX_RADIUS_RATIO = 0.018
MIN_KEYPOINT_BBOX_RADIUS_PX = 4
POINT_VISIBILITY_UNLABELED = 0
POINT_VISIBILITY_LABELED_OCCLUDED = 1
POINT_VISIBILITY_VISIBLE = 2


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


def clear_false_cubies() -> None:
    for obj in list(bpy.data.objects):
        if obj.name.startswith(FALSE_CUBIE_PREFIX):
            bpy.data.objects.remove(obj, do_unlink=True)


def background_cubie_collection() -> bpy.types.Collection:
    collection = bpy.data.collections.get(BACKGROUND_CUBIE_COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(BACKGROUND_CUBIE_COLLECTION_NAME)
        bpy.context.scene.collection.children.link(collection)
    return collection


def false_cubie_collection() -> bpy.types.Collection:
    collection = bpy.data.collections.get(FALSE_CUBIE_COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(FALSE_CUBIE_COLLECTION_NAME)
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


def camera_visible_face_labels(axes: dict[str, Vector], visible_normals: list[Vector]) -> set[str]:
    visible_keys = {axis_key(normal) for normal in visible_normals}
    return {
        face
        for face, axis in axes.items()
        if axis_key(axis) in visible_keys
    }


def cubie_outer_face_labels(obj: bpy.types.Object, axes: dict[str, Vector]) -> set[str]:
    return {
        face
        for face, axis in axes.items()
        if round(obj.location.dot(axis)) == 1
    }


def false_cubie_candidates(
    objects: list[bpy.types.Object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    visible_normals = visible_world_normals()
    axes = camera_relative_axes(visible_normals)
    visible_faces = camera_visible_face_labels(axes, visible_normals)
    candidates = []

    for obj in objects:
        cubie_faces = cubie_outer_face_labels(obj, axes)
        visible_cubie_faces = cubie_faces & visible_faces
        hidden_cubie_faces = cubie_faces - visible_faces
        if not visible_cubie_faces or not hidden_cubie_faces:
            continue
        for face in sorted(hidden_cubie_faces):
            candidates.append({
                "target": obj,
                "face": face,
                "axis": axes[face],
                "target_faces": sorted(cubie_faces),
                "target_visible_faces": sorted(visible_cubie_faces),
            })

    return candidates, {
        "visible_normals": [[normal.x, normal.y, normal.z] for normal in visible_normals],
        "axes": {name: [axis.x, axis.y, axis.z] for name, axis in axes.items()},
        "visible_faces": sorted(visible_faces),
        "candidate_count": len(candidates),
    }


def random_false_cubie_material() -> bpy.types.Material:
    missing = [
        name
        for name in FALSE_CUBIE_COLOR_MATERIAL_NAMES
        if bpy.data.materials.get(name) is None
    ]
    if missing:
        raise ValueError(
            "Missing false cubie color materials in Blender file: "
            + ", ".join(missing)
        )
    return bpy.data.materials[random.choice(FALSE_CUBIE_COLOR_MATERIAL_NAMES)]


def set_cubie_solid_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = 0


def randomize_false_cubie(
    source_objects: list[bpy.types.Object],
    probability: float,
) -> dict[str, object] | None:
    clear_false_cubies()
    if probability <= 0.0 or random.random() >= probability:
        return None

    candidates, geometry = false_cubie_candidates(source_objects)
    if not candidates:
        return {
            **geometry,
            "placed": False,
            "reason": "no_candidate_faces",
        }

    source = random.choice(source_objects)
    candidate = random.choice(candidates)
    target = candidate["target"]
    face = str(candidate["face"])
    axis = candidate["axis"]
    material = random_false_cubie_material()

    cubie = source.copy()
    cubie.data = source.data.copy()
    cubie.animation_data_clear()
    cubie.name = f"{FALSE_CUBIE_PREFIX}.000"
    cubie.location = target.location + axis
    cubie.rotation_mode = "QUATERNION"
    cubie.rotation_quaternion = source.rotation_quaternion.copy()
    cubie.scale = (1.0, 1.0, 1.0)
    set_cubie_solid_material(cubie, material)

    false_cubie_collection().objects.link(cubie)
    bpy.context.view_layer.update()

    return {
        **geometry,
        "placed": True,
        "name": cubie.name,
        "source": source.name,
        "target": target.name,
        "color": material.name,
        "face": face,
        "target_faces": candidate["target_faces"],
        "target_visible_faces": candidate["target_visible_faces"],
        "location": [cubie.location.x, cubie.location.y, cubie.location.z],
    }


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
    false_cubie_probability: float = DEFAULT_FALSE_CUBIE_PROBABILITY,
) -> tuple[list[str], int, dict[str, object], dict[str, object] | None]:
    clear_background_cubies()
    clear_false_cubies()
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
    false_cubie = randomize_false_cubie(
        objects,
        probability=false_cubie_probability,
    )
    return moves, background_cubies, camera_randomization, false_cubie


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
    return any(normal.dot(toward_camera) > 0.0 for normal in normals)


def ray_to_point(camera: bpy.types.Object, point: Vector) -> tuple[Vector, Vector, float]:
    origin = camera.matrix_world.translation
    offset = point - origin
    distance = offset.length
    return origin, offset.normalized(), distance


def hand_occludes_point(
    point: Vector,
    hand_names: set[str],
    depsgraph: bpy.types.Depsgraph,
    distance_tolerance: float = 0.02,
) -> bool:
    if not hand_names:
        return False
    scene = bpy.context.scene
    camera = get_camera()
    origin, direction, point_distance = ray_to_point(camera, point)
    hit, location, _normal, _polygon_index, hit_object, _matrix = scene.ray_cast(
        depsgraph,
        origin,
        direction,
        distance=point_distance + distance_tolerance,
    )
    if not hit or hit_object is None:
        return False
    if hit_object.name not in hand_names:
        return False
    return (location - origin).length < point_distance - distance_tolerance


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
    for index, spec in enumerate(cube_pose_keypoint_specs()):
        point = spec["point"]
        if not isinstance(point, Vector):
            continue
        name = str(spec["name"])
        normals = spec["normals"]
        projected = project_world_point(point, camera, scene)
        hand_occluded = False
        if projected is None:
            x, y = 0.0, 0.0
            visibility = POINT_VISIBILITY_UNLABELED
        elif not isinstance(normals, list) or not point_faces_camera(point, normals, camera.location):
            x, y = projected
            visibility = POINT_VISIBILITY_UNLABELED
        else:
            x, y = projected
            hand_occluded = hand_occludes_point(point, visible_hand_names, depsgraph)
            visibility = POINT_VISIBILITY_LABELED_OCCLUDED if hand_occluded else POINT_VISIBILITY_VISIBLE

        projected_by_name[name] = (x, y, visibility)
        metadata_by_name[name] = {
            "index": index,
            "name": name,
            "world_point": [point.x, point.y, point.z],
            "pixel_point": [x, y],
            "normalized_point": list(normalize_point(x, y, width, height)),
            "visibility": visibility,
            "hand_occluded": hand_occluded,
        }

    for piece_index, corner_name in enumerate([name for name in CUBE_KEYPOINT_NAMES if name.isupper()], start=1):
        main_name = canonical_cube_keypoint_name(corner_name)
        main_keypoint = projected_piece_keypoint(main_name, projected_by_name)
        ordered_faces = [
            face
            for face, _point in clockwise_keypoints(
                main_keypoint,
                [
                    (
                        face,
                        projected_piece_keypoint(
                            corner_keypoint_name_for_faces(corner_name, {face}),
                            projected_by_name,
                        ),
                    )
                    for face in corner_name
                ],
            )
        ]
        ordered_faces = rotate_sequence_to_start(ordered_faces, corner_name[0])
        ordered_edge_facing = [
            corner_keypoint_name_for_faces(corner_name, {ordered_faces[index - 1], face})
            for index, face in enumerate(ordered_faces)
        ]
        ordered_center_facing = [
            corner_keypoint_name_for_faces(corner_name, {face})
            for face in ordered_faces
        ]
        line, pose_object = pose_object_from_keypoints(
            class_id=CORNER_CLASS_ID,
            name=f"corner_{piece_index}",
            keypoint_names=[main_name, *ordered_edge_facing, *ordered_center_facing],
            projected_by_name=projected_by_name,
            metadata_by_name=metadata_by_name,
            width=width,
            height=height,
        )
        if line is not None and pose_object is not None:
            lines.append(line)
            pose_objects.append(pose_object)

    for piece_index, edge_spec in enumerate(edge_piece_name_specs(), start=1):
        edge_line_with_source_indexes = sorted(
            [
                (source_index, str(name), projected_piece_keypoint(str(name), projected_by_name))
                for source_index, name in enumerate(edge_spec["edge_line"])
            ],
            key=lambda item: left_to_right_key(item[2]),
        )
        edge_line_names = [name for _source_index, name, _point in edge_line_with_source_indexes]
        edge_line_source_indexes = [source_index for source_index, _name, _point in edge_line_with_source_indexes]
        center_side_names = [
            str(center_side[source_index])
            for center_side in edge_spec["center_sides"]
            for source_index in edge_line_source_indexes
        ]
        line, pose_object = pose_object_from_keypoints(
            class_id=EDGE_CLASS_ID,
            name=f"edge_{piece_index}",
            keypoint_names=[*edge_line_names, *center_side_names],
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
        "source_keypoints": KEYPOINT_COUNT,
        "visible_hands": sorted(visible_hand_names),
        "candidates": len(CUBE_KEYPOINT_NAMES),
        "piece_candidates": 20,
    }


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
        EDGE_CLASS_ID: (0, 95, 95),
    }
    point_colors = {
        CORNER_CLASS_ID: (255, 255, 255),
        EDGE_CLASS_ID: (0, 255, 255),
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

        label_anchor = None
        if labeled_points:
            label_anchor = np.asarray(labeled_points, dtype=np.float32).mean(axis=0)

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
            label = str(keypoint.get("label", int(keypoint["index"]) + 1))
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_point = np.asarray([float(x), float(y)], dtype=np.float32)
            if label_anchor is not None:
                label_point = (label_point * 0.65) + (label_anchor * 0.35)
            text_x = int(round(float(label_point[0]) - text_size[0] / 2))
            text_y = int(round(float(label_point[1]) + text_size[1] / 2))
            pad = 2
            cv2.rectangle(
                output,
                (text_x - pad, text_y - text_size[1] - pad),
                (text_x + text_size[0] + pad, text_y + baseline + pad),
                (0, 0, 0),
                thickness=-1,
            )
            cv2.putText(
                output,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                draw_color,
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(path), output)


def write_yolo_pose(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def generate_dataset(
    output_dir: Path,
    count: int,
    scramble_length: int,
    width: int,
    height: int,
    background_cubie_probability: float,
    max_background_cubies: int,
    false_cubie_probability: float,
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
        moves, background_cubies, camera_randomization, false_cubie = randomize_scene(
            scramble_length,
            background_cubie_probability=background_cubie_probability,
            max_background_cubies=max_background_cubies,
            false_cubie_probability=false_cubie_probability,
        )
        objects = cube_objects()
        render_objects = list(objects)
        if false_cubie is not None and false_cubie.get("placed"):
            if false_cubie_object := bpy.data.objects.get(str(false_cubie["name"])):
                render_objects.append(false_cubie_object)

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
                render_objects,
                probability=cube_rotational_blur_probability,
                max_degrees=max_cube_rotational_blur_degrees,
                samples=cube_rotational_blur_samples,
            )
            yolo_lines, pose_objects, pose_geometry = render_pose_keypoints(mask_path)
            write_yolo_pose(label_path, yolo_lines)
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
            "false_cubie": false_cubie,
            "cube_rotational_blur": rotational_blur,
            "hands": hand_randomization,
            "pose_geometry": pose_geometry,
            "keypoints": pose_objects,
            "objects": len(yolo_lines),
        })
        mask_message = f", {mask_path}" if mask_path is not None else ""
        print(
            f"Wrote {image_path}{mask_message}, and {label_path} "
            f"with {len(yolo_lines)} pose keypoint objects, "
            f"{background_cubies} background cubies "
            f"{'with' if false_cubie and false_cubie.get('placed') else 'without'} false cubie "
            f"and {'rotational blur' if rotational_blur else 'no rotational blur'}"
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
        "--false-cubie-probability",
        type=float,
        default=DEFAULT_FALSE_CUBIE_PROBABILITY,
        help=(
            "Probability of adding one unlabeled cubie attached to a hidden exterior "
            "face of a partially visible cubie."
        ),
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
        help="Skip writing visualization masks; segmentation labels are still written.",
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
        false_cubie_probability=args.false_cubie_probability,
        cube_rotational_blur_probability=args.cube_rotational_blur_probability,
        max_cube_rotational_blur_degrees=args.max_cube_rotational_blur_degrees,
        cube_rotational_blur_samples=args.cube_rotational_blur_samples,
        render_masks=not args.no_masks,
    )


if __name__ == "__main__":
    main()
