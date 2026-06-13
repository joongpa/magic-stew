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
import struct
import sys
import tempfile
import urllib.request
import zlib

import bpy
from mathutils import Quaternion, Vector

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

SOLVED_LOCATION_PROP = "synthetic_solved_location"
SOLVED_ROTATION_PROP = "synthetic_solved_rotation"
DEFAULT_OUTPUT_DIR = "synthetic_cube_dataset"
DEFAULT_IMAGE_SIZE = 512
CLASS_NAMES = ["center_sticker", "edge_sticker", "corner_sticker"]
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
    value: class_id
    for class_id, class_name in enumerate(CLASS_NAMES)
    for value in INSTANCE_VALUES[class_name.removesuffix("_sticker")]
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
MIN_SEGMENT_PIXELS = 8


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


def randomize_camera() -> None:
    camera = get_camera()
    camera.location = random_point_on_shell(CAMERA_DISTANCE_RANGE, CAMERA_ELEVATION_DEGREES)
    camera.data.lens = CAMERA_LENS_MM
    look_at(camera, random_target())


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


def randomize_scene(
    scramble_length: int = DEFAULT_SCRAMBLE_LENGTH,
    background_cubie_probability: float = DEFAULT_BACKGROUND_CUBIE_PROBABILITY,
    max_background_cubies: int = DEFAULT_MAX_BACKGROUND_CUBIES,
) -> tuple[list[str], int]:
    clear_background_cubies()
    objects = cube_objects()
    reset_cube(objects)

    randomize_camera()
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
    return moves, background_cubies


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


def write_grayscale_png(path: Path, pixels: bytearray, width: int, height: int) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw_rows = bytearray()
    row_stride = width
    for y in range(height):
        raw_rows.append(0)
        start = y * row_stride
        raw_rows.extend(pixels[start : start + row_stride])

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk("IHDR".encode(), struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += chunk("IDAT".encode(), zlib.compress(bytes(raw_rows), level=6))
    png += chunk("IEND".encode(), b"")
    path.write_bytes(png)


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


def material_pass_index_for_hit(
    hit_object: bpy.types.Object,
    polygon_index: int,
    depsgraph: bpy.types.Depsgraph,
) -> int:
    actual_hit_object = hit_object.evaluated_get(depsgraph)
    if polygon_index < 0 or polygon_index >= len(actual_hit_object.data.polygons):
        return 0
    material_index = actual_hit_object.data.polygons[polygon_index].material_index
    if material_index < 0 or material_index >= len(actual_hit_object.material_slots):
        return 0

    material = actual_hit_object.material_slots[material_index].material

    if material is None:
        return 0
    return int(material.pass_index)


def sticker_mask_value_for_hit(
    hit_object: bpy.types.Object,
    polygon_index: int,
    depsgraph: bpy.types.Depsgraph,
) -> int:
    cubie_value = int(hit_object.pass_index)
    material_value = material_pass_index_for_hit(hit_object, polygon_index, depsgraph)
    sticker_color_index = STICKER_COLOR_INDEX_BY_MATERIAL_PASS_INDEX.get(material_value)
    if sticker_color_index is None:
        return 0
    return STICKER_TO_MASK_VALUE.get((sticker_color_index, cubie_value), 0)


def render_mask(path: Path, objects: list[bpy.types.Object]) -> bytearray:
    scene = bpy.context.scene
    camera = get_camera()
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cubie_names = {obj.name for obj in objects}
    pixels = bytearray(width * height)

    bpy.context.view_layer.update()
    for y in range(height):
        for x in range(width):
            origin, direction = camera_ray(camera, scene, x, y)
            hit, _location, _normal, polygon_index, hit_object, _matrix = scene.ray_cast(
                depsgraph,
                origin,
                direction,
            )
            if hit and hit_object is not None and hit_object.name in cubie_names:
                pixels[y * width + x] = sticker_mask_value_for_hit(
                    hit_object,
                    polygon_index,
                    depsgraph,
                )

    write_grayscale_png(path, pixels, width, height)
    return pixels


def normalize_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return (
        max(0.0, min(x / width, 1.0)),
        max(0.0, min(y / height, 1.0)),
    )


def yolo_segments_with_cv2(pixels: bytearray, width: int, height: int) -> list[str]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "OpenCV is required to write non-rectangular YOLO segmentation labels. "
            "Install cv2 in Blender's Python environment before generating training data."
        ) from None

    mask = np.frombuffer(bytes(pixels), dtype=np.uint8).reshape((height, width))
    lines = []
    for label in sorted(int(value) for value in np.unique(mask) if int(value) in MASK_VALUE_TO_CLASS):
        binary = (mask == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < MIN_SEGMENT_PIXELS:
                continue

            epsilon = 0.0008 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(approx) < 3:
                continue

            values = [str(MASK_VALUE_TO_CLASS[label])]
            for x, y in approx:
                nx, ny = normalize_point(float(x), float(y), width, height)
                values.append(f"{nx:.6f}")
                values.append(f"{ny:.6f}")
            lines.append(" ".join(values))
    return lines


def write_yolo_segments(path: Path, pixels: bytearray, width: int, height: int) -> list[str]:
    lines = yolo_segments_with_cv2(pixels, width, height)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return lines


def generate_dataset(
    output_dir: Path,
    count: int,
    scramble_length: int,
    width: int,
    height: int,
    background_cubie_probability: float,
    max_background_cubies: int,
) -> None:
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    configure_render(width, height)

    metadata = []
    for index in range(count):
        moves, background_cubies = randomize_scene(
            scramble_length,
            background_cubie_probability=background_cubie_probability,
            max_background_cubies=max_background_cubies,
        )
        objects = cube_objects()

        stem = f"{(index):06d}"
        image_path = images_dir / f"{stem}.png"
        mask_path = masks_dir / f"{stem}.png"
        label_path = labels_dir / f"{stem}.txt"

        render_png(image_path)
        mask_pixels = render_mask(mask_path, objects)
        yolo_lines = write_yolo_segments(label_path, mask_pixels, width, height)

        metadata.append({
            "image": str(image_path),
            "mask": str(mask_path),
            "label": str(label_path),
            "moves": moves,
            "cubie_labels": {obj.name: obj.pass_index for obj in objects},
            "background_cubies": background_cubies,
            "sticker_mask_values": {
                str(mask_value): {
                    "sticker_color_index": sticker_color_index,
                    "cubie_pass_index": cubie_value,
                    "material_pass_index": sticker_color_index + 1,
                    "class": CLASS_NAMES[MASK_VALUE_TO_CLASS[mask_value]],
                }
                for mask_value, (sticker_color_index, cubie_value) in MASK_VALUE_TO_STICKER.items()
            },
            "segments": len(yolo_lines),
        })
        print(
            f"Wrote {image_path}, {mask_path}, and {label_path} "
            f"with {background_cubies} background cubies"
        )

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "path: " + str(output_dir.resolve()) + "\n"
        "train: images\n"
        "val: images\n"
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
    )


if __name__ == "__main__":
    main()
