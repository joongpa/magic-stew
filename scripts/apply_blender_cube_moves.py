"""Apply Rubik's Cube face turns and scene randomization for synthetic data.

Run inside Blender, for example:

    Blender /path/to/cube.blend --background --python scripts/apply_blender_cube_moves.py -- --count 100 --output-dir ./synthetic_cube

Assumptions:
- The cube is modeled as separate cubie objects, one object per movable cubie.
- Object origins are centered on their cubies.
- The cube core is at the world origin.
- Cubies are 1 meter wide, so cubie origins sit on the -1/0/1 grid.
- Cubie objects use quaternion rotation mode.
- Cubie object pass_index values are configured as grayscale mask labels.
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
from mathutils import Quaternion, Vector


# Common Blender convention: X right, Z up, and negative Y faces the camera/front.
# Use "+Y" instead if your modeled front face is on positive Y.
FRONT_AXIS = "-Y"

# Random scene defaults for synthetic data generation.
DEFAULT_SCRAMBLE_LENGTH = 100
CAMERA_DISTANCE_RANGE = (5.2, 7.2)
CAMERA_ELEVATION_DEGREES = (12.0, 42.0)
CAMERA_TARGET_JITTER = 0.25
CAMERA_LENS_MM = 35.0

LIGHT_DISTANCE_RANGE = (3.0, 6.0)
LIGHT_ELEVATION_DEGREES = (20.0, 70.0)
BACKGROUND_NAME = "SyntheticDataBackground"
BACKGROUND_MATERIAL_NAME = "SyntheticDataBackgroundMaterial"
MASK_MATERIAL_PREFIX = "SyntheticDataMaskValue"
BACKGROUND_DISTANCE = 4.0
BACKGROUND_SIZE = 12.0
BACKGROUND_IMAGE_URL = "https://picsum.photos/256/256"
BACKGROUND_IMAGE_PATH = f"{tempfile.gettempdir()}/synthetic_cube_background.jpg"

SOLVED_LOCATION_PROP = "synthetic_solved_location"
SOLVED_ROTATION_PROP = "synthetic_solved_rotation"
DEFAULT_OUTPUT_DIR = "synthetic_cube_dataset"
DEFAULT_IMAGE_SIZE = 512


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


def randomize_scene(scramble_length: int = DEFAULT_SCRAMBLE_LENGTH) -> list[str]:
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
    return moves


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


def mask_material(label: int) -> bpy.types.Material:
    material_name = f"{MASK_MATERIAL_PREFIX}_{label:03d}"
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(material_name)
        material.use_nodes = True

        nodes = material.node_tree.nodes
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        material.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])

    emission = material.node_tree.nodes.get("Emission")
    if emission is not None:
        value = max(0, min(label, 255)) / 255.0
        emission.inputs["Color"].default_value = (value, value, value, 1.0)
        emission.inputs["Strength"].default_value = 1.0

    return material


def use_object_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    if not obj.material_slots:
        obj.data.materials.append(material)

    for slot in obj.material_slots:
        slot.link = "OBJECT"
        slot.material = material


def render_mask(path: Path, objects: list[bpy.types.Object]) -> None:
    scene = bpy.context.scene
    world = scene.world
    background = bpy.data.objects.get(BACKGROUND_NAME)
    original_world_color = tuple(world.color) if world is not None else None
    original_film_transparent = scene.render.film_transparent
    original_color_mode = scene.render.image_settings.color_mode
    original_view_transform = scene.view_settings.view_transform
    original_look = scene.view_settings.look
    original_exposure = scene.view_settings.exposure
    original_gamma = scene.view_settings.gamma
    mask_objects = objects + ([background] if background is not None else [])
    original_materials = {
        obj.name: [
            {
                "material": slot.material,
                "link": slot.link,
            }
            for slot in obj.material_slots
        ]
        for obj in mask_objects
    }
    background_was_hidden = background.hide_render if background is not None else None

    try:
        if world is not None:
            world.color = (0.0, 0.0, 0.0)
        scene.render.film_transparent = False
        scene.render.image_settings.color_mode = "RGB"
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        if background is not None:
            background.hide_render = False
            use_object_material(background, mask_material(0))

        for obj in objects:
            use_object_material(obj, mask_material(obj.pass_index))

        render_png(path)
    finally:
        for obj in mask_objects:
            if not original_materials[obj.name]:
                for slot in obj.material_slots:
                    slot.material = None
                continue

            for slot, saved_slot in zip(obj.material_slots, original_materials[obj.name]):
                slot.material = saved_slot["material"]
                slot.link = saved_slot["link"]

        if background is not None and background_was_hidden is not None:
            background.hide_render = background_was_hidden
        if world is not None and original_world_color is not None:
            world.color = original_world_color
        scene.render.film_transparent = original_film_transparent
        scene.render.image_settings.color_mode = original_color_mode
        scene.view_settings.view_transform = original_view_transform
        scene.view_settings.look = original_look
        scene.view_settings.exposure = original_exposure
        scene.view_settings.gamma = original_gamma
        bpy.context.view_layer.update()


def generate_dataset(
    output_dir: Path,
    count: int,
    scramble_length: int,
    width: int,
    height: int,
) -> None:
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    configure_render(width, height)

    metadata = []
    for index in range(count):
        moves = randomize_scene(scramble_length)
        objects = cube_objects()

        stem = f"{(840 + index):06d}"
        image_path = images_dir / f"{stem}.png"
        mask_path = masks_dir / f"{stem}.png"

        render_png(image_path)
        render_mask(mask_path, objects)

        metadata.append({
            "image": str(image_path),
            "mask": str(mask_path),
            "moves": moves,
            "labels": {obj.name: obj.pass_index for obj in objects},
        })
        print(f"Wrote {image_path} and {mask_path}")

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scramble-length", type=int, default=DEFAULT_SCRAMBLE_LENGTH)
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_SIZE)

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
    )


if __name__ == "__main__":
    main()
