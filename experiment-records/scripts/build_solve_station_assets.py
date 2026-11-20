"""Build the high-detail local solve-station asset library used by synthetic renders.

Run with Blender, for example:

    Blender --background --python scripts/build_solve_station_assets.py -- \
        --output syntheticAssets/SolveStationAssets.blend

The asset is authored around a three-unit cube at the world origin.  Local +Y points
away from the camera; the timer and solver's torso are on +Y.  Hands
are deliberately omitted; the forearms end at rounded wrists outside the cube.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


ASSET_COLLECTION_NAME = "SolveStation_Complete"
ROOT_NAME = f"{ASSET_COLLECTION_NAME}_ROOT"


def cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("syntheticAssets/SolveStationAssets.blend"),
    )
    argv = sys.argv
    return parser.parse_args(argv[argv.index("--") + 1 :] if "--" in argv else [])


def clear_file() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)


def set_bsdf(material: bpy.types.Material, name: str, value: object) -> None:
    node = material.node_tree.nodes.get("Principled BSDF")
    if node is not None and name in node.inputs:
        node.inputs[name].default_value = value


def material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    roughness: float = 0.5,
    metallic: float = 0.0,
    specular: float = 0.35,
    emission: tuple[float, float, float, float] | None = None,
    emission_strength: float = 0.0,
    subsurface: float = 0.0,
) -> bpy.types.Material:
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    result.diffuse_color = color
    set_bsdf(result, "Base Color", color)
    set_bsdf(result, "Roughness", roughness)
    set_bsdf(result, "Metallic", metallic)
    set_bsdf(result, "Specular IOR Level", specular)
    set_bsdf(result, "Specular", specular)
    if emission is not None:
        set_bsdf(result, "Emission Color", emission)
        set_bsdf(result, "Emission", emission)
        set_bsdf(result, "Emission Strength", emission_strength)
    if subsurface:
        set_bsdf(result, "Subsurface Weight", subsurface)
        set_bsdf(result, "Subsurface", subsurface)
    return result


def add_fine_bump(
    mat: bpy.types.Material,
    *,
    scale: float,
    strength: float,
    distance: float,
) -> None:
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.65
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = distance
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def wood_material(name: str) -> bpy.types.Material:
    mat = material(name, (0.22, 0.075, 0.025, 1.0), roughness=0.42, specular=0.28)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.2
    noise.inputs["Detail"].default_value = 7.0
    noise.inputs["Roughness"].default_value = 0.62
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (0.35, 7.5, 2.0)
    texcoord = nodes.new("ShaderNodeTexCoord")
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.025, 0.007, 0.002, 1.0)
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[1].color = (0.42, 0.13, 0.035, 1.0)
    ramp.color_ramp.elements[1].position = 0.78
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.06
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def link_object(
    obj: bpy.types.Object,
    collection: bpy.types.Collection,
    root: bpy.types.Object,
) -> bpy.types.Object:
    for existing in list(obj.users_collection):
        existing.objects.unlink(obj)
    collection.objects.link(obj)
    obj.parent = root
    return obj


def smooth(obj: bpy.types.Object) -> None:
    if obj.type != "MESH":
        return
    for polygon in obj.data.polygons:
        polygon.use_smooth = True


def rounded_box(
    name: str,
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    mat: bpy.types.Material,
    collection: bpy.types.Collection,
    root: bpy.types.Object,
    *,
    bevel: float = 0.08,
    segments: int = 4,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel > 0.0:
        modifier = obj.modifiers.new("Edge softening", "BEVEL")
        modifier.width = min(bevel, min(dimensions) * 0.45)
        modifier.segments = segments
    obj.data.materials.append(mat)
    return link_object(obj, collection, root)


def disc(
    name: str,
    location: tuple[float, float, float],
    radius: float,
    depth: float,
    mat: bpy.types.Material,
    collection: bpy.types.Collection,
    root: bpy.types.Object,
    *,
    vertices: int = 64,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    modifier = obj.modifiers.new("Edge softening", "BEVEL")
    modifier.width = min(depth * 0.22, radius * 0.08)
    modifier.segments = 3
    smooth(obj)
    return link_object(obj, collection, root)


def tapered_segment(
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    start_radius: float,
    end_radius: float,
    mat: bpy.types.Material,
    collection: bpy.types.Collection,
    root: bpy.types.Object,
) -> bpy.types.Object:
    a = Vector(start)
    b = Vector(end)
    direction = b - a
    midpoint = (a + b) * 0.5
    bpy.ops.mesh.primitive_cone_add(
        vertices=64,
        radius1=start_radius,
        radius2=end_radius,
        depth=direction.length,
        location=midpoint,
    )
    obj = bpy.context.object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    obj.data.materials.append(mat)
    modifier = obj.modifiers.new("Soft silhouette", "BEVEL")
    modifier.width = 0.09
    modifier.segments = 3
    smooth(obj)
    return link_object(obj, collection, root)


def sphere(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    mat: bpy.types.Material,
    collection: bpy.types.Collection,
    root: bpy.types.Object,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    smooth(obj)
    return link_object(obj, collection, root)


def extruded_outline(
    name: str,
    outline: list[tuple[float, float]],
    y_front: float,
    depth: float,
    mat: bpy.types.Material,
    collection: bpy.types.Collection,
    root: bpy.types.Object,
) -> bpy.types.Object:
    front_y = y_front
    back_y = y_front + depth
    vertices = [(x, front_y, z) for x, z in outline] + [(x, back_y, z) for x, z in outline]
    count = len(outline)
    faces = [tuple(range(count)), tuple(range(count, count * 2))]
    for index in range(count):
        next_index = (index + 1) % count
        faces.append((index, next_index, count + next_index, count + index))
    mesh = bpy.data.meshes.new(f"{name}.mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.parent = root
    obj.data.materials.append(mat)
    modifier = obj.modifiers.new("Cloth silhouette", "BEVEL")
    modifier.width = 0.18
    modifier.segments = 5
    return obj


def create_materials() -> dict[str, bpy.types.Material]:
    mats = {
        "wood": wood_material("SolveStation.Material.DeskWood"),
        "edge": material("SolveStation.Material.DeskEdge", (0.018, 0.02, 0.022, 1), roughness=0.34),
        "mat": material("SolveStation.Material.MatFabric", (0.025, 0.045, 0.075, 1), roughness=0.93),
        "mat_print": material("SolveStation.Material.MatPrint", (0.03, 0.2, 0.34, 1), roughness=0.76),
        "lcd": material("SolveStation.Material.LCD", (0.018, 0.038, 0.035, 1), roughness=0.16, specular=0.62),
        "lcd_digit": material(
            "SolveStation.Material.LCDDigit",
            (0.21, 0.42, 0.27, 1),
            roughness=0.28,
            emission=(0.16, 0.42, 0.22, 1),
            emission_strength=0.35,
        ),
        "metal": material("SolveStation.Material.Metal", (0.21, 0.23, 0.26, 1), roughness=0.28, metallic=0.72),
        "keyboard": material("SolveStation.Material.Keyboard", (0.018, 0.019, 0.022, 1), roughness=0.35),
        "screen": material(
            "SolveStation.Material.LaptopScreen",
            (0.025, 0.07, 0.13, 1),
            roughness=0.18,
            emission=(0.04, 0.18, 0.42, 1),
            emission_strength=0.8,
        ),
        "phone": material("SolveStation.Material.PhoneBody", (0.035, 0.038, 0.044, 1), roughness=0.2, metallic=0.38),
        "phone_screen": material(
            "SolveStation.Material.PhoneScreen",
            (0.055, 0.025, 0.09, 1),
            roughness=0.12,
            emission=(0.18, 0.045, 0.35, 1),
            emission_strength=0.52,
        ),
        "camera_glass": material("SolveStation.Material.CameraGlass", (0.006, 0.012, 0.018, 1), roughness=0.08, metallic=0.42),
        "shirt": material("SolveStation.Material.Shirt", (0.018, 0.022, 0.03, 1), roughness=0.91),
        "shirt_seam": material("SolveStation.Material.ShirtSeam", (0.06, 0.07, 0.085, 1), roughness=0.88),
        "wall": material("SolveStation.Material.Wall", (0.34, 0.36, 0.38, 1), roughness=0.94),
        "baseboard": material("SolveStation.Material.Baseboard", (0.065, 0.07, 0.075, 1), roughness=0.64),
        "skin": material(
            "SolveStation.Material.Skin",
            (0.55, 0.25, 0.13, 1),
            roughness=0.52,
            specular=0.25,
            subsurface=0.08,
        ),
    }
    add_fine_bump(mats["mat"], scale=115.0, strength=0.2, distance=0.025)
    add_fine_bump(mats["shirt"], scale=145.0, strength=0.16, distance=0.018)
    add_fine_bump(mats["skin"], scale=95.0, strength=0.035, distance=0.012)
    return mats


def build_desk_and_mat(collection, root, mats) -> None:
    rounded_box(
        "SolveStation.Desk.Top",
        (0.0, 0.65, -1.82),
        (15.0, 10.5, 0.62),
        mats["wood"], collection, root,
        bevel=0.18, segments=6,
    )
    rounded_box(
        "SolveStation.Desk.FrontEdge",
        (0.0, -4.48, -1.78),
        (14.65, 0.18, 0.52),
        mats["edge"], collection, root,
        bevel=0.07,
    )
    mat_base = rounded_box(
        "SolveStation.Mat.Base",
        (0.0, -0.15, -1.5098),
        (9.8, 6.15, 0.001),
        mats["mat"], collection, root,
        bevel=0.0004, segments=2,
    )
    mat_base.visible_shadow = False
    center_print = rounded_box(
        "SolveStation.Mat.CenterPrint",
        (0.0, 0.18, -1.5092),
        (5.5, 3.8, 0.0002),
        mats["mat_print"], collection, root,
        bevel=0.00008, segments=2,
    )
    center_print.visible_shadow = False
    for side in (-1.0, 1.0):
        strip = rounded_box(
            f"SolveStation.Mat.Accent.{'L' if side < 0 else 'R'}",
            (side * 4.05, -0.05, -1.5092),
            (0.48, 4.7, 0.0002),
            mats["mat_print"], collection, root,
            bevel=0.00008,
        )
        strip.rotation_euler.z = math.radians(side * 8.0)
        strip.visible_shadow = False


def build_room_background(collection, root, mats) -> None:
    rounded_box(
        "SolveStation.Room.Wall",
        (0.0, 5.9, 2.55),
        (20.0, 0.34, 9.2),
        mats["wall"],
        collection,
        root,
        bevel=0.08,
    )
    rounded_box(
        "SolveStation.Room.Baseboard",
        (0.0, 5.7, -1.0),
        (20.0, 0.18, 0.55),
        mats["baseboard"],
        collection,
        root,
        bevel=0.06,
    )


def build_timer(collection, root, mats) -> None:
    # Blender --python does not put the script directory on sys.path.
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from stackmat_timer_assets import build_timer_variants

    build_timer_variants(collection, root, material)


def build_phone(collection, root, mats) -> None:
    x, y, z = 4.0, 0.65, -1.36
    body = rounded_box("SolveStation.Phone.Body", (x, y, z), (1.55, 3.05, 0.24), mats["phone"], collection, root, bevel=0.22, segments=9)
    body.rotation_euler.z = math.radians(-11.0)
    screen = rounded_box("SolveStation.Phone.Screen", (x, y, z + 0.135), (1.38, 2.82, 0.025), mats["phone_screen"], collection, root, bevel=0.16, segments=8)
    screen.rotation_euler.z = body.rotation_euler.z
    for index, offset in enumerate((-0.25, 0.0, 0.25)):
        line = rounded_box(
            f"SolveStation.Phone.UI.{index}",
            (x + offset * 0.65, y - 0.28 + index * 0.28, z + 0.16),
            (0.48, 0.08, 0.018),
            mats["lcd_digit"], collection, root,
            bevel=0.025,
        )
        line.rotation_euler.z = body.rotation_euler.z


def build_laptop(collection, root, mats) -> None:
    # The laptop is across the table from the solver. Its display faces +Y,
    # toward the solver, while the Stackmat remains on the solver's side.
    x, y = -5.0, -2.75
    rounded_box("SolveStation.Laptop.Base", (x, y, -1.35), (5.35, 3.55, 0.27), mats["metal"], collection, root, bevel=0.17, segments=7)
    rounded_box("SolveStation.Laptop.KeyboardInset", (x, y - 0.52, -1.195), (4.65, 1.68, 0.025), mats["keyboard"], collection, root, bevel=0.08)
    rounded_box("SolveStation.Laptop.Trackpad", (x, y + 0.92, -1.19), (2.0, 0.82, 0.022), mats["edge"], collection, root, bevel=0.08)
    rows, cols = 5, 12
    for row in range(rows):
        for col in range(cols):
            key_x = x - 2.05 + col * 0.37
            key_y = y - 1.12 + row * 0.34
            rounded_box(
                f"SolveStation.Laptop.Key.{row:02d}.{col:02d}",
                (key_x, key_y, -1.145),
                (0.29, 0.25, 0.055),
                mats["keyboard"], collection, root,
                bevel=0.045, segments=3,
            )
    screen_y, screen_z = y - 1.72, 0.28
    rounded_box("SolveStation.Laptop.ScreenBack", (x, screen_y, screen_z), (5.35, 0.19, 3.05), mats["metal"], collection, root, bevel=0.16, segments=7)
    rounded_box("SolveStation.Laptop.Screen", (x, screen_y + 0.105, screen_z), (4.86, 0.028, 2.58), mats["screen"], collection, root, bevel=0.08)
    disc("SolveStation.Laptop.Webcam", (x, screen_y + 0.125, screen_z + 1.37), 0.045, 0.025, mats["camera_glass"], collection, root, vertices=32)
    for side in (-1.0, 1.0):
        rounded_box(
            f"SolveStation.Laptop.Hinge.{'L' if side < 0 else 'R'}",
            (x + side * 1.85, screen_y + 0.02, -1.14),
            (0.68, 0.32, 0.24),
            mats["edge"], collection, root,
            bevel=0.09,
        )


def build_solver(collection, root, mats) -> None:
    # Only the chest and shoulders normally survive an ROI crop.  A deeply
    # beveled chest volume reads as a shirt from frontal and overhead cameras
    # without inventing a head or face outside this asset's scope.
    rounded_box(
        "SolveStation.Shirt.Torso",
        (0.0, 3.95, 0.25),
        (6.65, 1.3, 5.05),
        mats["shirt"],
        collection,
        root,
        bevel=0.55,
        segments=10,
    )
    for side, label in ((-1.0, "L"), (1.0, "R")):
        shoulder = (side * 2.95, 3.5, 1.75)
        elbow = (side * 3.25, 1.65, 0.2)
        # The wrist ends beneath the cube silhouette.  This keeps the handless
        # intermediate asset usable without exposing a conspicuous rounded stump.
        wrist = (side * 1.12, 0.08, -0.08)
        tapered_segment(f"SolveStation.Shirt.Sleeve.{label}", shoulder, elbow, 0.78, 0.62, mats["shirt"], collection, root)
        sphere(f"SolveStation.Shirt.SleeveJoint.{label}", elbow, (0.66, 0.66, 0.66), mats["shirt"], collection, root)
        tapered_segment(f"SolveStation.Arm.Forearm.{label}", elbow, wrist, 0.55, 0.29, mats["skin"], collection, root)
        sphere(f"SolveStation.Arm.Elbow.{label}", elbow, (0.55, 0.55, 0.55), mats["skin"], collection, root)
        sphere(f"SolveStation.Arm.Wrist.{label}", wrist, (0.29, 0.31, 0.29), mats["skin"], collection, root)


def build_asset() -> bpy.types.Collection:
    clear_file()
    scene = bpy.context.scene
    collection = bpy.data.collections.new(ASSET_COLLECTION_NAME)
    scene.collection.children.link(collection)
    root = bpy.data.objects.new(ROOT_NAME, None)
    collection.objects.link(root)
    mats = create_materials()
    build_desk_and_mat(collection, root, mats)
    build_room_background(collection, root, mats)
    build_timer(collection, root, mats)
    build_phone(collection, root, mats)
    build_laptop(collection, root, mats)
    build_solver(collection, root, mats)
    return collection


def add_preview_scene(asset_collection: bpy.types.Collection) -> None:
    preview = bpy.data.collections.new("PreviewOnly")
    bpy.context.scene.collection.children.link(preview)
    bpy.ops.object.camera_add(location=(9.6, -11.2, 7.3))
    camera = bpy.context.object
    camera.name = "Preview.Camera"
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = (Vector((0.0, 0.5, -0.25)) - camera.location).to_track_quat("-Z", "Y")
    camera.data.lens = 48.0
    for old in list(camera.users_collection):
        old.objects.unlink(camera)
    preview.objects.link(camera)
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type="AREA", location=(-3.5, -4.0, 8.5))
    key = bpy.context.object
    key.name = "Preview.Key"
    key.data.energy = 1250.0
    key.data.shape = "DISK"
    key.data.size = 4.0
    key.rotation_mode = "QUATERNION"
    key.rotation_quaternion = (Vector((0, 0, 0)) - key.location).to_track_quat("-Z", "Y")
    for old in list(key.users_collection):
        old.objects.unlink(key)
    preview.objects.link(key)

    bpy.ops.object.light_add(type="AREA", location=(5.0, 2.0, 5.5))
    fill = bpy.context.object
    fill.name = "Preview.Fill"
    fill.data.energy = 650.0
    fill.data.size = 3.0
    fill.rotation_mode = "QUATERNION"
    fill.rotation_quaternion = (Vector((0, 0, 0)) - fill.location).to_track_quat("-Z", "Y")
    for old in list(fill.users_collection):
        old.objects.unlink(fill)
    preview.objects.link(fill)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.025, 0.025, 0.032)
    scene.view_settings.look = "AgX - Medium High Contrast"


def main() -> None:
    args = cli_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    asset = build_asset()
    add_preview_scene(asset)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    print(f"Wrote {output} with {len(asset.all_objects)} solve-station objects")


if __name__ == "__main__":
    main()
