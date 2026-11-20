import math
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(bpy.path.abspath("//"))
TEXTURE_DIR = ROOT / "syntheticAssets" / "rigged_hand_textures"
OUTPUT_BLEND = ROOT / "rigged_hand_textured_direct.blend"
OUTPUT_PREVIEW = ROOT / "rigged_hand_textured_direct_preview.png"
TEXTURE_SIZE = 2048
TEXTURE_DIR.mkdir(parents=True, exist_ok=True)


def set_input(node, name, value):
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def point_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def new_bake_image(name, filename, colorspace):
    old = bpy.data.images.get(name)
    if old:
        bpy.data.images.remove(old)
    image = bpy.data.images.new(name, width=TEXTURE_SIZE, height=TEXTURE_SIZE, alpha=False)
    image.colorspace_settings.name = colorspace
    image.file_format = "PNG"
    image.filepath_raw = str(TEXTURE_DIR / filename)
    return image


def activate_bake_target(material, image, x=-50, y=-600):
    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    node.name = f"Bake Target - {image.name}"
    node.image = image
    node.location = (x, y)
    material.node_tree.nodes.active = node
    node.select = True
    return node


hand = max(
    (obj for obj in bpy.data.objects if obj.type == "MESH"),
    key=lambda obj: len(obj.data.vertices),
)
source_material = hand.data.materials[0]
nodes = source_material.node_tree.nodes
links = source_material.node_tree.links
output = next(node for node in nodes if node.type == "OUTPUT_MATERIAL")
skin = nodes.get("Skin Shader")
tone_ramp = nodes.get("Natural Skin Tones")
roughness = nodes.get("Roughness Variation")
if not all((skin, tone_ramp, roughness)):
    raise RuntimeError("Expected the prepared skin material before baking")

# Create an explicit UV layout on the hand mesh. Vertex groups, armature modifier,
# weights, shape, and pose are not changed by this operation.
bpy.context.view_layer.objects.active = hand
for obj in bpy.context.selected_objects:
    obj.select_set(False)
hand.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
if not hand.data.uv_layers:
    hand.data.uv_layers.new(name="SkinUV")
else:
    hand.data.uv_layers.active.name = "SkinUV"
bpy.ops.uv.smart_project(
    angle_limit=math.radians(55.0),
    island_margin=0.006,
    area_weight=0.15,
    correct_aspect=True,
    scale_to_bounds=True,
)
bpy.ops.object.mode_set(mode="OBJECT")

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 1
scene.render.bake.margin = 16
scene.render.bake.use_clear = True
scene.render.bake.normal_space = "TANGENT"

base_image = new_bake_image("Hand Base Color", "hand_basecolor.png", "sRGB")
rough_image = new_bake_image("Hand Roughness", "hand_roughness.png", "Non-Color")
normal_image = new_bake_image("Hand Normal", "hand_normal.png", "Non-Color")

# Bake base color through emission to capture color only, independent of lighting.
emission = nodes.new("ShaderNodeEmission")
emission.name = "Temporary Bake Emission"
links.new(tone_ramp.outputs["Color"], emission.inputs["Color"])
links.new(emission.outputs["Emission"], output.inputs["Surface"])
activate_bake_target(source_material, base_image)
bpy.ops.object.bake(type="EMIT")
base_image.save()
print(f"BAKED_BASE={base_image.filepath_raw}")

# Bake scalar roughness as a grayscale image.
links.new(roughness.outputs["Result"], emission.inputs["Color"])
activate_bake_target(source_material, rough_image)
bpy.ops.object.bake(type="EMIT")
rough_image.save()
print(f"BAKED_ROUGHNESS={rough_image.filepath_raw}")

# Restore the skin shader and bake its bump network to a tangent-space normal map.
links.new(skin.outputs["BSDF"], output.inputs["Surface"])
activate_bake_target(source_material, normal_image)
bpy.ops.object.bake(type="NORMAL")
normal_image.save()
print(f"BAKED_NORMAL={normal_image.filepath_raw}")

# Replace the entire procedural graph with a conventional UV/image material.
direct = bpy.data.materials.get("Baked Realistic Skin")
if direct is None:
    direct = bpy.data.materials.new("Baked Realistic Skin")
direct.use_nodes = True
direct.diffuse_color = (0.55, 0.245, 0.16, 1.0)
direct_nodes = direct.node_tree.nodes
direct_links = direct.node_tree.links
direct_nodes.clear()

direct_output = direct_nodes.new("ShaderNodeOutputMaterial")
direct_output.location = (720, 80)
direct_skin = direct_nodes.new("ShaderNodeBsdfPrincipled")
direct_skin.location = (420, 80)
set_input(direct_skin, "Metallic", 0.0)
set_input(direct_skin, "IOR", 1.4)
set_input(direct_skin, "Specular IOR Level", 0.34)
set_input(direct_skin, "Subsurface Weight", 0.115)
set_input(direct_skin, "Subsurface Radius", (1.0, 0.44, 0.22))
set_input(direct_skin, "Subsurface Scale", 0.007)
direct_links.new(direct_skin.outputs["BSDF"], direct_output.inputs["Surface"])

uv = direct_nodes.new("ShaderNodeUVMap")
uv.uv_map = "SkinUV"
uv.location = (-720, 100)

base_node = direct_nodes.new("ShaderNodeTexImage")
base_node.name = "Base Color Texture"
base_node.label = "Baked base color"
base_node.image = base_image
base_node.location = (-450, 300)
direct_links.new(uv.outputs["UV"], base_node.inputs["Vector"])
direct_links.new(base_node.outputs["Color"], direct_skin.inputs["Base Color"])

rough_node = direct_nodes.new("ShaderNodeTexImage")
rough_node.name = "Roughness Texture"
rough_node.label = "Baked roughness"
rough_node.image = rough_image
rough_node.location = (-450, 20)
direct_links.new(uv.outputs["UV"], rough_node.inputs["Vector"])
direct_links.new(rough_node.outputs["Color"], direct_skin.inputs["Roughness"])

normal_node = direct_nodes.new("ShaderNodeTexImage")
normal_node.name = "Normal Texture"
normal_node.label = "Baked skin detail"
normal_node.image = normal_image
normal_node.location = (-450, -260)
normal_map = direct_nodes.new("ShaderNodeNormalMap")
normal_map.location = (100, -170)
normal_map.uv_map = "SkinUV"
normal_map.inputs["Strength"].default_value = 0.72
direct_links.new(uv.outputs["UV"], normal_node.inputs["Vector"])
direct_links.new(normal_node.outputs["Color"], normal_map.inputs["Color"])
direct_links.new(normal_map.outputs["Normal"], direct_skin.inputs["Normal"])

hand.data.materials.clear()
hand.data.materials.append(direct)
for polygon in hand.data.polygons:
    polygon.use_smooth = True

# Pack the three PNG maps into the .blend as well as keeping external copies.
for image in (base_image, rough_image, normal_image):
    image.pack()
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND), compress=True)

# Build a temporary studio scene to verify the image-based material visually.
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 640
scene.render.resolution_y = 640
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(OUTPUT_PREVIEW)
scene.view_settings.look = "Medium High Contrast"
scene.view_settings.exposure = 1.25

world = scene.world or bpy.data.worlds.new("Preview World")
scene.world = world
world.use_nodes = True
background = world.node_tree.nodes.get("Background")
background.inputs["Color"].default_value = (0.075, 0.090, 0.125, 1.0)
background.inputs["Strength"].default_value = 0.55

corners = [hand.matrix_world @ Vector(corner) for corner in hand.bound_box]
center = sum(corners, Vector()) / len(corners)
extent = max(max(c[i] for c in corners) - min(c[i] for c in corners) for i in range(3))

camera = bpy.data.objects.get("Camera")
scene.camera = camera
camera.data.lens = 57
camera.location = center + Vector((extent * 0.36, extent * 2.30, extent * 0.10))
point_at(camera, center + Vector((0, 0, extent * 0.02)))

for obj in list(bpy.data.objects):
    if obj.type == "LIGHT":
        bpy.data.objects.remove(obj, do_unlink=True)
    elif obj.type == "MESH" and obj != hand and len(obj.data.vertices) <= 8:
        obj.hide_render = True


def add_area(name, offset, energy, size, color):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    data.color = color
    light = bpy.data.objects.new(name, data)
    scene.collection.objects.link(light)
    light.location = center + Vector(offset)
    point_at(light, center)


add_area("Key", (-extent * 1.4, extent * 1.4, extent * 1.6), 5200, extent * 2.0, (1.0, 0.74, 0.60))
add_area("Fill", (extent * 1.4, extent * 0.9, extent * 0.3), 3100, extent * 1.8, (0.58, 0.72, 1.0))
add_area("Rim", (0, -extent * 1.0, extent * 1.25), 4300, extent * 1.2, (1.0, 0.48, 0.30))

bpy.ops.render.render(write_still=True)
print(f"SAVED_BLEND={OUTPUT_BLEND}")
print(f"SAVED_PREVIEW={OUTPUT_PREVIEW}")
