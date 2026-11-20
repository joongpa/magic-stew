import math
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(bpy.path.abspath("//")).parent
OUTPUT_BLEND = ROOT / "rigged_hand_textured.blend"
OUTPUT_PREVIEW = ROOT / "rigged_hand_textured_preview.png"


def set_input(node, name, value):
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def point_camera(camera, target):
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


# Identify the actual skinned mesh without relying solely on its downloaded name.
hand = max(
    (obj for obj in bpy.data.objects if obj.type == "MESH"),
    key=lambda obj: len(obj.data.vertices),
)

# Build a UV-independent procedural skin shader.
material = bpy.data.materials.get("Realistic Procedural Skin")
if material is None:
    material = bpy.data.materials.new("Realistic Procedural Skin")
material.use_nodes = True
material.diffuse_color = (0.55, 0.245, 0.16, 1.0)

nodes = material.node_tree.nodes
links = material.node_tree.links
nodes.clear()

output = nodes.new("ShaderNodeOutputMaterial")
output.name = "Skin Material Output"
output.location = (900, 80)

skin = nodes.new("ShaderNodeBsdfPrincipled")
skin.name = "Skin Shader"
skin.label = "Skin: soft specular + subsurface scattering"
skin.location = (620, 80)
set_input(skin, "Metallic", 0.0)
set_input(skin, "Roughness", 0.47)
set_input(skin, "IOR", 1.4)
set_input(skin, "Specular IOR Level", 0.34)
set_input(skin, "Subsurface Weight", 0.115)
set_input(skin, "Subsurface Radius", (1.0, 0.44, 0.22))
set_input(skin, "Subsurface Scale", 0.007)
links.new(skin.outputs["BSDF"], output.inputs["Surface"])

texcoord = nodes.new("ShaderNodeTexCoord")
texcoord.location = (-1100, 80)

mapping = nodes.new("ShaderNodeMapping")
mapping.name = "Skin Texture Scale"
mapping.location = (-900, 80)
links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])

# Broad mottling suggests blood flow and natural variation without painted UVs.
macro = nodes.new("ShaderNodeTexNoise")
macro.name = "Broad Skin Mottling"
macro.location = (-660, 280)
macro.noise_dimensions = "3D"
macro.inputs["Scale"].default_value = 3.2
macro.inputs["Detail"].default_value = 5.0
macro.inputs["Roughness"].default_value = 0.68
macro.inputs["Distortion"].default_value = 0.18
links.new(mapping.outputs["Vector"], macro.inputs["Vector"])

color_ramp = nodes.new("ShaderNodeValToRGB")
color_ramp.name = "Natural Skin Tones"
color_ramp.location = (-390, 290)
ramp = color_ramp.color_ramp
ramp.elements.remove(ramp.elements[1])
stops = (
    (0.23, (0.285, 0.075, 0.040, 1.0)),
    (0.43, (0.500, 0.165, 0.095, 1.0)),
    (0.60, (0.665, 0.300, 0.195, 1.0)),
    (0.78, (0.760, 0.410, 0.300, 1.0)),
)
for index, (position, color) in enumerate(stops):
    element = ramp.elements[0] if index == 0 else ramp.elements.new(position)
    element.position = position
    element.color = color
links.new(macro.outputs["Fac"], color_ramp.inputs["Fac"])
links.new(color_ramp.outputs["Color"], skin.inputs["Base Color"])

# Two bands of microstructure create pores and faint skin grain.
micro = nodes.new("ShaderNodeTexNoise")
micro.name = "Skin Grain"
micro.location = (-660, -110)
micro.noise_dimensions = "3D"
micro.inputs["Scale"].default_value = 78.0
micro.inputs["Detail"].default_value = 7.0
micro.inputs["Roughness"].default_value = 0.72
links.new(mapping.outputs["Vector"], micro.inputs["Vector"])

pores = nodes.new("ShaderNodeTexNoise")
pores.name = "Fine Pores"
pores.location = (-660, -380)
pores.noise_dimensions = "3D"
pores.inputs["Scale"].default_value = 235.0
pores.inputs["Detail"].default_value = 4.0
pores.inputs["Roughness"].default_value = 0.78
links.new(mapping.outputs["Vector"], pores.inputs["Vector"])

grain_mix = nodes.new("ShaderNodeMix")
grain_mix.name = "Layered Skin Detail"
grain_mix.data_type = "FLOAT"
grain_mix.blend_type = "MULTIPLY"
grain_mix.location = (-310, -150)
grain_mix.inputs[0].default_value = 0.38
links.new(micro.outputs["Fac"], grain_mix.inputs[2])
links.new(pores.outputs["Fac"], grain_mix.inputs[3])

bump = nodes.new("ShaderNodeBump")
bump.name = "Subtle Pore Relief"
bump.location = (320, -120)
bump.inputs["Strength"].default_value = 0.16
bump.inputs["Distance"].default_value = 0.00055
links.new(grain_mix.outputs["Result"], bump.inputs["Height"])
links.new(bump.outputs["Normal"], skin.inputs["Normal"])

# Slight roughness variation keeps highlights organic.
rough_ramp = nodes.new("ShaderNodeMapRange")
rough_ramp.name = "Roughness Variation"
rough_ramp.location = (60, -360)
rough_ramp.inputs["From Min"].default_value = 0.0
rough_ramp.inputs["From Max"].default_value = 1.0
rough_ramp.inputs["To Min"].default_value = 0.40
rough_ramp.inputs["To Max"].default_value = 0.56
links.new(micro.outputs["Fac"], rough_ramp.inputs["Value"])
links.new(rough_ramp.outputs["Result"], skin.inputs["Roughness"])

hand.data.materials.clear()
hand.data.materials.append(material)
for polygon in hand.data.polygons:
    polygon.use_smooth = True

# Save the material-enhanced asset before adding temporary preview-only objects.
# Original object, rig, camera, light, animation, and helper names remain unchanged.
material.use_fake_user = True
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND), compress=True)

# Hide the tiny helper cube only in the temporary preview render.
for obj in bpy.data.objects:
    if obj.type == "MESH" and obj != hand and len(obj.data.vertices) <= 8:
        obj.hide_render = True

# Neutral studio preview setup.
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 640
scene.render.resolution_y = 640
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
scene.render.filepath = str(OUTPUT_PREVIEW)
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.look = "Medium High Contrast"
scene.view_settings.exposure = 1.25

world = scene.world or bpy.data.worlds.new("Studio World")
scene.world = world
world.use_nodes = True
world_bg = world.node_tree.nodes.get("Background")
world_bg.inputs["Color"].default_value = (0.075, 0.090, 0.125, 1.0)
world_bg.inputs["Strength"].default_value = 0.55

# Evaluate the visual center from the mesh bounding box.
corners = [hand.matrix_world @ Vector(corner) for corner in hand.bound_box]
center = sum(corners, Vector()) / len(corners)
extent = max((max(c[i] for c in corners) - min(c[i] for c in corners)) for i in range(3))

camera = bpy.data.objects.get("Camera")
if camera is None:
    camera_data = bpy.data.cameras.new("Preview Camera")
    camera = bpy.data.objects.new("Preview Camera", camera_data)
    scene.collection.objects.link(camera)
scene.camera = camera
camera.data.lens = 57
camera.location = center + Vector((extent * 0.36, extent * 2.30, extent * 0.10))
point_camera(camera, center + Vector((0, 0, extent * 0.02)))

# Replace scene lights with a three-point studio arrangement.
for obj in list(bpy.data.objects):
    if obj.type == "LIGHT":
        bpy.data.objects.remove(obj, do_unlink=True)


def add_area(name, location, energy, size, color):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = location
    point_camera(obj, center)
    return obj


add_area(
    "Key Softbox",
    center + Vector((-extent * 1.4, extent * 1.4, extent * 1.6)),
    5200,
    extent * 2.0,
    (1.0, 0.74, 0.60),
)
add_area(
    "Fill Softbox",
    center + Vector((extent * 1.4, extent * 0.9, extent * 0.3)),
    3100,
    extent * 1.8,
    (0.58, 0.72, 1.0),
)
add_area(
    "Rim Light",
    center + Vector((0, -extent * 1.0, extent * 1.25)),
    4300,
    extent * 1.2,
    (1.0, 0.48, 0.30),
)

# A matte floor/background catches a soft shadow without competing with the hand.
floor_data = bpy.data.meshes.new("Preview Floor Mesh")
floor = bpy.data.objects.new("Preview Floor", floor_data)
scene.collection.objects.link(floor)
floor_z = min(c.z for c in corners) - extent * 0.035
verts = [
    (-extent * 2.2, -extent * 2.2, floor_z),
    (extent * 2.2, -extent * 2.2, floor_z),
    (extent * 2.2, extent * 2.2, floor_z),
    (-extent * 2.2, extent * 2.2, floor_z),
]
floor_data.from_pydata(verts, [], [(0, 1, 2, 3)])
floor_material = bpy.data.materials.new("Preview Charcoal")
floor_material.diffuse_color = (0.025, 0.030, 0.040, 1.0)
floor_material.use_nodes = True
floor_bsdf = floor_material.node_tree.nodes.get("Principled BSDF")
set_input(floor_bsdf, "Base Color", (0.025, 0.030, 0.040, 1.0))
set_input(floor_bsdf, "Roughness", 0.72)
floor.data.materials.append(floor_material)

bpy.ops.render.render(write_still=True)
print(f"SAVED_BLEND={OUTPUT_BLEND}")
print(f"SAVED_PREVIEW={OUTPUT_PREVIEW}")
