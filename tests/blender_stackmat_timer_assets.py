"""Blender integration check (run against the rebuilt asset library).

    Blender --background --python tests/blender_stackmat_timer_assets.py

Checks every variant and repeated generation without rendering a dataset.
"""

import importlib.util
from pathlib import Path
from unittest.mock import patch
import bpy

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "cube_moves", ROOT / "scripts/apply_blender_cube_moves.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
variants = ("gen2", "gen3", "gen4", "gen5", "gan_timer2")
for variant in variants:
    m.clear_background_scene()
    coll = m.append_background_asset_collection(
        m.solve_station_asset_blend_path(),
        m.SOLVE_STATION_COLLECTION_NAME,
        m.BACKGROUND_ASSET_COLLECTION_PREFIX + ".Test",
    )
    m.link_collection_to_collection(coll, m.background_scene_collection())
    assert len(m.solve_station_family_objects(coll, ("Timer",))) == 5
    with patch.object(m.random, "choice", return_value=variant):
        metadata = m.select_solve_station_timer(coll)
    timers = m.solve_station_family_objects(coll, ("Timer",))
    assert len(timers) == 1 and metadata["variant"] == variant
    obj = timers[0]
    assert not obj.hide_render and not obj.hide_viewport
    assert abs(obj.dimensions.x - 7.84) < 0.001, obj.dimensions[:]
    assert obj.dimensions.y < 2.1 and obj.dimensions.z < 0.6
    before = [tuple(mat.diffuse_color) for mat in obj.data.materials]
    m.randomize_solve_station_materials(coll)
    assert before == [tuple(mat.diffuse_color) for mat in obj.data.materials]
    layout = m.randomize_solve_station_layout(coll)
    assert layout["timer"]["bounds"][3] - layout["timer"]["bounds"][2] < 2.5
    print("PASS", variant, metadata, "vertices", len(obj.data.vertices))
m.clear_background_scene()
assert not [o for o in bpy.data.objects if o.get("timer_variant")]
# Exercise the actual entry point, metadata, and cleanup across repeated samples.
for seed in range(8):
    m.random.seed(seed)
    result = m.add_solve_station_background()
    timers = [o for o in bpy.data.objects if o.get("timer_variant")]
    assert len(timers) == 1
    assert result["timer"]["variant"] == timers[0]["timer_variant"]
    assert result["layout"]["timer"]["bounds"]
    m.clear_background_scene()
assert not [o for o in bpy.data.objects if o.get("timer_variant")]
print("PASS repeated generation / placement / cleanup")
