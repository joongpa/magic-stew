"""Apply Rubik's Cube face turns and scene randomization for synthetic data.

Run inside Blender, for example:

    Blender /path/to/cube.blend --background --python scripts/apply_blender_cube_moves.py -- --count 100 --output-dir ./synthetic_cube --gpu

Assumptions:
- The cube is modeled as separate cubie objects, one object per movable cubie.
- Object origins are centered on their cubies.
- The cube core is at the world origin.
- Cubies are 1 meter wide, so cubie origins sit on the -1/0/1 grid.
- Cubie objects use quaternion rotation mode.
- Cubie object pass_index values identify physical cubies.
- The six physical center stickers form one color-ordered cube-core instance.
  Other qualifying stickers and landmarks have one keypoint per instance.
- Blender axes are X=right, Z=up, and Y=front/back. Change FRONT_AXIS if needed.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import site
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from itertools import chain, permutations, product
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector

PROJECT = Path(__file__).resolve().parents[1]
VENV = PROJECT / ".venv"

# Blender's embedded Python does not automatically see packages installed in
# the project's virtual environment.  Unix and Windows use different venv
# layouts, so try both rather than assuming the author's machine.
site_package_candidates = [
    VENV / "Lib" / "site-packages",  # Windows
    *sorted((VENV / "lib").glob("python*/site-packages")),  # macOS/Linux
]
site_packages = next(
    (path for path in site_package_candidates if path.is_dir()),
    None,
)
if site_packages is None:
    raise RuntimeError(
        f"Could not find the virtual-environment site-packages directory under {VENV}. "
        "Create the project environment first (for example: py -m venv .venv and "
        ".venv\\Scripts\\python -m pip install -e \".[dev]\")."
    )
site.addsitedir(str(site_packages))


# Common Blender convention: X right, Z up, and negative Y faces the camera/front.
# Use "+Y" instead if your modeled front face is on positive Y.
FRONT_AXIS = "-Y"

# Random scene defaults for synthetic data generation.
DEFAULT_SCRAMBLE_LENGTH = 30
CAMERA_TARGET_JITTER = 0.10
CAMERA_VIEWPOINT_WEIGHTS = (
    ("front", 0.24),
    ("front_diagonal", 0.28),
    ("over_shoulder", 0.18),
    ("behind_diagonal", 0.18),
    ("low_behind_diagonal", 0.12),
    ("low_side", 0.12),
    ("overhead", 0.12),
)
CAMERA_ROI_LENS_RANGE = (42.0, 49.0)
# Keep close cubes at their existing framing, then ramp optical zoom from 1x
# at six feet to 2x at fifteen feet without changing camera positions. Target
# jitter and sensor shifts still allow modest off-center variation.
CAMERA_CUBE_ZOOM_SCALE_RANGE = (1.0, 2.0)
CAMERA_CUBE_ZOOM_DISTANCE_RANGE_FEET = (6.0, 15.0)
# The behind-palm hard case sits 14–18 scene units away. Compensate its lens
# against this closer framing baseline even below the six-foot zoom threshold.
CAMERA_BEHIND_PALM_REFERENCE_DISTANCE = 10.0
CAMERA_ROI_SHIFT_X_RANGE = (-0.035, 0.035)
CAMERA_ROI_SHIFT_Y_RANGE = (-0.03, 0.03)
CAMERA_ROLL_DEGREES_RANGE = (-45.0, 45.0)
# The existing framing is calibrated around four feet from the cube. Varying
# distance and focal length together preserves the base framing while varying
# perspective compression; the extra distance-based zoom above tightens it.
CAMERA_SCENE_UNITS_PER_FOOT = 3.0
CAMERA_DISTANCE_BUCKETS_FEET = (
    ("near", 0.35, (3.0, 6.0)),
    ("medium", 0.40, (6.0, 10.0)),
    ("far", 0.25, (10.0, 15.0)),
)
CAMERA_LOW_BEHIND_DISTANCE_BUCKETS_FEET = (
    ("medium", 0.35, (8.0, 11.0)),
    ("far", 0.65, (11.0, 15.0)),
)

SYNTHETIC_LIGHT_PREFIX = "SyntheticDataLight"
SYNTHETIC_WINDOW_BLOCKER_PREFIX = "SyntheticWindowBlocker"
LIGHTING_SETUP_WEIGHTS = (
    ("ceiling_lights", 0.76),
    ("desk_lamp", 0.24),
)
SUNLIGHT_WINDOW_PROBABILITY = 0.42
# Yellow and blue dominate to stress discrimination between sticker colors.
LIGHT_TEMPERATURE_PROFILES = (
    # name, probability, indoor sources, window daylight, tint target, tint amount
    ("hard_yellow", 0.38, (2400.0, 3400.0), (3000.0, 4300.0), (1.0, 0.52, 0.015), 0.72),
    ("hard_blue", 0.38, (7600.0, 12000.0), (8200.0, 14000.0), (0.015, 0.16, 1.0), 0.72),
    ("hard_red", 0.06, (2500.0, 4200.0), (3200.0, 5200.0), (1.0, 0.025, 0.012), 0.58),
    ("hard_magenta", 0.06, (3000.0, 5200.0), (4000.0, 6500.0), (1.0, 0.015, 0.48), 0.56),
    ("hard_cyan", 0.06, (5200.0, 8200.0), (6500.0, 10000.0), (0.01, 0.88, 1.0), 0.56),
    ("hard_green", 0.06, (3400.0, 6200.0), (4800.0, 7600.0), (0.025, 1.0, 0.10), 0.52),
)
DEFAULT_MIXED_LIGHTING_PROBABILITY = 0.15
MIXED_LIGHTING_PROBABILITY = DEFAULT_MIXED_LIGHTING_PROBABILITY
MIXED_LIGHTING_PROFILES = (
    # Name, weight, key temperature range, accent RGB.
    ("warm_key_blue_accent", 0.50, (3800.0, 4700.0), (0.30, 0.17, 1.0)),
    ("cool_key_amber_accent", 0.20, (5600.0, 7200.0), (1.0, 0.35, 0.08)),
    ("neutral_key_magenta_accent", 0.15, (4500.0, 5500.0), (0.82, 0.09, 0.68)),
    ("warm_key_teal_accent", 0.15, (3200.0, 4300.0), (0.06, 0.66, 0.82)),
)
LIGHT_TARGET_JITTER_RANGE = (-0.35, 0.35)
LIGHT_TEMPERATURE_KELVIN_RANGE = (2700.0, 7500.0)
FLOOR_LAMP_DISTANCE_FEET_RANGE = (10.0, 15.0)
FLOOR_LAMP_HEIGHT_FEET_RANGE = (5.5, 7.0)
FLOOR_LAMP_ENERGY_RANGE = (12000.0, 28000.0)
FLOOR_LAMP_SIZE_RANGE = (4.5, 7.0)
WORLD_AMBIENT_LEVEL_RANGE = (0.045, 0.18)
BACKGROUND_NAME = "SyntheticDataBackground"
BACKGROUND_MATERIAL_NAME = "SyntheticDataBackgroundMaterial"
BACKGROUND_DISTANCE = 4.0
BACKGROUND_SIZE = 12.0
BACKGROUND_SCENE_COLLECTION_NAME = "SyntheticBackgroundScene"
BACKGROUND_SCENE_PREFIX = "SyntheticBackgroundObject"
SECONDARY_CUBE_OBJECT_PREFIX = f"{BACKGROUND_SCENE_PREFIX}.secondary_cube."
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
SOLVE_STATION_MAT_PRINT_DIR = PROJECT / "syntheticAssets" / "stackmat_prints"
SOLVE_STATION_MAT_PRINTS = (
    "turn_trails",
    "prism_grid",
    "steel_cubes",
    "pixel_orbit",
    "lime_shift",
    "cubicle_sakura",
    "cubicle_aurora",
    "cubicle_angstrom",
    "scs_poly_dark",
    "scs_cosmic",
    "scs_strokes",
)
SOLVE_STATION_MAT_PRINT_PROBABILITY = 0.80
SOLVE_STATION_SHIRT_PRINT_DIR = PROJECT / "syntheticAssets" / "shirt_prints"
SOLVE_STATION_SHIRT_PRINTS = (
    "championship_cube",
    "yellow_grid_right",
    "yellow_grid_left",
    "yellow_isometric",
    "hand_drawn_cube",
    "cubing_fiesta",
    "angular_cube",
    "problem_solver",
    "scs_graffiti",
    "scs_vintage",
    "scs_speed_cube",
    "cubicle_embroidered",
    "cubicle_nationals",
    "cubicle_worlds",
)
SOLVE_STATION_SHIRT_PRINT_PROBABILITY = 0.60
# Conditional on the shirt having a print.
SOLVE_STATION_SHIRT_POCKET_PROBABILITY = 0.30
SOLVE_STATION_SHIRT_PRINT_MATERIAL = "SyntheticSolveStationShirtPrint"
SOLVE_STATION_LAPTOP_PROBABILITY = 0.48
LAPTOP_GLOW_PROBABILITY = 0.10
LAPTOP_GLOW_COLOR_PROFILES = (
    ("electric_blue", 0.18, (0.12, 0.38, 1.0, 1.0)),
    ("cyan", 0.16, (0.05, 0.88, 1.0, 1.0)),
    ("violet", 0.15, (0.52, 0.14, 1.0, 1.0)),
    ("magenta", 0.13, (1.0, 0.10, 0.48, 1.0)),
    ("amber", 0.13, (1.0, 0.38, 0.05, 1.0)),
    ("green", 0.12, (0.08, 1.0, 0.30, 1.0)),
    ("cool_white", 0.13, (0.82, 0.92, 1.0, 1.0)),
)
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
STICKER_OBJECT_NAME_PREFIX = "StickerlessMaterialRadialPart."
SEGMENTATION_CLASS_NAMES = ("cube", "sticker")
STICKER_KEYPOINT_CLASS_NAMES = ("white", "yellow", "orange", "red", "blue", "green")
STICKER_KEYPOINT_CLASS_ID_BY_MATERIAL = {
    "White": 0,
    "Yellow": 1,
    "Orange": 2,
    "Red": 3,
    "Blue": 4,
    "Green": 5,
}
STICKER_KEYPOINT_MIN_VISIBLE_AREA_RATIO = 0.10
# Sticker faces within this many degrees of edge-on have too little visual
# evidence to produce a useful center keypoint. This uses the same convention
# as point_faces_camera: 0 degrees is exactly edge-on and 90 degrees is
# face-on.
STICKER_KEYPOINT_MIN_PLANE_ANGLE_DEGREES = 5.0
STICKER_KEYPOINT_MASK_COLORS = {
    # OpenCV uses BGR ordering; these correspond to the sticker colors.
    0: (255, 255, 255),  # white
    1: (0, 238, 245),    # yellow
    2: (0, 146, 242),    # orange
    3: (1, 32, 231),     # red
    4: (227, 177, 101),  # blue
    5: (1, 227, 80),     # green
}
MIN_VISIBLE_STICKER_AREA_PX_AT_128 = 8.0
POSE_LABEL_DIRECTORY = "pose"
POSE_LABEL_MODE_INDEPENDENT = "independent"
POSE_LABEL_MODE_ORDERED = "ordered"
POSE_LABEL_MODES = (POSE_LABEL_MODE_INDEPENDENT, POSE_LABEL_MODE_ORDERED)
MASK_MODE_SEGMENTATION = "segmentation"
MASK_MODE_KEYPOINTS = "keypoints"
MASK_MODE_NONE = "none"
MASK_MODES = (MASK_MODE_SEGMENTATION, MASK_MODE_KEYPOINTS, MASK_MODE_NONE)
ORDERED_POSE_CLASS_ID = 0
ORDERED_KEYPOINT_NAMES = (
    # Corners: 0..7.  The order follows the existing canonical cube frame.
    "UBL", "UBR", "UFR", "UFL",
    "DBL", "DBR", "DFR", "DFL",
    # Edge midpoints: 8..19.  First the top/bottom face perimeters, then
    # the four vertical edges.
    "UB", "UR", "UF", "UL",
    "DB", "DR", "DF", "DL",
    "BR", "BL", "FR", "FL",
    # Face centers: 20..25.
    "U", "D", "F", "B", "R", "L",
)
ORDERED_KEYPOINT_COUNT = len(ORDERED_KEYPOINT_NAMES)
ORDERED_KEYPOINT_SURFACE_COORDINATE = 1.5
# Annotation widths are specified at the former 256x256 reference resolution
# and scaled with the output image so overlays remain visually proportional.
ANNOTATION_REFERENCE_SIZE = 256.0
WIREFRAME_LINE_BASE_WIDTH = 2
WIREFRAME_LINE_SHADOW_BASE_WIDTH = 5
WIREFRAME_LINE_COLOR = (0, 255, 255)
WIREFRAME_LINE_SHADOW_COLOR = (0, 0, 0)
POSE_SEGMENT_FILL_BASE_WIDTH = 9
POSE_SEGMENT_OUTLINE_BASE_WIDTH = 2
# Keep keypoint-mask dots compact so they mark intersections without covering
# the rendered sticker detail.  Values are scaled from the 256px reference.
POSE_POINT_BASE_RADIUS = 1
POSE_POINT_OUTLINE_WIDTH = 1
# Display-space RGB samples taken from the flat sticker centers in the six
# predefined GAN sticker-set images on TheCubicle's product page. These are
# visual references rather than manufacturer color specifications; the page
# itself says the product images are intended to represent the shades.
STICKER_COLOR_PALETTES = {
    "full_bright": {
        "White": (255, 255, 255),
        "Yellow": (245, 238, 0),
        "Red": (231, 32, 1),
        "Orange": (242, 146, 0),
        "Blue": (101, 177, 227),
        "Green": (80, 227, 1),
    },
    "full_bright_sky_blue": {
        "White": (255, 255, 255),
        "Yellow": (245, 238, 0),
        "Red": (231, 32, 1),
        "Orange": (242, 146, 0),
        "Blue": (0, 118, 205),
        "Green": (80, 227, 1),
    },
    "half_bright": {
        "White": (255, 255, 255),
        "Yellow": (249, 240, 1),
        "Red": (154, 0, 0),
        "Orange": (242, 148, 0),
        "Blue": (0, 62, 195),
        "Green": (79, 226, 0),
    },
    "half_bright_sky_blue": {
        "White": (255, 255, 255),
        "Yellow": (249, 240, 1),
        "Red": (154, 0, 0),
        "Orange": (242, 148, 0),
        "Blue": (1, 123, 208),
        "Green": (79, 226, 0),
    },
    "factory_half_bright": {
        "White": (255, 255, 255),
        "Yellow": (245, 238, 0),
        "Red": (154, 0, 0),
        "Orange": (238, 83, 1),
        "Blue": (0, 54, 128),
        "Green": (80, 227, 1),
    },
    "classic": {
        "White": (255, 255, 255),
        "Yellow": (236, 203, 0),
        "Red": (154, 0, 0),
        "Orange": (203, 53, 0),
        "Blue": (0, 61, 188),
        "Green": (30, 133, 24),
    },
}
STICKER_COLOR_PALETTE_SOURCE = "https://www.thecubicle.com/products/gan-3x3-ff-sticker-set"
STICKER_ROUGHNESS_RANGE = (0.18, 0.38)
STICKER_SPECULAR_RANGE = (0.45, 0.75)
STICKER_COAT_WEIGHT_RANGE = (0.08, 0.24)
STICKER_COAT_ROUGHNESS_RANGE = (0.12, 0.32)
CUBE_LOGO_DIR = PROJECT / "syntheticAssets" / "cube_logos"
CUBE_LOGO_NAMES = ("gan", "dayan", "moyu", "geometric", "mgc", "mofangjiaoshi")
# Roll independently for each cube across all presets and modes.
CUBE_LOGO_PROBABILITY = 0.60
CUBE_LOGO_ORIGINAL_COLOR_PROBABILITY = 0.25
CUBE_LOGO_MATERIAL_NAME = "SyntheticCubeLogo"
CUBE_LOGO_UV_NAME = "SyntheticCubeLogoUV"
# Disable procedural scratches/dents; retain randomized sticker color and gloss.
STICKER_IMPERFECTIONS_ENABLED = False
STICKER_IMPERFECTION_FINE_SCALE_RANGE = (120.0, 240.0)
STICKER_IMPERFECTION_SCRATCH_SCALE_RANGE = (28.0, 70.0)
STICKER_IMPERFECTION_BUMP_STRENGTH_RANGE = (0.012, 0.035)
STICKER_IMPERFECTION_BUMP_DISTANCE_RANGE = (0.008, 0.022)
CORE_FILL_OBJECT_NAMES = ("StickerlessBlackCoreFill", "StickerlessBlackCoreSphereFill")
# Keep discrete scrambles enabled for varied cube states. Generate partial
# layer turns by default so ordered-keypoint datasets contain examples of
# layer misalignment. The CLI still exposes an explicit opt-out for legacy
# rigid-cube datasets; solved cubes retain orientation-only x/y/z moves.
CUBE_MISALIGNMENT_ENABLED = True
SCRAMBLES_ENABLED = True
PARTIAL_TURN_APPLIED_PROBABILITY = 0.70
SPECIAL_PARTIAL_TURN_PROBABILITY = 0.30
# Preserve the former relative mix among ordinary partial states. Corner cuts
# are deliberately kept out of slice and simultaneous-turn states.
SINGLE_PARTIAL_STATE_WEIGHTS = (
    ("mid_turn", 0.50),
    ("corner_cut", 0.20),
)
SLICE_MOVES = ("M", "E", "S")
OPPOSITE_FACE_PAIRS = (("R", "L"), ("U", "D"), ("F", "B"))
MID_TURN_DEGREES_RANGE = (-45.0, 45.0)
CORNER_CUT_SMALL_DEGREES_RANGE = (0.0, 10.0)
CORNER_CUT_ADJACENT_DEGREES_RANGE = (5.0, 45.0)
# Seam counterparts are only visually useful once the layer has moved beyond
# the small alignment tolerance used for corner-cut samples.
PARTIAL_TURN_INTERSECTION_MIN_DEGREES = 10.0
# Keep seam counterparts enabled by default, but expose a single switch for
# dataset ablations. Base keypoint labels are unaffected when this is False.
PARTIAL_TURN_SEAM_KEYPOINTS_ENABLED = True
# Preserve the small boundary tolerance used by the distance-based keypoint
# visibility test. It only covers numerical/render-boundary differences around
# the mathematical point; it does not exempt turn seams or any cube piece.
KEYPOINT_DISTANCE_TOLERANCE = 0.02
CUBE_FLOAT_PROBABILITY = 0.55
CUBE_DESK_NO_HANDS_PROBABILITY = 0.10
CUBE_HANDS_TABLE_LIFT = 0.35
CUBE_TABLE_CLEARANCE_RANGE = (0.02, 0.18)
CUBE_FLOAT_CLEARANCE_RANGE = (0.12, 0.55)
CUBE_FLOAT_TILT_DEGREES = (-30.0, 30.0)
CUBE_POSE_XY_JITTER_RANGE = (-0.25, 0.25)
CUBE_POSE_ROTATION_DEGREES = (-30.0, 30.0)
# Disable deliberate edge placement with the tighter framing. The optional
# augmentation remains available by raising this probability explicitly.
# The overshoot is measured in normalized image coordinates, so at 128x128 it
# corresponds to roughly 1-3 pixels of cube geometry outside the frame.
CUBE_EDGE_CLIP_PROBABILITY = 0.0
CUBE_EDGE_CLIP_OVERSHOOT_RANGE = (0.005, 0.025)
DEFAULT_CUBE_ROTATIONAL_BLUR_PROBABILITY = 1.0
DEFAULT_MAX_CUBE_ROTATIONAL_BLUR_DEGREES = 20.0
DEFAULT_CUBE_ROTATIONAL_BLUR_SAMPLES = 7
# The larger range is reserved for hands and cubies participating in a
# partial layer turn. The global range is intentionally specified in degrees
# so the whole-image blur stays in the desired 3-8 degree band.
GLOBAL_ROTATIONAL_BLUR_DEGREES_RANGE = (3.0, 8.0)
LOCAL_ROTATIONAL_BLUR_FRACTION_RANGE = (1.00, 1.50)
MAX_LOCAL_ROTATIONAL_BLUR_DEGREES = 30.0
HAND_LOCAL_ROTATIONAL_BLUR_FRACTION = 0.50
DEFAULT_DEGRADE_PROBABILITY = 0.85
DEGRADE_GAUSSIAN_BLUR_PROBABILITY = 0.35
DEGRADE_MOTION_BLUR_PROBABILITY = 0.70
DEGRADE_NOISE_PROBABILITY = 0.45
DEGRADE_VIDEO_COMPRESSION_PROBABILITY = 0.60
DEGRADE_RESCALE_PROBABILITY = 0.25
# CRF 40 was producing occasional overly harsh block/compression artifacts;
# keep the lower bound unchanged while softening the worst case further.
VIDEO_COMPRESSION_CRF_RANGE = (28, 30)
VIDEO_COMPRESSION_CONTEXT_FRAMES = 5
VIDEO_COMPRESSION_FRAME_RATE = 30
VIDEO_COMPRESSION_MOTION_JITTER_PX_RANGE = (0.35, 1.25)
HAND_NAMES = ("left-hand", "right-hand")
# Normal samples always show hands.  The separate desk_no_hands probability
# controls the approximately 10% no-hand population.
HAND_VISIBILITY_WEIGHTS = (
    ("onehandgrip", 0.10),
    ("both", 0.90),
)
# Randomize hand placement while retaining the shared cube pose and the
# permitted R/L partial-turn motion.
HAND_RANDOMIZATION_ENABLED = True
HAND_FIXED_SKIN_COLOR = (0.75, 0.45, 0.30, 1.0)
# Keep lateral hand nudging narrow so the hand stays beside the cube instead
# of moving into its top/side geometry.  This is in the cube-local frame.
HAND_X_OFFSET_RANGE = (-0.1, 0.1)
HAND_Z_OFFSET_RANGE = (-0.5, 0.5)
# The imported hand asset is authored facing the opposite direction from the
# solve station. Keep this fixed asset-to-solver orientation separate from
# pose-local quarter-turn randomization.
HAND_SOLVER_FACING_ROTATION_DEGREES = 180.0
HAND_ORIGIN_X_ROTATION_DEGREES = (-90.0, 0.0, 90.0)
HAND_POSES_WITHOUT_ORIGIN_X_ROTATION = (
    "HandPose.StandardHold",
    "HandPose.OneHandGrip",
)
HAND_POSE_ACTION_NAMES = (
    "HandPose.StandardHold",
    "HandPose.OneHandGrip",
    "HandPose.StandardGrip",
    "HandPose.PostD",
    "HandPose.PostS",
    "HandPose.PostU",
)
HAND_TEXTURE_DIR = PROJECT / "syntheticAssets" / "rigged_hand_textures"
HAND_TEXTURE_SOURCE_BLEND_CANDIDATES = (
    PROJECT / "rigged_hand_textured_direct.blend",
    PROJECT / "syntheticAssets" / "rigged_hand_textured_direct.blend",
)
HAND_TEXTURE_UV_LAYER_NAME = "SkinUV"
HAND_ASSET_OBJECT_NAMES = (
    "left-hand",
    "left-hand-rig",
    "right-hand",
    "right-hand-rig",
)
HAND_ASSET_BLEND_PATH = PROJECT / "syntheticAssets" / "rigged_hand.blend"
HAND_SKIN_MATERIAL_NAME = "SyntheticHandTexturedMaterial"
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
HAND_ASSET_IMPORTED = False

SOLVED_LOCATION_PROP = "synthetic_solved_location"
SOLVED_ROTATION_PROP = "synthetic_solved_rotation"
DEFAULT_OUTPUT_DIR = "data/synthetic_cube_core_dataset"
RFDETR_KEYPOINT_NATIVE_INPUT_SIZE = (576, 576)
DEFAULT_IMAGE_SIZE = 144
RFDETR_KEYPOINT_DATASET_RESOLUTIONS = ((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),)
DEFAULT_HARD_CASE = "none"
SOLID_BACKGROUND_PATCH_HARD_CASE = "solid_background_patch"
DIAGONAL_CAMERA_ROLL_HARD_CASE = "diagonal_camera_roll"
SHARP_CUBE_SHADOWS_HARD_CASE = "sharp_cube_shadows"
SOLVED_CUBE_HARD_CASE = "solved_cube"
NEAR_SIDE_FACE_HARD_CASE = "near_side_face"
PALM_GAP_HARD_CASE = "palm_gap"
# Automatic dataset-generation mixture.  The solved state is sampled
# independently from the deliberate hard-case factor, so a solved cube can
# appear in either an ordinary scene or a non-solved hard-case layout.
HARD_CASE_GLOBAL_PROBABILITY = 0.40
GLOBAL_SOLVED_CUBE_PROBABILITY = 0.15
NON_SOLVED_HARD_CASES = (
    SOLID_BACKGROUND_PATCH_HARD_CASE,
    DIAGONAL_CAMERA_ROLL_HARD_CASE,
    SHARP_CUBE_SHADOWS_HARD_CASE,
    NEAR_SIDE_FACE_HARD_CASE,
    PALM_GAP_HARD_CASE,
)
HARD_CASE_AUTO_SAMPLING_ENABLED = True
# Procedural hard-case ranges. Each sample is still drawn from these ranges;
# they are broad enough to avoid memorizing one composition.
PALM_HULL_DISTANCE_RANGE = (2.50, 3.80)
PALM_HULL_WIDTH_RANGE = (3.00, 4.20)
PALM_HULL_HEIGHT_RANGE = (2.40, 3.40)
PALM_HULL_DEPTH_RANGE = (0.08, 0.18)
PALM_HAND_LIFT_RANGE = (0.85, 1.20)
SOLID_PATCH_BACKING_WIDTH_RANGE = (2.80, 4.10)
SOLID_PATCH_BACKING_DEPTH_RANGE = (0.16, 0.32)
SOLID_PATCH_BACKING_HEIGHT_RANGE = (0.75, 1.35)
SOLID_PATCH_GAP_RANGE = (0.012, 0.045)
# Secondary cubes are tabletop objects behind the target, rather than
# side-by-side distractors. The layout modes below include a controlled subset
# whose projected silhouettes overlap the target from the camera.
SECONDARY_CUBE_BEHIND_DEPTH_RANGE = (2.20, 4.20)
SECONDARY_CUBE_LATERAL_OVERLAP_RANGE = (-0.78, 0.78)
SECONDARY_CUBE_LATERAL_NEAR_RANGE = (-1.15, 1.15)
SECONDARY_CUBE_LATERAL_OFFSET_RANGE = (1.35, 2.10)
SECONDARY_CUBE_LAYOUT_WEIGHTS = (0.52, 0.33, 0.15)
SECONDARY_CUBE_MIN_SILHOUETTE_OVERLAP = 0.12
SECONDARY_CUBE_SCALE_RANGE = (0.45, 0.65)
# Deliberate cases have dedicated builders, and the automatic mixture below
# selects among them per image during ordinary generation. Add a case-specific
# builder to HARD_CASE_BUILDERS when a new failure is found; the builder should
# set only the requested scene variables and return the same scene metadata
# tuple as randomize_scene().
HARD_CASE_BUILDERS: dict[str, object] = {}
# Purging Blender's orphan-ID graph is more expensive than ordinary object
# cleanup, so do it periodically during long-running generation.
MEMORY_CLEANUP_INTERVAL = 32
DEFAULT_CYCLES_SAMPLES = 128
CYCLES_ADAPTIVE_THRESHOLD = 0.03
CYCLES_ADAPTIVE_MIN_SAMPLES = 16
INDEPENDENT_CLASS_NAMES = ["corner", "corner-edge", "corner-edge-center"]
CORNER_CLASS_ID = 0
CORNER_EDGE_CLASS_ID = 1
CORNER_EDGE_CENTER_CLASS_ID = 2
CENTER_CLASS_ID = 3
INDEPENDENT_CLASS_MASK_FILL_COLORS = {
    CORNER_CLASS_ID: (90, 90, 90),
    CORNER_EDGE_CLASS_ID: (0, 95, 95),
    CORNER_EDGE_CENTER_CLASS_ID: (95, 0, 95),
}
INDEPENDENT_CLASS_MASK_POINT_COLORS = {
    CORNER_CLASS_ID: (255, 255, 255),
    CORNER_EDGE_CLASS_ID: (0, 255, 255),
    CORNER_EDGE_CENTER_CLASS_ID: (255, 120, 255),
}
CUBE_CORE_CLASS_NAME = "cube-core"
COMBINED_KEYPOINT_CLASS_NAMES = (
    *INDEPENDENT_CLASS_NAMES, *STICKER_KEYPOINT_CLASS_NAMES, CUBE_CORE_CLASS_NAME,
)
CUBE_CORE_CLASS_ID = COMBINED_KEYPOINT_CLASS_NAMES.index(CUBE_CORE_CLASS_NAME)
CUBE_CORE_KEYPOINT_SHAPE = [len(STICKER_KEYPOINT_CLASS_NAMES), 3]
STICKER_KEYPOINT_DATASET_CLASS_ID_OFFSET = len(INDEPENDENT_CLASS_NAMES)
STICKER_KEYPOINT_DATASET_CLASS_ID_BY_MATERIAL = {
    material_name: STICKER_KEYPOINT_DATASET_CLASS_ID_OFFSET + class_id
    for material_name, class_id in STICKER_KEYPOINT_CLASS_ID_BY_MATERIAL.items()
}
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
# A face whose normal is nearly perpendicular to the camera ray is seen almost
# edge-on.  Its geometric keypoints are poorly supported by image evidence,
# so do not train on them.  The angle is measured from face-on (0 degrees) to
# edge-on (90 degrees).
FACE_VISIBILITY_MIN_PLANE_ANGLE_DEGREES = 10.0
CORNER_FACE_VISIBILITY_MIN_PLANE_ANGLE_DEGREES = 5.0
# Keep a named independent-mode alias because this policy is part of the
# independent-keypoint dataset contract.
INDEPENDENT_FACE_VISIBILITY_MIN_PLANE_ANGLE_DEGREES = FACE_VISIBILITY_MIN_PLANE_ANGLE_DEGREES
# Visibility=1 is a valid Ultralytics pose target, but it still creates a
# positive detection for a point that may have no visible image evidence.  The
# independent detector is intended to find visual intersections, so omit such
# points by default.  The CLI can opt back into the old behavior for studies
# that explicitly want occluded-point inference.
INDEPENDENT_INCLUDE_OCCLUDED_KEYPOINTS = False


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
    yaw_degrees = random.uniform(*CUBE_POSE_ROTATION_DEGREES)
    if floating:
        pitch_degrees = random.uniform(*CUBE_FLOAT_TILT_DEGREES)
        roll_degrees = random.uniform(*CUBE_FLOAT_TILT_DEGREES)
    else:
        # A tabletop cube may spin around the table normal, but it must not
        # pitch or roll. Whole-cube x/y/z orientation moves are applied
        # separately for solved samples before this scene-level pose.
        pitch_degrees = 0.0
        roll_degrees = 0.0
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


def transformed_min_z(
    objects: list[bpy.types.Object],
    rotation: Quaternion | Matrix,
) -> float:
    rotation_matrix = (
        rotation.to_matrix().to_4x4()
        if isinstance(rotation, Quaternion)
        else rotation
    )
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
    apply: bool = True,
    force_on_table: bool = False,
) -> dict[str, object]:
    floating = not force_on_table and random.random() < CUBE_FLOAT_PROBABILITY
    rotation, rotation_metadata = random_cube_pose_rotation(floating)
    if force_on_table:
        clearance = 0.0
    else:
        clearance_range = CUBE_FLOAT_CLEARANCE_RANGE if floating else CUBE_TABLE_CLEARANCE_RANGE
        clearance = random.uniform(*clearance_range)
    min_z = transformed_min_z(bounds_objects or objects, rotation)
    translation = Vector((
        random.uniform(*CUBE_POSE_XY_JITTER_RANGE),
        random.uniform(*CUBE_POSE_XY_JITTER_RANGE),
        BACKGROUND_TABLE_Z + clearance - min_z,
    ))
    bounds_for_projection = bounds_objects or objects
    transform = Matrix.Translation(translation) @ rotation.to_matrix().to_4x4()
    edge_clip_requested = random.random() < CUBE_EDGE_CLIP_PROBABILITY
    edge_clip_side: str | None = None
    edge_clip_overshoot = 0.0
    projected_bounds = None

    if edge_clip_requested:
        # Solve a small one-dimensional image-space placement problem. A
        # direct world-space XY range is useful for ordinary off-center
        # variation, but it does not produce a consistent amount of clipping
        # across camera viewpoints. Nudge a sampled pose toward one frame edge
        # using finite-difference image gradients, while preserving its Z
        # placement and therefore its table clearance.
        projected_bounds = projected_object_group_bounds(
            bounds_for_projection,
            transform,
        )
        if projected_bounds is not None:
            edge_clip_side = random.choice(("left", "right", "bottom", "top"))
            edge_index = {
                "left": 0,
                "right": 1,
                "bottom": 2,
                "top": 3,
            }[edge_clip_side]
            edge_clip_overshoot = random.uniform(*CUBE_EDGE_CLIP_OVERSHOOT_RANGE)
            target_edge = (
                -edge_clip_overshoot
                if edge_clip_side in ("left", "bottom")
                else 1.0 + edge_clip_overshoot
            )
            probe_distance = 0.25
            probe_x = Matrix.Translation(Vector((probe_distance, 0.0, 0.0))) @ transform
            probe_y = Matrix.Translation(Vector((0.0, probe_distance, 0.0))) @ transform
            projected_x = projected_object_group_bounds(bounds_for_projection, probe_x)
            projected_y = projected_object_group_bounds(bounds_for_projection, probe_y)
            if projected_x is not None and projected_y is not None:
                gradient = Vector((
                    (projected_x[edge_index] - projected_bounds[edge_index]) / probe_distance,
                    (projected_y[edge_index] - projected_bounds[edge_index]) / probe_distance,
                ))
                gradient_length_squared = gradient.length_squared
                if gradient_length_squared > 1e-8:
                    correction = (
                        (target_edge - projected_bounds[edge_index])
                        * gradient
                        / gradient_length_squared
                    )
                    translation.x += correction.x
                    translation.y += correction.y
                    transform = Matrix.Translation(translation) @ rotation.to_matrix().to_4x4()
                    projected_bounds = projected_object_group_bounds(
                        bounds_for_projection,
                        transform,
                    )

    if apply:
        transform = apply_global_pose(objects, rotation, translation)
    if projected_bounds is None:
        projected_bounds = projected_object_group_bounds(
            bounds_for_projection,
            transform,
        )

    actual_edge_clip = bool(
        projected_bounds is not None
        and (
            projected_bounds[0] < 0.0
            or projected_bounds[1] > 1.0
            or projected_bounds[2] < 0.0
            or projected_bounds[3] > 1.0
        )
    )
    return {
        "mode": "on_table" if force_on_table else ("floating" if floating else "near_table"),
        "desk_no_hands": force_on_table,
        "clearance": clearance,
        "table_z": BACKGROUND_TABLE_Z,
        "translation": [translation.x, translation.y, translation.z],
        "transform": matrix_rows(transform),
        "projected_bounds": list(projected_bounds) if projected_bounds is not None else None,
        "edge_clip_requested": edge_clip_requested,
        "edge_clip_applied": actual_edge_clip,
        "edge_clip_side": edge_clip_side,
        "edge_clip_overshoot": edge_clip_overshoot if actual_edge_clip else 0.0,
        **rotation_metadata,
    }


def objects_on_face(objects: list[bpy.types.Object], axis: Vector) -> list[bpy.types.Object]:
    return [obj for obj in objects if round(obj.location.dot(axis)) == 1]


def partial_turn_axis(move: str) -> Vector:
    """Return the signed axis used by an outer-face or slice turn."""
    axes = face_specs()
    if move in axes:
        return axes[move]
    # Singmaster slice directions follow their neighboring face turns:
    # M like L, E like D, and S like F. The direction is randomized below,
    # but retaining this convention keeps metadata and geometry consistent.
    slice_axes = {
        "M": axes["L"],
        "E": axes["D"],
        "S": axes["F"],
    }
    try:
        return slice_axes[move]
    except KeyError as exc:
        raise ValueError(f"Unsupported partial-turn move {move!r}.") from exc


def partial_turn_layer_coordinate(move: str) -> int:
    """Return the cubie-layer coordinate selected by a partial-turn move."""
    return 0 if move in SLICE_MOVES else 1


def objects_on_partial_turn_layer(
    objects: list[bpy.types.Object],
    move: str,
) -> list[bpy.types.Object]:
    axis = partial_turn_axis(move)
    coordinate = partial_turn_layer_coordinate(move)
    return [obj for obj in objects if round(obj.location.dot(axis)) == coordinate]


def snap_to_cube_grid(location: Vector) -> Vector:
    return Vector((round(location.x), round(location.y), round(location.z)))


# Include both face turns and whole-cube quarter-turns in generated scrambles.
# Solved samples use the same move application path in orientation-only mode,
# restricted to x/y/z whole-cube turns.
SCRAMBLE_MOVE_NAMES = tuple("RLUDBFxyz")
ORIENTATION_ONLY_MOVE_NAMES = tuple("xyz")


def whole_cube_move_axis(move: str) -> Vector:
    """Return the axis for a Singmaster whole-cube quarter turn.

    The existing face turns use -90 degrees around the named face normal.
    With that convention, x is R L' M', y is U D' E', and z is F B' S.
    """
    if move == "x":
        return Vector((1.0, 0.0, 0.0))
    if move == "y":
        return Vector((0.0, 0.0, 1.0))
    if move == "z":
        return face_specs()["F"]
    raise ValueError(f"Unsupported whole-cube move {move!r}; expected x, y, or z.")


def rotate_whole_cube(objects: list[bpy.types.Object], move: str) -> None:
    """Apply a whole-cube turn to every cubie, including the six centers."""
    axis = whole_cube_move_axis(move)
    turn = Quaternion(axis, -math.pi / 2.0)
    for obj in objects:
        obj.rotation_mode = "QUATERNION"
        obj.location = snap_to_cube_grid(turn @ obj.location)
        obj.rotation_quaternion = turn @ obj.rotation_quaternion
    bpy.context.view_layer.update()


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


def apply_scramble_move(objects: list[bpy.types.Object], move: str) -> None:
    if move in "RLUDBF":
        rotate_face(objects, move)
    elif move in "xyz":
        rotate_whole_cube(objects, move)
    else:
        raise ValueError(f"Unsupported scramble move {move!r}.")


def random_move_sequence(length: int, move_names: tuple[str, ...]) -> list[str]:
    moves = []
    for _ in range(length):
        choices = [
            move
            for move in move_names
            if not moves or move != moves[-1]
        ]
        moves.append(random.choice(choices))
    return moves


def random_scramble_moves(length: int) -> list[str]:
    return random_move_sequence(length, SCRAMBLE_MOVE_NAMES)


def random_orientation_moves(length: int) -> list[str]:
    # Keep at least one orientation move even when callers request a zero
    # scramble length, since solved samples still need a randomized cube
    # orientation.
    return random_move_sequence(max(1, length), ORIENTATION_ONLY_MOVE_NAMES)


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
    if move not in (*face_specs(), *SLICE_MOVES):
        raise ValueError(
            f"Unsupported move {move!r}; expected an outer face or one of M, E, S."
        )

    axis = partial_turn_axis(move)
    moving = objects_on_partial_turn_layer(objects, move)
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
    if not CUBE_MISALIGNMENT_ENABLED:
        return {"mode": "none"}

    if random.random() >= PARTIAL_TURN_APPLIED_PROBABILITY:
        return {"mode": "none"}

    def apply_operation(move: str) -> dict[str, object]:
        angle_degrees = random.uniform(*MID_TURN_DEGREES_RANGE)
        direction = random_turn_direction()
        moving = rotate_face_continuous(objects, move, angle_degrees, direction)
        return {
            "move": move,
            "angle_degrees": angle_degrees,
            "direction": direction,
            "objects": moving,
        }

    def state_from_operations(
        mode: str,
        operations: list[dict[str, object]],
    ) -> dict[str, object]:
        state: dict[str, object] = {
            "mode": mode,
            "operations": operations,
            "moves": [operation["move"] for operation in operations],
        }
        # Retain the original top-level fields for single-turn consumers and
        # for metadata readers that predate the generalized operation list.
        if len(operations) == 1:
            operation = operations[0]
            state.update({
                "face": operation["move"],
                "angle_degrees": operation["angle_degrees"],
                "direction": operation["direction"],
                "objects": operation["objects"],
            })
        return state

    if random.random() < SPECIAL_PARTIAL_TURN_PROBABILITY:
        if random.random() < 0.5:
            move = random.choice(SLICE_MOVES)
            return state_from_operations("slice_turn", [apply_operation(move)])

        # Full simultaneous misalignments are limited to opposite layers.
        # Independent angles make examples such as U=33 and D=25 possible.
        moves = list(random.choice(OPPOSITE_FACE_PAIRS))
        random.shuffle(moves)
        return state_from_operations(
            "simultaneous_turn",
            [apply_operation(move) for move in moves],
        )

    state = random.choices(
        [mode for mode, _weight in SINGLE_PARTIAL_STATE_WEIGHTS],
        weights=[weight for _mode, weight in SINGLE_PARTIAL_STATE_WEIGHTS],
        k=1,
    )[0]
    face = random.choice(list("RLUDBF"))
    if state == "mid_turn":
        return state_from_operations("mid_turn", [apply_operation(face)])

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
        "operations": [
            {
                "move": face,
                "angle_degrees": small_angle_degrees,
                "direction": small_direction,
                "objects": small_objects,
            },
            {
                "move": adjacent_face,
                "angle_degrees": adjacent_angle_degrees,
                "direction": adjacent_direction,
                "objects": adjacent_objects,
            },
        ],
        "moves": [face, adjacent_face],
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


def scene_scramble_moves(scramble_length: int, solved: bool) -> list[str]:
    """Return scramble-mode or orientation-only cube moves for a scene."""
    if solved:
        # A solved state still needs random whole-cube orientation, even when
        # ordinary face scrambles are disabled by the CLI or a preset.
        return random_orientation_moves(scramble_length)
    if not SCRAMBLES_ENABLED:
        return []
    return random_scramble_moves(scramble_length)


def scene_move_mode(solved: bool) -> str:
    if solved:
        return "orientation_only"
    return "scramble" if SCRAMBLES_ENABLED else "none"


def scene_layer_render_state(objects: list[bpy.types.Object]) -> dict[str, object]:
    """Return a partial-turn state only when mid-turn augmentation is enabled."""
    if not CUBE_MISALIGNMENT_ENABLED:
        return {"mode": "none"}
    return randomize_layer_render_state(objects)


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


def random_camera_distance(
    distance_buckets: tuple[tuple[str, float, tuple[float, float]], ...] | None = None,
) -> tuple[str, float, float]:
    distance_buckets = distance_buckets or CAMERA_DISTANCE_BUCKETS_FEET
    bucket_name, _weight, distance_range = random.choices(
        distance_buckets,
        weights=[weight for _name, weight, _range in distance_buckets],
        k=1,
    )[0]
    distance_feet = random.uniform(*distance_range)
    return bucket_name, distance_feet, distance_feet * CAMERA_SCENE_UNITS_PER_FOOT


def camera_zoom_scale_for_distance(distance_feet: float) -> float:
    near_feet, far_feet = CAMERA_CUBE_ZOOM_DISTANCE_RANGE_FEET
    minimum_scale, maximum_scale = CAMERA_CUBE_ZOOM_SCALE_RANGE
    if far_feet <= near_feet:
        return maximum_scale
    distance_fraction = (distance_feet - near_feet) / (far_feet - near_feet)
    distance_fraction = max(0.0, min(1.0, distance_fraction))
    return minimum_scale + (maximum_scale - minimum_scale) * distance_fraction


def randomize_camera(
    viewpoint_weights: tuple[tuple[str, float], ...] | None = None,
    distance_buckets: tuple[tuple[str, float, tuple[float, float]], ...] | None = None,
) -> dict[str, object]:
    camera = get_camera()
    viewpoint_weights = viewpoint_weights or CAMERA_VIEWPOINT_WEIGHTS
    viewpoint = random.choices(
        [name for name, _weight in viewpoint_weights],
        weights=[weight for _name, weight in viewpoint_weights],
        k=1,
    )[0]
    if viewpoint == "front":
        azimuth_degrees = -90.0 + random.uniform(-17.0, 17.0)
        elevation_degrees = random.uniform(18.0, 36.0)
        reference_distance = 11.75
    elif viewpoint == "front_diagonal":
        azimuth_degrees = random.choice((-45.0, -135.0)) + random.uniform(-14.0, 14.0)
        elevation_degrees = random.uniform(20.0, 43.0)
        reference_distance = 12.25
    elif viewpoint == "over_shoulder":
        azimuth_degrees = random.choice((58.0, 122.0)) + random.uniform(-11.0, 11.0)
        elevation_degrees = random.uniform(57.0, 68.0)
        reference_distance = 13.75
    elif viewpoint == "behind_diagonal":
        azimuth_degrees = random.choice((42.0, 138.0)) + random.uniform(-13.0, 13.0)
        elevation_degrees = random.uniform(27.0, 48.0)
        reference_distance = 12.5
    elif viewpoint == "low_behind_diagonal":
        azimuth_degrees = random.choice((32.0, 148.0)) + random.uniform(-12.0, 12.0)
        elevation_degrees = random.uniform(34.0, 50.0)
        reference_distance = 16.0
    elif viewpoint == "low_side":
        azimuth_degrees = random.choice((0.0, 180.0)) + random.uniform(-13.0, 13.0)
        elevation_degrees = random.uniform(10.0, 20.0)
        reference_distance = 13.0
    else:
        azimuth_degrees = random.choice((-135.0, -90.0, -45.0, 45.0, 90.0, 135.0))
        azimuth_degrees += random.uniform(-15.0, 15.0)
        elevation_degrees = random.uniform(64.0, 80.0)
        reference_distance = 13.5
    roll_degrees = random.uniform(*CAMERA_ROLL_DEGREES_RANGE)

    if viewpoint == "low_behind_diagonal" and distance_buckets is None:
        distance_buckets = CAMERA_LOW_BEHIND_DISTANCE_BUCKETS_FEET
    distance_bucket, distance_feet, distance = random_camera_distance(distance_buckets)
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
    zoom_scale = camera_zoom_scale_for_distance(distance_feet)
    camera.data.lens = framing_lens * distance / reference_distance * zoom_scale
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
        "cube_zoom_scale": zoom_scale,
        "roi_crop": {
            "method": "camera_lens_and_sensor_shift",
            "shift_x": camera.data.shift_x,
            "shift_y": camera.data.shift_y,
        },
    }


def aim_camera_at_final_cube(
    camera_randomization: dict[str, object],
    cube_pose: dict[str, object],
) -> None:
    """Follow the known cube center with bounded, zoom-aware framing offsets.

    This is a constant-time transform update, not a mesh projection or a
    visibility check. It runs after hand lift, and again if camera recovery
    replaces the sampled camera. Camera position, lens, and roll stay fixed.
    """
    camera = get_camera()
    zoom = max(1.0, float(camera_randomization.get("cube_zoom_scale", 1.0)))
    if camera_randomization.get("viewpoint") == "behind_palm":
        zoom *= max(
            1.0,
            float(camera_randomization["distance"]) / CAMERA_BEHIND_PALM_REFERENCE_DISTANCE,
        )
    # Preserve the original draws so repeated calls don't compound offsets.
    offsets = camera_randomization.setdefault("sampled_framing_offsets", {
        "target": list(camera_randomization.get("target", (0.0, 0.0, 0.0))),
        "shift_x": camera.data.shift_x,
        "shift_y": camera.data.shift_y,
    })
    target_offset = Vector(tuple(
        max(-CAMERA_TARGET_JITTER, min(CAMERA_TARGET_JITTER, float(value))) / zoom
        for value in offsets["target"]
    ))
    target = Vector(cube_pose["translation"]) + target_offset
    look_at(camera, target)
    camera.rotation_quaternion = camera.rotation_quaternion @ Quaternion(
        Vector((0.0, 0.0, -1.0)),
        math.radians(float(camera_randomization.get("roll_degrees", 0.0))),
    )
    for axis, bounds in (("x", CAMERA_ROI_SHIFT_X_RANGE), ("y", CAMERA_ROI_SHIFT_Y_RANGE)):
        value = max(bounds[0], min(bounds[1], float(offsets[f"shift_{axis}"]))) / zoom
        setattr(camera.data, f"shift_{axis}", value)
    camera_randomization["target"] = list(target)
    camera_randomization["target_follows_final_cube"] = True
    camera_randomization["framing_offset_scale"] = 1.0 / zoom
    camera_randomization.setdefault("roi_crop", {}).update({
        "shift_x": camera.data.shift_x,
        "shift_y": camera.data.shift_y,
    })
    bpy.context.view_layer.update()


def camera_cube_visibility_probe(objects: list[bpy.types.Object]) -> dict[str, object]:
    """Check whether room geometry blocks rays from the camera to the cube."""
    scene = bpy.context.scene
    camera = get_camera()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cube_mesh_names = cube_render_mesh_names(objects)
    origin = camera.matrix_world.translation
    visible_rays = 0
    missed_rays = 0
    rays_cast = 0
    blocker_counts: dict[str, int] = {}
    required_visible_rays = max(3, len(objects) // 8)

    for obj in objects:
        target = obj.matrix_world.translation
        offset = target - origin
        if offset.length <= 1e-6:
            continue
        rays_cast += 1
        direction = offset.normalized()
        ray_origin = origin
        for _ in range(32):
            hit, location, _normal, _polygon_index, hit_object, _matrix = scene.ray_cast(
                depsgraph,
                ray_origin,
                direction,
                distance=offset.length + 0.05,
            )
            if not hit or hit_object is None:
                missed_rays += 1
                break
            if hit_object.name in cube_mesh_names:
                visible_rays += 1
                break
            if (
                getattr(hit_object, "hide_render", False)
                or not getattr(hit_object, "visible_camera", True)
            ):
                ray_origin = location + direction * 1e-4
                continue
            blocker_counts[hit_object.name] = blocker_counts.get(hit_object.name, 0) + 1
            break
        if visible_rays >= required_visible_rays:
            break

    dominant_blocker = max(blocker_counts, key=blocker_counts.get) if blocker_counts else None
    return {
        "accepted": visible_rays >= required_visible_rays,
        "visible_cube_rays": visible_rays,
        "required_visible_cube_rays": required_visible_rays,
        "rays_cast": rays_cast,
        "missed_rays": missed_rays,
        "dominant_blocker": dominant_blocker,
        "dominant_blocker_rays": blocker_counts.get(dominant_blocker, 0) if dominant_blocker else 0,
        "blockers": blocker_counts,
    }


def ensure_camera_has_clear_cube_view(
    objects: list[bpy.types.Object],
    initial_randomization: dict[str, object],
    max_general_attempts: int = 12,
    max_fallback_attempts: int = 12,
) -> dict[str, object]:
    """Retry cameras hidden behind room geometry, favoring known-open views as fallback."""
    camera = get_camera()
    best: tuple[int, Matrix, float, float, float, dict[str, object], dict[str, object]] | None = None
    attempt = 0
    randomization = initial_randomization

    fallback_viewpoints = (("front", 0.55), ("front_diagonal", 0.45))
    fallback_distances = (("near_fallback", 1.0, (3.0, 4.5)),)
    total_attempts = max_general_attempts + max_fallback_attempts
    while attempt < total_attempts:
        attempt += 1
        bpy.context.view_layer.update()
        probe = camera_cube_visibility_probe(objects)
        score = int(probe["visible_cube_rays"])
        if best is None or score > best[0]:
            best = (
                score,
                camera.matrix_world.copy(),
                camera.data.lens,
                camera.data.shift_x,
                camera.data.shift_y,
                dict(randomization),
                probe,
            )
        if probe["accepted"]:
            randomization["blocking_retries"] = attempt - 1
            randomization["visibility_probe"] = probe
            return randomization

        if attempt < max_general_attempts:
            randomization = randomize_camera()
        else:
            randomization = randomize_camera(
                viewpoint_weights=fallback_viewpoints,
                distance_buckets=fallback_distances,
            )

    if best is None:
        raise RuntimeError("Camera visibility probing produced no candidates.")
    _score, matrix_world, lens, shift_x, shift_y, randomization, probe = best
    camera.matrix_world = matrix_world
    camera.data.lens = lens
    camera.data.shift_x = shift_x
    camera.data.shift_y = shift_y
    bpy.context.view_layer.update()
    randomization["blocking_retries"] = total_attempts - 1
    randomization["visibility_probe"] = probe
    randomization["visibility_fallback_exhausted"] = True
    return randomization


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


def blend_light_color(
    base_color: tuple[float, float, float],
    tint_color: tuple[float, float, float] | None,
    tint_amount: float,
) -> tuple[float, float, float]:
    """Blend a restrained tint into a temperature-derived light color."""
    if tint_color is None or tint_amount <= 0.0:
        return base_color
    amount = max(0.0, min(tint_amount, 1.0))
    return tuple(
        max(0.0, min((1.0 - amount) * base + amount * tint, 1.0))
        for base, tint in zip(base_color, tint_color)
    )


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
    color_override: tuple[float, float, float] | None = None,
    color_tint: tuple[float, float, float] | None = None,
    tint_amount: float = 0.0,
) -> dict[str, object]:
    """Create one intentionally placed source and return reproducibility metadata."""
    name = f"{SYNTHETIC_LIGHT_PREFIX}.{role}"
    data = bpy.data.lights.new(name, type=light_type)
    light = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(light)
    light.location = location
    look_at(light, target)

    color = color_override or blend_light_color(
        kelvin_to_rgb(temperature_kelvin),
        color_tint,
        tint_amount,
    )
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
        "color_tint": list(color_tint) if color_tint is not None else None,
        "tint_amount": tint_amount,
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


def add_floor_lamp_source(
    target: Vector,
    temperature_kelvin: float,
    color_tint: tuple[float, float, float] | None = None,
    tint_amount: float = 0.0,
) -> dict[str, object]:
    """Add a broad, elevated lamp-like key light 10–15 feet from the cube."""
    table_forward, table_right = camera_table_basis()
    side = random.choice((-1.0, 1.0))
    distance_feet = random.uniform(*FLOOR_LAMP_DISTANCE_FEET_RANGE)
    distance = distance_feet * CAMERA_SCENE_UNITS_PER_FOOT
    horizontal_direction = (
        table_right * side * random.uniform(0.82, 1.0)
        + table_forward * random.uniform(-0.35, 0.35)
    ).normalized()
    location = target + horizontal_direction * distance
    location.z = (
        BACKGROUND_TABLE_Z
        + random.uniform(*FLOOR_LAMP_HEIGHT_FEET_RANGE)
        * CAMERA_SCENE_UNITS_PER_FOOT
    )
    lamp_target = Vector((
        target.x + random.uniform(-0.4, 0.4),
        target.y + random.uniform(-0.4, 0.4),
        random.uniform(BACKGROUND_TABLE_Z, 0.1),
    ))
    return add_light_source(
        "floor_lamp",
        "AREA",
        location,
        lamp_target,
        random.uniform(*FLOOR_LAMP_ENERGY_RANGE),
        temperature_kelvin + random.uniform(-220.0, 220.0),
        size=random.uniform(*FLOOR_LAMP_SIZE_RANGE),
        size_y=random.uniform(*FLOOR_LAMP_SIZE_RANGE),
        color_tint=color_tint,
        tint_amount=tint_amount,
    )


def configure_world_lighting(
    strength_range: tuple[float, float],
    temperature_range: tuple[float, float],
    color_tint: tuple[float, float, float] | None = None,
    tint_amount: float = 0.0,
) -> dict[str, object]:
    world = bpy.context.scene.world or bpy.data.worlds.new("SyntheticDataWorld")
    bpy.context.scene.world = world
    strength = random.uniform(*strength_range)
    temperature_kelvin = random.uniform(*temperature_range)
    temperature_color = kelvin_to_rgb(temperature_kelvin)
    tint = blend_light_color(temperature_color, color_tint, tint_amount)
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
        "color_tint": list(color_tint) if color_tint is not None else None,
        "tint_amount": tint_amount,
    }


def solve_station_metadata(background_randomization: dict[str, object] | None) -> dict[str, object] | None:
    if not background_randomization:
        return None
    objects = background_randomization.get("objects")
    if not isinstance(objects, list):
        return None
    return next(
        (
            item
            for item in objects
            if isinstance(item, dict) and item.get("family") == "solve_station"
        ),
        None,
    )


def laptop_screen_objects() -> list[bpy.types.Object]:
    return [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH"
        and not obj.hide_render
        and any(
            material is not None
            and material.name.startswith("SolveStation.Material.LaptopScreen")
            for material in obj.data.materials
        )
    ]


def add_laptop_screen_glow(
    background_randomization: dict[str, object] | None,
    target: Vector,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    station = solve_station_metadata(background_randomization)
    requested = bool(station and station.get("laptop_glow_enabled"))
    metadata: dict[str, object] = {
        "enabled": False,
        "probability": LAPTOP_GLOW_PROBABILITY,
    }
    if not requested:
        return None, metadata

    screens = laptop_screen_objects()
    if not screens:
        metadata["reason"] = "laptop_screen_not_found"
        return None, metadata

    screen = screens[0]
    corners = [screen.matrix_world @ Vector(corner) for corner in screen.bound_box]
    minimum = Vector((
        min(point.x for point in corners),
        min(point.y for point in corners),
        min(point.z for point in corners),
    ))
    maximum = Vector((
        max(point.x for point in corners),
        max(point.y for point in corners),
        max(point.z for point in corners),
    ))
    screen_center = (minimum + maximum) * 0.5
    glow_target = Vector((
        target.x + random.uniform(-0.35, 0.35),
        target.y + random.uniform(-0.35, 0.35),
        target.z + random.uniform(-0.25, 0.35),
    ))
    toward_cube = glow_target - screen_center
    if toward_cube.length <= 1e-6:
        toward_cube = Vector((0.0, 1.0, 0.0))
    light_location = screen_center + toward_cube.normalized() * 0.16
    extents = sorted((maximum.x - minimum.x, maximum.y - minimum.y, maximum.z - minimum.z))
    stored_glow_color = station.get("laptop_glow_color") if station else None
    glow_color = (
        tuple(float(channel) for channel in stored_glow_color[:3])
        if isinstance(stored_glow_color, list) and len(stored_glow_color) >= 3
        else kelvin_to_rgb(8000.0)
    )
    source = add_light_source(
        "laptop_screen",
        "AREA",
        light_location,
        glow_target,
        random.uniform(1600.0, 3000.0),
        random.uniform(6500.0, 10000.0),
        size=max(0.8, extents[-1] * random.uniform(0.55, 0.78)),
        size_y=max(0.5, extents[-2] * random.uniform(0.55, 0.78)),
        color_override=glow_color,
    )
    metadata.update({
        "enabled": True,
        "screen_object": screen.name,
        "screen_center": list(screen_center),
        "target": list(glow_target),
        "emission_strength": station.get("laptop_emission_strength") if station else None,
        "color_name": station.get("laptop_glow_color_name") if station else None,
        "color": list(glow_color),
    })
    if station is not None:
        station["laptop_glow"] = metadata
    return source, metadata


def add_colored_wall_wash(
    background_randomization: dict[str, object] | None,
    color: tuple[float, float, float],
) -> dict[str, object] | None:
    """Aim a colored accent at the solve-station wall, away from the cube."""
    station = solve_station_metadata(background_randomization)
    collection_name = station.get("collection") if station else None
    collection = bpy.data.collections.get(collection_name) if isinstance(collection_name, str) else None
    if collection is None:
        return None
    wall = next(
        (obj for obj in collection.all_objects
         if obj.type == "MESH" and "SolveStation.Room.Wall" in obj.name and not obj.hide_render),
        None,
    )
    if wall is None:
        return None
    interior_y = min(corner[1] for corner in wall.bound_box)
    target = wall.matrix_world @ Vector((
        random.uniform(-3.0, 3.0),
        interior_y,
        random.uniform(0.0, 1.5),
    ))
    outward = (wall.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()
    location = target - outward * random.uniform(2.3, 3.4)
    location.z += random.uniform(0.5, 1.5)
    return add_light_source(
        "colored_wall_wash",
        "AREA",
        location,
        target,
        random.uniform(1150.0, 1950.0),
        6500.0,
        size=random.uniform(6.0, 9.0),
        size_y=random.uniform(4.0, 6.0),
        color_override=color,
    )


def randomize_light(background_randomization: dict[str, object] | None = None) -> dict[str, object]:
    """Sample a room-wide cast or combine a white key with colored accents."""
    clear_scene_lights()
    accent_color = None
    if random.random() < MIXED_LIGHTING_PROBABILITY:
        setup = "ceiling_lights"
        temperature_profile, _profile_weight, room_temperature_range, accent_color = random.choices(
            MIXED_LIGHTING_PROFILES,
            weights=[profile[1] for profile in MIXED_LIGHTING_PROFILES],
            k=1,
        )[0]
        window_temperature_range = None
        color_tint = None
        tint_amount = 0.0
    else:
        setup = random.choices(
            [name for name, _weight in LIGHTING_SETUP_WEIGHTS],
            weights=[weight for _name, weight in LIGHTING_SETUP_WEIGHTS],
            k=1,
        )[0]
        (
            temperature_profile,
            _profile_weight,
            room_temperature_range,
            window_temperature_range,
            color_tint,
            tint_amount,
        ) = random.choices(
            LIGHT_TEMPERATURE_PROFILES,
            weights=[profile[1] for profile in LIGHT_TEMPERATURE_PROFILES],
            k=1,
        )[0]
    base_temperature = random.uniform(*room_temperature_range)
    target = lighting_target()
    sources: list[dict[str, object]] = []
    sources.append(add_floor_lamp_source(target, base_temperature, color_tint, tint_amount))

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
                color_tint=color_tint, tint_amount=tint_amount,
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
            color_tint=color_tint, tint_amount=tint_amount,
        ))
        sources.append(add_light_source(
            "room_fill", "AREA",
            Vector((random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5), random.uniform(4.5, 6.5))),
            Vector((0.0, 0.0, BACKGROUND_TABLE_Z)),
            random.uniform(140.0, 280.0), base_temperature + random.uniform(-300.0, 300.0),
            size=random.uniform(5.0, 8.0), size_y=random.uniform(5.0, 8.0),
            color_tint=color_tint, tint_amount=tint_amount,
        ))
        world_ranges = ((0.10, 0.22), room_temperature_range)
        exposure_range = (-0.05, 0.28)

    window_metadata = None
    if window_temperature_range is not None and random.random() < SUNLIGHT_WINDOW_PROBABILITY:
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
        window_temperature = random.uniform(*window_temperature_range)
        sources.append(add_light_source(
            "window_sun", "AREA", window_location, patch_target,
            random.uniform(900.0, 1700.0), window_temperature,
            size=random.uniform(0.22, 0.42), size_y=random.uniform(0.22, 0.42),
            color_tint=color_tint, tint_amount=tint_amount,
        ))
        window_metadata = {
            "enabled": True,
            "probability": SUNLIGHT_WINDOW_PROBABILITY,
            "aperture": aperture,
            "effect": "rectangular_sun_patch",
        }
        world_ranges = ((0.14, 0.28), room_temperature_range)
        exposure_range = (exposure_range[0] - 0.16, exposure_range[1] - 0.08)

    conditions = [setup]
    if window_metadata:
        conditions.append("sunlight_window")
    if accent_color is not None:
        # Place a weaker colored fill to the side and behind the cube so
        # the overhead key still sets most of the foreground color.
        table_forward, table_right = camera_table_basis()
        fill_location = (
            target + table_forward * random.uniform(1.8, 3.5)
            + table_right * random.choice((-1.0, 1.0)) * random.uniform(3.5, 5.0)
        )
        fill_location.z = random.uniform(2.8, 4.6)
        sources.append(add_light_source(
            "colored_fill", "AREA", fill_location, target,
            random.uniform(120.0, 205.0), 6500.0,
            size=random.uniform(2.5, 4.0), size_y=random.uniform(2.5, 4.0),
            color_override=accent_color,
        ))
        conditions.append("mixed_light_colors")
        wall_wash = add_colored_wall_wash(background_randomization, accent_color)
        if wall_wash is not None:
            sources.append(wall_wash)
            conditions.append("colored_wall_wash")
        world_ranges = ((0.10, 0.18), (4500.0, 5600.0))

    laptop_source, laptop_glow_metadata = add_laptop_screen_glow(
        background_randomization,
        target,
    )
    if laptop_source is not None:
        sources.append(laptop_source)
    world = configure_world_lighting(world_ranges[0], world_ranges[1], color_tint, tint_amount)

    exposure = random.uniform(*exposure_range)
    bpy.context.scene.view_settings.exposure = exposure
    bpy.context.view_layer.update()
    primary = sources[0]
    return {
        "mode": "standard",
        "setup": setup,
        "source_count": len(sources),
        "sources": sources,
        "conditions": conditions,
        "temperature_profile": {
            "name": temperature_profile,
            "base_kelvin": base_temperature,
            "room_range_kelvin": list(room_temperature_range),
            "window_range_kelvin": list(window_temperature_range) if window_temperature_range else None,
            "tint_color": list(color_tint) if color_tint is not None else None,
            "tint_amount": tint_amount,
        },
        "sunlight_window": window_metadata or {
            "enabled": False,
            "probability": SUNLIGHT_WINDOW_PROBABILITY if window_temperature_range is not None else 0.0,
        },
        "laptop_glow": laptop_glow_metadata,
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


def srgb_channel_to_linear(channel: int) -> float:
    """Convert an 8-bit sRGB channel to Blender's linear color value."""
    normalized = channel / 255.0
    if normalized <= 0.04045:
        return normalized / 12.92
    return ((normalized + 0.055) / 1.055) ** 2.4


def randomize_sticker_imperfections(material: bpy.types.Material) -> dict[str, object] | None:
    """Add subtle procedural scratches and dents to a sticker shader."""
    if not STICKER_IMPERFECTIONS_ENABLED or not material.use_nodes:
        return None

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return None

    node_prefix = "SyntheticSticker.Imperfection."
    for node in list(nodes):
        if node.name.startswith(node_prefix):
            nodes.remove(node)
    for link in list(bsdf.inputs["Normal"].links):
        links.remove(link)

    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.name = f"{node_prefix}Coordinates"

    fine_noise = nodes.new("ShaderNodeTexNoise")
    fine_noise.name = f"{node_prefix}FineNoise"
    fine_noise.inputs["Scale"].default_value = random.uniform(
        *STICKER_IMPERFECTION_FINE_SCALE_RANGE
    )
    fine_noise.inputs["Detail"].default_value = random.uniform(2.0, 5.0)
    fine_noise.inputs["Roughness"].default_value = random.uniform(0.55, 0.8)

    scratch_wave = nodes.new("ShaderNodeTexWave")
    scratch_wave.name = f"{node_prefix}Scratches"
    scratch_wave.wave_type = "BANDS"
    scratch_wave.bands_direction = random.choice(("X", "Y", "Z"))
    scratch_wave.inputs["Scale"].default_value = random.uniform(
        *STICKER_IMPERFECTION_SCRATCH_SCALE_RANGE
    )
    scratch_wave.inputs["Distortion"].default_value = random.uniform(3.0, 8.0)
    scratch_wave.inputs["Detail"].default_value = random.uniform(2.0, 5.0)
    scratch_wave.inputs["Detail Scale"].default_value = random.uniform(1.5, 3.5)

    scratch_ramp = nodes.new("ShaderNodeValToRGB")
    scratch_ramp.name = f"{node_prefix}ScratchMask"
    scratch_ramp.color_ramp.interpolation = "CONSTANT"
    scratch_ramp.color_ramp.elements[0].position = random.uniform(0.40, 0.52)
    scratch_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    scratch_ramp.color_ramp.elements[1].position = random.uniform(0.58, 0.72)
    scratch_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    combine = nodes.new("ShaderNodeMixRGB")
    combine.name = f"{node_prefix}Combine"
    combine.blend_type = "MULTIPLY"
    combine.inputs["Fac"].default_value = random.uniform(0.28, 0.52)

    bump = nodes.new("ShaderNodeBump")
    bump.name = f"{node_prefix}Bump"
    bump.inputs["Strength"].default_value = random.uniform(
        *STICKER_IMPERFECTION_BUMP_STRENGTH_RANGE
    )
    bump.inputs["Distance"].default_value = random.uniform(
        *STICKER_IMPERFECTION_BUMP_DISTANCE_RANGE
    )
    if "Invert" in bump.inputs:
        bump.inputs["Invert"].default_value = random.random() < 0.25

    links.new(texcoord.outputs["Object"], fine_noise.inputs["Vector"])
    links.new(texcoord.outputs["Object"], scratch_wave.inputs["Vector"])
    links.new(scratch_wave.outputs["Fac"], scratch_ramp.inputs["Fac"])
    links.new(fine_noise.outputs["Fac"], combine.inputs[1])
    links.new(scratch_ramp.outputs["Color"], combine.inputs[2])
    links.new(combine.outputs["Color"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return {
        "enabled": True,
        "fine_scale": fine_noise.inputs["Scale"].default_value,
        "scratch_scale": scratch_wave.inputs["Scale"].default_value,
        "scratch_direction": scratch_wave.bands_direction,
        "bump_strength": bump.inputs["Strength"].default_value,
        "bump_distance": bump.inputs["Distance"].default_value,
    }


def randomize_sticker_material_gloss() -> dict[str, object]:
    """Randomize sticker shades from one coherent product-page palette and gloss."""
    palette_name, palette = random.choice(tuple(STICKER_COLOR_PALETTES.items()))
    material_settings: dict[str, object] = {
        "palette": palette_name,
        "palette_source": STICKER_COLOR_PALETTE_SOURCE,
    }
    for name in STICKER_COLOR_MATERIAL_NAMES:
        material = bpy.data.materials.get(name)
        if material is None:
            continue

        material.use_nodes = True
        rgb = palette[name]
        linear_rgb = tuple(srgb_channel_to_linear(channel) for channel in rgb)
        rgba = (*linear_rgb, 1.0)
        material.diffuse_color = rgba
        set_principled_input(material, "Base Color", rgba)
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
        imperfections = randomize_sticker_imperfections(material)

        material_settings[name] = {
            "rgb": list(rgb),
            "linear_rgba": list(rgba),
            "roughness": roughness,
            "specular": specular,
            "coat_weight": coat_weight,
            "coat_roughness": coat_roughness,
            "imperfections": imperfections,
        }
    return material_settings


def add_cube_logo_uv(
    obj: bpy.types.Object,
    cubie: bpy.types.Object,
    white_polygons: list[bpy.types.MeshPolygon],
) -> bool:
    """Map the white center's outward face onto a square in mesh space."""
    mesh = obj.data
    if mesh.uv_layers.get(CUBE_LOGO_UV_NAME) is not None:
        return True

    solved_location = Vector(cubie.get(SOLVED_LOCATION_PROP, cubie.location))
    solved_rotation = Quaternion(cubie.get(SOLVED_ROTATION_PROP, cubie.rotation_quaternion))
    relative_transform = cubie.matrix_world.inverted() @ obj.matrix_world
    solved_transform = solved_rotation.to_matrix() @ relative_transform.to_3x3()
    outward = (solved_transform.inverted() @ solved_location).normalized()
    front = [polygon for polygon in white_polygons if polygon.normal.dot(outward) > 0.92]
    if not front:
        return False

    up = Vector((0.0, 0.0, 1.0)) if abs(outward.z) < 0.9 else Vector((0.0, 1.0, 0.0))
    right = up.cross(outward).normalized()
    up = outward.cross(right)
    coordinates = {
        index: (mesh.vertices[index].co.dot(right), mesh.vertices[index].co.dot(up))
        for polygon in front
        for index in polygon.vertices
    }
    xs, ys = zip(*coordinates.values())
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    side = min(max(xs) - min(xs), max(ys) - min(ys))
    if side <= 1e-6:
        return False

    uv = mesh.uv_layers.new(name=CUBE_LOGO_UV_NAME, do_init=False)
    for polygon in white_polygons:
        for loop_index in polygon.loop_indices:
            uv.data[loop_index].uv = (-1.0, -1.0)
    for polygon in front:
        for loop_index, vertex_index in zip(polygon.loop_indices, polygon.vertices):
            x, y = coordinates[vertex_index]
            uv.data[loop_index].uv = (
                (x - center_x) / side + 0.5,
                (y - center_y) / side + 0.5,
            )
    return True


def randomize_cube_logo(cubies: list[bpy.types.Object]) -> dict[str, object] | None:
    """Print at most one logo, on this cube's physical white center."""
    use_logo = random.random() < CUBE_LOGO_PROBABILITY
    for cubie in center_cubie_objects(cubies):
        # Prefer detached stickers. Older assets put the white face directly
        # on the cubie mesh, so inspect that mesh last.
        for obj in reversed(mesh_objects_in_roots([cubie])):
            for index, material in enumerate(obj.data.materials):
                if material is None or material.name != "White":
                    continue
                slot = obj.material_slots[index]
                if slot.material and slot.material.name.startswith(CUBE_LOGO_MATERIAL_NAME):
                    slot.link = "DATA"
                if not use_logo or slot.material != material or not material.use_nodes:
                    continue
                bsdf = material.node_tree.nodes.get("Principled BSDF")
                white_polygons = [
                    polygon for polygon in obj.data.polygons if polygon.material_index == index
                ]
                if bsdf is None or not add_cube_logo_uv(obj, cubie, white_polygons):
                    continue

                logo_name = random.choice(CUBE_LOGO_NAMES)
                image = bpy.data.images.load(
                    str(CUBE_LOGO_DIR / f"{logo_name}.png"), check_existing=True
                )
                logo_material = material.copy()
                logo_material.name = CUBE_LOGO_MATERIAL_NAME
                nodes = logo_material.node_tree.nodes
                links = logo_material.node_tree.links
                uv = nodes.new("ShaderNodeUVMap")
                uv.uv_map = CUBE_LOGO_UV_NAME
                texture = nodes.new("ShaderNodeTexImage")
                texture.image = image
                texture.extension = "CLIP"
                links.new(uv.outputs["UV"], texture.inputs["Vector"])

                ink = texture.outputs["Color"]
                metadata: dict[str, object] = {"name": logo_name, "color_mode": "original"}
                if random.random() >= CUBE_LOGO_ORIGINAL_COLOR_PROBABILITY:
                    tint = nodes.new("ShaderNodeHueSaturation")
                    tint.inputs["Hue"].default_value = random.random()
                    tint.inputs["Saturation"].default_value = random.uniform(0.5, 0.95)
                    tint.inputs["Value"].default_value = random.uniform(0.12, 0.29)
                    links.new(ink, tint.inputs["Color"])
                    ink = tint.outputs["Color"]
                    metadata["color_mode"] = "dark_tint"
                    metadata["hsv_adjustment"] = {
                        name.lower(): float(tint.inputs[name].default_value)
                        for name in ("Hue", "Saturation", "Value")
                    }

                base_color = nodes["Principled BSDF"].inputs["Base Color"]
                mix = nodes.new("ShaderNodeMixRGB")
                mix.inputs["Color1"].default_value = base_color.default_value
                if base_color.is_linked:
                    links.new(base_color.links[0].from_socket, mix.inputs["Color1"])
                links.new(texture.outputs["Alpha"], mix.inputs["Fac"])
                links.new(ink, mix.inputs["Color2"])
                links.new(mix.outputs["Color"], base_color)

                # The mesh slot stays White for the sticker-label pipeline.
                slot.link = "OBJECT"
                slot.material = logo_material
                return metadata
    return None


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


def _background_collection_objects(collection: bpy.types.Collection) -> list[bpy.types.Object]:
    all_objects = getattr(collection, "all_objects", None)
    if all_objects is not None:
        return list(all_objects)

    objects: list[bpy.types.Object] = []
    seen: set[int] = set()

    def visit(current: bpy.types.Collection) -> None:
        for obj in current.objects:
            pointer = obj.as_pointer()
            if pointer in seen:
                continue
            seen.add(pointer)
            objects.append(obj)
        for child in current.children:
            visit(child)

    visit(collection)
    return objects


def clear_background_scene() -> None:
    """Remove the generated background and its owned datablocks.

    The imported solve-station collection also contains objects whose names do
    not use the generated prefix, so name-based cleanup leaves a complete
    asset behind on every iteration. Walk the generator-owned collection
    instead. Orphan mesh/material/image datablocks are reclaimed in batches by
    cleanup_blender_memory() to keep this hot path inexpensive.
    """
    generated_objects: list[bpy.types.Object] = []
    generated_collections: list[bpy.types.Collection] = []
    background_collection = bpy.data.collections.get(BACKGROUND_SCENE_COLLECTION_NAME)
    if background_collection is not None:
        generated_objects.extend(_background_collection_objects(background_collection))
        generated_collections.extend(list(background_collection.children))

    # Include stale generated collections left by older versions of the script.
    for collection in list(bpy.data.collections):
        if collection.name.startswith(BACKGROUND_ASSET_COLLECTION_PREFIX):
            if all(collection is not existing for existing in generated_collections):
                generated_collections.append(collection)
            for obj in _background_collection_objects(collection):
                if all(obj is not existing for existing in generated_objects):
                    generated_objects.append(obj)

    for obj in generated_objects:
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

    def remove_collection_tree(collection: bpy.types.Collection) -> None:
        for child in list(collection.children):
            remove_collection_tree(child)
        if collection.name in bpy.data.collections:
            bpy.data.collections.remove(collection, do_unlink=True)

    for collection in generated_collections:
        remove_collection_tree(collection)


def cleanup_blender_memory() -> None:
    """Periodically reclaim orphan IDs and Python-side temporary arrays."""
    purge = getattr(bpy.data, "orphans_purge", None)
    if callable(purge):
        try:
            purge(do_recursive=True)
        except TypeError:
            purge()
    gc.collect()


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


def add_procedural_camera_hull(
    collection: bpy.types.Collection,
    name: str,
    center: Vector,
    camera_direction: Vector,
    camera_right: Vector,
    camera_up: Vector,
    width: float,
    height: float,
    depth: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    """Create a shallow irregular silhouette in camera space.

    This is intentionally procedural: the asymmetric outline approximates the
    broad convex envelope of a visible palm/hand without copying hand mesh
    pixels into the label. It is placed behind the cube and the rigged hands,
    so only the palm-gap region remains visible.
    """
    outline = (
        (-0.96, -0.46),
        (-0.82, 0.18),
        (-0.48, 0.63),
        (0.02, 0.78),
        (0.58, 0.64),
        (0.96, 0.30),
        (0.86, -0.48),
        (0.32, -0.82),
        (-0.30, -0.78),
    )
    front = center + camera_direction * (depth * 0.5)
    back = center - camera_direction * (depth * 0.5)
    vertices = []
    for plane_center in (front, back):
        vertices.extend(
            plane_center
            + camera_right * (x * width * 0.5)
            + camera_up * (y * height * 0.5)
            for x, y in outline
        )
    count = len(outline)
    faces = [
        tuple(range(count)),
        tuple(range(count, 2 * count))[::-1],
    ]
    faces.extend(
        (index, (index + 1) % count, count + (index + 1) % count, count + index)
        for index in range(count)
    )
    mesh = bpy.data.meshes.new(f"{name}.Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.pass_index = 0
    obj.data.materials.append(material)
    bevel = obj.modifiers.new("Soft procedural palm-hull edge", "BEVEL")
    bevel.width = min(width, height) * 0.018
    bevel.segments = 2
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
    *,
    print_path: Path | None = None,
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
    if print_path is not None:
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.load(str(print_path), check_existing=True)
        texture.extension = "EXTEND"
        links.new(texcoord.outputs["Generated"], texture.inputs["Vector"])
        links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
        return

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


def configure_station_shirt_print(
    asset_collection: bpy.types.Collection,
    fabric: bpy.types.Material,
    design: str | None,
    *,
    chest_pocket: bool = False,
) -> bool:
    """Alpha-blend a chest print into the torso's existing fabric shader."""
    torso = next(
        (obj for obj in asset_collection.all_objects
         if obj.type == "MESH" and "SolveStation.Shirt.Torso" in obj.name),
        None,
    )
    if torso is None:
        return False
    mesh = torso.data
    slot = next(
        (torso.material_slots[index] for index, material in enumerate(mesh.materials)
         if material == fabric),
        None,
    )
    if slot is None:
        return False
    if slot.material and slot.material.name.startswith(SOLVE_STATION_SHIRT_PRINT_MATERIAL):
        slot.link = "DATA"
    if design is None or slot.material != fabric or not fabric.use_nodes:
        return False
    if fabric.node_tree.nodes.get("Principled BSDF") is None:
        return False

    # The torso faces local -Y. Map the print in physical X/Z proportions
    # so the artwork follows the shirt when the solve station is repositioned.
    xs, _, zs = zip(*torso.bound_box)
    width, height = max(xs) - min(xs), max(zs) - min(zs)
    if chest_pocket:
        # Local +X is the wearer's left on this front-facing torso.
        side = 1.75 * min(width * random.uniform(0.17, 0.23), height * 0.30)
        center_x = min(xs) + width * 0.66
        center_z = min(zs) + height * 0.72
    else:
        side = min(width * random.uniform(0.48, 0.62), height * 0.70)
        center_x = (min(xs) + max(xs)) * 0.5
        center_z = min(zs) + height * 0.61
    if side <= 1e-6 or not any(polygon.normal.y < -0.9 for polygon in mesh.polygons):
        return False
    uv_name = "SolveStationShirtPrintUV"
    uv = mesh.uv_layers.get(uv_name)
    if uv is None:
        uv = mesh.uv_layers.new(name=uv_name, do_init=False)
    for polygon in mesh.polygons:
        for loop_index, vertex_index in zip(polygon.loop_indices, polygon.vertices):
            x, _, z = mesh.vertices[vertex_index].co
            uv.data[loop_index].uv = (
                ((x - center_x) / side + 0.5, (z - center_z) / side + 0.5)
                if polygon.normal.y < -0.9
                else (-1.0, -1.0)
            )

    printed = fabric.copy()
    printed.name = SOLVE_STATION_SHIRT_PRINT_MATERIAL
    nodes = printed.node_tree.nodes
    links = printed.node_tree.links
    coords = nodes.new("ShaderNodeUVMap")
    coords.uv_map = uv_name
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(
        str(SOLVE_STATION_SHIRT_PRINT_DIR / f"{design}.png"), check_existing=True
    )
    texture.extension = "CLIP"
    links.new(coords.outputs["UV"], texture.inputs["Vector"])
    color = nodes["Principled BSDF"].inputs["Base Color"]
    mix = nodes.new("ShaderNodeMixRGB")
    mix.inputs["Color1"].default_value = color.default_value
    if color.is_linked:
        links.new(color.links[0].from_socket, mix.inputs["Color1"])
    links.new(texture.outputs["Alpha"], mix.inputs["Fac"])
    links.new(texture.outputs["Color"], mix.inputs["Color2"])
    links.new(mix.outputs["Color"], color)
    slot.link = "OBJECT"
    slot.material = printed
    return True


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
    mat_design = (
        random.choice(SOLVE_STATION_MAT_PRINTS)
        if random.random() < SOLVE_STATION_MAT_PRINT_PROBABILITY
        else None
    )
    mat_pattern = "cube_print" if mat_design else random.choices(
        ("solid", "heather", "stripes", "mottled"),
        weights=(0.3, 0.2, 0.28, 0.22),
        k=1,
    )[0]
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
    shirt_design = shirt_print_placement = None
    shirt_material = randomized_materials.get("SolveStation.Material.Shirt")
    if shirt_material is not None:
        configure_station_color_pattern(
            shirt_material,
            shirt_color,
            shirt_accent_color,
            shirt_pattern,
        )
        if random.random() < SOLVE_STATION_SHIRT_PRINT_PROBABILITY:
            shirt_design = random.choice(SOLVE_STATION_SHIRT_PRINTS)
            shirt_print_placement = (
                "chest_pocket" if random.random() < SOLVE_STATION_SHIRT_POCKET_PROBABILITY
                else "center_chest"
            )
        if not configure_station_shirt_print(
            asset_collection,
            shirt_material,
            shirt_design,
            chest_pocket=shirt_print_placement == "chest_pocket",
        ):
            shirt_design = shirt_print_placement = None
    mat_material = randomized_materials.get("SolveStation.Material.MatFabric")
    if mat_material is not None:
        print_path = (
            SOLVE_STATION_MAT_PRINT_DIR / f"{mat_design}.jpg" if mat_design else None
        )
        configure_station_color_pattern(
            mat_material,
            mat_color,
            mat_print_color,
            mat_pattern,
            print_path=print_path,
        )
        # The full-bleed print replaces the separate accents on the legacy mat.
        for obj in list(asset_collection.all_objects):
            if "SolveStation.Mat.CenterPrint" in obj.name or "SolveStation.Mat.Accent." in obj.name:
                obj.hide_render = print_path is not None
                obj.hide_viewport = print_path is not None
    return {
        "shirt_color": list(shirt_color),
        "shirt_accent_color": list(shirt_accent_color),
        "shirt_pattern": shirt_pattern,
        "shirt_design": shirt_design,
        "shirt_print_placement": shirt_print_placement,
        "skin_color": list(skin_color),
        "wall_color": list(wall_color),
        "mat_color": list(mat_color),
        "mat_print_color": list(mat_print_color),
        "mat_pattern": mat_pattern,
        "mat_design": mat_design,
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
    laptop_glow_enabled: bool = False,
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
            if laptop_glow_enabled:
                target_x = laptop_side * random.uniform(3.3, 4.0)
                target_y = random.uniform(-2.9, -2.4)
                yaw = laptop_side * random.uniform(32.0, 50.0)
            else:
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
            if laptop_glow_enabled:
                target_x = laptop_side * 3.8
                target_y = -2.8
                yaw = laptop_side * 40.0
            else:
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
            "glow_facing_cube": laptop_glow_enabled,
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


def move_rear_station_cluster_for_camera(
    asset_collection: bpy.types.Collection,
    camera_direction: Vector,
) -> dict[str, object]:
    """Move existing rear desk props out of the central rear-camera ray.

    Rear hand views can otherwise look directly through the solve timer and
    solver cluster.  The props remain present and randomized, but are placed
    on a plausible desk side so they do not create an accidental full-frame
    occlusion unrelated to the palm-gap factor.
    """
    side = -1.0 if camera_direction.x >= 0.0 else 1.0
    placements: dict[str, object] = {}
    groups = (
        ("timer", ("Timer",), side * random.uniform(3.4, 4.3), random.uniform(2.25, 2.95)),
        ("solver", ("Shirt", "Arm"), side * random.uniform(3.1, 4.0), random.uniform(1.85, 2.60)),
    )
    for name, families, target_x, target_y in groups:
        objects = solve_station_family_objects(asset_collection, families)
        if not objects:
            continue
        base = {obj.name: obj.matrix_world.copy() for obj in objects}
        yaw = random.uniform(-8.0, 8.0)
        bounds = set_object_group_local_pose(objects, base, target_x, target_y, yaw)
        placements[name] = {
            "center": [target_x, target_y],
            "yaw_degrees": yaw,
            "bounds": list(bounds),
            "rear_camera_side_layout": True,
        }
    return placements


def move_solve_station_cluster_for_camera(
    background_randomization: dict[str, object],
) -> dict[str, object] | None:
    """Move the solve-station torso/solver aside after a camera is blocked.

    The station is appended before the cube receives its final global pose.
    That pose can move the cube into a torso that was clear during the earlier
    probe, so this recovery is intentionally available at the final probe as
    well.  Return ``None`` for scenes without the solve-station background.
    """
    station_entries = background_randomization.get("objects", [])
    if not isinstance(station_entries, list):
        return None
    station_metadata = next(
        (
            entry
            for entry in station_entries
            if isinstance(entry, dict) and entry.get("family") == "solve_station"
        ),
        None,
    )
    if not isinstance(station_metadata, dict):
        return None
    collection_name = station_metadata.get("collection")
    if not isinstance(collection_name, str):
        return None
    asset_collection = bpy.data.collections.get(collection_name)
    if asset_collection is None:
        return None

    camera_direction, _camera_right, _camera_up = camera_basis()
    placements = move_rear_station_cluster_for_camera(
        asset_collection,
        camera_direction,
    )
    if placements:
        station_metadata["rear_camera_cluster_layout"] = placements
    return placements or None


def remove_solve_station_forearm_stubs(
    asset_collection: bpy.types.Collection,
) -> list[str]:
    """Remove the authored skin forearm pieces now replaced by rigged hands."""
    removed = []
    for obj in list(asset_collection.all_objects):
        if obj.name.startswith("SolveStation.Arm."):
            removed.append(obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)
    return removed


def select_solve_station_timer(asset_collection: bpy.types.Collection) -> dict[str, object]:
    """Keep one reference model before computing footprints or moving props.

    Delete unused variants instead of hiding them: hidden meshes would still
    enter placement bounds and later camera-facing transforms.
    """
    timers = solve_station_family_objects(asset_collection, ("Timer",))
    variants = sorted({obj.get("timer_variant") for obj in timers if obj.get("timer_variant")})
    if not variants:
        raise RuntimeError(
            "The solve-station library still contains the legacy timer. "
            "Rebuild it with scripts/build_solve_station_assets.py."
        )
    selected = random.choice(variants)
    reference = None
    for obj in timers:
        if obj.get("timer_variant") != selected:
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        obj.hide_render = False
        obj.hide_viewport = False
        obj.hide_set(False)
        reference = obj.get("timer_reference")
    bpy.context.view_layer.update()
    return {"variant": selected, "reference": reference, "asset_version": 2}


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

    timer_metadata = select_solve_station_timer(asset_collection)
    removed_forearm_stubs = remove_solve_station_forearm_stubs(asset_collection)

    laptop_glow_enabled = random.random() < LAPTOP_GLOW_PROBABILITY
    laptop_visible = laptop_glow_enabled or random.random() < SOLVE_STATION_LAPTOP_PROBABILITY
    phone_visible = random.random() < SOLVE_STATION_PHONE_PROBABILITY
    other_cube_visible = random.random() < SOLVE_STATION_OTHER_CUBE_PROBABILITY
    set_solve_station_family_visible(asset_collection, "Laptop", laptop_visible)
    set_solve_station_family_visible(asset_collection, "Phone", phone_visible)
    layout_metadata = randomize_solve_station_layout(
        asset_collection,
        laptop_glow_enabled=laptop_glow_enabled,
    )
    phone_layout = layout_metadata.get("phone")
    if isinstance(phone_layout, dict) and phone_layout.get("visible") is False:
        phone_visible = False
    material_metadata = randomize_solve_station_materials(asset_collection)
    laptop_emission_strength = 0.0
    laptop_glow_color = None
    laptop_glow_color_name = None
    if laptop_glow_enabled:
        laptop_emission_strength = random.uniform(4.5, 8.5)
        laptop_glow_color_name, _color_weight, laptop_glow_color = random.choices(
            LAPTOP_GLOW_COLOR_PROFILES,
            weights=[weight for _name, weight, _color in LAPTOP_GLOW_COLOR_PROFILES],
            k=1,
        )[0]
        for laptop_object in solve_station_family_objects(asset_collection, ("Laptop",)):
            for material in laptop_object.data.materials:
                if material is not None and material.name.startswith("SolveStation.Material.LaptopScreen"):
                    set_principled_input(material, "Base Color", laptop_glow_color)
                    set_principled_input(material, "Emission Color", laptop_glow_color)
                    set_principled_input(material, "Emission", laptop_glow_color)
                    set_principled_input(material, "Emission Strength", laptop_emission_strength)

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
        "removed_forearm_stubs": removed_forearm_stubs,
        "timer": timer_metadata,
        "laptop_visible": laptop_visible,
        "laptop_glow_enabled": laptop_glow_enabled,
        "laptop_emission_strength": laptop_emission_strength,
        "laptop_glow_color": list(laptop_glow_color) if laptop_glow_color else None,
        "laptop_glow_color_name": laptop_glow_color_name,
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


def projected_bounds_overlap_ratio(
    first: tuple[float, float, float, float] | None,
    second: tuple[float, float, float, float] | None,
) -> float:
    """Return intersection over the smaller projected bounding-box area."""
    if first is None or second is None:
        return 0.0
    left = max(first[0], second[0])
    right = min(first[1], second[1])
    bottom = max(first[2], second[2])
    top = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, top - bottom)
    first_area = max(0.0, first[1] - first[0]) * max(0.0, first[3] - first[2])
    second_area = max(0.0, second[1] - second[0]) * max(0.0, second[3] - second[2])
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area > 1e-8 else 0.0


def bounds_intersect_with_margin(
    first: tuple[float, float, float, float, float, float],
    second: tuple[float, float, float, float, float, float],
    margin: float = 0.0,
) -> bool:
    """Return whether two world-space AABBs intersect after adding a margin."""
    return all(
        first[index * 2] - margin < second[index * 2 + 1]
        and first[index * 2 + 1] + margin > second[index * 2]
        for index in range(3)
    )


def ensure_secondary_cube_clearance(
    main_objects: list[bpy.types.Object],
    clearance_margin: float = 0.04,
    max_attempts: int = 16,
) -> dict[str, object] | None:
    """Move a secondary cube backward until its 3D bounds cannot clip the target."""
    secondary_objects = [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and obj.name.startswith(SECONDARY_CUBE_OBJECT_PREFIX)
    ]
    if not secondary_objects:
        return None

    main_bounds = transformed_object_group_bounds(main_objects, Matrix.Identity(4))
    secondary_bounds = transformed_object_group_bounds(
        secondary_objects,
        Matrix.Identity(4),
    )
    if not bounds_intersect_with_margin(main_bounds, secondary_bounds, clearance_margin):
        return {
            "applied": False,
            "margin": clearance_margin,
            "attempts": 0,
            "shift": [0.0, 0.0, 0.0],
        }

    table_forward, _table_right = camera_table_basis()
    roots = [obj for obj in secondary_objects if obj.parent is None]
    if not roots:
        roots = secondary_objects
    total_shift = Vector((0.0, 0.0, 0.0))
    attempts = 0
    while attempts < max_attempts:
        step = table_forward * (0.45 if attempts < 8 else 0.8)
        for root in roots:
            root.matrix_world = Matrix.Translation(step) @ root.matrix_world
        total_shift += step
        attempts += 1
        bpy.context.view_layer.update()
        secondary_bounds = transformed_object_group_bounds(
            secondary_objects,
            Matrix.Identity(4),
        )
        if not bounds_intersect_with_margin(main_bounds, secondary_bounds, clearance_margin):
            break

    return {
        "applied": True,
        "margin": clearance_margin,
        "attempts": attempts,
        "shift": list(total_shift),
        "clear": not bounds_intersect_with_margin(
            main_bounds,
            secondary_bounds,
            clearance_margin,
        ),
    }


def duplicate_secondary_cube(
    objects: list[bpy.types.Object],
    core_objects: list[bpy.types.Object],
    background_randomization: dict[str, object],
    obstacle_objects: list[bpy.types.Object] | None = None,
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
    secondary_move_count = 0 if secondary_solved else random.randint(8, 22)
    secondary_moves = scene_scramble_moves(secondary_move_count, secondary_solved)
    secondary_scramble_objects = duplicate_objects + duplicate_core_objects
    for move in secondary_moves:
        apply_scramble_move(secondary_scramble_objects, move)

    table_forward, table_right = camera_table_basis()
    rotation = (
        Quaternion(Vector((0.0, 0.0, 1.0)), random.uniform(0.0, math.tau))
        @ Quaternion(Vector((1.0, 0.0, 0.0)), math.radians(random.uniform(-9.0, 9.0)))
        @ Quaternion(Vector((0.0, 1.0, 0.0)), math.radians(random.uniform(-9.0, 9.0)))
    )
    pose_roots = duplicate_objects + duplicate_core_objects
    render_parts = duplicate_cube_parts + duplicate_core_parts
    secondary_scale = random.uniform(*SECONDARY_CUBE_SCALE_RANGE)
    scale_transform = Matrix.Diagonal((secondary_scale, secondary_scale, secondary_scale, 1.0))
    placement_rotation = rotation.to_matrix().to_4x4() @ scale_transform
    main_projected_bounds = projected_object_group_bounds(
        objects + core_objects,
        Matrix.Identity(4),
    )
    collision_groups = [objects + core_objects]
    if obstacle_objects:
        visible_obstacles = [
            obj
            for obj in obstacle_objects
            if obj.type == "MESH" and not obj.hide_render and not obj.hide_get()
        ]
        if visible_obstacles:
            collision_groups.append(visible_obstacles)
    obstacle_bounds = [
        transformed_object_group_bounds(group, Matrix.Identity(4))
        for group in collision_groups
        if group
    ]
    layout_mode = random.choices(
        ("silhouette_intersect", "behind_near", "behind_offset"),
        weights=SECONDARY_CUBE_LAYOUT_WEIGHTS,
        k=1,
    )[0]
    if layout_mode == "silhouette_intersect":
        lateral_range = SECONDARY_CUBE_LATERAL_OVERLAP_RANGE
        placement_zone = "behind_overlap"
    elif layout_mode == "behind_near":
        lateral_range = SECONDARY_CUBE_LATERAL_NEAR_RANGE
        placement_zone = "behind_near"
    else:
        lateral_range = SECONDARY_CUBE_LATERAL_OFFSET_RANGE
        placement_zone = "behind_offset"
    clearance = random.uniform(0.02, 0.08)
    cropped_candidate = None
    for placement_attempt in range(64):
        right_offset = random.uniform(*lateral_range)
        if layout_mode == "behind_offset":
            right_offset *= random.choice((-1.0, 1.0))
        depth_offset = random.uniform(*SECONDARY_CUBE_BEHIND_DEPTH_RANGE)
        horizontal_location = (
            table_right * right_offset + table_forward * depth_offset
        )
        min_z = transformed_min_z(render_parts, placement_rotation)
        translation = Vector((
            horizontal_location.x,
            horizontal_location.y,
            BACKGROUND_TABLE_Z + clearance - min_z,
        ))
        candidate_transform = (
            Matrix.Translation(translation) @ placement_rotation
        )
        candidate_world_bounds = transformed_object_group_bounds(
            render_parts,
            candidate_transform,
        )
        if any(
            bounds_intersect_with_margin(candidate_world_bounds, obstacle_bounds_item, 0.06)
            for obstacle_bounds_item in obstacle_bounds
        ):
            continue
        projected_bounds = projected_object_group_bounds(
            render_parts,
            candidate_transform,
        )
        if projected_bounds is None:
            continue
        intersects_frame = (
            projected_bounds[1] > 0.0
            and projected_bounds[0] < 1.0
            and projected_bounds[3] > 0.0
            and projected_bounds[2] < 1.0
        )
        if not intersects_frame:
            continue
        silhouette_overlap = projected_bounds_overlap_ratio(
            main_projected_bounds,
            projected_bounds,
        )
        if (
            layout_mode == "silhouette_intersect"
            and silhouette_overlap < SECONDARY_CUBE_MIN_SILHOUETTE_OVERLAP
        ):
            continue
        cropped_candidate = (
            placement_attempt + 1,
            right_offset,
            translation,
            candidate_transform,
            projected_bounds,
            silhouette_overlap,
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
        silhouette_overlap,
    ) = cropped_candidate
    world_bounds = transformed_object_group_bounds(render_parts, candidate_transform)

    transform = candidate_transform
    for obj in pose_roots:
        obj.matrix_world = transform @ obj.matrix_world
        obj.rotation_mode = "QUATERNION"
    bpy.context.view_layer.update()
    actual_projected_bounds = projected_object_group_bounds(
        render_parts,
        Matrix.Identity(4),
    )
    metadata = {
        "model": "duplicate_of_main_cube",
        "white_center_logo": randomize_cube_logo(duplicate_objects),
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
        "layout_mode": layout_mode,
        "crop_attempts": crop_attempts,
        "placement_attempts": crop_attempts,
        "right_offset": right_offset,
        "depth_offset": depth_offset,
        "behind_main_cube": True,
        "scale": secondary_scale,
        "silhouette_overlap_ratio": silhouette_overlap,
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


def hand_texture_source_blend_path() -> Path | None:
    return next(
        (path for path in HAND_TEXTURE_SOURCE_BLEND_CANDIDATES if path.exists()),
        None,
    )


def ensure_hand_texture_uvs() -> bool:
    """Copy the baked asset's UV layout onto both imported rigged hands."""
    targets = [bpy.data.objects.get(name) for name in HAND_NAMES]
    if not all(target is not None for target in targets):
        return False
    if all(target.data.uv_layers.get(HAND_TEXTURE_UV_LAYER_NAME) for target in targets):
        return True

    source_path = hand_texture_source_blend_path()
    if source_path is None:
        return False

    with bpy.data.libraries.load(str(source_path), link=False) as (data_from, data_to):
        if "hand" not in data_from.objects:
            return False
        data_to.objects = ["hand"]

    source = data_to.objects[0] if data_to.objects else None
    if source is None or source.type != "MESH":
        return False
    source_uv = source.data.uv_layers.get(HAND_TEXTURE_UV_LAYER_NAME)
    if source_uv is None:
        bpy.data.objects.remove(source, do_unlink=True)
        return False

    try:
        for target in targets:
            if len(target.data.loops) != len(source.data.loops):
                raise RuntimeError(
                    "The rigged hand mesh topology does not match the baked hand UV asset."
                )
            target_uv = target.data.uv_layers.get(HAND_TEXTURE_UV_LAYER_NAME)
            if target_uv is None:
                target_uv = target.data.uv_layers.new(name=HAND_TEXTURE_UV_LAYER_NAME)
            for index, source_loop in enumerate(source_uv.data):
                target_uv.data[index].uv = source_loop.uv
            target.data.uv_layers.active_index = target.data.uv_layers.find(
                HAND_TEXTURE_UV_LAYER_NAME
            )
    finally:
        source_mesh = source.data
        bpy.data.objects.remove(source, do_unlink=True)
        if source_mesh.users == 0:
            bpy.data.meshes.remove(source_mesh)
    return True


def ensure_hand_assets_loaded() -> None:
    """Replace cube.blend's static hand meshes with the rigged pose asset."""
    global HAND_ASSET_IMPORTED
    if HAND_ASSET_IMPORTED:
        return

    existing_asset = all(
        bpy.data.objects.get(name) is not None for name in HAND_ASSET_OBJECT_NAMES
    ) and all(
        bpy.data.actions.get(name) is not None for name in HAND_POSE_ACTION_NAMES
    )
    if existing_asset:
        ensure_hand_texture_uvs()
        HAND_ASSET_IMPORTED = True
        return

    if not HAND_ASSET_BLEND_PATH.exists():
        raise FileNotFoundError(
            f"Rigged hand asset not found: {HAND_ASSET_BLEND_PATH}. "
            "Place rigged_hand.blend in syntheticAssets."
        )

    for name in HAND_ASSET_OBJECT_NAMES:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)

    object_names = set(HAND_ASSET_OBJECT_NAMES)
    with bpy.data.libraries.load(str(HAND_ASSET_BLEND_PATH), link=False) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects if name in object_names]
        data_to.actions = [
            name for name in data_from.actions if name in HAND_POSE_ACTION_NAMES
        ]

    target_collection = bpy.context.scene.collection
    for obj in data_to.objects:
        if obj is not None and not obj.users_collection:
            target_collection.objects.link(obj)

    missing_objects = [name for name in HAND_ASSET_OBJECT_NAMES if bpy.data.objects.get(name) is None]
    missing_actions = [name for name in HAND_POSE_ACTION_NAMES if bpy.data.actions.get(name) is None]
    if missing_objects or missing_actions:
        details = []
        if missing_objects:
            details.append(f"objects: {', '.join(missing_objects)}")
        if missing_actions:
            details.append(f"actions: {', '.join(missing_actions)}")
        raise RuntimeError(f"Could not import rigged hand asset ({'; '.join(details)}).")

    ensure_hand_texture_uvs()
    HAND_ASSET_IMPORTED = True


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


def hand_rig_objects() -> dict[str, bpy.types.Object]:
    rigs = {
        name: bpy.data.objects.get(f"{name}-rig")
        for name in HAND_NAMES
    }
    if any(rig is None for rig in rigs.values()):
        missing = [name for name, rig in rigs.items() if rig is None]
        raise ValueError(f"Missing hand armatures: {', '.join(missing)}")
    if any(rig.type != "ARMATURE" for rig in rigs.values() if rig is not None):
        raise ValueError("Imported hand rig objects must be armatures.")
    return {name: rig for name, rig in rigs.items() if rig is not None}


def hand_object_states(
    objects: dict[str, bpy.types.Object],
) -> dict[str, dict[str, object]]:
    rigs = hand_rig_objects()
    return {
        name: {
            "matrix_world": obj.matrix_world.copy(),
            "rig_matrix_world": rigs[name].matrix_world.copy(),
            "rotation_mode": obj.rotation_mode,
            "rig_rotation_mode": rigs[name].rotation_mode,
            "hide_render": obj.hide_render,
            "hide_viewport": obj.hide_viewport,
            "hide": obj.hide_get(),
            "rig_hide_render": rigs[name].hide_render,
            "rig_hide_viewport": rigs[name].hide_viewport,
            "rig_hide": rigs[name].hide_get(),
            "action": (
                rigs[name].animation_data.action
                if rigs[name].animation_data is not None
                else None
            ),
            "materials": [material for material in obj.data.materials],
        }
        for name, obj in objects.items()
    }


def restore_hand_objects(
    objects: dict[str, bpy.types.Object],
    states: dict[str, dict[str, object]],
) -> None:
    rigs = hand_rig_objects()
    for name, obj in objects.items():
        state = states[name]
        rig = rigs[name]
        if rig.animation_data is None:
            rig.animation_data_create()
        rig.animation_data.action = state["action"]
        rig.matrix_world = state["rig_matrix_world"]
        rig.rotation_mode = state["rig_rotation_mode"]
        rig.hide_render = state["rig_hide_render"]
        rig.hide_viewport = state["rig_hide_viewport"]
        rig.hide_set(state["rig_hide"])
        obj.matrix_world = state["matrix_world"]
        obj.rotation_mode = state["rotation_mode"]
        obj.hide_render = state["hide_render"]
        obj.hide_viewport = state["hide_viewport"]
        obj.hide_set(state["hide"])
        obj.data.materials.clear()
        for material in state["materials"]:
            obj.data.materials.append(material)
    bpy.context.view_layer.update()


def assign_hand_pose(rig: bpy.types.Object, action_name: str) -> None:
    action = bpy.data.actions.get(action_name)
    if action is None:
        raise ValueError(f"Hand pose action not found: {action_name}")
    if rig.animation_data is None:
        rig.animation_data_create()
    rig.animation_data.action = action
    # Blender 5 uses explicit action slots. Assigning only ``action`` can
    # leave the rig bound to the previous slot, so different named poses may
    # evaluate to the same visible hand pose.
    if hasattr(rig.animation_data, "action_slot") and len(action.slots):
        suitable_slots = getattr(rig.animation_data, "action_suitable_slots", ())
        slot = suitable_slots[0] if suitable_slots else action.slots[0]
        rig.animation_data.action_slot = slot


def hand_pose_assignments(
    visibility: str,
    one_hand_side: str | None = None,
) -> dict[str, str]:
    standard_hold = "HandPose.StandardHold"
    one_handgrip = "HandPose.OneHandGrip"
    other_poses = tuple(
        name
        for name in HAND_POSE_ACTION_NAMES
        if name not in (standard_hold, one_handgrip)
    )

    if visibility == "both":
        standard_side = random.choice(HAND_NAMES)
        other_side = next(name for name in HAND_NAMES if name != standard_side)
        return {
            standard_side: standard_hold,
            other_side: random.choice(other_poses),
        }

    if visibility == "onehandgrip":
        visible_side = one_hand_side or random.choice(HAND_NAMES)
        if visible_side not in HAND_NAMES:
            raise ValueError(f"Unknown one-hand side: {visible_side}")
        hidden_side = next(name for name in HAND_NAMES if name != visible_side)
        return {visible_side: one_handgrip, hidden_side: standard_hold}

    return {
        "left-hand": standard_hold,
        "right-hand": "HandPose.StandardGrip",
    }


def apply_shared_cube_pose(
    cube_objects_to_transform: list[bpy.types.Object],
    hand_rigs_to_transform: dict[str, bpy.types.Object],
    cube_pose: dict[str, object],
) -> Matrix:
    rows = cube_pose.get("transform")
    if not isinstance(rows, list):
        raise ValueError("Cube pose is missing its shared transform matrix.")
    transform = matrix_from_rows(rows)

    # The shared pose moves the cube and hands together for the global tilt and
    # translation. The cube's discrete scramble is already present in the
    # cubie transforms, but it must not rotate the hand rigs: hand placement is
    # randomized independently of cube orientation. The permitted R/L local
    # hand turns are applied separately before this shared global transform.
    for obj in cube_objects_to_transform:
        obj.matrix_world = transform @ obj.matrix_world
        obj.rotation_mode = "QUATERNION"
    for rig in hand_rigs_to_transform.values():
        rig.matrix_world = transform @ rig.matrix_world
        rig.rotation_mode = "QUATERNION"
    bpy.context.view_layer.update()
    return transform


def apply_partial_turn_to_hands(
    layer_state: dict[str, object],
    hand_rigs: dict[str, bpy.types.Object],
) -> list[str]:
    """Move the hand holding a side layer with an R/L partial turn.

    The hand placement is authored independently from the cube scramble. A
    transient R turn moves the right hand with the R layer, and an L turn
    moves the left hand with the L layer. U/D/F/B, slice, and other partial
    turns leave both hands in place. Simultaneous and corner-cut states can
    contain multiple components; apply the same rule independently to each.
    """
    moved_hands: list[str] = []
    for face, angle, direction, _object_names in _partial_turn_layers(layer_state):
        if face not in {"R", "L"}:
            continue
        if not isinstance(angle, (int, float)):
            continue
        if not isinstance(direction, (int, float)):
            direction = 1.0

        hand_name = "right-hand" if face == "R" else "left-hand"
        rig = hand_rigs.get(hand_name)
        if rig is None:
            continue
        face_axis = partial_turn_axis(face)
        turn = Quaternion(
            face_axis,
            float(direction) * math.radians(float(angle)),
        ).to_matrix().to_4x4()
        # A face turn rotates around the face-center axis, not the cube's
        # origin. For R/L that pivot is the corresponding +/-1 face center.
        turn_about_face = (
            Matrix.Translation(face_axis)
            @ turn
            @ Matrix.Translation(-face_axis)
        )
        rig.matrix_world = turn_about_face @ rig.matrix_world
        rig.rotation_mode = "QUATERNION"
        if hand_name not in moved_hands:
            moved_hands.append(hand_name)

    if moved_hands:
        bpy.context.view_layer.update()
    return moved_hands


def cube_pose_with_hand_lift(
    cube_pose: dict[str, object],
    lift: float,
) -> dict[str, object]:
    if lift <= 0.0:
        return cube_pose
    rows = cube_pose.get("transform")
    translation = cube_pose.get("translation")
    if not isinstance(rows, list) or not isinstance(translation, list):
        raise ValueError("Cube pose is missing its transform or translation metadata.")
    lifted = dict(cube_pose)
    lifted_rows = [list(row) for row in rows]
    lifted_rows[2][3] += lift
    lifted_translation = list(translation)
    lifted_translation[2] += lift
    lifted["transform"] = lifted_rows
    lifted["translation"] = lifted_translation
    lifted["hands_table_lift"] = lift
    lifted["effective_clearance"] = float(cube_pose.get("clearance", 0.0)) + lift
    return lifted


def cube_pose_on_table(cube_pose: dict[str, object]) -> dict[str, object]:
    """Move an existing cube pose down until its lowest point touches the table."""
    clearance = max(0.0, float(cube_pose.get("clearance", 0.0)))
    on_table = dict(cube_pose)
    rows = cube_pose.get("transform")
    translation = cube_pose.get("translation")
    if clearance > 0.0:
        if not isinstance(rows, list) or not isinstance(translation, list):
            raise ValueError("Cube pose is missing its transform or translation metadata.")
        on_table_rows = [list(row) for row in rows]
        on_table_rows[2][3] -= clearance
        on_table_translation = list(translation)
        on_table_translation[2] -= clearance
        on_table["transform"] = on_table_rows
        on_table["translation"] = on_table_translation
    on_table["mode"] = "on_table"
    on_table["clearance"] = 0.0
    on_table.pop("effective_clearance", None)
    return on_table


def hand_skin_material(color: tuple[float, float, float, float]) -> bpy.types.Material:
    texture_paths = {
        "base_color": HAND_TEXTURE_DIR / "hand_basecolor.png",
        "roughness": HAND_TEXTURE_DIR / "hand_roughness.png",
        "normal": HAND_TEXTURE_DIR / "hand_normal.png",
    }
    missing = [str(path) for path in texture_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing rigged hand texture asset(s): " + ", ".join(missing)
        )

    material = bpy.data.materials.get(HAND_SKIN_MATERIAL_NAME)
    if material is None:
        material = bpy.data.materials.new(HAND_SKIN_MATERIAL_NAME)
    material.use_nodes = True
    material.diffuse_color = color
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.name = "Hand Material Output"
    output.location = (620, 80)
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.name = "Principled BSDF"
    shader.location = (340, 80)
    set_principled_input(material, "Metallic", 0.0)
    set_principled_input(material, "Specular IOR Level", 0.28)
    set_principled_input(material, "Specular", 0.28)
    set_principled_input(material, "IOR", 1.4)
    set_principled_input(material, "Subsurface Weight", 0.08)
    set_principled_input(material, "Subsurface Radius", (1.0, 0.44, 0.22))
    set_principled_input(material, "Subsurface Scale", 0.007)
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    uv_layer_available = all(
        obj.data.uv_layers.get(HAND_TEXTURE_UV_LAYER_NAME) is not None
        for obj in bpy.data.objects
        if obj.name in HAND_NAMES and obj.type == "MESH"
    )
    if uv_layer_available:
        coordinates = nodes.new("ShaderNodeUVMap")
        coordinates.uv_map = HAND_TEXTURE_UV_LAYER_NAME
    else:
        coordinates = nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Hand Texture Coordinates"
    coordinates.location = (-900, 80)

    def add_texture(
        name: str,
        path: Path,
        colorspace: str,
        location: tuple[float, float],
    ) -> bpy.types.ShaderNodeTexImage:
        image = bpy.data.images.load(str(path), check_existing=True)
        image.colorspace_settings.name = colorspace
        texture = nodes.new("ShaderNodeTexImage")
        texture.name = name
        texture.image = image
        texture.location = location
        links.new(
            coordinates.outputs["UV"] if uv_layer_available else coordinates.outputs["Generated"],
            texture.inputs["Vector"],
        )
        return texture

    base_texture = add_texture(
        "Hand Base Color Texture",
        texture_paths["base_color"],
        "sRGB",
        (-620, 280),
    )
    roughness_texture = add_texture(
        "Hand Roughness Texture",
        texture_paths["roughness"],
        "Non-Color",
        (-620, 40),
    )
    normal_texture = add_texture(
        "Hand Normal Texture",
        texture_paths["normal"],
        "Non-Color",
        (-620, -220),
    )
    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.name = "Hand Normal Map"
    normal_map.location = (80, -180)
    normal_map.inputs["Strength"].default_value = 0.72
    links.new(base_texture.outputs["Color"], shader.inputs["Base Color"])
    links.new(roughness_texture.outputs["Color"], shader.inputs["Roughness"])
    links.new(normal_texture.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])
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
    forced_visibility: str | None = None,
) -> dict[str, object] | None:
    if not objects:
        return None

    one_hand_side = None
    if forced_visibility in ("left", "right"):
        # Keep old forced-side values as compatibility aliases, but make the
        # resulting render state explicitly OneHandGrip-only.
        visibility = "onehandgrip"
        one_hand_side = f"{forced_visibility}-hand"
    elif forced_visibility == "none":
        visibility = "none"
    elif not HAND_RANDOMIZATION_ENABLED:
        visibility = "both"
    else:
        visibility = forced_visibility or random.choices(
            [mode for mode, _weight in HAND_VISIBILITY_WEIGHTS],
            weights=[weight for _mode, weight in HAND_VISIBILITY_WEIGHTS],
            k=1,
        )[0]
    if not HAND_RANDOMIZATION_ENABLED and visibility == "both":
        assignments = {
            "left-hand": "HandPose.StandardHold",
            "right-hand": "HandPose.StandardHold",
        }
    else:
        assignments = hand_pose_assignments(visibility, one_hand_side=one_hand_side)
    visible_hands = (
        set(HAND_NAMES)
        if visibility == "both"
        else {
            name
            for name, pose_name in assignments.items()
            if pose_name == "HandPose.OneHandGrip"
        }
    )
    randomization: dict[str, object] = {
        "visibility": visibility,
        "objects": {},
    }
    if visibility == "onehandgrip":
        randomization["one_hand_side"] = next(iter(visible_hands))
    skin_color = (
        random_hand_skin_color()
        if HAND_RANDOMIZATION_ENABLED
        else HAND_FIXED_SKIN_COLOR
    )
    skin_material = hand_skin_material(skin_color)
    apply_hand_material(objects, skin_material)
    randomization["skin_color"] = list(skin_color)
    randomization["skin_material"] = skin_material.name
    randomization["follows_cube_pose"] = False
    rigs = hand_rig_objects()

    for name, obj in objects.items():
        rig = rigs[name]
        pose_name = assignments[name]
        assign_hand_pose(rig, pose_name)
        base_matrix = states[name]["rig_matrix_world"]
        if HAND_RANDOMIZATION_ENABLED:
            x_offset = random.uniform(*HAND_X_OFFSET_RANGE)
            z_offset = random.uniform(*HAND_Z_OFFSET_RANGE)
            x_rotation_degrees = (
                0.0
                if pose_name in HAND_POSES_WITHOUT_ORIGIN_X_ROTATION
                else random.choice(HAND_ORIGIN_X_ROTATION_DEGREES)
            )
        else:
            x_offset = 0.0
            z_offset = 0.0
            x_rotation_degrees = 0.0
        hidden = name not in visible_hands
        location_offset = Vector((x_offset, 0.0, z_offset))
        # The imported hand rigs already contain the correct placement for
        # every pose. Randomization only nudges and applies the permitted
        # pose-local quarter-turn to that authored placement.
        hand_rotation = Quaternion(
            Vector((1.0, 0.0, 0.0)),
            math.radians(x_rotation_degrees),
        ).to_matrix().to_4x4()
        solver_facing_rotation = Quaternion(
            Vector((0.0, 0.0, 1.0)),
            math.radians(HAND_SOLVER_FACING_ROTATION_DEGREES),
        ).to_matrix().to_4x4()
        # The nudge is authored in the same local frame as the pose. Apply
        # the pose rotation first so a +/-90-degree pose rotates the nudge
        # with the hand instead of leaving it in cube/world X-Z axes.
        local_matrix = (
            solver_facing_rotation
            @ hand_rotation
            @ Matrix.Translation(location_offset)
            @ base_matrix
        )
        rig.matrix_world = local_matrix
        obj.rotation_mode = "QUATERNION"
        obj.hide_render = hidden
        obj.hide_viewport = hidden
        obj.hide_set(hidden)
        rig.hide_render = hidden
        rig.hide_viewport = hidden
        rig.hide_set(hidden)

        randomization["objects"][name] = {
            "visible": not hidden,
            "pose": pose_name,
            "x_offset": x_offset,
            "z_offset": z_offset,
            "x_rotation_degrees": x_rotation_degrees,
            "solver_facing_rotation_degrees": HAND_SOLVER_FACING_ROTATION_DEGREES,
            "randomization_enabled": HAND_RANDOMIZATION_ENABLED,
            "location": [rig.location.x, rig.location.y, rig.location.z],
        }

    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    return randomization


def randomize_scene(
    scramble_length: int = DEFAULT_SCRAMBLE_LENGTH,
    background_object_count: int = DEFAULT_BACKGROUND_OBJECT_COUNT,
    desk_no_hands: bool = False,
    solved: bool = False,
) -> tuple[
    list[str],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    sticker_materials = randomize_sticker_material_gloss()
    objects = cube_objects()
    core_objects = core_fill_objects()
    reset_cube(objects)
    reset_core_fill_objects(core_objects)

    camera_randomization = randomize_camera()
    background_randomization = randomize_background(background_object_count)
    # Background walls and large props do not exist when the first camera is
    # sampled. Probe after loading them and retry cameras whose cube rays are
    # blocked before any expensive rendering begins.
    clear_scene_lights()
    camera_randomization = ensure_camera_has_clear_cube_view(
        objects,
        camera_randomization,
    )
    # Build the rig after appending the background so lights bundled in an asset
    # cannot accidentally leak into, or conflict with, the selected setup.
    light_randomization = randomize_light(background_randomization)
    background_randomization["world_color"] = light_randomization["world"]["display_color"]
    background_randomization["world_lighting"] = light_randomization["world"]
    moves = scene_scramble_moves(scramble_length, solved)
    scramble_objects = objects + core_objects
    wireframe_scramble_transform = Matrix.Identity(4)
    for move in moves:
        if move in "xyz":
            whole_cube_turn = Quaternion(whole_cube_move_axis(move), -math.pi / 2.0)
            wireframe_scramble_transform = (
                whole_cube_turn.to_matrix().to_4x4() @ wireframe_scramble_transform
            )
        apply_scramble_move(scramble_objects, move)

    layer_state = scene_layer_render_state(objects)

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
    # Keep the cube in its local pose until the hands have been placed. The
    # shared transform is applied to both the cube and the hand armatures in
    # generate_dataset(), so the hand placement remains authored at the cube's
    # origin instead of being transformed around its own pivot.
    cube_pose = randomize_cube_global_pose(
        pose_objects,
        bounds_objects=pose_bounds,
        apply=False,
        force_on_table=desk_no_hands,
    )
    cube_pose["solved"] = solved
    cube_pose["wireframe_scramble_transform"] = matrix_rows(wireframe_scramble_transform)
    return (
        moves,
        layer_state,
        background_randomization,
        camera_randomization,
        light_randomization,
        cube_pose,
        sticker_materials,
    )


def build_hard_case_scene(
    case_name: str,
    *,
    scramble_length: int,
    background_object_count: int,
    solved: bool = False,
) -> tuple[
    list[str],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Build one deliberate failure case with shared ordinary-scene variation.

    This is a registry-backed extension point rather than a second set of
    loosely related flags.  A future case can be added by defining a builder
    near this function and registering it in ``HARD_CASE_BUILDERS``.  Builders
    should reset the cube and explicitly choose the scramble, camera,
    background, lighting, materials, and cube pose they need.  Background and
    lighting builders should normally start from the ordinary distributions
    and then add only the bounded hard-case override.  They must return the
    same seven metadata values as ``randomize_scene``.

    Keeping this branch separate is important: the failure factor remains
    intentional, while shared ordinary-scene variation is inherited through
    explicit helper calls rather than duplicated ad hoc ranges.
    """

    builder = HARD_CASE_BUILDERS.get(case_name)
    if builder is None:
        available = ", ".join(sorted(HARD_CASE_BUILDERS)) or "(none registered yet)"
        raise ValueError(
            f"Unknown hard case {case_name!r}. Registered hard cases: {available}. "
            "Add its builder to HARD_CASE_BUILDERS before generating it."
        )
    if not callable(builder):
        raise TypeError(f"Hard-case builder {case_name!r} is not callable.")
    return builder(
        scramble_length=scramble_length,
        background_object_count=background_object_count,
        solved=solved,
    )


def build_solid_background_patch_case(
    *,
    scramble_length: int,
    background_object_count: int,
    solved: bool = False,
) -> tuple[
    list[str],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Build the realistic solid-color background-patch failure case.

    The target cube is held just above or in front of a nearby,
    ordinary-looking desk accessory: a thin dark base with a solid-color top
    visible across the small gap behind the cube.  The accessory is
    deliberately background geometry, not a second cube, so the visibility
    map keeps those pixels as class 0. This reproduces the observed failure
    where a contiguous solid-color patch is absorbed into the cube mask.
    """

    clear_background_scene()
    hide_flat_background()
    background_collection = background_scene_collection()

    variant_index = random.randrange(1000000)
    patch_side = random.choice((-1.0, 1.0))
    patch_profiles = (
        ("red", (0.46, 0.018, 0.012, 1.0)),
        ("burgundy", (0.22, 0.012, 0.018, 1.0)),
        ("amber", (0.52, 0.19, 0.018, 1.0)),
        ("blue", (0.018, 0.11, 0.28, 1.0)),
        ("violet", (0.20, 0.025, 0.30, 1.0)),
        ("green", (0.035, 0.25, 0.07, 1.0)),
        ("yellow", (0.58, 0.48, 0.025, 1.0)),
    )
    patch_profile_name, patch_color = random.choices(
        patch_profiles,
        weights=(3.0, 0.7, 2.5, 0.8, 0.7, 0.6, 3.0),
        k=1,
    )[0]
    backing_color = random.choice(
        (
            (0.10, 0.12, 0.14, 1.0),
            (0.16, 0.17, 0.18, 1.0),
            (0.20, 0.18, 0.16, 1.0),
        )
    )
    backing_width = random.uniform(*SOLID_PATCH_BACKING_WIDTH_RANGE)
    backing_depth = random.uniform(*SOLID_PATCH_BACKING_DEPTH_RANGE)
    backing_height = random.uniform(*SOLID_PATCH_BACKING_HEIGHT_RANGE)
    patch_width = backing_width * random.uniform(0.80, 0.96)
    patch_depth = backing_depth * random.uniform(0.78, 0.94)
    patch_height = backing_height * random.uniform(0.68, 0.90)
    accessory_yaw_degrees = random.uniform(-18.0, 18.0)
    accessory_tilt_degrees = random.uniform(-12.0, 12.0)

    camera = get_camera()
    azimuth_degrees = random.uniform(-58.0, -32.0)
    elevation_degrees = random.uniform(24.0, 38.0)
    distance = random.uniform(13.0, 17.0)
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    camera_target = Vector((
        random.uniform(-0.15, 0.15),
        random.uniform(-0.15, 0.15),
        random.uniform(-0.40, -0.15),
    ))
    camera.location = camera_target + Vector((
        distance * math.cos(elevation) * math.cos(azimuth),
        distance * math.cos(elevation) * math.sin(azimuth),
        distance * math.sin(elevation),
    ))
    look_at(camera, camera_target)
    camera.data.type = "PERSP"
    zoom_scale = camera_zoom_scale_for_distance(
        distance / CAMERA_SCENE_UNITS_PER_FOOT
    )
    camera.data.lens = random.uniform(34.0, 42.0) * zoom_scale
    camera.data.shift_x = 0.0
    camera.data.shift_y = 0.0
    camera_roll_degrees = random.uniform(*CAMERA_ROLL_DEGREES_RANGE)
    camera.rotation_quaternion = camera.rotation_quaternion @ Quaternion(
        Vector((0.0, 0.0, -1.0)),
        math.radians(camera_roll_degrees),
    )
    table_forward, table_right = camera_table_basis()
    table_up = Vector((0.0, 0.0, 1.0))
    table_frame = Matrix((table_right, table_forward, table_up)).transposed().to_quaternion()
    accessory_rotation = Quaternion(
        table_up,
        math.radians(accessory_yaw_degrees),
    ) @ table_frame
    accessory_rotation = Quaternion(
        table_right,
        math.radians(accessory_tilt_degrees),
    ) @ accessory_rotation

    # Use the ordinary solve-station background and lighting distribution;
    # the solid patch is the only deliberate background-specific factor.
    background_randomization = randomize_background(background_object_count)
    station_metadata = solve_station_metadata(background_randomization)
    if station_metadata is not None:
        # Keep this false-positive case free of a second cube by design.
        station_metadata["other_cube_visible"] = False
    background_collection = background_scene_collection()
    backing_material = noisy_material(
        f"{BACKGROUND_SCENE_MATERIAL_PREFIX}.hard_case.backing",
        backing_color,
        textured=False,
    )
    patch_material = glossy_material(
        f"{BACKGROUND_SCENE_MATERIAL_PREFIX}.hard_case.{patch_profile_name}_patch",
        patch_color,
        roughness=0.42,
        specular=0.28,
    )

    cube_half_depth = random.uniform(0.72, 1.00)
    gap_width = random.uniform(*SOLID_PATCH_GAP_RANGE)
    side_gap = random.uniform(*SOLID_PATCH_GAP_RANGE)
    table_depth = random.uniform(0.65, 1.10)
    accessory_side_offset = random.uniform(-0.28, 0.28)
    # Keep a raised accessory panel just behind the held cube, like a laptop
    # screen, tablet, or upright book. The small physical gap is along the
    # desk depth; the dark backing/bezel remains visible around the solid face.
    accessory_center = (
        camera_target
        + table_right * accessory_side_offset
        + table_forward * table_depth
    )
    backing_location = Vector((
        accessory_center.x,
        accessory_center.y,
        BACKGROUND_TABLE_Z + backing_height * 0.5,
    ))
    patch_location = Vector((
        accessory_center.x,
        accessory_center.y,
        BACKGROUND_TABLE_Z + backing_height * 0.5,
    )) - table_forward * ((backing_depth - patch_depth) * 0.5)

    backing = add_background_box(
        background_collection,
        f"{BACKGROUND_SCENE_PREFIX}.hard_case.backing",
        backing_location,
        (backing_width, backing_depth, backing_height),
        backing_material,
        rotation=accessory_rotation,
    )
    backing.pass_index = 0

    patch = add_background_box(
        background_collection,
        f"{BACKGROUND_SCENE_PREFIX}.hard_case.solid_{patch_profile_name}_patch",
        patch_location,
        (patch_width, patch_depth, patch_height),
        patch_material,
        rotation=accessory_rotation,
    )
    patch.pass_index = 0
    bevel = patch.modifiers.new("Soft accessory edges", "BEVEL")
    bevel.width = 0.035
    bevel.segments = 2

    forced_hand_visibility = "left" if patch_side > 0.0 else "right"
    if random.random() < 0.12:
        forced_hand_visibility = "both"
    camera_randomization = {
        "viewpoint": "front_diagonal",
        "location": list(camera.location),
        "target": list(camera_target),
        "azimuth_degrees": azimuth_degrees,
        "elevation_degrees": elevation_degrees,
        "distance": distance,
        "distance_feet": distance / CAMERA_SCENE_UNITS_PER_FOOT,
        "distance_bucket": "near",
        "reference_distance": 12.25,
        "roll_degrees": camera_roll_degrees,
        "lens_mm": camera.data.lens,
        "framing_lens_mm": camera.data.lens,
        "cube_zoom_scale": zoom_scale,
        "roi_crop": {
            "method": "hard_case_controlled_camera",
            "shift_x": camera.data.shift_x,
            "shift_y": camera.data.shift_y,
        },
    }

    light_randomization = randomize_light(background_randomization)
    background_randomization["world_color"] = light_randomization["world"]["display_color"]
    background_randomization["world_lighting"] = light_randomization["world"]

    objects = cube_objects()
    core_objects = core_fill_objects()
    reset_cube(objects)
    reset_core_fill_objects(core_objects)
    moves = scene_scramble_moves(scramble_length, solved)
    wireframe_scramble_transform = Matrix.Identity(4)
    for move in moves:
        if move in "xyz":
            whole_cube_turn = Quaternion(whole_cube_move_axis(move), -math.pi / 2.0)
            wireframe_scramble_transform = (
                whole_cube_turn.to_matrix().to_4x4() @ wireframe_scramble_transform
            )
        apply_scramble_move(objects + core_objects, move)
    # Deliberate cases keep the ordinary transient-turn distribution.  The
    # hard-case factor controls the scene condition; it must not silently
    # remove mid-turn/corner-cut examples from the training mixture.
    layer_state = scene_layer_render_state(objects)
    pose_objects = objects + core_objects
    yaw_degrees = random.uniform(-28.0, 28.0)
    pitch_degrees = random.uniform(-8.0, 8.0)
    roll_degrees = random.uniform(-8.0, 8.0)
    rotation = (
        Quaternion(Vector((0.0, 0.0, 1.0)), math.radians(yaw_degrees))
        @ Quaternion(Vector((1.0, 0.0, 0.0)), math.radians(pitch_degrees))
        @ Quaternion(Vector((0.0, 1.0, 0.0)), math.radians(roll_degrees))
    )
    clearance = random.uniform(0.04, 0.22)
    min_z = transformed_min_z(pose_objects, rotation)
    translation = Vector((
        random.uniform(*CUBE_POSE_XY_JITTER_RANGE),
        random.uniform(*CUBE_POSE_XY_JITTER_RANGE),
        BACKGROUND_TABLE_Z + clearance - min_z,
    ))
    transform = Matrix.Translation(translation) @ rotation.to_matrix().to_4x4()
    cube_pose = {
        "mode": "near_table",
        "desk_no_hands": False,
        "clearance": clearance,
        "table_z": BACKGROUND_TABLE_Z,
        "translation": list(translation),
        "transform": matrix_rows(transform),
        "yaw_degrees": yaw_degrees,
        "pitch_degrees": pitch_degrees,
        "roll_degrees": roll_degrees,
        "solved": solved,
        "wireframe_scramble_transform": matrix_rows(wireframe_scramble_transform),
    }

    background_randomization["mode"] = "solid_background_patch"
    background_randomization["forced_hand_visibility"] = forced_hand_visibility
    background_randomization["families"] = [
        family
        for family in background_randomization.get("families", [])
        if family != "other_cube"
    ] + ["solid_background_patch"]
    background_randomization["object_count"] = int(
        background_randomization.get("object_count", 0)
    ) + 2
    background_randomization["variant"] = {
        "index": variant_index,
        "patch_profile": patch_profile_name,
        "patch_color": list(patch_color),
        "backing_color": list(backing_color),
        "world_strength": light_randomization["world"]["strength"],
        "patch_side": patch_side,
        "accessory_yaw_degrees": accessory_yaw_degrees,
        "accessory_tilt_degrees": accessory_tilt_degrees,
        "table_depth": table_depth,
        "gap_width": gap_width,
        "side_gap": side_gap,
        "accessory_side_offset": accessory_side_offset,
        "cube_half_depth": cube_half_depth,
        "patch_width": patch_width,
        "patch_depth": patch_depth,
        "patch_height": patch_height,
        "backing_width": backing_width,
        "backing_depth": backing_depth,
        "backing_height": backing_height,
        "accessory_rotation": "table_aligned_with_bounded_yaw_and_tilt",
        "variation_policy": "controlled_ranges_preserve_background_patch_cause",
    }
    background_objects = background_randomization.setdefault("objects", [])
    if isinstance(background_objects, list):
        background_objects.append({
            "family": "solid_background_patch",
            "type": "desk_accessory_panel",
            "target_role": "background_only",
            "object_names": [backing.name, patch.name],
            "patch_material": patch_profile_name,
            "patch_side": patch_side,
            "secondary_cube_visible": False,
        })
    sticker_materials = randomize_sticker_material_gloss()
    return (
        moves,
        layer_state,
        background_randomization,
        camera_randomization,
        light_randomization,
        cube_pose,
        sticker_materials,
    )


def apply_hard_case_lighting_override(
    light_randomization: dict[str, object],
    lighting_mode: str,
) -> dict[str, object]:
    """Retain normal lighting diversity while making a lighting case deliberate."""
    sources = light_randomization.get("sources", [])
    if not isinstance(sources, list):
        return light_randomization
    area_sources = [
        source
        for source in sources
        if isinstance(source, dict)
        and source.get("type") == "AREA"
        and source.get("role") not in {"colored_fill", "colored_wall_wash"}
    ]
    if not area_sources:
        return light_randomization

    key = random.choice(area_sources)
    key_object = bpy.data.objects.get(key.get("name"))
    if key_object is None or key_object.type != "LIGHT":
        return light_randomization

    if lighting_mode == "sharp":
        key_object.data.energy = random.uniform(950.0, 1900.0)
        key_object.data.size = random.uniform(0.16, 0.55)
        key_object.data.size_y = random.uniform(0.18, 0.75)
        for source in area_sources:
            if source is key:
                continue
            light_object = bpy.data.objects.get(source.get("name"))
            if light_object is not None and light_object.type == "LIGHT":
                light_object.data.energy *= random.uniform(0.08, 0.24)
    elif lighting_mode == "split":
        key_object.data.energy = random.uniform(850.0, 1700.0)
        key_object.data.size = random.uniform(0.30, 0.95)
        key_object.data.size_y = random.uniform(0.35, 1.25)
        for source in area_sources:
            if source is key:
                continue
            light_object = bpy.data.objects.get(source.get("name"))
            if light_object is not None and light_object.type == "LIGHT":
                light_object.data.energy *= random.uniform(0.12, 0.34)

    light_randomization["hard_case_override"] = {
        "mode": lighting_mode,
        "key_source": key.get("name"),
        "key_energy": key_object.data.energy,
        "key_size": key_object.data.size,
        "source_count": len(area_sources),
    }
    return light_randomization


def build_controlled_cube_case(
    *,
    case_name: str,
    scramble_length: int,
    background_object_count: int,
    camera_mode: str,
    lighting_mode: str,
    forced_hand_visibility: str,
    solved: bool = False,
    backdrop: bool = False,
) -> tuple[
    list[str],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Build a controlled failure factor on top of the ordinary scene distribution."""

    side_face_angle_degrees = None
    if camera_mode == "diagonal_roll":
        azimuth_degrees = random.choice((-45.0, -135.0)) + random.uniform(-10.0, 10.0)
        elevation_degrees = random.uniform(20.0, 40.0)
        roll_range = (34.0, 45.0) if random.random() < 0.70 else (24.0, 34.0)
        camera_roll_degrees = random.choice((-1.0, 1.0)) * random.uniform(*roll_range)
    elif camera_mode == "near_side":
        side_face_angle_degrees = random.uniform(9.0, 13.0)
        side_face_sign = random.choice((-1.0, 1.0))
        azimuth_degrees = -90.0 + side_face_sign * side_face_angle_degrees
        elevation_degrees = random.uniform(18.0, 32.0)
        camera_roll_degrees = random.uniform(-30.0, 30.0)
    elif camera_mode == "behind_palm":
        azimuth_degrees = random.choice((42.0, 138.0)) + random.uniform(-8.0, 8.0)
        elevation_degrees = random.uniform(28.0, 48.0)
        camera_roll_degrees = random.uniform(-45.0, 45.0)
    elif camera_mode == "solved":
        azimuth_degrees = random.choice((-56.0, -44.0)) + random.uniform(-8.0, 8.0)
        elevation_degrees = random.uniform(18.0, 34.0)
        camera_roll_degrees = random.choice((-1.0, 1.0)) * random.uniform(10.0, 24.0)
    else:
        azimuth_degrees = random.choice((-52.0, -38.0)) + random.uniform(-8.0, 8.0)
        elevation_degrees = random.uniform(20.0, 40.0)
        camera_roll_degrees = random.uniform(*CAMERA_ROLL_DEGREES_RANGE)

    distance = (
        random.uniform(14.0, 18.0)
        if camera_mode == "behind_palm"
        else random.uniform(8.5, 11.5)
    )
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    camera_target = Vector((
        random.uniform(-0.14, 0.14),
        random.uniform(-0.14, 0.14),
        random.uniform(-0.38, -0.12),
    ))
    camera = get_camera()
    camera.location = camera_target + Vector((
        distance * math.cos(elevation) * math.cos(azimuth),
        distance * math.cos(elevation) * math.sin(azimuth),
        distance * math.sin(elevation),
    ))
    look_at(camera, camera_target)
    camera.data.type = "PERSP"
    zoom_scale = camera_zoom_scale_for_distance(
        distance / CAMERA_SCENE_UNITS_PER_FOOT
    )
    framing_lens = random.uniform(41.0, 49.0)
    reference_distance = (
        CAMERA_BEHIND_PALM_REFERENCE_DISTANCE
        if camera_mode == "behind_palm" else distance
    )
    camera.data.lens = framing_lens * distance / reference_distance * zoom_scale
    camera.data.shift_x = random.uniform(*CAMERA_ROI_SHIFT_X_RANGE)
    camera.data.shift_y = random.uniform(*CAMERA_ROI_SHIFT_Y_RANGE)
    camera.rotation_quaternion = camera.rotation_quaternion @ Quaternion(
        Vector((0.0, 0.0, -1.0)),
        math.radians(camera_roll_degrees),
    )
    camera_direction, camera_right, camera_up = camera_basis()

    # Keep the hard-case builder separate so it can control the failure factor,
    # but inherit the ordinary solve-station background distribution. This
    # preserves variation in the desk, room, mat, laptop/phone visibility,
    # clothing, wall elements, and their materials/layouts.
    background_randomization = randomize_background(background_object_count)
    background_collection = background_scene_collection()
    if camera_mode == "behind_palm":
        # The solve-station timer and solver/arm cluster are camera-facing
        # props in the ordinary layout. A rear palm view can legitimately
        # place them outside the crop; hide only those occluders so the
        # deliberate case does not become a malformed all-background frame.
        rear_view_hidden = []
        for obj in solve_station_family_objects(
            background_collection,
            ("Room", "Timer", "Shirt", "Arm"),
        ):
            obj.hide_render = True
            rear_view_hidden.append(obj.name)
        station_metadata = solve_station_metadata(background_randomization)
        if station_metadata is not None:
            station_metadata["rear_view_hidden_families"] = [
                "Room",
                "Timer",
                "Shirt",
                "Arm",
            ]
            station_metadata["rear_view_hidden_object_count"] = len(rear_view_hidden)
    backdrop_object_names: list[str] = []
    backdrop_obj: bpy.types.Object | None = None
    backdrop_initial_center: Vector | None = None
    backdrop_geometry: dict[str, object] = {}
    if backdrop:
        # Use a dimmed procedural skin tone so the hull blends with the palm
        # while still being a distinct background-only surface in the label.
        # Keep the backdrop in the visible mid-to-light skin range.  Very dark
        # tones can turn a large procedural silhouette into an opaque black
        # slab under the station's randomized lighting, which is not the
        # intended palm-gap appearance.
        hull_color = scaled_color(
            random_color_from_ranges(HAND_SKIN_TONE_RANGES[:4]),
            random.uniform(0.45, 0.75),
        )
        hull_material = noisy_material(
            f"{BACKGROUND_SCENE_MATERIAL_PREFIX}.hard_case.{case_name}.palm_hull",
            hull_color,
            textured=False,
        )
        # Place the hull several scene units down the camera ray, so it is
        # unambiguously behind the cube and hands.  Clamp its world height to
        # the tabletop band after applying a small camera-space offset; this
        # avoids the downward ray putting the backdrop under the desk.
        hull_center = (
            camera_target
            + camera_direction * random.uniform(*PALM_HULL_DISTANCE_RANGE)
            + camera_right * random.uniform(-0.16, 0.16)
            + camera_up * random.uniform(-0.08, 0.18)
        )
        hull_center.z = min(
            BACKGROUND_TABLE_Z + 2.0,
            max(BACKGROUND_TABLE_Z + 0.05, hull_center.z),
        )
        backdrop_initial_center = hull_center.copy()
        backdrop_obj = add_procedural_camera_hull(
            background_collection,
            f"{BACKGROUND_SCENE_PREFIX}.hard_case.{case_name}.palm_hull",
            hull_center,
            camera_direction,
            camera_right,
            camera_up,
            width=random.uniform(*PALM_HULL_WIDTH_RANGE),
            height=random.uniform(*PALM_HULL_HEIGHT_RANGE),
            depth=random.uniform(*PALM_HULL_DEPTH_RANGE),
            material=hull_material,
        )
        backdrop_object_names.append(backdrop_obj.name)
        backdrop_geometry = {
            "initial_center": list(backdrop_initial_center),
            "width": float(backdrop_obj.dimensions.x),
            "height": float(backdrop_obj.dimensions.z),
            "depth": float(backdrop_obj.dimensions.y),
        }

    # Start from the same lighting distribution as ordinary samples. Sharp
    # and split-shadow cases then apply a bounded override to that sampled rig,
    # so they retain variation in setup, color temperature, window light,
    # laptop glow, and world exposure.
    light_randomization = randomize_light(background_randomization)
    if lighting_mode in {"sharp", "split"}:
        light_randomization = apply_hard_case_lighting_override(
            light_randomization,
            lighting_mode,
        )
    background_randomization["world_color"] = light_randomization["world"]["display_color"]
    background_randomization["world_lighting"] = light_randomization["world"]

    objects = cube_objects()
    core_objects = core_fill_objects()
    reset_cube(objects)
    reset_core_fill_objects(core_objects)
    moves = scene_scramble_moves(scramble_length, solved)
    wireframe_scramble_transform = Matrix.Identity(4)
    for move in moves:
        if move in "xyz":
            whole_cube_turn = Quaternion(whole_cube_move_axis(move), -math.pi / 2.0)
            wireframe_scramble_transform = (
                whole_cube_turn.to_matrix().to_4x4() @ wireframe_scramble_transform
            )
        apply_scramble_move(objects + core_objects, move)
    layer_state = scene_layer_render_state(objects)

    if camera_mode == "near_side":
        yaw_degrees = random.uniform(-3.0, 3.0)
        pitch_degrees = random.uniform(-5.0, 5.0)
        cube_roll_degrees = random.uniform(-4.0, 4.0)
    elif solved:
        # Solved samples still need clearly varied cube orientation. Use the
        # same full three-axis ranges as ordinary solved poses rather than a
        # nearly upright special-case range.
        yaw_degrees = random.uniform(*CUBE_POSE_ROTATION_DEGREES)
        pitch_degrees = random.uniform(*CUBE_FLOAT_TILT_DEGREES)
        cube_roll_degrees = random.uniform(*CUBE_FLOAT_TILT_DEGREES)
    else:
        yaw_degrees = random.uniform(-26.0, 26.0)
        pitch_degrees = random.uniform(-10.0, 10.0)
        cube_roll_degrees = random.uniform(-9.0, 9.0)
    rotation = (
        Quaternion(Vector((0.0, 0.0, 1.0)), math.radians(yaw_degrees))
        @ Quaternion(Vector((1.0, 0.0, 0.0)), math.radians(pitch_degrees))
        @ Quaternion(Vector((0.0, 1.0, 0.0)), math.radians(cube_roll_degrees))
    )
    clearance = random.uniform(0.05, 0.24)
    min_z = transformed_min_z(objects + core_objects, rotation)
    translation = Vector((
        random.uniform(*CUBE_POSE_XY_JITTER_RANGE),
        random.uniform(*CUBE_POSE_XY_JITTER_RANGE),
        BACKGROUND_TABLE_Z + clearance - min_z,
    ))
    transform = Matrix.Translation(translation) @ rotation.to_matrix().to_4x4()
    if backdrop_obj is not None and backdrop_initial_center is not None:
        # Re-anchor the procedural palm envelope to the cube's actual pose,
        # including the later hand lift. The camera target is intentionally
        # not used as the final anchor because it can sit below the held cube
        # for elevated viewpoints.
        cube_center = Vector(translation)
        horizontal_view_direction = Vector((camera_direction.x, camera_direction.y, 0.0))
        if horizontal_view_direction.length <= 1e-6:
            horizontal_view_direction = Vector((0.0, 1.0, 0.0))
        horizontal_view_direction.normalize()
        hull_center = (
            cube_center
            + horizontal_view_direction * random.uniform(*PALM_HULL_DISTANCE_RANGE)
            + camera_right * random.uniform(-0.16, 0.16)
            + Vector((0.0, 0.0, random.uniform(-0.15, 0.20)))
        )
        backdrop_delta = hull_center - backdrop_initial_center
        for vertex in backdrop_obj.data.vertices:
            vertex.co += backdrop_delta
        backdrop_geometry["final_center"] = list(backdrop_initial_center + backdrop_delta)
    cube_pose = {
        "mode": "near_table",
        "desk_no_hands": False,
        "clearance": clearance,
        "table_z": BACKGROUND_TABLE_Z,
        "translation": list(translation),
        "transform": matrix_rows(transform),
        "yaw_degrees": yaw_degrees,
        "pitch_degrees": pitch_degrees,
        "roll_degrees": cube_roll_degrees,
        "solved": solved,
        "wireframe_scramble_transform": matrix_rows(wireframe_scramble_transform),
    }
    variant = {
        "index": random.randrange(1000000),
        "camera_mode": camera_mode,
        "lighting_mode": lighting_mode,
        "solved": solved,
        "azimuth_degrees": azimuth_degrees,
        "elevation_degrees": elevation_degrees,
        "camera_roll_degrees": camera_roll_degrees,
        "camera_distance": distance,
        "cube_yaw_degrees": yaw_degrees,
        "cube_pitch_degrees": pitch_degrees,
        "cube_roll_degrees": cube_roll_degrees,
        "world_strength": light_randomization["world"]["strength"],
    }
    if side_face_angle_degrees is not None:
        variant["side_face_angle_degrees"] = side_face_angle_degrees
    if backdrop_geometry:
        variant["palm_hull_geometry"] = backdrop_geometry
    background_randomization["mode"] = case_name
    background_randomization["forced_hand_visibility"] = forced_hand_visibility
    background_randomization["variant"] = variant
    background_randomization["hard_case_factor"] = {
        "camera_mode": camera_mode,
        "lighting_mode": lighting_mode,
        "solved": solved,
    }
    background_randomization["families"] = list(
        background_randomization.get("families", [])
    ) + (["dark_gap_backdrop"] if backdrop else [])
    background_randomization["object_count"] = int(
        background_randomization.get("object_count", 0)
    ) + len(backdrop_object_names)
    background_objects = background_randomization.setdefault("objects", [])
    if isinstance(background_objects, list) and backdrop_object_names:
        background_objects.append({
            "family": "dark_gap_backdrop",
            "type": "background_only",
            "target_role": "background_only",
            "object_names": backdrop_object_names,
            "secondary_cube_visible": False,
        })
    camera_randomization = {
        "viewpoint": camera_mode,
        "location": list(camera.location),
        "target": list(camera_target),
        "azimuth_degrees": azimuth_degrees,
        "elevation_degrees": elevation_degrees,
        "distance": distance,
        "distance_feet": distance / CAMERA_SCENE_UNITS_PER_FOOT,
        "distance_bucket": "near",
        "reference_distance": reference_distance,
        "roll_degrees": camera_roll_degrees,
        "lens_mm": camera.data.lens,
        "framing_lens_mm": framing_lens,
        "cube_zoom_scale": zoom_scale,
        "roi_crop": {
            "method": "hard_case_controlled_camera",
            "shift_x": camera.data.shift_x,
            "shift_y": camera.data.shift_y,
        },
    }
    sticker_materials = randomize_sticker_material_gloss()
    return (
        moves,
        layer_state,
        background_randomization,
        camera_randomization,
        light_randomization,
        cube_pose,
        sticker_materials,
    )


def build_diagonal_camera_roll_case(**kwargs: object) -> tuple[object, ...]:
    return build_controlled_cube_case(
        case_name=DIAGONAL_CAMERA_ROLL_HARD_CASE,
        camera_mode="diagonal_roll",
        lighting_mode="standard",
        forced_hand_visibility=random.choice(("left", "right", "both")),
        **kwargs,
    )


def build_sharp_cube_shadows_case(**kwargs: object) -> tuple[object, ...]:
    return build_controlled_cube_case(
        case_name=SHARP_CUBE_SHADOWS_HARD_CASE,
        camera_mode="standard",
        lighting_mode="sharp",
        forced_hand_visibility=random.choice(("left", "right", "both")),
        **kwargs,
    )


def build_solved_cube_case(**kwargs: object) -> tuple[object, ...]:
    # The automatic sampler can independently mark any hard case as solved;
    # this named builder remains an explicit solved-cube benchmark case.
    kwargs.pop("solved", None)
    return build_controlled_cube_case(
        case_name=SOLVED_CUBE_HARD_CASE,
        camera_mode="solved",
        lighting_mode=random.choice(("standard", "standard", "split")),
        forced_hand_visibility=random.choice(("left", "right", "left", "right", "both")),
        solved=True,
        **kwargs,
    )


def build_near_side_face_case(**kwargs: object) -> tuple[object, ...]:
    return build_controlled_cube_case(
        case_name=NEAR_SIDE_FACE_HARD_CASE,
        camera_mode="near_side",
        lighting_mode="split",
        forced_hand_visibility=random.choice(("left", "right", "both")),
        **kwargs,
    )


def build_palm_gap_case(**kwargs: object) -> tuple[object, ...]:
    return build_controlled_cube_case(
        case_name=PALM_GAP_HARD_CASE,
        camera_mode="behind_palm",
        lighting_mode=random.choice(("sharp", "split")),
        forced_hand_visibility=random.choice(("both", "both", "both", "left", "right")),
        backdrop=True,
        **kwargs,
    )


HARD_CASE_BUILDERS.update({
    SOLID_BACKGROUND_PATCH_HARD_CASE: build_solid_background_patch_case,
    DIAGONAL_CAMERA_ROLL_HARD_CASE: build_diagonal_camera_roll_case,
    SHARP_CUBE_SHADOWS_HARD_CASE: build_sharp_cube_shadows_case,
    SOLVED_CUBE_HARD_CASE: build_solved_cube_case,
    NEAR_SIDE_FACE_HARD_CASE: build_near_side_face_case,
    PALM_GAP_HARD_CASE: build_palm_gap_case,
})


def configure_nvidia_gpu() -> str:
    """Configure Cycles to render on NVIDIA GPUs, preferring OptiX.

    OptiX is generally fastest on RTX cards because it can use hardware ray
    tracing. CUDA is kept as a fallback for older NVIDIA cards or installations
    where the OptiX backend is unavailable. Fail loudly if the flag was given
    but no NVIDIA device can actually be configured.
    """
    cycles_addon = bpy.context.preferences.addons.get("cycles")
    if cycles_addon is None:
        raise RuntimeError(
            "The Cycles Blender add-on is unavailable; cannot enable NVIDIA GPU rendering."
        )

    cycles_preferences = cycles_addon.preferences
    failures = []
    for backend in ("OPTIX", "CUDA"):
        try:
            cycles_preferences.compute_device_type = backend
            cycles_preferences.get_devices()
            devices = list(cycles_preferences.devices)
        except Exception as exc:
            failures.append(f"{backend}: {exc}")
            continue

        gpu_devices = [
            device
            for device in devices
            if str(getattr(device, "type", "CPU")).upper() != "CPU"
        ]
        if not gpu_devices:
            failures.append(f"{backend}: no NVIDIA devices detected")
            continue

        # Use all detected NVIDIA devices and avoid adding the CPU as a
        # rendering device, which can reduce throughput for this workload.
        for device in devices:
            device.use = device in gpu_devices
        bpy.context.scene.cycles.device = "GPU"
        names = ", ".join(str(device.name) for device in gpu_devices)
        print(f"Using NVIDIA Cycles backend {backend}: {names}")
        return backend

    details = "; ".join(failures) if failures else "no backend was available"
    raise RuntimeError(
        "--gpu was requested, but Blender could not configure an NVIDIA Cycles device. "
        f"Tried OptiX and CUDA ({details})."
    )


def configure_render(width: int, height: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = DEFAULT_CYCLES_SAMPLES
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = CYCLES_ADAPTIVE_THRESHOLD
    scene.cycles.adaptive_min_samples = CYCLES_ADAPTIVE_MIN_SAMPLES
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 4
    scene.cycles.transparent_max_bounces = 4
    scene.cycles.use_light_tree = True
    scene.render.use_persistent_data = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False


def mesh_objects_in_roots(roots: list[bpy.types.Object]) -> list[bpy.types.Object]:
    """Return mesh objects in the supplied object hierarchies without duplicates."""
    objects = []
    seen: set[int] = set()
    for root in roots:
        for obj in object_tree(root):
            if obj.type != "MESH" or obj.as_pointer() in seen:
                continue
            seen.add(obj.as_pointer())
            objects.append(obj)
    return objects


def set_mesh_objects_render_visible(
    roots: list[bpy.types.Object],
) -> list[tuple[bpy.types.Object, bool, bool, bool]]:
    """Temporarily expose cube meshes for ray casts and return their states."""
    objects = mesh_objects_in_roots(roots)
    root_names = {obj.name for obj in roots}
    seen = {obj.as_pointer() for obj in objects}
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.as_pointer() in seen:
            continue
        dotted_name = f".{obj.name}."
        if any(f".{root_name}." in dotted_name for root_name in root_names):
            objects.append(obj)
            seen.add(obj.as_pointer())
    original_visibility = [
        (obj, obj.hide_render, obj.hide_viewport, obj.hide_get())
        for obj in objects
    ]
    for obj in objects:
        obj.hide_render = False
        obj.hide_viewport = False
        obj.hide_set(False)
    bpy.context.view_layer.update()
    return original_visibility


def restore_mesh_object_visibility(
    states: list[tuple[bpy.types.Object, bool, bool, bool]],
) -> None:
    """Restore mesh visibility after a temporary ray-cast pass."""
    for obj, hide_render, hide_viewport, hide in states:
        obj.hide_render = hide_render
        obj.hide_viewport = hide_viewport
        obj.hide_set(hide)
    bpy.context.view_layer.update()


def cube_render_mesh_names(roots: list[bpy.types.Object]) -> set[str]:
    """Return cube body and detached sticker mesh names used by ray casts."""
    names = {obj.name for obj in mesh_objects_in_roots(roots)}
    root_names = {obj.name for obj in roots}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        dotted_name = f".{obj.name}."
        if any(f".{root_name}." in dotted_name for root_name in root_names):
            names.add(obj.name)
    return names


def sticker_render_mesh_names(roots: list[bpy.types.Object]) -> set[str]:
    """Return the names of detached sticker objects, one name per sticker."""
    names = set()
    for obj in mesh_objects_in_roots(roots):
        if obj.name.startswith(STICKER_OBJECT_NAME_PREFIX):
            names.add(obj.name)
            continue
        materials = [material.name for material in obj.data.materials if material is not None]
        if (
            obj.parent is not None
            and len(materials) == 1
            and materials[0] in STICKER_COLOR_MATERIAL_NAMES
        ):
            names.add(obj.name)
    return names


def camera_frame_bounds(
    camera: bpy.types.Object,
    scene: bpy.types.Scene,
) -> tuple[float, float, float, float, float]:
    frame = camera.data.view_frame(scene=scene)
    xs = [corner.x for corner in frame]
    ys = [corner.y for corner in frame]
    zs = [corner.z for corner in frame]
    return min(xs), max(xs), min(ys), max(ys), sum(zs) / len(zs)


def camera_ray_sampler(
    camera: bpy.types.Object,
    scene: bpy.types.Scene,
    width: int,
    height: int,
) -> Callable[[int, int], tuple[Vector, Vector]]:
    """Cache the camera projection for a visibility pass."""
    left, right, bottom, top, frame_z = camera_frame_bounds(camera, scene)
    frame_x = [left + ((x + 0.5) / width) * (right - left) for x in range(width)]
    frame_y = [bottom + (1.0 - ((y + 0.5) / height)) * (top - bottom) for y in range(height)]
    camera_matrix = camera.matrix_world.copy()
    rotation = camera_matrix.to_3x3()

    if camera.data.type == "ORTHO":
        direction = (rotation @ Vector((0.0, 0.0, -1.0))).normalized()

        def orthographic_ray(x: int, y: int) -> tuple[Vector, Vector]:
            return camera_matrix @ Vector((frame_x[x], frame_y[y], 0.0)), direction

        return orthographic_ray

    origin = camera_matrix.translation

    def perspective_ray(x: int, y: int) -> tuple[Vector, Vector]:
        return origin, (rotation @ Vector((frame_x[x], frame_y[y], frame_z))).normalized()

    return perspective_ray


def render_visibility_class_map(
    main_cube_roots: list[bpy.types.Object],
    hand_roots: list[bpy.types.Object],
    width: int,
    height: int,
    sticker_face_keys: dict[tuple[str, int], str] | None = None,
) -> tuple[object, dict[str, object]]:
    """Ray-cast visible ownership and a separate visible-sticker map.

    The ownership map is 0=background, 1=target cube, 2=hand, 3=secondary
    cube.  The sticker maps contain one binary raster per detached sticker
    object; they are separate because sticker pixels are also part of the
    whole-cube mask.
    """
    try:
        import numpy as np
    except ImportError:
        raise RuntimeError("NumPy is required to write visibility labels.") from None

    main_cube_names = cube_render_mesh_names(main_cube_roots)
    sticker_names = sticker_render_mesh_names(main_cube_roots)
    sticker_instance_by_parent_material = {}
    for sticker_name in sticker_names:
        sticker_object = bpy.data.objects.get(sticker_name)
        if sticker_object is None or sticker_object.parent is None:
            continue
        for material in sticker_object.data.materials:
            if material is not None and material.name in STICKER_COLOR_MATERIAL_NAMES:
                sticker_instance_by_parent_material[
                    (sticker_object.parent.name, material.name)
                ] = sticker_name
    hand_names = {
        obj.name
        for root in hand_roots
        for obj in object_tree(root)
        if obj.type == "MESH"
    }
    secondary_cube_names = {
        obj.name
        for obj in bpy.data.objects
        if obj.type == "MESH" and obj.name.startswith(SECONDARY_CUBE_OBJECT_PREFIX)
    }
    visibility = np.zeros((height, width), dtype=np.uint8)
    sticker_mask_keys = (
        sorted(set(sticker_face_keys.values()))
        if sticker_face_keys is not None
        else sorted(sticker_names)
    )
    sticker_masks = {
        name: np.zeros((height, width), dtype=np.uint8)
        for name in sticker_mask_keys
    }
    tracked_names = main_cube_names | hand_names | secondary_cube_names
    tracked_objects = {
        obj.as_pointer(): obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and obj.name in tracked_names
    }
    original_visibility = [
        (obj, obj.hide_render, obj.hide_viewport, obj.hide_get())
        for obj in tracked_objects.values()
    ]
    try:
        for obj in tracked_objects.values():
            # Cube/secondary geometry must be queryable even when its render
            # visibility was temporarily changed by the generator. Hands are
            # different: preserve their current state so hidden hands do not
            # occlude a hand-free ground-truth render.
            if obj.name in main_cube_names or obj.name in secondary_cube_names:
                obj.hide_render = False
                obj.hide_viewport = False
                obj.hide_set(False)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        camera = get_camera()
        scene = bpy.context.scene
        sample_ray = camera_ray_sampler(camera, scene, width, height)
        for y in range(height):
            for x in range(width):
                origin, direction = sample_ray(x, y)
                ray_origin = origin
                hit = False
                hit_object = None
                for _ in range(32):
                    hit, location, _normal, _polygon_index, hit_object, _matrix = scene.ray_cast(
                        depsgraph,
                        ray_origin,
                        direction,
                    )
                    if not hit or hit_object is None:
                        break
                    if (
                        getattr(hit_object, "visible_camera", True)
                        and not getattr(hit_object, "hide_render", False)
                    ):
                        break
                    ray_origin = location + direction * 1e-4
                if not hit or hit_object is None:
                    continue
                object_name = hit_object.name
                if object_name in main_cube_names:
                    visibility[y, x] = 1
                    sticker_key = None
                    if sticker_face_keys is not None:
                        sticker_key = sticker_face_keys.get(
                            (object_name, int(_polygon_index))
                        )
                    elif object_name in sticker_masks:
                        sticker_key = object_name
                    if sticker_key is None:
                        # Older cube files keep sticker faces as material slots
                        # on cubies. When detached sticker objects also exist,
                        # map the parent face back to that same sticker object
                        # so disjoint visible regions remain one instance.
                        polygon_index = int(_polygon_index)
                        materials = getattr(getattr(hit_object, "data", None), "materials", None)
                        if (
                            materials is not None
                            and 0 <= polygon_index < len(getattr(hit_object.data, "polygons", ()))
                        ):
                            material_index = hit_object.data.polygons[polygon_index].material_index
                            if 0 <= material_index < len(materials):
                                material_name = materials[material_index].name
                                if material_name in STICKER_COLOR_MATERIAL_NAMES:
                                    sticker_key = sticker_instance_by_parent_material.get(
                                        (object_name, material_name),
                                        f"{object_name}:{material_name}",
                                    )
                                    if sticker_key not in sticker_masks:
                                        sticker_masks[sticker_key] = np.zeros((height, width), dtype=np.uint8)
                    if sticker_key is not None:
                        sticker_masks[sticker_key][y, x] = 1
                elif object_name in hand_names:
                    visibility[y, x] = 2
                elif object_name in secondary_cube_names:
                    visibility[y, x] = 3
    finally:
        for obj, hide_render, hide_viewport, hide in original_visibility:
            obj.hide_render = hide_render
            obj.hide_viewport = hide_viewport
            obj.hide_set(hide)
        bpy.context.view_layer.update()

    return visibility, sticker_masks


def render_segmentation_masks(
    main_cube_roots: list[bpy.types.Object],
    width: int,
    height: int,
) -> object:
    """Ray-cast the whole-cube mask directly without creating an image file."""
    visibility, _sticker_masks = render_visibility_class_map(main_cube_roots, [], width, height)
    return (visibility == 1).astype("uint8")


def filter_visible_sticker_masks(
    sticker_masks: dict[str, object],
    width: int,
    height: int,
) -> dict[str, object]:
    """Keep sticker instances with at least 8 reference-resolution area pixels."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError("OpenCV and NumPy are required to filter sticker masks.") from None

    minimum_area = MIN_VISIBLE_STICKER_AREA_PX_AT_128 * (width * height) / (128.0 * 128.0)
    filtered = {}
    for name, mask in sticker_masks.items():
        if not isinstance(mask, np.ndarray):
            continue
        ys, xs = np.where(mask != 0)
        if len(xs) < 3:
            continue
        points = np.column_stack((xs, ys)).astype(np.int32).reshape(-1, 1, 2)
        hull = cv2.convexHull(points)
        if cv2.contourArea(hull) >= minimum_area:
            filtered[name] = mask
    return filtered


def sticker_material_name(obj: bpy.types.Object) -> str | None:
    for material in getattr(getattr(obj, "data", None), "materials", ()):
        if material is not None and material.name in STICKER_KEYPOINT_CLASS_ID_BY_MATERIAL:
            return material.name
    return None


def sticker_polygon_face_groups(
    obj: bpy.types.Object,
    polygons: list[object],
) -> list[tuple[str, list[object]]]:
    """Split a sticker mesh into its distinct axis-aligned sticker faces."""
    groups = []
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        for sign in (-1, 1):
            axis = axis_vector(axis_index, float(sign))
            selected = [
                polygon
                for polygon in polygons
                if polygon.normal.length > 1e-6
                and polygon.normal.normalized().dot(axis) > 0.92
            ]
            if sum(float(getattr(polygon, "area", 0.0)) for polygon in selected) < 0.05:
                continue
            groups.append((f"{axis_name}{'p' if sign > 0 else 'n'}", selected))
    return groups


def sticker_surface_spec(
    obj: bpy.types.Object,
    material_name: str,
    polygons: list[object],
    mask_key: str,
) -> dict[str, object] | None:
    """Build one physical-sticker center and projected-area source."""
    if not polygons:
        return None
    transform = obj.matrix_world
    world_vertices = []
    world_polygons = []
    normal = Vector((0.0, 0.0, 0.0))
    for polygon in polygons:
        polygon_vertices = [
            transform @ obj.data.vertices[int(vertex_index)].co
            for vertex_index in polygon.vertices
        ]
        if len(polygon_vertices) < 3:
            continue
        world_vertices.extend(polygon_vertices)
        world_polygons.append(polygon_vertices)
        weight = max(float(getattr(polygon, "area", 0.0)), 1e-6)
        transformed_normal = transform.to_3x3() @ polygon.normal
        if transformed_normal.length > 1e-6:
            normal += transformed_normal.normalized() * weight
    if not world_vertices:
        return None
    # Blender's MeshPolygon.center can be badly biased for the concave ngons
    # used by the radial sticker meshes. Use the midpoint of the face's
    # tangential vertex bounds instead, retaining the face depth separately.
    anchor_polygon = max(
        polygons,
        key=lambda polygon: max(float(getattr(polygon, "area", 0.0)), 1e-6),
    )
    center_local = anchor_polygon.center.copy()
    local_normal = anchor_polygon.normal.normalized()
    normal_axis = max(range(3), key=lambda axis_index: abs(local_normal[axis_index]))
    for axis_index in range(3):
        if axis_index == normal_axis:
            continue
        coordinates = [
            float(obj.data.vertices[int(vertex_index)].co[axis_index])
            for polygon in polygons
            for vertex_index in polygon.vertices
        ]
        if coordinates:
            center_local[axis_index] = (min(coordinates) + max(coordinates)) * 0.5
    center = transform @ center_local
    if normal.length <= 1e-6:
        normal = (world_vertices[1] - world_vertices[0]).cross(
            world_vertices[2] - world_vertices[0]
        )
    if normal.length <= 1e-6:
        return None
    # Detached sticker meshes have a small bevel/thickness. Their volume
    # centroid is therefore inside the cubie, which would make an otherwise
    # visible center look occluded by its own cubie. Use the outward-facing
    # front polygon center instead.
    if obj.name.startswith(STICKER_OBJECT_NAME_PREFIX):
        parent_center = (
            obj.parent.matrix_world.translation.copy()
            if obj.parent is not None
            else center.copy()
        )
        outward_hint = center - parent_center
        if outward_hint.length > 1e-6:
            front_polygon = max(
                polygons,
                key=lambda polygon: (
                    max(float(getattr(polygon, "area", 0.0)), 1e-6)
                    * max(
                        0.0,
                        (transform.to_3x3() @ polygon.normal).normalized().dot(
                            outward_hint.normalized()
                        ),
                    )
                ),
            )
            center_local = front_polygon.center.copy()
            local_normal = front_polygon.normal.normalized()
            normal_axis = max(
                range(3),
                key=lambda axis_index: abs(local_normal[axis_index]),
            )
            for axis_index in range(3):
                if axis_index == normal_axis:
                    continue
                coordinates = [
                    float(obj.data.vertices[int(vertex_index)].co[axis_index])
                    for polygon in polygons
                    for vertex_index in polygon.vertices
                ]
                if coordinates:
                    center_local[axis_index] = (min(coordinates) + max(coordinates)) * 0.5
            center = transform @ center_local
            normal = transform.to_3x3() @ front_polygon.normal
            if normal.dot(outward_hint) < 0.0:
                normal.negate()
            center += normal.normalized() * 0.003
    return {
        "name": mask_key,
        "mask_key": mask_key,
        "source_object": obj.name,
        "material_name": material_name,
        "class_id": STICKER_KEYPOINT_CLASS_ID_BY_MATERIAL[material_name],
        "class": STICKER_KEYPOINT_CLASS_NAMES[
            STICKER_KEYPOINT_CLASS_ID_BY_MATERIAL[material_name]
        ],
        "point": center,
        "normal": normal.normalized(),
        "world_polygons": world_polygons,
        "polygon_indices": [int(polygon.index) for polygon in polygons],
    }


def sticker_keypoint_specs(main_cube_roots: list[bpy.types.Object]) -> list[dict[str, object]]:
    """Return one center spec for every physical sticker in the cube."""
    specs = []
    covered_parent_materials: set[tuple[str, str]] = set()
    sticker_names = sticker_render_mesh_names(main_cube_roots)
    for sticker_name in sorted(sticker_names):
        obj = bpy.data.objects.get(sticker_name)
        if obj is None or not sticker_name.startswith(STICKER_OBJECT_NAME_PREFIX):
            continue
        material_name = sticker_material_name(obj)
        if material_name is None:
            continue
        if obj.parent is not None:
            covered_parent_materials.add((obj.parent.name, material_name))
        polygons = list(getattr(getattr(obj, "data", None), "polygons", ()))
        face_candidates = []
        for face_name, face_polygons in sticker_polygon_face_groups(obj, polygons):
            candidate = sticker_surface_spec(
                obj,
                material_name,
                face_polygons,
                f"{obj.name}:face:{face_name}",
            )
            if candidate is not None:
                face_candidates.append(candidate)
        if face_candidates:
            spec = dict(face_candidates[0])
            spec["name"] = obj.name
            spec["mask_key"] = obj.name
            spec["source_cubie"] = obj.parent.name if obj.parent is not None else None
            spec["face_candidates"] = face_candidates
            spec["polygon_indices"] = [
                int(polygon.index)
                for candidate in face_candidates
                for polygon_index in candidate["polygon_indices"]
                for polygon in polygons
                if int(polygon.index) == int(polygon_index)
            ]
            specs.append(spec)

    # Support older cube assets whose sticker faces are material slots on the
    # cubie instead of detached sticker objects.
    for obj in main_cube_roots:
        if obj.type != "MESH":
            continue
        materials = getattr(obj.data, "materials", ())
        for material_name in STICKER_COLOR_MATERIAL_NAMES:
            if (obj.name, material_name) in covered_parent_materials:
                continue
            material_indices = {
                index
                for index, material in enumerate(materials)
                if material is not None and material.name == material_name
            }
            material_polygons = [
                polygon
                for polygon in getattr(obj.data, "polygons", ())
                if polygon.material_index in material_indices
            ]
            face_candidates = []
            for face_name, face_polygons in sticker_polygon_face_groups(
                obj,
                material_polygons,
            ):
                candidate = sticker_surface_spec(
                    obj,
                    material_name,
                    face_polygons,
                    f"{obj.name}:{material_name}:face:{face_name}",
                )
                if candidate is not None:
                    face_candidates.append(candidate)
            if face_candidates:
                spec = dict(face_candidates[0])
                spec["name"] = f"{obj.name}:{material_name}"
                spec["mask_key"] = f"{obj.name}:{material_name}"
                spec["source_cubie"] = obj.name
                spec["face_candidates"] = face_candidates
                spec["polygon_indices"] = [
                    int(polygon.index)
                    for candidate in face_candidates
                    for polygon_index in candidate["polygon_indices"]
                    for polygon in material_polygons
                    if int(polygon.index) == int(polygon_index)
                ]
                specs.append(spec)
    return specs


def projected_sticker_area_mask(
    spec: dict[str, object],
    width: int,
    height: int,
) -> object:
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError("OpenCV and NumPy are required for sticker keypoint labels.") from None

    mask = np.zeros((height, width), dtype=np.uint8)
    camera = get_camera()
    scene = bpy.context.scene
    for world_polygon in spec.get("world_polygons", []):
        projected = [
            project_world_point_unclipped(point, camera, scene)
            for point in world_polygon
        ]
        if any(point is None for point in projected) or len(projected) < 3:
            continue
        points = np.round(np.asarray(projected, dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask


def select_sticker_face_spec(
    spec: dict[str, object],
    visible_mask: object,
    width: int,
    height: int,
) -> dict[str, object]:
    """Choose the physical face whose projection owns this sticker's pixels."""
    try:
        import numpy as np
    except ImportError:
        raise RuntimeError("NumPy is required for sticker keypoint labels.") from None
    candidates = spec.get("face_candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return spec
    if not isinstance(visible_mask, np.ndarray) or not np.any(visible_mask):
        return {**spec, **candidates[0], "name": spec["name"], "mask_key": spec["mask_key"]}
    best_candidate = candidates[0]
    best_overlap = -1
    for candidate in candidates:
        candidate_mask = projected_sticker_area_mask(candidate, width, height)
        overlap = int(np.count_nonzero((candidate_mask != 0) & (visible_mask != 0)))
        if overlap > best_overlap:
            best_overlap = overlap
            best_candidate = candidate
    return {
        **spec,
        **best_candidate,
        "name": spec["name"],
        "mask_key": spec["mask_key"],
        "face_candidates": candidates,
    }


def sticker_keypoint_bbox(
    visible_mask: object,
    point: tuple[float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    try:
        import numpy as np
    except ImportError:
        raise RuntimeError("NumPy is required for sticker keypoint labels.") from None
    if isinstance(visible_mask, np.ndarray):
        ys, xs = np.where(visible_mask != 0)
        if len(xs) > 0:
            x_min = float(xs.min())
            x_max = float(xs.max() + 1)
            y_min = float(ys.min())
            y_max = float(ys.max() + 1)
            return (
                (x_min + x_max) / 2.0,
                (y_min + y_max) / 2.0,
                max(1.0, x_max - x_min),
                max(1.0, y_max - y_min),
            )
    radius = max(2.0, float(scaled_annotation_size(4, width, height)))
    x, y = point
    return x, y, min(width, radius * 2.0), min(height, radius * 2.0)


def render_sticker_keypoints(
    label_path: Path,
    mask_path: Path,
    image_path: Path,
    main_cube_roots: list[bpy.types.Object],
    hand_roots: list[bpy.types.Object],
    width: int,
    height: int,
    landmark_lines: list[str] | None = None,
    landmark_pose_objects: list[dict[str, object]] | None = None,
) -> tuple[list[str], dict[str, object]]:
    """Write landmark, sticker and six-center cube-core instances."""
    try:
        import numpy as np
    except ImportError:
        raise RuntimeError("NumPy is required for sticker keypoint labels.") from None

    specs = sticker_keypoint_specs(main_cube_roots)
    visibility_map, sticker_masks = render_visibility_class_map(
        main_cube_roots,
        hand_roots,
        width,
        height,
    )
    depsgraph = bpy.context.evaluated_depsgraph_get()
    camera = get_camera()
    scene = bpy.context.scene
    lines = []
    keypoints = []
    core_by_color: dict[int, dict[str, object]] = {}
    center_cubies = {obj.name: obj for obj in center_cubie_objects()}
    cube_center = cubie_world_center(list(center_cubies.values()))
    counts = {
        "visible": 0,
        "occluded": 0,
        "excluded": 0,
        "out_of_frame": 0,
        "angle_filtered": 0,
    }
    for base_spec in specs:
        center_cubie = center_cubies.get(base_spec["source_cubie"])
        if center_cubie is not None:
            outward = center_cubie.matrix_world.translation - cube_center
            spec = max(base_spec["face_candidates"], key=lambda face: face["normal"].dot(outward))
            point = spec["point"]
            projected = project_world_point(point, camera, scene)
            occluded_by = None
            visibility = POINT_VISIBILITY_UNLABELED
            if projected is not None:
                # A back-facing or occluded center still has a coordinate target.
                occluded_by = object_occluding_point(point, depsgraph)
                front_facing = spec["normal"].dot(camera.location - point) > 0.0
                visibility = (
                    POINT_VISIBILITY_VISIBLE if front_facing and occluded_by is None
                    else POINT_VISIBILITY_LABELED_OCCLUDED
                )
            color_id = int(spec["class_id"])
            if color_id in core_by_color:
                raise ValueError(f"Multiple center stickers for {spec['class']}")
            core_by_color[color_id] = {
                "source_object": spec["source_object"],
                "source_cubie": center_cubie.name,
                "class": spec["class"],
                "sticker_class_id": color_id,
                "pixel_point": projected if projected is not None else (0.0, 0.0),
                "normalized_point": (
                    normalize_point(*projected, width, height)
                    if projected is not None else (0.0, 0.0)
                ),
                "visibility": visibility,
                "occluded_by": occluded_by,
            }
            continue
        visible_mask = sticker_masks.get(base_spec["mask_key"])
        spec = select_sticker_face_spec(base_spec, visible_mask, width, height)
        point = spec["point"]
        projected = project_world_point(point, camera, scene)
        projected_mask = projected_sticker_area_mask(spec, width, height)
        projected_area = int(np.count_nonzero(projected_mask))
        visible_area = (
            int(np.count_nonzero(visible_mask))
            if isinstance(visible_mask, np.ndarray)
            else 0
        )
        area_ratio = visible_area / projected_area if projected_area > 0 else 0.0
        if projected is None:
            counts["out_of_frame"] += 1
            counts["excluded"] += 1
            continue
        if not point_faces_camera(
            point,
            [spec["normal"]],
            camera.location,
            min_plane_angle_degrees=STICKER_KEYPOINT_MIN_PLANE_ANGLE_DEGREES,
        ):
            counts["angle_filtered"] += 1
            counts["excluded"] += 1
            continue
        occluded_by = object_occluding_point(point, depsgraph)
        is_occluded = occluded_by is not None
        if not is_occluded:
            visibility = POINT_VISIBILITY_VISIBLE
            counts["visible"] += 1
        elif area_ratio > STICKER_KEYPOINT_MIN_VISIBLE_AREA_RATIO:
            visibility = POINT_VISIBILITY_LABELED_OCCLUDED
            counts["occluded"] += 1
        else:
            counts["excluded"] += 1
            continue
        bbox = sticker_keypoint_bbox(visible_mask, projected, width, height)
        line = yolo_pose_line(
            STICKER_KEYPOINT_DATASET_CLASS_ID_OFFSET + int(spec["class_id"]),
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3],
            [(projected[0], projected[1], visibility)],
            width,
            height,
        )
        lines.append(line)
        keypoints.append({
            "name": spec["name"],
            "source_object": spec["source_object"],
            "material": spec["material_name"],
            "class": spec["class"],
            "class_id": STICKER_KEYPOINT_DATASET_CLASS_ID_OFFSET + int(spec["class_id"]),
            "sticker_class_id": int(spec["class_id"]),
            "pixel_point": [projected[0], projected[1]],
            "normalized_point": list(normalize_point(projected[0], projected[1], width, height)),
            "visibility": visibility,
            "occluded": is_occluded,
            "occluded_by": occluded_by,
            "visible_area_pixels": visible_area,
            "projected_area_pixels": projected_area,
            "visible_area_ratio": area_ratio,
        })

    missing = [
        color for color_id, color in enumerate(STICKER_KEYPOINT_CLASS_NAMES)
        if color_id not in core_by_color
    ]
    if missing:
        raise ValueError(f"Missing cube-core center stickers: {', '.join(missing)}")
    core_keypoints = [core_by_color[color_id] for color_id in range(CUBE_CORE_KEYPOINT_SHAPE[0])]
    core_points = [(*item["pixel_point"], item["visibility"]) for item in core_keypoints]
    cube_mask = visibility_map == 1
    bbox = (
        sticker_keypoint_bbox(cube_mask, (0.0, 0.0), width, height)
        if np.any(cube_mask) else keypoint_bbox(core_points, width, height)
    )
    padding = " 0 0 0" * (CUBE_CORE_KEYPOINT_SHAPE[0] - 1)
    combined_lines = [line + padding for line in chain(landmark_lines or (), lines)]
    if bbox is not None:
        combined_lines.append(yolo_pose_line(CUBE_CORE_CLASS_ID, *bbox, core_points, width, height))
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        "\n".join(combined_lines) + ("\n" if combined_lines else ""),
        encoding="utf-8",
    )
    write_sticker_keypoint_mask(
        mask_path,
        image_path,
        keypoints,
        width,
        height,
        landmark_pose_objects=landmark_pose_objects,
        core_keypoints=core_keypoints,
    )
    return combined_lines, {
        "format": "YOLO pose combined landmark, sticker-center and cube-core keypoints",
        "class_names": list(COMBINED_KEYPOINT_CLASS_NAMES),
        "landmark_class_names": list(INDEPENDENT_CLASS_NAMES),
        "sticker_class_names": list(STICKER_KEYPOINT_CLASS_NAMES),
        "keypoint_shape": CUBE_CORE_KEYPOINT_SHAPE,
        "min_face_plane_angle_degrees": STICKER_KEYPOINT_MIN_PLANE_ANGLE_DEGREES,
        "min_visible_area_ratio_for_occluded": STICKER_KEYPOINT_MIN_VISIBLE_AREA_RATIO,
        "keypoints": keypoints,
        "landmark_keypoints": [
            keypoint
            for pose_object in (landmark_pose_objects or [])
            for keypoint in pose_object.get("keypoints", [])
        ],
        "landmark_pose_objects": landmark_pose_objects or [],
        "cube_core_keypoints": core_keypoints,
        "cube_core_labeled": bbox is not None,
        "counts": counts,
        "candidate_count": len(specs),
    }


def write_sticker_keypoint_mask(
    path: Path,
    image_path: Path,
    keypoints: list[dict[str, object]],
    width: int,
    height: int,
    *,
    core_keypoints: list[dict[str, object]],
    landmark_pose_objects: list[dict[str, object]] | None = None,
) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError("OpenCV is required for sticker keypoint masks.") from None
    image = read_image_file(image_path, cv2, np, cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (height, width):
        raise RuntimeError(f"Could not read rendered image for sticker keypoint mask: {image_path}")
    output = image.copy()
    if landmark_pose_objects:
        draw_pose_keypoint_mask(output, landmark_pose_objects, width, height, cv2, np)
    point_radius = scaled_annotation_size(POSE_POINT_BASE_RADIUS + 1, width, height)
    outline_radius = point_radius + max(1, POSE_POINT_OUTLINE_WIDTH)
    for keypoint in keypoints:
        class_id = int(keypoint["sticker_class_id"])
        color = STICKER_KEYPOINT_MASK_COLORS[class_id]
        x, y = keypoint["pixel_point"]
        center = (round(float(x)), round(float(y)))
        cv2.circle(output, center, outline_radius, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(output, center, point_radius, color, thickness=-1, lineType=cv2.LINE_AA)
    diamond_radius = scaled_annotation_size(POSE_POINT_BASE_RADIUS + 3, width, height)
    for keypoint in sorted(core_keypoints, key=lambda item: item["visibility"]):
        if keypoint["visibility"] == POINT_VISIBILITY_UNLABELED:
            continue
        color = STICKER_KEYPOINT_MASK_COLORS[int(keypoint["sticker_class_id"])]
        x, y = (round(float(value)) for value in keypoint["pixel_point"])
        for radius, fill in ((diamond_radius + 1, (0, 0, 0)), (diamond_radius, color)):
            vertices = np.asarray(
                [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(output, vertices, fill, lineType=cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not write_image_file(path, output, cv2):
        raise RuntimeError(f"Could not write sticker keypoint mask: {path}")


def segmentation_lines_from_mask(
    mask: object,
    class_id: int,
    one_instance: bool = False,
) -> list[str]:
    """Return YOLO polygon lines for a mask's components or one instance."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "OpenCV and NumPy are required to write YOLO segmentation annotations."
        ) from None

    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise TypeError("Segmentation mask must be a 2-D NumPy array.")
    mask_height, mask_width = mask.shape[:2]
    if one_instance:
        ys, xs = np.where(mask != 0)
        if len(xs) == 0:
            return []
        if len(xs) >= 3:
            points = np.column_stack((xs, ys)).astype(np.int32)
            contour = cv2.convexHull(points.reshape(-1, 1, 2))
            points = contour.reshape(-1, 2).astype(float)
        else:
            x_min, x_max = float(xs.min()), float(xs.max() + 1)
            y_min, y_max = float(ys.min()), float(ys.max() + 1)
            points = np.asarray(
                ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)),
                dtype=float,
            )
        normalized_points = []
        for x, y in points:
            normalized_points.extend((
                max(0.0, min(float(x) / mask_width, 1.0)),
                max(0.0, min(float(y) / mask_height, 1.0)),
            ))
        return [
            str(class_id)
            + " "
            + " ".join(f"{value:.6f}" for value in normalized_points)
        ]

    binary_mask = np.where(mask != 0, 255, 0).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    lines = []
    for contour in contours:
        if len(contour) < 3 or cv2.contourArea(contour) <= 0.0:
            continue
        points = contour.reshape(-1, 2).astype(float)
        if len(points) < 3:
            continue
        normalized_points = []
        for x, y in points:
            normalized_points.extend((
                max(0.0, min(float(x) / mask_width, 1.0)),
                max(0.0, min(float(y) / mask_height, 1.0)),
            ))
        lines.append(
            str(class_id)
            + " "
            + " ".join(f"{value:.6f}" for value in normalized_points)
        )
    return lines


def write_yolo_cube_sticker_segmentation_label(
    cube_mask: object,
    sticker_masks: object,
    label_path: Path,
) -> int:
    """Write the cube and one polygon per connected visible sticker component.

    YOLO polygon rows cannot represent disconnected regions as one instance.
    The exact physical-sticker grouping remains available in the in-memory
    masks and metadata, while the label file preserves each visible region.
    """
    try:
        import numpy as np
    except ImportError:
        raise RuntimeError("NumPy is required to write YOLO segmentation annotations.") from None

    if not isinstance(cube_mask, np.ndarray):
        raise TypeError("Cube mask must be a NumPy array.")
    lines = segmentation_lines_from_mask(cube_mask, 0)
    if isinstance(sticker_masks, np.ndarray):
        sticker_masks = {"sticker": sticker_masks}
    if not isinstance(sticker_masks, dict):
        raise TypeError("Sticker masks must be a dict of object name to NumPy mask.")
    for _sticker_name, sticker_mask in sorted(sticker_masks.items()):
        if not isinstance(sticker_mask, np.ndarray) or sticker_mask.shape != cube_mask.shape:
            raise ValueError("Every sticker mask must match the cube mask dimensions.")
        lines.extend(segmentation_lines_from_mask(sticker_mask, 1))
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def write_segmentation_mask(
    path: Path,
    cube_mask: object,
    sticker_masks: object,
) -> None:
    """Write an RGB verification mask: blue cube, green stickers, black background."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError("OpenCV and NumPy are required to write segmentation masks.") from None

    if not isinstance(cube_mask, np.ndarray):
        raise TypeError("Cube mask must be a NumPy array.")
    if isinstance(sticker_masks, np.ndarray):
        sticker_masks = {"sticker": sticker_masks}
    if not isinstance(sticker_masks, dict):
        raise TypeError("Sticker masks must be a dict of object name to NumPy mask.")
    sticker_mask = np.zeros_like(cube_mask, dtype=np.uint8)
    for mask in sticker_masks.values():
        if not isinstance(mask, np.ndarray) or mask.shape != cube_mask.shape:
            raise ValueError("Every sticker mask must match the cube mask dimensions.")
        sticker_mask |= (mask != 0).astype(np.uint8)
    output = np.zeros((*cube_mask.shape, 3), dtype=np.uint8)
    output[cube_mask != 0] = (180, 80, 0)
    output[sticker_mask != 0] = (0, 210, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not write_image_file(path, output, cv2):
        raise RuntimeError(f"Could not write segmentation mask: {path}")


def write_yolo_segmentation_label(
    mask: object,
    label_path: Path,
    class_id_map: dict[int, int],
) -> int:
    """Convert an in-memory class map into one YOLO polygon label file."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "OpenCV and NumPy are required to write YOLO segmentation annotations."
        ) from None

    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise TypeError("Segmentation mask must be a 2-D NumPy array.")
    mask_height, mask_width = mask.shape[:2]
    if mask_width <= 0 or mask_height <= 0:
        raise RuntimeError("Segmentation mask has invalid dimensions.")

    lines = []
    for source_class_id, yolo_class_id in sorted(class_id_map.items()):
        binary_mask = np.where(mask == source_class_id, 255, 0).astype(np.uint8)
        contours, _hierarchy = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            if len(contour) < 3 or cv2.contourArea(contour) <= 0.0:
                continue
            points = contour.reshape(-1, 2).astype(float)
            if len(points) < 3:
                continue
            normalized_points = []
            for x, y in points:
                normalized_points.extend((
                    max(0.0, min(float(x) / mask_width, 1.0)),
                    max(0.0, min(float(y) / mask_height, 1.0)),
                ))
            lines.append(
                str(yolo_class_id)
                + " "
                + " ".join(f"{value:.6f}" for value in normalized_points)
            )

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def render_png(path: Path) -> None:
    """Publish a still only after Blender has successfully written it."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    render_settings = bpy.context.scene.render
    previous_filepath = render_settings.filepath
    try:
        with tempfile.TemporaryDirectory(prefix=".render-", dir=path.parent) as directory:
            staged_path = Path(directory) / path.name
            render_settings.filepath = str(staged_path)
            result = bpy.ops.render.render(write_still=True)
            if (
                "FINISHED" not in result
                or not staged_path.is_file()
                or staged_path.stat().st_size == 0
            ):
                raise RuntimeError(f"Blender did not write the expected render: {path}")
            staged_path.replace(path)
    finally:
        render_settings.filepath = previous_filepath


def read_image_file(path: Path, cv2: object, np: object, flags: int) -> object | None:
    """Read an image through Python file I/O for Windows path compatibility."""
    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)


def write_image_file(path: Path, image: object, cv2: object) -> bool:
    """Write an image through Python file I/O for Windows path compatibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix.lower() or ".png", image)
    if not success:
        return False
    try:
        path.write_bytes(encoded.tobytes())
    except OSError:
        return False
    return True


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


def layer_turn_blur_object_axes(
    layer_state: dict[str, object],
    cube_pose: dict[str, object],
    hand_rig_names: set[str],
) -> dict[str, tuple[Vector, float]]:
    """Return world-space blur axes for objects participating in layer motion.

    A corner cut or simultaneous turn can contain multiple layer turns.
    Objects shared by multiple layers use the axis of the larger turn. Only
    the hand corresponding to an R/L component receives localized blur,
    matching the actual hand transform; U/D/F/B and slice components do not
    move either hand.
    """
    mode = layer_state.get("mode")
    if mode == "none":
        return {}

    transform_rows = cube_pose.get("transform")
    if not isinstance(transform_rows, list):
        raise ValueError("Cube pose is missing its shared transform matrix.")
    world_rotation = matrix_from_rows(transform_rows).to_3x3()
    if mode not in {"mid_turn", "corner_cut", "slice_turn", "simultaneous_turn"}:
        return {}

    object_axes: dict[str, tuple[float, Vector]] = {}
    hand_axes: dict[str, Vector] = {}
    for face, angle, direction, object_names in _partial_turn_layers(layer_state):
        if not isinstance(angle, (int, float)):
            continue
        if not isinstance(direction, (int, float)):
            direction = 1.0

        magnitude = abs(float(angle))
        axis = (world_rotation @ partial_turn_axis(face)).normalized()
        if float(direction) < 0.0:
            axis = -axis
        if face in {"R", "L"}:
            hand_name = "right-hand" if face == "R" else "left-hand"
            hand_axes[f"{hand_name}-rig"] = axis
        for name in object_names:
            if not isinstance(name, str):
                continue
            previous = object_axes.get(name)
            if previous is None or magnitude > previous[0]:
                object_axes[name] = (magnitude, axis)

    result = {
        name: (axis, 1.0)
        for name, (_magnitude, axis) in object_axes.items()
    }
    for name in hand_rig_names:
        hand_axis = hand_axes.get(name)
        if hand_axis is not None:
            result[name] = (hand_axis, HAND_LOCAL_ROTATIONAL_BLUR_FRACTION)
    return result


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


def set_rotational_blur_sample_transforms(
    objects: list[bpy.types.Object],
    base_transforms: dict[str, tuple[Vector, Quaternion]],
    global_axis: Vector,
    global_angle_radians: float,
    local_angle_radians: float,
    local_object_axes: dict[str, tuple[Vector, float]],
    local_pivot: Vector | None = None,
) -> None:
    """Apply separate whole-scene and localized motion to one blur sample."""
    global_rotation = Quaternion(global_axis, global_angle_radians)
    if local_pivot is None:
        local_pivot = Vector((0.0, 0.0, 0.0))
    for obj in objects:
        base_location, base_rotation = base_transforms[obj.name]
        local_motion = local_object_axes.get(obj.name)
        if local_motion is not None:
            local_axis, local_magnitude_scale = local_motion
            motion_rotation = Quaternion(
                local_axis,
                local_angle_radians * local_magnitude_scale,
            )
            obj.location = local_pivot + motion_rotation @ (base_location - local_pivot)
        else:
            motion_rotation = global_rotation
            obj.location = motion_rotation @ base_location
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = motion_rotation @ base_rotation
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
        image = read_image_file(sample_path, cv2, np, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not read rendered blur sample: {sample_path}")
        image_float = image.astype(np.float32)
        accumulator = image_float if accumulator is None else accumulator + image_float

    averaged = np.clip(accumulator / len(sample_paths), 0, 255).astype(np.uint8)
    if not write_image_file(output_path, averaged, cv2):
        raise RuntimeError(f"Could not write averaged rotational blur render: {output_path}")


def render_cube_rotational_blur_png(
    path: Path,
    objects: list[bpy.types.Object],
    probability: float,
    max_degrees: float,
    samples: int,
    local_object_axes: dict[str, tuple[Vector, float]] | None = None,
    local_pivot: Vector | None = None,
) -> dict[str, object] | None:
    if probability <= 0.0 or max_degrees <= 0.0 or samples < 2 or random.random() >= probability:
        render_png(path)
        return None

    axis = random_point_on_unit_sphere()
    # Keep the whole-image blur in the requested 3-8 degree range. Objects in
    # local_object_axes receive the larger range around their layer-turn axis,
    # with a per-object scale for the less-coupled hand motion.
    global_min_degrees = min(GLOBAL_ROTATIONAL_BLUR_DEGREES_RANGE[0], max_degrees)
    global_max_degrees = min(GLOBAL_ROTATIONAL_BLUR_DEGREES_RANGE[1], max_degrees)
    global_magnitude_degrees = random.uniform(
        global_min_degrees,
        max(global_min_degrees, global_max_degrees),
    )
    available_object_names = {obj.name for obj in objects}
    local_object_axes = {
        name: (axis.normalized(), float(magnitude_scale))
        for name, (axis, magnitude_scale) in (local_object_axes or {}).items()
        if name in available_object_names and axis.length_squared > 1e-12
    }
    local_magnitude_degrees = (
        min(
            MAX_LOCAL_ROTATIONAL_BLUR_DEGREES,
            random.uniform(
                max_degrees * LOCAL_ROTATIONAL_BLUR_FRACTION_RANGE[0],
                max_degrees * LOCAL_ROTATIONAL_BLUR_FRACTION_RANGE[1],
            ),
        )
        if local_object_axes
        else global_magnitude_degrees
    )
    base_transforms = object_transforms(objects)
    sample_paths = []
    blur_error = None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".blur-", dir=path.parent) as temp_dir:
            temp_path = Path(temp_dir)
            global_angles = rotational_blur_sample_angles(global_magnitude_degrees, samples)
            local_angles = rotational_blur_sample_angles(local_magnitude_degrees, samples)
            for sample_index, (global_angle, local_angle) in enumerate(zip(global_angles, local_angles)):
                set_rotational_blur_sample_transforms(
                    objects,
                    base_transforms,
                    axis,
                    global_angle,
                    local_angle,
                    local_object_axes,
                    local_pivot=local_pivot,
                )
                sample_path = temp_path / f"rotational_blur_{sample_index:03d}.png"
                try:
                    render_png(sample_path)
                except (OSError, RuntimeError) as error:
                    print(
                        f"Blur subframe {sample_index + 1}/{samples} for {path.name} failed. "
                        f"Retrying once. {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    render_png(sample_path)
                sample_paths.append(sample_path)

            averaged_path = temp_path / "averaged.png"
            average_rendered_pngs(sample_paths, averaged_path)
            averaged_path.replace(path)
    except (OSError, RuntimeError) as error:
        blur_error = error
    finally:
        restore_object_transforms(objects, base_transforms)

    if blur_error is not None:
        print(
            f"Rotational blur failed for {path.name}. Rendering the base pose. {blur_error}",
            file=sys.stderr,
            flush=True,
        )
        render_png(path)
        return None

    return {
        "axis": [axis.x, axis.y, axis.z],
        "magnitude_degrees": global_magnitude_degrees,
        "global_magnitude_degrees": global_magnitude_degrees,
        "local_magnitude_degrees": local_magnitude_degrees if local_object_axes else None,
        "hand_local_magnitude_degrees": (
            local_magnitude_degrees * HAND_LOCAL_ROTATIONAL_BLUR_FRACTION
            if any(
                magnitude_scale < 1.0
                for _axis, magnitude_scale in local_object_axes.values()
            )
            else None
        ),
        "local_object_count": len(local_object_axes),
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


def apply_video_compression(
    frame: object,
    cv2: object,
    np: object,
) -> tuple[object, dict[str, object]]:
    """Round-trip a frame through a short H.264/yuv420p video sequence."""
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg).exists():
        raise RuntimeError(
            "Video degradation requires ffmpeg with libx264. "
            "Install ffmpeg or set it on PATH."
        )

    height, width = frame.shape[:2]
    context_frames = VIDEO_COMPRESSION_CONTEXT_FRAMES
    center_index = context_frames // 2
    jitter_x = random.uniform(*VIDEO_COMPRESSION_MOTION_JITTER_PX_RANGE)
    jitter_y = random.uniform(*VIDEO_COMPRESSION_MOTION_JITTER_PX_RANGE)
    if random.random() < 0.5:
        jitter_x = -jitter_x
    if random.random() < 0.5:
        jitter_y = -jitter_y
    crf = random.randint(*VIDEO_COMPRESSION_CRF_RANGE)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        source_pattern = temp_path / "source_%03d.png"
        decoded_pattern = temp_path / "decoded_%03d.png"
        video_path = temp_path / "compressed.mp4"
        for frame_index in range(context_frames):
            relative_index = frame_index - center_index
            transform = np.asarray(
                (
                    (1.0, 0.0, relative_index * jitter_x / center_index),
                    (0.0, 1.0, relative_index * jitter_y / center_index),
                ),
                dtype=np.float32,
            )
            context_frame = cv2.warpAffine(
                frame,
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            source_path = temp_path / f"source_{frame_index:03d}.png"
            if not write_image_file(source_path, context_frame, cv2):
                raise RuntimeError(f"Could not write temporary video frame: {source_path}")

        encode_command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(VIDEO_COMPRESSION_FRAME_RATE),
            "-i",
            str(source_pattern),
            "-frames:v",
            str(context_frames),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(context_frames + 1),
            "-bf",
            "2",
            "-an",
            str(video_path),
        ]
        decode_command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(context_frames),
            str(decoded_pattern),
        ]
        try:
            subprocess.run(
                encode_command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            subprocess.run(
                decode_command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "Video degradation requires ffmpeg with libx264. "
                "Install ffmpeg or set it on PATH."
            ) from error
        except subprocess.CalledProcessError as error:
            details = error.stderr.strip() if error.stderr else "unknown ffmpeg error"
            raise RuntimeError(f"Video degradation failed: {details}") from error

        decoded_path = temp_path / f"decoded_{center_index + 1:03d}.png"
        decoded = read_image_file(decoded_path, cv2, np, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError(f"Could not read decoded video frame: {decoded_path}")
        if decoded.shape[:2] != (height, width):
            decoded = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_LINEAR)

    metadata = {
        "codec": "h264",
        "pixel_format": "yuv420p",
        "crf": crf,
        "preset": "veryfast",
        "frame_rate": VIDEO_COMPRESSION_FRAME_RATE,
        "context_frames": context_frames,
        "center_frame": center_index,
        "motion_jitter_px": [jitter_x, jitter_y],
        "encoder": ffmpeg,
    }
    return decoded, metadata


def apply_camera_degradation(
    path: Path,
    probability: float,
) -> dict[str, object] | None:
    """Apply post-render camera artifacts."""
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

    image = read_image_file(path, cv2, np, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(
            "Could not read rendered image for degradation: "
            f"{path.resolve()} (exists={path.exists()})"
        )

    metadata: dict[str, object] = {}
    alpha = image[:, :, 3] if image.ndim == 3 and image.shape[2] == 4 else None
    frame = image[:, :, :3] if alpha is not None else image

    if random.random() < DEGRADE_RESCALE_PROBABILITY:
        scale = random.uniform(0.70, 0.95)
        height, width = frame.shape[:2]
        small = cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        frame = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
        metadata["rescale"] = {"scale": scale}

    if random.random() < DEGRADE_GAUSSIAN_BLUR_PROBABILITY:
        kernel = odd_kernel(random.choice((3, 3, 5)))
        sigma = random.uniform(0.25, 0.9)
        frame = cv2.GaussianBlur(frame, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
        metadata["gaussian_blur"] = {"kernel": kernel, "sigma": sigma}

    if random.random() < DEGRADE_MOTION_BLUR_PROBABILITY:
        kernel_size = odd_kernel(random.choice((3, 5, 7)))
        angle = random.uniform(0.0, 180.0)
        frame = cv2.filter2D(frame, -1, motion_blur_kernel(kernel_size, angle))
        metadata["motion_blur"] = {"kernel": kernel_size, "angle_degrees": angle}

    if random.random() < DEGRADE_NOISE_PROBABILITY:
        sigma = random.uniform(1.0, 7.0)
        noise = np.random.normal(0.0, sigma, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
        metadata["sensor_noise"] = {"sigma": sigma}

    if random.random() < DEGRADE_VIDEO_COMPRESSION_PROBABILITY:
        frame, metadata["video_compression"] = apply_video_compression(frame, cv2, np)

    if alpha is not None:
        frame = np.dstack([frame, alpha])

    if not write_image_file(path, frame, cv2):
        raise RuntimeError(f"Could not write degraded image: {path}")
    return metadata or None


def normalize_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return (
        max(0.0, min(x / width, 1.0)),
        max(0.0, min(y / height, 1.0)),
    )


def scaled_annotation_size(value: int, width: int, height: int) -> int:
    """Scale an annotation width/radius from the 256px reference size."""
    scale = min(width, height) / ANNOTATION_REFERENCE_SIZE
    return max(1, int(round(value * scale)))


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


def project_world_point_unclipped(
    point: Vector,
    camera: bpy.types.Object,
    scene: bpy.types.Scene,
) -> tuple[float, float] | None:
    """Project a point without dropping it just because it is outside the frame."""
    co = world_to_camera_view(scene, camera, point)
    if co.z < 0.0:
        return None
    return co.x * scene.render.resolution_x, (1.0 - co.y) * scene.render.resolution_y


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


def center_cubie_objects(
    objects: list[bpy.types.Object] | None = None,
) -> list[bpy.types.Object]:
    if objects is None:
        objects = cube_objects()
    centers = []
    for obj in objects:
        solved_location = Vector(obj.get(SOLVED_LOCATION_PROP, obj.location))
        outer_axis_count = sum(
            1
            for axis_index in range(3)
            if abs(round(solved_location[axis_index])) == 1
        )
        if outer_axis_count == 1:
            centers.append(obj)
    return sorted(centers, key=lambda obj: obj.name)


def edge_cubie_objects() -> list[bpy.types.Object]:
    """Return the twelve cubies whose solved locations lie on cube edges."""
    edges = []
    for obj in cube_objects():
        solved_location = Vector(obj.get(SOLVED_LOCATION_PROP, obj.location))
        outer_axis_count = sum(
            1
            for axis_index in range(3)
            if abs(round(solved_location[axis_index])) == 1
        )
        if outer_axis_count == 2:
            edges.append(obj)
    return sorted(edges, key=lambda obj: obj.name)


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
        relative_one_axes = (0, 1, 2)
        local_offset = Vector(tuple(-inner_offset[axis_index] for axis_index in range(3)))
        point = obj.matrix_world @ local_offset
        normals = [
            (obj.matrix_world.to_3x3() @ axis_vector(axis_index, -inner_offset[axis_index] * 2.0)).normalized()
            for axis_index in relative_one_axes
        ]
        specs.append({
            "name": f"{obj.name}:{INDEPENDENT_CLASS_NAMES[CORNER_CLASS_ID]}:012",
            "class_id": CORNER_CLASS_ID,
            "class": INDEPENDENT_CLASS_NAMES[CORNER_CLASS_ID],
            "source_object": obj.name,
            "source_piece_type": "corner",
            "relative_point": [1, 1, 1],
            "point": point,
            "normals": normals,
        })
    return specs


def corner_edge_center_relative_keypoint_specs() -> list[dict[str, object]]:
    """Return the four face-corner points attached to each center cubie."""
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
        outer_axis = best_axis_index
        outer_sign = best_sign
        free_axes = [axis_index for axis_index in range(3) if axis_index != outer_axis]
        local_normal = axis_vector(outer_axis, outer_sign)
        for first_sign, second_sign in product((-1.0, 1.0), repeat=2):
            local_offset = axis_vector(outer_axis, outer_sign * 0.5)
            local_offset[free_axes[0]] = first_sign * 0.5
            local_offset[free_axes[1]] = second_sign * 0.5
            point = obj.matrix_world @ local_offset
            normal = (obj.matrix_world.to_3x3() @ local_normal).normalized()
            specs.append({
                "name": (
                    f"{obj.name}:{INDEPENDENT_CLASS_NAMES[CORNER_EDGE_CENTER_CLASS_ID]}:"
                    f"{free_axes[0]}{int(first_sign)}{free_axes[1]}{int(second_sign)}"
                ),
                "class_id": CORNER_EDGE_CENTER_CLASS_ID,
                "class": INDEPENDENT_CLASS_NAMES[CORNER_EDGE_CENTER_CLASS_ID],
                "source_object": obj.name,
                "source_piece_type": "center",
                "relative_point": [
                    int(axis_index == outer_axis) for axis_index in range(3)
                ],
                "point": point,
                "normals": [normal],
            })
    return specs


def edge_relative_keypoint_specs() -> list[dict[str, object]]:
    """Return corner-edge points attached to the physical edge cubies.

    Each edge cubie owns the two points at the ends of its exposed edge.  The
    previous independent-keypoint generator created these same geometric
    points from the adjacent corner cubies, which made them remain stationary
    when an edge cubie turned with a layer.  Their image locations should
    instead follow the edge cubie's accumulated transform.
    """
    specs = []
    for obj in edge_cubie_objects():
        solved_location = Vector(obj.get(SOLVED_LOCATION_PROP, obj.location))
        outer_axes = [
            axis_index
            for axis_index in range(3)
            if abs(round(solved_location[axis_index])) == 1
        ]
        free_axes = [axis_index for axis_index in range(3) if axis_index not in outer_axes]
        if len(outer_axes) != 2 or len(free_axes) != 1:
            raise ValueError(f"Expected an edge cubie with two outer axes: {obj.name}")
        solved_rotation = Quaternion(
            obj.get(SOLVED_ROTATION_PROP, (1.0, 0.0, 0.0, 0.0))
        )
        solved_rotation_inverse = solved_rotation.to_matrix().transposed()
        local_outer_faces = []
        for axis_index in outer_axes:
            world_sign = 1.0 if solved_location[axis_index] >= 0.0 else -1.0
            local_normal = solved_rotation_inverse @ axis_vector(axis_index, world_sign)
            local_axis = max(range(3), key=lambda candidate: abs(local_normal[candidate]))
            local_sign = 1.0 if local_normal[local_axis] >= 0.0 else -1.0
            if abs(local_normal[local_axis]) < 0.5:
                raise ValueError(f"Could not map edge face to a local axis: {obj.name}")
            local_outer_faces.append((local_axis, local_sign))
        local_outer_axes = {axis_index for axis_index, _sign in local_outer_faces}
        local_free_axis = next(axis_index for axis_index in range(3) if axis_index not in local_outer_axes)
        base_local_offset = Vector((0.0, 0.0, 0.0))
        for local_axis, local_sign in local_outer_faces:
            base_local_offset[local_axis] = local_sign * 0.5
        normals = [
            (obj.matrix_world.to_3x3() @ axis_vector(local_axis, local_sign)).normalized()
            for local_axis, local_sign in local_outer_faces
        ]
        for free_sign in (-1.0, 1.0):
            local_offset = base_local_offset.copy()
            local_offset[local_free_axis] = free_sign * 0.5
            point = obj.matrix_world @ local_offset
            specs.append({
                "name": f"{obj.name}:{INDEPENDENT_CLASS_NAMES[CORNER_EDGE_CLASS_ID]}:{local_free_axis}{int(free_sign)}",
                "class_id": CORNER_EDGE_CLASS_ID,
                "class": INDEPENDENT_CLASS_NAMES[CORNER_EDGE_CLASS_ID],
                "source_object": obj.name,
                "source_piece_type": "edge",
                "relative_point": [
                    1 if axis_index == local_free_axis else 0
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
            "name": f"{obj.name}:center",
            "class_id": CENTER_CLASS_ID,
            "class": "center",
            "source_object": obj.name,
            "source_piece_type": "center",
            "relative_point": "center",
            "point": point,
            "normals": [normal],
        })
    return specs


def cube_keypoint_specs_from_corners_and_centers(
    layer_state: dict[str, object] | None = None,
    cube_pose: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Return the independent intersection-point candidates.

    There are 8 corner points, 24 corner-edge points owned by edge cubies,
    and 24 corner-edge-center points owned by center cubies.  Face-center
    points are intentionally excluded from this three-class detector.
    """
    specs = [
        *corner_relative_keypoint_specs(),
        *edge_relative_keypoint_specs(),
        *corner_edge_center_relative_keypoint_specs(),
    ]
    expected_count = 8 + 24 + 24
    if len(specs) != expected_count:
        raise ValueError(f"Expected {expected_count} independent keypoints, got {len(specs)}")
    return add_partial_turn_intersection_specs(specs, layer_state, cube_pose)


def _partial_turn_layers(
    layer_state: dict[str, object] | None,
) -> list[tuple[str, float, float, set[str]]]:
    """Return the partial-turn operations recorded in a layer state.

    The object lists are captured after each turn is applied. Keeping the
    operations in application order lets the intersection-label helper undo
    them in reverse order, including for corner cuts and simultaneous turns.
    """
    if not isinstance(layer_state, dict):
        return []
    if layer_state.get("mode") not in {
        "mid_turn",
        "corner_cut",
        "slice_turn",
        "simultaneous_turn",
    }:
        return []

    operations: list[tuple[str, float, float, set[str]]] = []
    raw_operations = layer_state.get("operations")
    if isinstance(raw_operations, list):
        for raw_operation in raw_operations:
            if not isinstance(raw_operation, dict):
                continue
            move = raw_operation.get("move", raw_operation.get("face"))
            angle = raw_operation.get("angle_degrees")
            direction = raw_operation.get("direction", 1.0)
            object_names = raw_operation.get("objects")
            if not isinstance(move, str) or move not in (*face_specs(), *SLICE_MOVES):
                continue
            if not isinstance(angle, (int, float)):
                continue
            if not isinstance(direction, (int, float)):
                direction = 1.0
            if not isinstance(object_names, list):
                continue
            operations.append(
                (
                    move,
                    float(angle),
                    float(direction),
                    {name for name in object_names if isinstance(name, str)},
                )
            )
        if operations:
            return operations

    # Read the legacy fields as a fallback for hand-authored hard cases or
    # metadata produced before the generalized operation list was introduced.
    raw_operations = [("face", "angle_degrees", "direction", "objects")]
    if layer_state.get("mode") == "corner_cut":
        raw_operations.append(
            ("adjacent_face", "adjacent_angle_degrees", "adjacent_direction", "adjacent_objects")
        )
    for move_key, angle_key, direction_key, objects_key in raw_operations:
        move = layer_state.get(move_key)
        angle = layer_state.get(angle_key)
        direction = layer_state.get(direction_key, 1.0)
        object_names = layer_state.get(objects_key)
        if not isinstance(move, str) or move not in (*face_specs(), *SLICE_MOVES):
            continue
        if not isinstance(angle, (int, float)):
            continue
        if not isinstance(direction, (int, float)):
            direction = 1.0
        if not isinstance(object_names, list):
            continue
        operations.append(
            (
                move,
                float(angle),
                float(direction),
                {name for name in object_names if isinstance(name, str)},
            )
        )
    return operations


def _rotate_partial_turn_geometry(
    point: Vector,
    normals: list[Vector],
    face: str,
    angle: float,
    direction: float,
    cube_to_world: Matrix,
    inverse: bool = False,
) -> tuple[Vector, list[Vector]]:
    """Rotate a point and its face normals around the cube's turn axis."""
    pivot = cube_to_world.translation.copy()
    axis = (cube_to_world.to_3x3() @ partial_turn_axis(face)).normalized()
    signed_angle = -direction * angle if inverse else direction * angle
    rotation = Quaternion(axis, math.radians(signed_angle))
    return (
        pivot + rotation @ (point - pivot),
        [rotation @ normal for normal in normals],
    )


def _point_on_partial_turn_seam(
    point: Vector,
    move: str,
    cube_to_local: Matrix,
    tolerance: float = 1e-4,
) -> bool:
    """Return whether a point lies on an inner boundary of a turned layer."""
    local_point = cube_to_local @ point
    turn_axis = partial_turn_axis(move)
    expected_coordinates = (0.5,) if move not in SLICE_MOVES else (-0.5, 0.5)
    signed_coordinate = local_point.dot(turn_axis)
    return any(
        abs(signed_coordinate - expected_coordinate) <= tolerance
        for expected_coordinate in expected_coordinates
    )


def add_partial_turn_intersection_specs(
    specs: list[dict[str, object]],
    layer_state: dict[str, object] | None,
    cube_pose: dict[str, object] | None,
) -> list[dict[str, object]]:
    """Add only the edge/center intersections on a partial-turn seam.

    The seam is the inner boundary plane of the turning layer: coordinate
    ``+0.5`` in the selected outward-face direction.  A seam crossing on an
    adjacent outer face has two center-sticker points and two edge-sticker
    points.  Only those points receive a turned-side counterpart; all other
    non-corner points remain singletons.  The ordinary aligned path retains
    its original 56 candidates.
    """
    if not PARTIAL_TURN_SEAM_KEYPOINTS_ENABLED:
        return specs
    operations = _partial_turn_layers(layer_state)
    if not operations or not isinstance(cube_pose, dict):
        return specs
    transform_rows = cube_pose.get("transform")
    if not isinstance(transform_rows, list):
        return specs

    cube_to_world = matrix_from_rows(transform_rows)
    cube_to_local = cube_to_world.inverted()

    augmented = list(specs)
    for spec in specs:
        if spec.get("source_piece_type") not in {"edge", "center"}:
            continue
        source_object = spec.get("source_object")
        point = spec.get("point")
        normals = spec.get("normals")
        if not isinstance(source_object, str) or not isinstance(point, Vector):
            continue
        if not isinstance(normals, list) or not all(isinstance(normal, Vector) for normal in normals):
            continue

        # The recorded point is in the final scene state.  For each seam, undo
        # only later operations that affect this source cubie, test the point
        # in that operation's pre-turn state, then reapply the later motions
        # to the virtual turned-side counterpart.
        for operation_index, (face, angle, direction, object_names) in enumerate(operations):
            if (
                source_object in object_names
                or abs(angle) <= PARTIAL_TURN_INTERSECTION_MIN_DEGREES
            ):
                continue

            seam_point = point.copy()
            seam_normals = [normal.copy() for normal in normals]
            later_operations = operations[operation_index + 1:]
            for later_face, later_angle, later_direction, later_objects in reversed(later_operations):
                if source_object not in later_objects:
                    continue
                seam_point, seam_normals = _rotate_partial_turn_geometry(
                    seam_point,
                    seam_normals,
                    later_face,
                    later_angle,
                    later_direction,
                    cube_to_world,
                    inverse=True,
                )

            if not _point_on_partial_turn_seam(seam_point, face, cube_to_local):
                continue

            intersection_point, intersection_normals = _rotate_partial_turn_geometry(
                seam_point,
                seam_normals,
                face,
                angle,
                direction,
                cube_to_world,
            )
            for later_face, later_angle, later_direction, later_objects in later_operations:
                if source_object not in later_objects:
                    continue
                intersection_point, intersection_normals = _rotate_partial_turn_geometry(
                    intersection_point,
                    intersection_normals,
                    later_face,
                    later_angle,
                    later_direction,
                    cube_to_world,
                )

            augmented.append({
                **spec,
                "name": f"{spec['name']}:partial_turn_seam:{face}",
                "point": intersection_point,
                "normals": intersection_normals,
                "intersection_source": "partial_turn_seam",
                "intersection_face": face,
            })
    return augmented


def ordered_cube_keypoint_specs() -> list[dict[str, object]]:
    """Return the fixed 26-point structural cube keypoint layout.

    The points are defined in the cube's canonical local frame rather than by
    whichever cubie happens to be visible.  This keeps the label order stable:
    eight corners, twelve outer-edge midpoints, and six face centers.
    """
    axes = face_specs()
    surface = ORDERED_KEYPOINT_SURFACE_COORDINATE

    def make_spec(name: str, faces: tuple[str, ...], index: int) -> dict[str, object]:
        point = Vector((0.0, 0.0, 0.0))
        normals = []
        for face in faces:
            axis = axes[face]
            point += axis * surface
            normals.append(axis.copy())
        return {
            "index": index,
            "name": name,
            "point": point,
            "normals": normals,
            "kind": "corner" if index < 8 else "edge" if index < 20 else "center",
        }

    specs: list[dict[str, object]] = []
    for index, name in enumerate(ORDERED_KEYPOINT_NAMES):
        if index < 8:
            faces = tuple(name)
        elif index < 20:
            faces = (name[0], name[1])
        else:
            faces = (name,)
        specs.append(make_spec(name, faces, index))
    return specs


def ordered_keypoint_source_cubie_center(name: str) -> Vector:
    """Return the solved cubie center that owns an ordered surface point."""
    center = Vector((0.0, 0.0, 0.0))
    for letter in name:
        axis_index, value, _normal = cube_keypoint_coordinate(letter)
        if abs(value) > 1.0:
            center[axis_index] = 1.0 if value > 0.0 else -1.0
    return center


def ordered_keypoint_local_geometry(
    spec: dict[str, object],
    layer_state: dict[str, object] | None,
    objects: list[bpy.types.Object] | None = None,
    cube_to_local: Matrix | None = None,
) -> tuple[Vector, list[Vector]]:
    """Return one ordered point and its normals after scramble/layer transforms.

    Ordered point names identify the surface point on a solved cubie. During a
    scramble or partial turn, the owning cubie can move independently from the
    rest of the cube. When the live Blender objects are available, use the
    actual object's accumulated transform so the target stays attached to the
    physical cubie that the rendered stickers belong to. The dimension-only
    path remains as a fallback for geometry helpers that do not have scene
    objects available.
    """
    local_point = spec["point"]
    local_normals = spec["normals"]
    if not isinstance(local_point, Vector) or not isinstance(local_normals, list):
        raise ValueError(f"Ordered keypoint {spec.get('name')} has invalid geometry.")

    source_center = ordered_keypoint_source_cubie_center(str(spec["name"]))

    # Face centers belong to the fixed cube core for the face-turn data.  They
    # do not follow an outer-layer cubie, so their canonical local geometry is
    # still correct.
    if source_center.length <= 1e-6:
        return local_point.copy(), [normal.copy() for normal in local_normals]

    if objects is not None and cube_to_local is not None:
        matches = []
        for obj in objects:
            solved_location = Vector(obj.get(SOLVED_LOCATION_PROP, obj.location))
            if (solved_location - source_center).length < 1e-6:
                matches.append(obj)
        if len(matches) != 1:
            raise ValueError(
                f"Could not identify the physical cubie for ordered keypoint "
                f"{spec['name']}: expected one match, got {len(matches)}"
            )

        obj = matches[0]
        solved_rotation = Quaternion(
            obj.get(SOLVED_ROTATION_PROP, (1.0, 0.0, 0.0, 0.0))
        )
        local_object_matrix = cube_to_local @ obj.matrix_world
        current_center = local_object_matrix.translation.copy()
        current_rotation = local_object_matrix.to_3x3()
        relative_rotation = current_rotation @ solved_rotation.to_matrix().transposed()
        transformed_point = current_center + relative_rotation @ (local_point - source_center)
        transformed_normals = [relative_rotation @ normal for normal in local_normals]
        return transformed_point, transformed_normals

    if layer_state is None or layer_state.get("mode") == "none":
        return local_point.copy(), [normal.copy() for normal in local_normals]

    matching_transform = None
    for initial_center, center, rotation in dimension_cubie_transforms(layer_state):
        if (initial_center - source_center).length < 1e-6:
            matching_transform = center, rotation
            break
    if matching_transform is None:
        raise ValueError(
            f"Could not find source cubie {source_center} for ordered keypoint {spec['name']}"
        )

    center, rotation = matching_transform
    transformed_point = center + rotation @ (local_point - source_center)
    transformed_normals = [rotation @ normal for normal in local_normals]
    return transformed_point, transformed_normals


def ordered_keypoint_symmetry_matrices() -> tuple[Matrix, ...]:
    """Return the 24 proper rotations of the canonical cube frame."""
    matrices: list[Matrix] = []
    for permutation in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            rows = [[0.0, 0.0, 0.0] for _ in range(3)]
            for row, column in enumerate(permutation):
                rows[row][column] = signs[row]
            matrix = Matrix(rows)
            if matrix.determinant() > 0.5:
                matrices.append(matrix)
    return tuple(matrices)


ORDERED_KEYPOINT_SYMMETRY_MATRICES = ordered_keypoint_symmetry_matrices()


def ordered_keypoint_symmetry_permutations() -> tuple[tuple[int, ...], ...]:
    """Return output-to-source permutations for all 24 cube symmetries.

    A permutation is indexed by the output slot: if ``permutation[j] == i``,
    rotating source point ``i`` by the associated symmetry lands at output
    point ``j``.  This convention lets a symmetry-aware loss gather target
    keypoints directly into the model's canonical output slots.
    """
    specs = ordered_cube_keypoint_specs()
    points = [spec["point"] for spec in specs]
    permutations_by_symmetry: list[tuple[int, ...]] = []
    for symmetry in ORDERED_KEYPOINT_SYMMETRY_MATRICES:
        output_to_source: list[int] = []
        for output_point in points:
            matches = [
                source_index
                for source_index, source_point in enumerate(points)
                if (symmetry @ source_point - output_point).length < 1e-6
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Cube symmetry did not map the ordered keypoint layout "
                    f"bijectively; output point {output_point} has {len(matches)} matches."
                )
            output_to_source.append(matches[0])
        if len(set(output_to_source)) != ORDERED_KEYPOINT_COUNT:
            raise ValueError("Cube symmetry permutation is not bijective.")
        permutations_by_symmetry.append(tuple(output_to_source))

    if len(permutations_by_symmetry) != 24:
        raise ValueError(
            "Expected 24 proper cube symmetries, "
            f"got {len(permutations_by_symmetry)}."
        )
    if len(set(permutations_by_symmetry)) != 24:
        raise ValueError("Cube symmetry permutations are not unique.")
    return tuple(permutations_by_symmetry)


ORDERED_KEYPOINT_SYMMETRY_PERMUTATIONS = ordered_keypoint_symmetry_permutations()


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


def point_faces_camera(
    point: Vector,
    normals: list[Vector],
    camera_location: Vector,
    min_plane_angle_degrees: float = FACE_VISIBILITY_MIN_PLANE_ANGLE_DEGREES,
) -> bool:
    toward_camera = camera_location - point
    if toward_camera.length <= 1e-6:
        return False
    min_dot = math.sin(math.radians(min_plane_angle_degrees))
    view_direction = toward_camera.normalized()
    return any(normal.normalized().dot(view_direction) > min_dot for normal in normals)


def ray_to_point(camera: bpy.types.Object, point: Vector) -> tuple[Vector, Vector, float]:
    origin = camera.matrix_world.translation
    offset = point - origin
    distance = offset.length
    return origin, offset.normalized(), distance


def object_occluding_point(
    point: Vector,
    depsgraph: bpy.types.Depsgraph,
) -> str | None:
    """Return the first render-visible object hit strictly before ``point``.

    Every cube piece participates in the same test, including at a partial-turn
    seam. A hit at the mathematical keypoint itself is not an occluder; only a
    hit strictly before that distance is.
    """
    scene = bpy.context.scene
    camera = get_camera()
    cube_mesh_names = cube_render_mesh_names([
        *cube_objects(),
        *core_fill_objects(),
    ])
    origin, direction, point_distance = ray_to_point(camera, point)
    traveled = 0.0
    epsilon = 0.001
    while traveled < point_distance - KEYPOINT_DISTANCE_TOLERANCE:
        hit, location, _normal, _polygon_index, hit_object, _matrix = scene.ray_cast(
            depsgraph,
            origin,
            direction,
            distance=(point_distance - traveled) + KEYPOINT_DISTANCE_TOLERANCE,
        )
        if not hit or hit_object is None:
            return None

        hit_distance = (location - origin).length
        traveled += hit_distance
        if (
            not getattr(hit_object, "visible_camera", True)
            or getattr(hit_object, "hide_render", False)
            or hit_object.name.startswith(SYNTHETIC_WINDOW_BLOCKER_PREFIX)
        ):
            origin = location + (direction * epsilon)
            traveled += epsilon
            continue
        if traveled >= point_distance:
            return None
        if (
            traveled >= point_distance - KEYPOINT_DISTANCE_TOLERANCE
            and hit_object.name in cube_mesh_names
        ):
            # The mathematical point can lie just beyond a bevel or shared
            # cubie boundary. Keep the small tolerance for cube support
            # geometry, but never use it to discard an external occluder such
            # as a hand that is genuinely in front of the point.
            return None
        return str(hit_object.name)

    return None


def dimension_cube_box_specs(
    layer_state: dict[str, object] | None = None,
) -> list[tuple[Vector, Matrix, Vector]]:
    """Return oriented boxes for the idealized cube or a partial turn."""
    if layer_state is None or layer_state.get("mode") == "none":
        return [
            (Vector((0.0, 0.0, 0.0)), Matrix.Identity(3), Vector((1.5, 1.5, 1.5)))
        ]

    operations = _partial_turn_layers(layer_state)
    if layer_state.get("mode") != "mid_turn" or len(operations) != 1:
        return [
            (center, rotation, Vector((0.5, 0.5, 0.5)))
            for _initial_center, center, rotation in dimension_cubie_transforms(layer_state)
        ]

    move, angle_degrees, direction, _object_names = operations[0]
    if move in SLICE_MOVES:
        return [
            (center, rotation, Vector((0.5, 0.5, 0.5)))
            for _initial_center, center, rotation in dimension_cubie_transforms(layer_state)
        ]

    face_axis = partial_turn_axis(move)
    axis_index = max(range(3), key=lambda index: abs(face_axis[index]))
    face_axis_sign = int(round(face_axis[axis_index]))
    turn = Quaternion(
        face_axis,
        math.radians(float(direction) * float(angle_degrees)),
    ).to_matrix()
    boxes = []
    half_extents = Vector((1.5, 1.5, 1.5))
    half_extents[axis_index] = 0.5
    for layer_coordinate in (-1.0, 0.0, 1.0):
        center = Vector((0.0, 0.0, 0.0))
        center[axis_index] = layer_coordinate
        rotation = turn if layer_coordinate == float(face_axis_sign) else Matrix.Identity(3)
        if layer_coordinate == float(face_axis_sign):
            center = face_axis.copy()
        boxes.append((center, rotation, half_extents.copy()))
    return boxes


def dimension_cubie_transforms(
    layer_state: dict[str, object],
) -> list[tuple[Vector, Vector, Matrix]]:
    """Transform dimension-defined cubies through all recorded layer turns."""
    operations = _partial_turn_layers(layer_state)
    if not operations:
        raise ValueError("A layer turn needs at least one valid operation.")

    transforms = []
    for x in (-1.0, 0.0, 1.0):
        for y in (-1.0, 0.0, 1.0):
            for z in (-1.0, 0.0, 1.0):
                initial_center = Vector((x, y, z))
                center = initial_center.copy()
                rotation = Matrix.Identity(3)
                for move, angle, direction, _object_names in operations:
                    axis = partial_turn_axis(move)
                    if round(center.dot(axis)) != partial_turn_layer_coordinate(move):
                        continue
                    turn = Quaternion(
                        axis,
                        math.radians(float(direction) * float(angle)),
                    ).to_matrix()
                    center = turn @ center
                    rotation = turn @ rotation
                transforms.append((initial_center, center, rotation))
    return transforms


def ray_box_intersection_distance(
    origin: Vector,
    direction: Vector,
    center: Vector,
    rotation: Matrix,
    half_extents: Vector,
) -> float | None:
    """Return the nearest non-negative ray distance for an oriented box."""
    inverse_rotation = rotation.transposed()
    local_origin = inverse_rotation @ (origin - center)
    local_direction = inverse_rotation @ direction
    lower = -float("inf")
    upper = float("inf")
    for axis in range(3):
        if abs(local_direction[axis]) <= 1e-8:
            if abs(local_origin[axis]) > half_extents[axis]:
                return None
            continue
        first = (-half_extents[axis] - local_origin[axis]) / local_direction[axis]
        second = (half_extents[axis] - local_origin[axis]) / local_direction[axis]
        lower = max(lower, min(first, second))
        upper = min(upper, max(first, second))
        if lower > upper:
            return None
    if upper < 0.0:
        return None
    return max(0.0, lower)


def wireframe_ray_hit_is_visible(
    point: Vector,
    face_normal: Vector,
    support_direction: Vector,
    main_cube_names: set[str],
    depsgraph: bpy.types.Depsgraph,
    allow_gap_if_unblocked: bool = False,
    candidate_object_names: set[str] | None = None,
    cube_transform: Matrix | None = None,
    cube_boxes: list[tuple[Vector, Matrix, Vector]] | None = None,
) -> bool:
    """Return whether a dimension-defined face point is frontmost and clear."""
    camera = get_camera()
    toward_camera = camera.matrix_world.translation - point
    if toward_camera.length <= 1e-6 or face_normal.dot(toward_camera.normalized()) <= 1e-4:
        return False

    scene = bpy.context.scene
    target_distance_tolerance = 0.035
    support_direction = support_direction.normalized()
    # The exact seam can be a small gap between beveled cubies. Probe both
    # adjacent surface strips as well as the seam itself. A hand or prop that
    # truly occludes the line also covers these immediate support probes.
    probe_points = [
        # Move well inside the adjacent flat sticker/surface.  The idealized
        # seam is at the edge of a beveled cubie, so probes immediately beside
        # it can still land in the bevel or the physical gap.
        point + support_direction * 0.12,
        point - support_direction * 0.12,
        point,
    ]
    for probe_point in probe_points:
        origin = camera.matrix_world.translation
        direction = (probe_point - origin).normalized()
        point_distance = (probe_point - origin).length
        if cube_transform is not None and cube_boxes is not None:
            inverse_cube_transform = cube_transform.inverted()
            local_origin = inverse_cube_transform @ origin
            local_direction = inverse_cube_transform.to_3x3() @ direction
            nearest_cube_distance = min(
                (
                    distance
                    for center, rotation, half_extents in cube_boxes
                    if (distance := ray_box_intersection_distance(
                        local_origin,
                        local_direction,
                        center,
                        rotation,
                        half_extents,
                    )) is not None
                ),
                default=None,
            )
            if (
                nearest_cube_distance is None
                or abs(nearest_cube_distance - point_distance) > target_distance_tolerance
            ):
                continue

        hit, location, _normal, _polygon_index, hit_object, _matrix = scene.ray_cast(
            depsgraph,
            origin,
            direction,
            distance=point_distance + target_distance_tolerance,
        )
        if hit and hit_object is not None:
            hit_distance = (location - origin).length
            if (
                getattr(hit_object, "visible_camera", True)
                and not getattr(hit_object, "hide_render", False)
                and not hit_object.name.startswith(SYNTHETIC_WINDOW_BLOCKER_PREFIX)
                and hit_object.name not in main_cube_names
                and hit_distance <= point_distance + target_distance_tolerance
            ):
                continue
        if allow_gap_if_unblocked or cube_transform is not None:
            return True
    return False


def interpolate_wireframe_boundary(
    start_t: float,
    end_t: float,
    start_visible: bool,
    visibility_at,
) -> float:
    """Refine a sampled visible/occluded transition along one line."""
    low = start_t
    high = end_t
    for _ in range(9):
        middle = (low + high) * 0.5
        if visibility_at(middle) == start_visible:
            low = middle
        else:
            high = middle
    return (low + high) * 0.5


def clip_wireframe_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    width: int,
    height: int,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip a projected segment to the image rectangle using Liang-Barsky."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    lower = 0.0
    upper = 1.0
    for coefficient, numerator in (
        (-dx, start[0]),
        (dx, (width - 1e-3) - start[0]),
        (-dy, start[1]),
        (dy, (height - 1e-3) - start[1]),
    ):
        if abs(coefficient) <= 1e-12:
            if numerator > 0.0:
                return None
            continue
        value = numerator / coefficient
        if coefficient < 0.0:
            lower = max(lower, value)
        else:
            upper = min(upper, value)
        if lower > upper:
            return None
    return (
        (start[0] + lower * dx, start[1] + lower * dy),
        (start[0] + upper * dx, start[1] + upper * dy),
    )


def merge_wireframe_segments(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Join adjacent collinear visible pieces into layer-length lines."""
    merged = list(segments)
    changed = True
    while changed:
        changed = False
        for first_index in range(len(merged)):
            first_start, first_end = merged[first_index]
            first_vector = (first_end[0] - first_start[0], first_end[1] - first_start[1])
            first_length = math.hypot(*first_vector)
            if first_length <= 1e-6:
                continue
            for second_index in range(first_index + 1, len(merged)):
                second_start, second_end = merged[second_index]
                second_vector = (second_end[0] - second_start[0], second_end[1] - second_start[1])
                second_length = math.hypot(*second_vector)
                if second_length <= 1e-6:
                    continue
                cross = abs(first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0])
                if cross > 0.025 * first_length * second_length:
                    continue
                if min(
                    math.dist(first_start, second_start),
                    math.dist(first_start, second_end),
                    math.dist(first_end, second_start),
                    math.dist(first_end, second_end),
                ) > 2.0:
                    continue
                direction = (first_vector[0] / first_length, first_vector[1] / first_length)
                points = [first_start, first_end, second_start, second_end]
                projection = [point[0] * direction[0] + point[1] * direction[1] for point in points]
                first = points[projection.index(min(projection))]
                last = points[projection.index(max(projection))]
                merged[first_index] = (first, last)
                del merged[second_index]
                changed = True
                break
            if changed:
                break
    return merged


def partial_turn_wireframe_line_specs(
    layer_state: dict[str, object],
) -> list[dict[str, object]]:
    """Build dimension-defined surface grids for a partially turned layer."""
    if layer_state.get("mode") in {"corner_cut", "slice_turn", "simultaneous_turn"}:
        # The cubie-based path handles both slice seams and two opposite
        # layers, while the slab path below is retained for the common single
        # outer-face turn.
        return corner_cut_wireframe_line_specs(layer_state)

    face = layer_state.get("face")
    angle_degrees = layer_state.get("angle_degrees")
    direction = layer_state.get("direction")
    if not isinstance(face, str) or not isinstance(angle_degrees, (int, float)):
        raise ValueError("A partial-turn wireframe needs a face and angle.")
    if not isinstance(direction, (int, float)):
        raise ValueError("A partial-turn wireframe needs a direction.")

    face_axis = face_specs()[face]
    turn_axis_index = max(range(3), key=lambda index: abs(face_axis[index]))
    face_axis_sign = int(round(face_axis[turn_axis_index]))
    layer_center = face_axis.copy()
    turn = Quaternion(
        face_axis,
        math.radians(float(direction) * float(angle_degrees)),
    ).to_matrix().to_4x4()
    layer_transform = (
        Matrix.Translation(layer_center)
        @ turn
        @ Matrix.Translation(-layer_center)
    )
    identity = Matrix.Identity(4)
    specs = []

    for layer_coordinate in (-1.0, 0.0, 1.0):
        transform = (
            layer_transform
            if layer_coordinate == float(face_axis_sign)
            else identity
        )
        rotation = transform.to_3x3()
        for normal_axis in range(3):
            tangent_axes = [axis for axis in range(3) if axis != normal_axis]
            for normal_sign in (-1.0, 1.0):
                # Along the turn axis, each slab contributes only its outer
                # face. Along the other axes, all three slabs contribute one
                # surface strip.
                if normal_axis == turn_axis_index and layer_coordinate != normal_sign:
                    continue
                for varying_axis, fixed_axis in (
                    (tangent_axes[0], tangent_axes[1]),
                    (tangent_axes[1], tangent_axes[0]),
                ):
                    varying_values = (
                        (layer_coordinate - 0.5, layer_coordinate + 0.5)
                        if varying_axis == turn_axis_index
                        else (-1.5, 1.5)
                    )
                    fixed_values = (
                        (layer_coordinate - 0.5, layer_coordinate + 0.5)
                        if fixed_axis == turn_axis_index
                        else (-1.5, -0.5, 0.5, 1.5)
                    )
                    for fixed_value in fixed_values:
                        start = Vector((0.0, 0.0, 0.0))
                        end = Vector((0.0, 0.0, 0.0))
                        start[normal_axis] = normal_sign * 1.5
                        end[normal_axis] = normal_sign * 1.5
                        start[fixed_axis] = fixed_value
                        end[fixed_axis] = fixed_value
                        start[varying_axis] = varying_values[0]
                        end[varying_axis] = varying_values[1]
                        specs.append({
                            "start": transform @ start,
                            "end": transform @ end,
                            "normal": (rotation @ axis_vector(normal_axis, normal_sign)).normalized(),
                            "support_direction": (rotation @ axis_vector(fixed_axis)).normalized(),
                        })
    return specs


def corner_cut_wireframe_line_specs(
    layer_state: dict[str, object],
) -> list[dict[str, object]]:
    """Build exposed surface-cell edges after sequential dimension turns."""
    specs = []
    cubie_transforms = dimension_cubie_transforms(layer_state)

    def inside_box(point: Vector, center: Vector, rotation: Matrix) -> bool:
        local_point = rotation.transposed() @ (point - center)
        return all(abs(local_point[axis]) < 0.5 + 1e-4 for axis in range(3))

    for cubie_index, (_initial_center, center, rotation) in enumerate(cubie_transforms):
        for normal_axis in range(3):
            for normal_sign in (-1.0, 1.0):
                local_normal = axis_vector(normal_axis, normal_sign)
                normal = (rotation @ local_normal).normalized()
                face_center = center + normal * 0.5
                if any(
                    other_index != cubie_index
                    and inside_box(
                        face_center + normal * 0.04,
                        other_center,
                        other_rotation,
                    )
                    for other_index, (_other_initial, other_center, other_rotation)
                    in enumerate(cubie_transforms)
                ):
                    continue
                tangent_axes = [axis for axis in range(3) if axis != normal_axis]
                for varying_axis, fixed_axis in (
                    (tangent_axes[0], tangent_axes[1]),
                    (tangent_axes[1], tangent_axes[0]),
                ):
                    for fixed_offset in (-0.5, 0.5):
                        local_start = local_normal * 0.5
                        local_end = local_normal * 0.5
                        local_start[fixed_axis] = fixed_offset
                        local_end[fixed_axis] = fixed_offset
                        local_start[varying_axis] = -0.5
                        local_end[varying_axis] = 0.5
                        specs.append({
                            "start": center + rotation @ local_start,
                            "end": center + rotation @ local_end,
                            "normal": normal,
                            "support_direction": (
                                rotation @ axis_vector(fixed_axis)
                            ).normalized(),
                        })
    return specs


def visible_wireframe_line_segments(
    line_specs: list[dict[str, object]],
    cube_transform: Matrix,
    cube_boxes: list[tuple[Vector, Matrix, Vector]],
    main_cube_roots: list[bpy.types.Object],
    width: int,
    height: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Project dimension-defined line specs, keeping only visible portions."""
    camera = get_camera()
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    main_cube_names = cube_render_mesh_names(main_cube_roots)
    segments = []

    for spec in line_specs:
        local_start = spec["start"]
        local_end = spec["end"]
        local_normal = spec["normal"]
        local_support_direction = spec["support_direction"]
        if not all(
            isinstance(value, Vector)
            for value in (local_start, local_end, local_normal, local_support_direction)
        ):
            continue
        start = cube_transform @ local_start
        end = cube_transform @ local_end
        normal = (cube_transform.to_3x3() @ local_normal).normalized()
        support_direction = (
            cube_transform.to_3x3() @ local_support_direction
        ).normalized()
        projected_start = project_world_point_unclipped(start, camera, scene)
        projected_end = project_world_point_unclipped(end, camera, scene)
        if projected_start is None or projected_end is None:
            continue

        pixel_length = math.dist(projected_start, projected_end)
        sample_count = max(32, min(192, int(math.ceil(pixel_length / 2.0)) + 1))
        visibility_cache: dict[float, bool] = {}

        def visibility_at(t: float) -> bool:
            cached = visibility_cache.get(t)
            if cached is not None:
                return cached
            visible = wireframe_ray_hit_is_visible(
                start.lerp(end, t),
                normal,
                support_direction,
                main_cube_names,
                depsgraph,
                cube_transform=cube_transform,
                cube_boxes=cube_boxes,
            )
            visibility_cache[t] = visible
            return visible

        samples = [index / (sample_count - 1) for index in range(sample_count)]
        visible = [visibility_at(t) for t in samples]
        index = 0
        while index < sample_count:
            if not visible[index]:
                index += 1
                continue
            run_start = index
            while index + 1 < sample_count and visible[index + 1]:
                index += 1
            run_end = index
            start_t = (
                0.0
                if run_start == 0
                else interpolate_wireframe_boundary(
                    samples[run_start - 1], samples[run_start], False, visibility_at
                )
            )
            end_t = (
                1.0
                if run_end == sample_count - 1
                else interpolate_wireframe_boundary(
                    samples[run_end], samples[run_end + 1], True, visibility_at
                )
            )
            if end_t - start_t > 1e-4:
                image_start = project_world_point_unclipped(
                    start.lerp(end, start_t), camera, scene
                )
                image_end = project_world_point_unclipped(
                    start.lerp(end, end_t), camera, scene
                )
                if image_start is not None and image_end is not None:
                    clipped = clip_wireframe_segment(image_start, image_end, width, height)
                    if clipped is not None and math.dist(*clipped) >= 1.0:
                        segments.append(clipped)
            index += 1

    segments = merge_wireframe_segments(segments)
    unique_segments = []
    seen = set()
    for start, end in segments:
        first = (round(start[0], 3), round(start[1], 3))
        second = (round(end[0], 3), round(end[1], 3))
        key = tuple(sorted((first, second)))
        if key in seen:
            continue
        seen.add(key)
        unique_segments.append((start, end))
    return unique_segments


def canonical_wireframe_line_specs() -> list[dict[str, object]]:
    """Return the standard 4-by-4 grid on each local cube face."""
    specs = []
    for normal_axis in range(3):
        tangent_axes = [axis for axis in range(3) if axis != normal_axis]
        for normal_sign in (-1.0, 1.0):
            for varying_axis, fixed_axis in (
                (tangent_axes[0], tangent_axes[1]),
                (tangent_axes[1], tangent_axes[0]),
            ):
                for fixed_value in (-1.5, -0.5, 0.5, 1.5):
                    start = [0.0, 0.0, 0.0]
                    end = [0.0, 0.0, 0.0]
                    start[normal_axis] = normal_sign * 1.5
                    end[normal_axis] = normal_sign * 1.5
                    start[fixed_axis] = fixed_value
                    end[fixed_axis] = fixed_value
                    start[varying_axis] = -1.5
                    end[varying_axis] = 1.5
                    specs.append({
                        "start": Vector(start),
                        "end": Vector(end),
                        "normal": axis_vector(normal_axis, normal_sign),
                        "support_direction": axis_vector(fixed_axis),
                    })
    return specs


def visible_canonical_wireframe_segments(
    cube_transform: Matrix,
    main_cube_roots: list[bpy.types.Object],
    width: int,
    height: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Project the standard grid for frames without a partial layer turn."""
    return visible_wireframe_line_segments(
        canonical_wireframe_line_specs(),
        cube_transform,
        dimension_cube_box_specs(),
        main_cube_roots,
        width,
        height,
    )


def write_wireframe_label(
    path: Path,
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    width: int,
    height: int,
) -> int:
    """Write one class plus two normalized endpoints for every visible segment."""
    lines = []
    for start, end in segments:
        normalized_start = normalize_point(start[0], start[1], width, height)
        normalized_end = normalize_point(end[0], end[1], width, height)
        lines.append(
            f"0 {normalized_start[0]:.6f} {normalized_start[1]:.6f} "
            f"{normalized_end[0]:.6f} {normalized_end[1]:.6f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def write_wireframe_overlay(
    path: Path,
    image_path: Path,
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> None:
    """Draw the exported visible wireframe over the final rendered image."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError("OpenCV is required to write wireframe verification renders.") from None

    image = read_image_file(image_path, cv2, np, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read rendered image for wireframe overlay: {image_path}")
    output = image.copy()
    line_width = scaled_annotation_size(WIREFRAME_LINE_BASE_WIDTH, image.shape[1], image.shape[0])
    shadow_width = scaled_annotation_size(
        WIREFRAME_LINE_SHADOW_BASE_WIDTH,
        image.shape[1],
        image.shape[0],
    )
    for start, end in segments:
        start_point = tuple(round(value) for value in start)
        end_point = tuple(round(value) for value in end)
        cv2.line(output, start_point, end_point, WIREFRAME_LINE_SHADOW_COLOR, shadow_width, cv2.LINE_AA)
        cv2.line(output, start_point, end_point, WIREFRAME_LINE_COLOR, line_width, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not write_image_file(path, output, cv2):
        raise RuntimeError(f"Could not write wireframe verification render: {path}")


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


def cube_pixel_owns_point(
    visibility_class_map: object | None,
    projected: tuple[float, float],
    width: int,
    height: int,
) -> bool:
    """Return whether the rendered pixel contains target-cube geometry."""
    if visibility_class_map is None:
        return False
    shape = getattr(visibility_class_map, "shape", ())
    if len(shape) != 2 or shape[0] != height or shape[1] != width:
        return False
    x = max(0, min(width - 1, round(float(projected[0]))))
    y = max(0, min(height - 1, round(float(projected[1]))))
    try:
        return int(visibility_class_map[y, x]) == 1
    except (IndexError, TypeError, ValueError):
        return False


def partial_turn_interior_seam_point(
    spec: dict[str, object],
    point: Vector,
    layer_state: dict[str, object] | None,
    cube_pose: dict[str, object] | None,
) -> bool:
    """Return whether a corner-edge point lies on an exposed turn seam.

    Edge-on outer stickers remain subject to the normal view-angle rule.  The
    exception is limited to edge cubies on the fixed side of a partial-turn
    boundary, where a turn can expose the cube interior. Corner-edge-center
    points remain subject to the normal view-angle rule. The test mirrors the
    seam reconstruction used for partial-turn intersection labels, including
    undoing later operations before checking the seam plane.
    """
    if not PARTIAL_TURN_SEAM_KEYPOINTS_ENABLED:
        return False
    if spec.get("source_piece_type") != "edge":
        return False
    if not isinstance(layer_state, dict) or not isinstance(cube_pose, dict):
        return False
    operations = _partial_turn_layers(layer_state)
    transform_rows = cube_pose.get("transform")
    source_object = spec.get("source_object")
    if not operations or not isinstance(transform_rows, list) or not isinstance(source_object, str):
        return False

    cube_to_world = matrix_from_rows(transform_rows)
    cube_to_local = cube_to_world.inverted()
    for operation_index, (face, angle, direction, object_names) in enumerate(operations):
        if source_object in object_names or abs(angle) <= PARTIAL_TURN_INTERSECTION_MIN_DEGREES:
            continue
        seam_point = point.copy()
        later_operations = operations[operation_index + 1:]
        for later_face, later_angle, later_direction, later_objects in reversed(later_operations):
            if source_object not in later_objects:
                continue
            seam_point, _seam_normals = _rotate_partial_turn_geometry(
                seam_point,
                [],
                later_face,
                later_angle,
                later_direction,
                cube_to_world,
                inverse=True,
            )
        if _point_on_partial_turn_seam(seam_point, face, cube_to_local):
            return True
    return False


def render_ordered_keypoint_label(
    cube_pose: dict[str, object],
    width: int,
    height: int,
    layer_state: dict[str, object] | None = None,
    min_plane_angle_degrees: float = FACE_VISIBILITY_MIN_PLANE_ANGLE_DEGREES,
    visibility_class_map: object | None = None,
) -> tuple[str | None, dict[str, object]]:
    """Project the ordered 26-point cube layout into one YOLO pose row."""
    transform_rows = cube_pose.get("transform")
    if not isinstance(transform_rows, list):
        raise ValueError("Cube pose is missing its transform matrix.")

    camera = get_camera()
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cube_to_world = matrix_from_rows(transform_rows)
    cube_to_local = cube_to_world.inverted()
    cube_rotation = cube_to_world.to_3x3()
    hand_object_names = {
        obj.name
        for obj in hand_objects().values()
    }

    projected_keypoints: list[tuple[float, float, int]] = []
    metadata: list[dict[str, object]] = []
    counts = {"unlabeled": 0, "occluded": 0, "visible": 0}
    label_objects = cube_objects()
    for spec in ordered_cube_keypoint_specs():
        local_point, local_normals = ordered_keypoint_local_geometry(
            spec,
            layer_state,
            objects=label_objects,
            cube_to_local=cube_to_local,
        )
        source_center = ordered_keypoint_source_cubie_center(str(spec["name"]))
        source_cubie_name = None
        if source_center.length > 1e-6:
            source_cubie_name = next(
                (
                    obj.name
                    for obj in label_objects
                    if (
                        Vector(obj.get(SOLVED_LOCATION_PROP, obj.location)) - source_center
                    ).length
                    < 1e-6
                ),
                None,
            )
        world_point = cube_to_world @ local_point
        world_normals = [cube_rotation @ normal for normal in local_normals]
        projected = project_world_point(world_point, camera, scene)
        visibility = POINT_VISIBILITY_UNLABELED
        occluded_by = None
        hand_occluded = False
        if projected is not None and point_faces_camera(
            world_point,
            world_normals,
            camera.location,
            min_plane_angle_degrees=min_plane_angle_degrees,
        ):
            occluded_by = object_occluding_point(
                world_point,
                depsgraph,
            )
            hand_occluded = bool(occluded_by and occluded_by in hand_object_names)
            visibility = (
                POINT_VISIBILITY_LABELED_OCCLUDED
                if occluded_by is not None
                else POINT_VISIBILITY_VISIBLE
            )

        if visibility == POINT_VISIBILITY_UNLABELED or projected is None:
            point_label = unlabeled_pose_keypoint()
        else:
            point_label = (float(projected[0]), float(projected[1]), visibility)
        projected_keypoints.append(point_label)
        counts[
            "visible"
            if visibility == POINT_VISIBILITY_VISIBLE
            else "occluded"
            if visibility == POINT_VISIBILITY_LABELED_OCCLUDED
            else "unlabeled"
        ] += 1
        metadata.append({
            "index": int(spec["index"]),
            "name": str(spec["name"]),
            "kind": str(spec["kind"]),
            "source_cubie": source_cubie_name,
            "local_point": list(local_point),
            "layer_mode": layer_state.get("mode", "none") if layer_state else "none",
            "world_point": list(world_point),
            "pixel_point": [point_label[0], point_label[1]],
            "normalized_point": list(normalize_point(point_label[0], point_label[1], width, height)),
            "visibility": visibility,
            "hand_occluded": hand_occluded,
            "occluded_by": occluded_by,
        })

    bbox = keypoint_bbox(projected_keypoints, width, height)
    if bbox is None:
        return None, {
            "keypoint_shape": [ORDERED_KEYPOINT_COUNT, 3],
            "keypoint_names": list(ORDERED_KEYPOINT_NAMES),
            "layer_mode": layer_state.get("mode", "none") if layer_state else "none",
            "min_face_plane_angle_degrees": min_plane_angle_degrees,
            "keypoints": metadata,
            **counts,
        }

    x_center, y_center, bbox_width, bbox_height = bbox
    return yolo_pose_line(
        ORDERED_POSE_CLASS_ID,
        x_center,
        y_center,
        bbox_width,
        bbox_height,
        projected_keypoints,
        width,
        height,
    ), {
        "keypoint_shape": [ORDERED_KEYPOINT_COUNT, 3],
        "keypoint_names": list(ORDERED_KEYPOINT_NAMES),
        "layer_mode": layer_state.get("mode", "none") if layer_state else "none",
        "min_face_plane_angle_degrees": min_plane_angle_degrees,
        "keypoints": metadata,
        "bbox_xywh": [x_center, y_center, bbox_width, bbox_height],
        **counts,
    }


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
        "class": INDEPENDENT_CLASS_NAMES[class_id],
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


def render_pose_keypoints(
    path: Path | None,
    image_path: Path | None = None,
    min_plane_angle_degrees: float = INDEPENDENT_FACE_VISIBILITY_MIN_PLANE_ANGLE_DEGREES,
    include_occluded_keypoints: bool = INDEPENDENT_INCLUDE_OCCLUDED_KEYPOINTS,
    layer_state: dict[str, object] | None = None,
    cube_pose: dict[str, object] | None = None,
    visibility_class_map: object | None = None,
    include_partial_turn_intersections: bool = False,
) -> tuple[list[str], list[dict[str, object]], dict[str, object]]:
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
    # The mathematical keypoint is on the cube boundary. A hit at that exact
    # distance is supporting geometry; any cube hit strictly before it is a
    # real occluder, including during a partial layer turn.
    lines = []
    pose_objects = []

    bpy.context.view_layer.update()
    projected_by_name: dict[str, tuple[float, float, int]] = {}
    metadata_by_name: dict[str, dict[str, object]] = {}
    keypoint_specs = cube_keypoint_specs_from_corners_and_centers(
        layer_state=layer_state,
        cube_pose=cube_pose,
    )
    if not include_partial_turn_intersections:
        keypoint_specs = [
            spec
            for spec in keypoint_specs
            if spec.get("intersection_source") != "partial_turn_seam"
        ]
    candidate_counts = {
        "projected": 0,
        "out_of_frame": 0,
        "angle_filtered": 0,
        "interior_seam_visible": 0,
        "visible": 0,
        "occluded": 0,
        "occluded_labeled": 0,
    }
    for index, spec in enumerate(keypoint_specs):
        point = spec["point"]
        if not isinstance(point, Vector):
            continue
        name = str(spec["name"])
        point_class_id = int(spec["class_id"])
        point_min_plane_angle_degrees = (
            CORNER_FACE_VISIBILITY_MIN_PLANE_ANGLE_DEGREES
            if point_class_id == CORNER_CLASS_ID
            else min_plane_angle_degrees
        )
        normals = spec["normals"]
        projected = project_world_point(point, camera, scene)
        hand_occluded = False
        occluded_by = None
        interior_seam_visible = False
        if projected is None:
            x, y = 0.0, 0.0
            visibility = POINT_VISIBILITY_UNLABELED
            raw_visibility = POINT_VISIBILITY_UNLABELED
            candidate_counts["out_of_frame"] += 1
        else:
            angle_visible = isinstance(normals, list) and point_faces_camera(
                point,
                normals,
                camera.location,
                min_plane_angle_degrees=point_min_plane_angle_degrees,
            )
            interior_seam_visible = (
                not angle_visible
                and partial_turn_interior_seam_point(
                    spec,
                    point,
                    layer_state,
                    cube_pose,
                )
                and cube_pixel_owns_point(
                    visibility_class_map,
                    projected,
                    width,
                    height,
                )
            )
        if projected is not None and not angle_visible and not interior_seam_visible:
            x, y = projected
            visibility = POINT_VISIBILITY_UNLABELED
            raw_visibility = POINT_VISIBILITY_UNLABELED
            candidate_counts["angle_filtered"] += 1
        elif projected is not None:
            x, y = projected
            candidate_counts["projected"] += 1
            if interior_seam_visible:
                candidate_counts["interior_seam_visible"] += 1
            occluded_by = object_occluding_point(
                point,
                depsgraph,
            )
            hand_occluded = occluded_by in visible_hand_names
            if occluded_by is not None:
                candidate_counts["occluded"] += 1
                raw_visibility = POINT_VISIBILITY_LABELED_OCCLUDED
                visibility = (
                    raw_visibility
                    if include_occluded_keypoints
                    else POINT_VISIBILITY_UNLABELED
                )
                if include_occluded_keypoints:
                    candidate_counts["occluded_labeled"] += 1
            else:
                candidate_counts["visible"] += 1
                raw_visibility = POINT_VISIBILITY_VISIBLE
                visibility = raw_visibility

        projected_by_name[name] = (x, y, visibility)
        metadata_by_name[name] = {
            "index": index,
            "name": name,
            "class": spec.get("class"),
            "class_id": spec.get("class_id"),
            "min_face_plane_angle_degrees": point_min_plane_angle_degrees,
            "source_object": spec.get("source_object"),
            "source_piece_type": spec.get("source_piece_type"),
            "relative_point": spec.get("relative_point"),
            "intersection_source": spec.get("intersection_source"),
            "intersection_face": spec.get("intersection_face"),
            "world_point": [point.x, point.y, point.z],
            "pixel_point": [x, y],
            "normalized_point": list(normalize_point(x, y, width, height)),
            "visibility": visibility,
            "raw_visibility": raw_visibility,
            "hand_occluded": hand_occluded,
            "occluded_by": occluded_by,
            "interior_seam_visible": interior_seam_visible,
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
        write_pose_keypoint_mask(path, pose_objects, width, height, image_path=image_path)

    return lines, pose_objects, {
        "keypoint_shape": [POSE_KEYPOINT_COUNT, 3],
        "source_keypoints": len(keypoint_specs),
        "base_source_keypoints": 56,
        "partial_turn_intersection_keypoints": sum(
            1 for spec in keypoint_specs if spec.get("intersection_source") == "partial_turn_seam"
        ),
        "partial_turn_seam_keypoints_enabled": (
            PARTIAL_TURN_SEAM_KEYPOINTS_ENABLED
            and include_partial_turn_intersections
        ),
        "partial_turn_intersections_included": include_partial_turn_intersections,
        "visible_hands": sorted(visible_hand_names),
        "candidates": len(keypoint_specs),
        "min_face_plane_angle_degrees": min_plane_angle_degrees,
        "corner_min_face_plane_angle_degrees": CORNER_FACE_VISIBILITY_MIN_PLANE_ANGLE_DEGREES,
        "include_occluded_keypoints": include_occluded_keypoints,
        "candidate_counts": candidate_counts,
        "source_piece_objects": (
            len(corner_cubie_objects())
            + len(edge_cubie_objects())
            + len(center_cubie_objects())
        ),
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
    image_path: Path | None = None,
) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "OpenCV and NumPy are required to write keypoint visualization masks. "
            "Install cv2 and numpy in Blender's Python environment before generating training data."
        ) from None

    if image_path is None:
        output = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        image = read_image_file(image_path, cv2, np, cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (height, width):
            raise RuntimeError(f"Could not read rendered image for keypoint mask: {image_path}")
        # Preserve the rendered image at full brightness; the compact colored
        # dots provide the annotation without dimming the underlying scene.
        output = image.copy()
    draw_pose_keypoint_mask(output, pose_objects, width, height, cv2, np)
    if not write_image_file(path, output, cv2):
        raise RuntimeError(f"Could not write keypoint visualization mask: {path}")


def draw_pose_keypoint_mask(
    output: object,
    pose_objects: list[dict[str, object]],
    width: int,
    height: int,
    cv2: object,
    np: object,
) -> None:
    """Draw the landmark overlay into an existing BGR image."""
    fill_colors = INDEPENDENT_CLASS_MASK_FILL_COLORS
    point_colors = INDEPENDENT_CLASS_MASK_POINT_COLORS
    segment_fill_width = scaled_annotation_size(POSE_SEGMENT_FILL_BASE_WIDTH, width, height)
    segment_outline_width = scaled_annotation_size(POSE_SEGMENT_OUTLINE_BASE_WIDTH, width, height)
    point_radius = scaled_annotation_size(POSE_POINT_BASE_RADIUS, width, height)
    for pose_object in pose_objects:
        class_id = int(pose_object["class_id"])
        if class_id not in fill_colors or class_id not in point_colors:
            raise ValueError(f"Unexpected independent keypoint class id in mask: {class_id}")
        fill_color = fill_colors[class_id]
        outline_color = point_colors[class_id]
        labeled_points = [
            keypoint["pixel_point"]
            for keypoint in pose_object["keypoints"]
            if int(keypoint["visibility"]) != POINT_VISIBILITY_UNLABELED
        ]
        if len(labeled_points) >= 3:
            points = np.asarray(labeled_points, dtype=np.float32)
            hull = cv2.convexHull(np.round(points).astype(np.int32))
            cv2.fillConvexPoly(output, hull, fill_color, lineType=cv2.LINE_AA)
            cv2.polylines(
                output,
                [hull],
                isClosed=True,
                color=outline_color,
                thickness=segment_outline_width,
                lineType=cv2.LINE_AA,
            )
        elif len(labeled_points) == 2:
            start = tuple(np.round(labeled_points[0]).astype(np.int32))
            end = tuple(np.round(labeled_points[1]).astype(np.int32))
            cv2.line(output, start, end, fill_color, thickness=segment_fill_width, lineType=cv2.LINE_AA)
            cv2.line(output, start, end, outline_color, thickness=segment_outline_width, lineType=cv2.LINE_AA)

        for keypoint in pose_object["keypoints"]:
            visibility = int(keypoint["visibility"])
            if visibility == POINT_VISIBILITY_UNLABELED:
                continue
            x, y = keypoint["pixel_point"]
            draw_color = outline_color
            if visibility == POINT_VISIBILITY_LABELED_OCCLUDED:
                draw_color = (0, 0, 255)
            center = (round(float(x)), round(float(y)))
            cv2.circle(
                output,
                center,
                point_radius + POSE_POINT_OUTLINE_WIDTH,
                (0, 0, 0),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                output,
                center,
                point_radius,
                draw_color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )


def write_ordered_keypoint_overlay(
    path: Path,
    image_path: Path,
    ordered_pose_metadata: dict[str, object],
    width: int,
    height: int,
) -> None:
    """Write a full-brightness image with visible ordered keypoints numbered 0-25."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "OpenCV and NumPy are required to write keypoint overlay masks."
        ) from None

    image = read_image_file(image_path, cv2, np, cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (height, width):
        raise RuntimeError(f"Could not read rendered image for keypoint overlay: {image_path}")
    output = image.copy()

    keypoints = ordered_pose_metadata.get("keypoints", [])
    if not isinstance(keypoints, list):
        raise TypeError("Ordered pose metadata must contain a keypoints list.")
    point_colors = {
        "corner": (255, 255, 255),
        "edge": (0, 255, 255),
        "center": (255, 255, 0),
    }
    point_radius = scaled_annotation_size(POSE_POINT_BASE_RADIUS, width, height)
    point_outline_width = POSE_POINT_OUTLINE_WIDTH
    font_scale = max(0.35, min(width, height) / ANNOTATION_REFERENCE_SIZE * 0.55)
    font_thickness = max(1, scaled_annotation_size(1, width, height))

    for keypoint in keypoints:
        if not isinstance(keypoint, dict):
            continue
        visibility = int(keypoint.get("visibility", POINT_VISIBILITY_UNLABELED))
        if visibility == POINT_VISIBILITY_UNLABELED:
            continue
        pixel_point = keypoint.get("pixel_point")
        index = keypoint.get("index")
        if not isinstance(pixel_point, list) or len(pixel_point) != 2 or not isinstance(index, int):
            continue
        x, y = (round(float(pixel_point[0])), round(float(pixel_point[1])))
        center = (x, y)
        point_kind = "corner" if index < 8 else "edge" if index < 20 else "center"
        color = (0, 0, 255) if visibility == POINT_VISIBILITY_LABELED_OCCLUDED else point_colors[point_kind]
        cv2.circle(
            output,
            center,
            point_radius + point_outline_width,
            (0, 0, 0),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(output, center, point_radius, color, thickness=-1, lineType=cv2.LINE_AA)

        label = str(index)
        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )
        text_x = x + point_radius + point_outline_width + 2
        if text_x + text_size[0] >= width:
            text_x = x - point_radius - point_outline_width - text_size[0] - 2
        text_y = y - point_radius - point_outline_width - 2
        if text_y - text_size[1] < 0:
            text_y = y + point_radius + point_outline_width + text_size[1] + 2
        text_x = max(0, min(text_x, width - text_size[0]))
        text_y = max(text_size[1], min(text_y, height - 1))
        cv2.putText(
            output,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    if not write_image_file(path, output, cv2):
        raise RuntimeError(f"Could not write ordered keypoint overlay: {path}")


def write_yolo_pose(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def atomic_write_text(path: Path, text: str | Iterable[str]) -> None:
    """Replace a text file without exposing a partially written document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            if isinstance(text, str):
                handle.write(text)
            else:
                handle.writelines(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_metadata_snapshot(path: Path, records: Iterable[dict[str, object]]) -> None:
    """Publish the existing JSON-array format with only one record encoded at a time."""

    def chunks() -> Iterator[str]:
        yield "["
        separator = "\n"
        for record in records:
            yield separator + "  " + json.dumps(record, indent=2).replace("\n", "\n  ")
            separator = ",\n"
        yield "\n]\n" if separator == ",\n" else "]\n"

    atomic_write_text(path, chunks())


def sample_indices_in_output_dir(output_dir: Path) -> set[int]:
    indices: set[int] = set()
    for directory_name in ("images", "masks", "labels", POSE_LABEL_DIRECTORY, "metadata"):
        directory = output_dir / directory_name
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.stem.isdigit():
                indices.add(int(path.stem))
    return indices


def load_metadata_records(path: Path) -> list[dict[str, object]]:
    """Load metadata, recovering the completed prefix after an interruption."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        position = 0
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text) or text[position] != "[":
            return []
        position += 1
        records: list[dict[str, object]] = []
        while True:
            while position < len(text) and text[position].isspace():
                position += 1
            if position >= len(text) or text[position] == "]":
                break
            try:
                item, end = decoder.raw_decode(text, position)
            except json.JSONDecodeError:
                break
            if isinstance(item, dict):
                records.append(item)
            position = end
            while position < len(text) and text[position].isspace():
                position += 1
            if position < len(text) and text[position] == ",":
                position += 1
                continue
            if position < len(text) and text[position] == "]":
                break
            break
        return records
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def metadata_sample_indices(records: Iterable[dict[str, object]]) -> set[int]:
    indices: set[int] = set()
    for record in records:
        for key in ("image", "mask", "label", "pose_label"):
            value = record.get(key)
            if not isinstance(value, str):
                continue
            stem = Path(value).stem
            if stem.isdigit():
                indices.add(int(stem))
    return indices


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
    hard_case: str = DEFAULT_HARD_CASE,
    nvidia_gpu: bool = False,
) -> None:
    if (width, height) not in RFDETR_KEYPOINT_DATASET_RESOLUTIONS:
        raise ValueError(
            "RF-DETR Keypoint dataset generation requires one of "
            f"{RFDETR_KEYPOINT_DATASET_RESOLUTIONS}; "
            f"received {width}x{height}."
        )
    if nvidia_gpu:
        configure_nvidia_gpu()
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    pose_dir = output_dir / POSE_LABEL_DIRECTORY
    metadata_dir = output_dir / "metadata"
    manifest_path = output_dir / "sticker_keypoint_manifest.json"
    keypoint_shape = CUBE_CORE_KEYPOINT_SHAPE
    metadata_path = output_dir / "metadata.json"
    metadata_records = load_metadata_records(metadata_path)
    published_indices = metadata_sample_indices(metadata_records)
    existing_indices = sample_indices_in_output_dir(output_dir) | published_indices
    if existing_indices:
        previous_shape = [1, 3]
        if manifest_path.is_file():
            previous_shape = json.loads(manifest_path.read_text(encoding="utf-8"))["keypoint_shape"]
        if previous_shape != keypoint_shape:
            raise ValueError(
                f"{output_dir} contains pose labels with shape {previous_shape}. "
                f"Use a different --output-dir for shape {keypoint_shape}."
            )
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        manifest_path,
        json.dumps(
            {
                "format": "combined_landmark_sticker_and_cube_core_keypoints_v1",
                "model": "RF-DETRKeypointPreview",
                "generation_mode": "standard",
                "blur_policy": {
                    "cube_rotational_blur": True,
                    "post_render_gaussian_blur": True,
                    "post_render_motion_blur": True,
                    "other_post_render_augmentations": [
                        "rescale",
                        "sensor_noise",
                        "video_compression",
                    ],
                },
                "model_native_input_resolution": list(RFDETR_KEYPOINT_NATIVE_INPUT_SIZE),
                "dataset_resolutions": [
                    list(resolution) for resolution in RFDETR_KEYPOINT_DATASET_RESOLUTIONS
                ],
                "class_names": list(COMBINED_KEYPOINT_CLASS_NAMES),
                "landmark_class_names": list(INDEPENDENT_CLASS_NAMES),
                "sticker_class_names": list(STICKER_KEYPOINT_CLASS_NAMES),
                "material_class_ids": STICKER_KEYPOINT_DATASET_CLASS_ID_BY_MATERIAL,
                "keypoint_shape": keypoint_shape,
                "cube_core_class_id": CUBE_CORE_CLASS_ID,
                "cube_core_keypoint_order": list(STICKER_KEYPOINT_CLASS_NAMES),
                "keypoints_per_sticker": 1,
                "landmark_keypoint_classes": {
                    "corner": CORNER_CLASS_ID,
                    "corner-edge": CORNER_EDGE_CLASS_ID,
                    "corner-edge-center": CORNER_EDGE_CENTER_CLASS_ID,
                },
                "partial_turn_seam_keypoints_included": False,
                "sticker_min_face_plane_angle_degrees": STICKER_KEYPOINT_MIN_PLANE_ANGLE_DEGREES,
                "visibility": {
                    "visible": POINT_VISIBILITY_VISIBLE,
                    "occluded": POINT_VISIBILITY_LABELED_OCCLUDED,
                    "excluded": POINT_VISIBILITY_UNLABELED,
                },
                "occluded_min_visible_sticker_area_ratio": STICKER_KEYPOINT_MIN_VISIBLE_AREA_RATIO,
                "mask_output": (
                    "Rendered image with circular landmark and sticker keypoints and "
                    "color-coded cube-core diamonds."
                ),
            },
            indent=2,
        )
        + "\n",
    )
    # Kept only to make the legacy manifest branch below inert while older
    # helper definitions remain available to existing imports.
    pose_label_mode = "combined_keypoints"
    min_face_plane_angle_degrees = 0.0
    include_occluded_independent_keypoints = False
    labels_dir = output_dir / "labels"
    # The old ordered/independent cube-pose manifests and segmentation class
    # manifest are intentionally not written for this sticker-keypoint format.
    if False:
        if pose_label_mode == POSE_LABEL_MODE_ORDERED:
            symmetry_manifest = output_dir / "ordered_keypoint_symmetries.json"
            atomic_write_text(
                symmetry_manifest,
                json.dumps(
                    {
                        "format": "ordered_cube_keypoint_symmetry_v1",
                        "keypoint_count": ORDERED_KEYPOINT_COUNT,
                        "keypoint_names": list(ORDERED_KEYPOINT_NAMES),
                        "permutation_convention": "permutation[output_index] = source_index",
                        "symmetry_scope": "full_cube_rotations",
                        "symmetry_count": 24,
                        "min_face_plane_angle_degrees": min_face_plane_angle_degrees,
                        "permutations": [
                            list(permutation)
                            for permutation in ORDERED_KEYPOINT_SYMMETRY_PERMUTATIONS
                        ],
                    },
                    indent=2,
                )
                + "\n",
            )
        else:
            independent_manifest = output_dir / "independent_keypoint_manifest.json"
            atomic_write_text(
                independent_manifest,
                json.dumps(
                    {
                        "format": "independent_cube_keypoints_v1",
                        "class_names": INDEPENDENT_CLASS_NAMES,
                        "keypoint_shape": [POSE_KEYPOINT_COUNT, 3],
                        "source_keypoint_count": 56,
                        "corner_source": "corner_cubies",
                        "corner_edge_source": "edge_cubies",
                        "corner_edge_center_source": "center_cubies",
                        "base_source_keypoint_count": 56,
                        "partial_turn_seam_keypoints_enabled": PARTIAL_TURN_SEAM_KEYPOINTS_ENABLED,
                        "partial_turn_intersections": (
                            "For mid-turn, slice, simultaneous, and corner-cut samples, only edge and "
                            "center-sticker intersections on the layer seam receive a "
                            "turned-side counterpart once the turn exceeds "
                            f"{PARTIAL_TURN_INTERSECTION_MIN_DEGREES:g} degrees; "
                            "corners remain singletons."
                        ),
                        "interior_seam_angle_override": (
                            "A corner-edge point that fails the ordinary view-angle "
                            "rule may still be visible when it lies on the fixed-side "
                            "partial-turn seam and its projected pixel is owned by the "
                            "target cube. Corner-edge-center points remain subject to "
                            "the ordinary rule. Hand/background ownership and the "
                            "normal occlusion test still exclude it."
                        ),
                        "min_face_plane_angle_degrees": min_face_plane_angle_degrees,
                        "corner_min_face_plane_angle_degrees": CORNER_FACE_VISIBILITY_MIN_PLANE_ANGLE_DEGREES,
                        "include_occluded_keypoints": include_occluded_independent_keypoints,
                    },
                    indent=2,
                )
                + "\n",
            )
    if False:
        atomic_write_text(
            labels_dir / "classes.json",
            json.dumps(
                {
                    str(class_id): name
                    for class_id, name in enumerate(SEGMENTATION_CLASS_NAMES)
                },
                indent=2,
            ),
        )

    ensure_hand_assets_loaded()
    next_sample_index = max(existing_indices, default=-1) + 1
    for index in range(count):
        auto_sample_hard_case = (
            hard_case == DEFAULT_HARD_CASE
            and HARD_CASE_AUTO_SAMPLING_ENABLED
        )
        if auto_sample_hard_case:
            active_hard_case = (
                random.choice(NON_SOLVED_HARD_CASES)
                if random.random() < HARD_CASE_GLOBAL_PROBABILITY
                else DEFAULT_HARD_CASE
            )
            cube_solved = random.random() < GLOBAL_SOLVED_CUBE_PROBABILITY
        else:
            active_hard_case = hard_case
            cube_solved = hard_case == SOLVED_CUBE_HARD_CASE
        # Keep the seeded scene draws aligned with the former 144 px path.
        random.random()
        configure_render(width, height)
        # Deliberate hard cases own their occlusion setup. Keep hands enabled
        # unless the case explicitly requests a hand-free scene through its
        # returned metadata.
        desk_no_hands = (
            random.random() < CUBE_DESK_NO_HANDS_PROBABILITY
            if active_hard_case == DEFAULT_HARD_CASE
            else False
        )
        if active_hard_case == DEFAULT_HARD_CASE:
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
                desk_no_hands=desk_no_hands,
                solved=cube_solved,
            )
        else:
            # Deliberate cases own their scene setup.  In particular, do not
            # call randomize_scene() here: that would erase the controlled
            # pose/background/occlusion the case is meant to reproduce.
            (
                moves,
                layer_state,
                background_randomization,
                camera_randomization,
                light_randomization,
                cube_pose,
                sticker_materials,
            ) = build_hard_case_scene(
                active_hard_case,
                scramble_length=scramble_length,
                background_object_count=background_object_count,
                solved=cube_solved,
            )
        base_cube_pose = cube_pose
        objects = cube_objects()
        core_objects = core_fill_objects()
        cube_pose_objects = list(objects) + core_objects
        cube_base_transforms = {
            obj.name: obj.matrix_world.copy() for obj in cube_pose_objects
        }

        sample_index = next_sample_index + index
        stem = f"{sample_index:06d}"
        image_path = images_dir / f"{stem}.png"
        mask_path = masks_dir / f"{stem}.png"
        pose_label_path = pose_dir / f"{stem}.txt"

        hands = hand_objects()
        hand_rigs = hand_rig_objects()
        hand_states = hand_object_states(hands)
        for obj in cube_pose_objects:
            obj.matrix_world = cube_base_transforms[obj.name]
        restore_hand_objects(hands, hand_states)
        hand_randomization = randomize_hands(
            hands,
            hand_states,
            forced_visibility=(
                background_randomization.get("forced_hand_visibility")
                if not desk_no_hands
                else "none"
            ),
        )
        hand_lift = (
            random.uniform(*PALM_HAND_LIFT_RANGE)
            if active_hard_case == PALM_GAP_HARD_CASE
            else CUBE_HANDS_TABLE_LIFT
        ) if (
            hand_randomization.get("visibility") != "none"
            and not desk_no_hands
        ) else 0.0
        hand_visible = hand_randomization.get("visibility") != "none"
        placement_cube_pose = base_cube_pose if hand_visible else cube_pose_on_table(base_cube_pose)
        cube_pose = cube_pose_with_hand_lift(placement_cube_pose, hand_lift)
        partial_turn_hands = apply_partial_turn_to_hands(layer_state, hand_rigs)
        apply_shared_cube_pose(cube_pose_objects, hand_rigs, cube_pose)
        hand_randomization["follows_cube_pose"] = True
        hand_randomization["follows_whole_cube_scramble"] = False
        hand_randomization["follows_local_layer_turn"] = partial_turn_hands
        hand_randomization["ignores_local_layer_turn"] = False
        hand_randomization["cube_vertical_lift"] = hand_lift
        hand_attempts = [{"attempt": 1, "accepted": True, "hands": hand_randomization}]

        if active_hard_case == PALM_GAP_HARD_CASE and hand_randomization is not None:
            # The palm-gap case holds the cube higher than ordinary desk
            # samples so the visible palm remains above the tabletop. The
            # procedural hull must follow that same lift or it can intersect
            # the table while the real hand is elevated.
            base_translation = base_cube_pose.get("translation")
            final_translation = cube_pose.get("translation")
            if isinstance(base_translation, list) and isinstance(final_translation, list):
                palm_lift_delta = float(final_translation[2]) - float(base_translation[2])
                if abs(palm_lift_delta) > 1e-6:
                    palm_objects = background_randomization.get("objects", [])
                    hull_names: list[str] = []
                    if isinstance(palm_objects, list):
                        for item in palm_objects:
                            if not isinstance(item, dict):
                                continue
                            if item.get("family") != "dark_gap_backdrop":
                                continue
                            names = item.get("object_names", [])
                            if isinstance(names, list):
                                hull_names.extend(
                                    name for name in names if isinstance(name, str)
                                )
                    for name in hull_names:
                        hull = bpy.data.objects.get(name)
                        if hull is not None:
                            hull.location.z += palm_lift_delta
                    variant = background_randomization.get("variant")
                    if isinstance(variant, dict):
                        geometry = variant.get("palm_hull_geometry")
                        if isinstance(geometry, dict):
                            final_center = geometry.get("final_center")
                            if isinstance(final_center, list) and len(final_center) == 3:
                                final_center[2] += palm_lift_delta
                            geometry["hand_lift_followed"] = palm_lift_delta

        aim_camera_at_final_cube(camera_randomization, cube_pose)

        # Place distractor geometry only after the primary cube and hands have
        # their final world transforms.  Placement bounds are therefore in the
        # same coordinate system as the rendered scene, and visible hands are
        # treated as collision obstacles.
        if active_hard_case != SOLID_BACKGROUND_PATCH_HARD_CASE:
            secondary_cube = duplicate_secondary_cube(
                objects,
                core_objects,
                background_randomization,
                obstacle_objects=list(hands.values()),
            )
            if secondary_cube is not None:
                background_randomization["secondary_cube"] = secondary_cube
        sticker_materials["white_center_logo"] = randomize_cube_logo(objects)

        # Hard-case builders use controlled cameras and therefore do not pass
        # through the ordinary pre-background camera probe.  Probe again after
        # the cube, hands, and secondary cube have their final transforms so a
        # wall, hand, or distractor cannot silently produce an all-background
        # render.  If necessary, recover with a clear camera before rendering.
        final_visibility_probe = camera_cube_visibility_probe(objects)
        if not final_visibility_probe["accepted"]:
            blockers = final_visibility_probe.get("blockers", {})
            station_is_blocking = isinstance(blockers, dict) and any(
                isinstance(name, str)
                and name.startswith(f"{BACKGROUND_SCENE_PREFIX}.solve_station.")
                for name in blockers
            )
            if station_is_blocking:
                move_solve_station_cluster_for_camera(background_randomization)
                final_visibility_probe = camera_cube_visibility_probe(objects)
        if not final_visibility_probe["accepted"]:
            camera_randomization = ensure_camera_has_clear_cube_view(
                objects,
                camera_randomization,
                max_general_attempts=6,
                max_fallback_attempts=6,
            )
            aim_camera_at_final_cube(camera_randomization, cube_pose)
            final_visibility_probe = camera_cube_visibility_probe(objects)
        background_randomization["final_visibility_probe"] = final_visibility_probe
        if not final_visibility_probe["accepted"]:
            raise RuntimeError(
                "Final camera visibility probe could not find any visible cube surface. "
                f"Probe: {final_visibility_probe}"
            )

        try:
            render_objects = list(objects) + list(hand_rigs.values())
            localized_blur_object_axes = layer_turn_blur_object_axes(
                layer_state,
                cube_pose,
                {
                    rig.name
                    for rig in hand_rigs.values()
                    if not rig.hide_render and not rig.hide_get()
                },
            )
            # Keep the local blur set limited to the actual partial-turn
            # direction(s). Hands remain in this set at a reduced magnitude.
            rotational_blur = render_cube_rotational_blur_png(
                image_path,
                render_objects,
                probability=cube_rotational_blur_probability,
                max_degrees=max_cube_rotational_blur_degrees,
                samples=cube_rotational_blur_samples,
                local_object_axes=localized_blur_object_axes,
                local_pivot=Vector(cube_pose["translation"]),
            )
            camera_degradation = apply_camera_degradation(
                image_path,
                probability=degrade_probability,
            )
            sticker_roots = list(objects) + core_objects
            cube_visibility_states = set_mesh_objects_render_visible(sticker_roots)
            try:
                landmark_lines, landmark_pose_objects, landmark_metadata = render_pose_keypoints(
                    path=None,
                    image_path=None,
                    include_occluded_keypoints=False,
                    layer_state=layer_state,
                    cube_pose=cube_pose,
                    include_partial_turn_intersections=False,
                )
                combined_lines, sticker_keypoint_metadata = render_sticker_keypoints(
                    pose_label_path,
                    mask_path,
                    image_path,
                    sticker_roots,
                    list(hand_rigs.values()),
                    width,
                    height,
                    landmark_lines=landmark_lines,
                    landmark_pose_objects=landmark_pose_objects,
                )
                sticker_keypoint_metadata["landmark_metadata"] = landmark_metadata
                pose_label_valid = True
            finally:
                restore_mesh_object_visibility(cube_visibility_states)
        finally:
            restore_hand_objects(hands, hand_states)

        metadata_entry = {
            "active_random_seed": ACTIVE_RANDOM_SEED,
            "hard_case": active_hard_case,
            "generation_mode": "standard",
            "blur_policy": {
                "cube_rotational_blur": True,
                "post_render_gaussian_blur": True,
                "post_render_motion_blur": True,
            },
            "hard_case_sampling": {
                "mode": "automatic" if auto_sample_hard_case else "fixed",
                "hard_case_probability": HARD_CASE_GLOBAL_PROBABILITY,
                "global_solved_probability": GLOBAL_SOLVED_CUBE_PROBABILITY,
                "cube_solved": cube_solved,
            },
            "image": str(image_path),
            "image_size": {
                "width": width,
                "height": height,
                "variant": "base",
            },
            "mask": str(mask_path),
            "pose_label": str(pose_label_path),
            "pose_label_valid": pose_label_valid,
            "sticker_keypoints": sticker_keypoint_metadata,
            "move_mode": scene_move_mode(cube_solved),
            "partial_turn_augmentation_enabled": CUBE_MISALIGNMENT_ENABLED,
            "partial_turn_probability": PARTIAL_TURN_APPLIED_PROBABILITY,
            "partial_turn_applied": layer_state.get("mode", "none") != "none",
            "moves": moves,
            "layer_state": layer_state,
            "cubie_labels": {obj.name: obj.pass_index for obj in objects},
            "camera": camera_randomization,
            "light": light_randomization,
            "background": background_randomization,
            "cube_pose": cube_pose,
            "desk_no_hands": desk_no_hands,
            "sticker_materials": sticker_materials,
            "cube_rotational_blur": rotational_blur,
            "camera_degradation": camera_degradation,
            "hands": hand_randomization,
            "hand_attempts": hand_attempts,
            "objects": len(combined_lines),
        }
        atomic_write_text(
            metadata_dir / f"{stem}.json",
            json.dumps(metadata_entry, separators=(",", ":")) + "\n",
        )
        if (index + 1) % MEMORY_CLEANUP_INTERVAL == 0:
            cleanup_blender_memory()
        mask_message = f", {mask_path}"
        blend_asset_count = background_randomization.get("blend_asset_count", 0)
        blend_asset_sources = background_randomization.get("blend_asset_sources", [])
        print(
            f"Wrote {image_path}{mask_message}, "
            f"and {pose_label_path} "
            f"with {metadata_entry['objects']} pose objects, "
            f"{blend_asset_count} blend background assets {blend_asset_sources}, "
            f"{'with' if rotational_blur else 'without'} rotational blur "
            f"and {'with' if camera_degradation else 'without'} camera degradation"
        )

    # Existing datasets may have records only in the combined file. Include
    # those records and any per-image files from this or an interrupted run.
    unpublished_paths = sorted(
        (path for path in metadata_dir.glob("*.json")
         if path.stem.isdigit() and int(path.stem) not in published_indices),
        key=lambda path: int(path.stem),
    )
    unpublished_records = (
        json.loads(path.read_text(encoding="utf-8")) for path in unpublished_paths
    )
    write_metadata_snapshot(metadata_path, chain(metadata_records, unpublished_records))

    data_yaml = output_dir / "data.yaml"
    atomic_write_text(
        data_yaml,
        "path: " + str(output_dir.resolve()) + "\n"
        "train: images\n"
        "val: images\n"
        + "pose: pose\n"
        + f"kpt_shape: {keypoint_shape}\n"
        + f"image_size: {DEFAULT_IMAGE_SIZE}\n"
        + f"available_image_sizes: [{DEFAULT_IMAGE_SIZE}]\n"
        + "model: RF-DETRKeypointPreview\n"
        + "pose_label_mode: combined_keypoints\n"
        + "names:\n"
        + "".join(
            f"  {index}: {name}\n"
            for index, name in enumerate(COMBINED_KEYPOINT_CLASS_NAMES)
        ),
    )


def cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Render with NVIDIA GPU acceleration; prefer OptiX and fall back to CUDA.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scramble-length", type=int, default=DEFAULT_SCRAMBLE_LENGTH)
    parser.add_argument("--width", type=int, choices=(DEFAULT_IMAGE_SIZE,), default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--height", type=int, choices=(DEFAULT_IMAGE_SIZE,), default=DEFAULT_IMAGE_SIZE)
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
        help="Probability of applying post-render blur, noise, rescale softness, and video codec artifacts.",
    )
    parser.add_argument(
        "--mixed-lighting-probability",
        type=float,
        default=DEFAULT_MIXED_LIGHTING_PROBABILITY,
        help="Chance of combining a white key with colored fill and background lights (default: 0.15).",
    )
    parser.add_argument(
        "--enable-mid-turns",
        "--enable-moves",
        dest="enable_mid_turns",
        action="store_true",
        default=True,
        help="Enable partial layer turns and corner-cut states (default); discrete scrambles remain enabled by default.",
    )
    parser.add_argument(
        "--no-mid-turns",
        "--disable-mid-turns",
        dest="enable_mid_turns",
        action="store_false",
        help="Disable partial layer turns and generate rigid-cube examples only.",
    )
    parser.add_argument(
        "--no-scrambles",
        action="store_true",
        help="Disable discrete cube scrambles while leaving scene/appearance augmentations active.",
    )

    argv = sys.argv
    args = parser.parse_args(argv[argv.index("--") + 1 :] if "--" in argv else [])
    if not 0.0 <= args.mixed_lighting_probability <= 1.0:
        parser.error("--mixed-lighting-probability must be between 0 and 1")
    return args


def main() -> None:
    global ACTIVE_RANDOM_SEED, CUBE_MISALIGNMENT_ENABLED, SCRAMBLES_ENABLED, MIXED_LIGHTING_PROBABILITY
    args = cli_args()
    CUBE_MISALIGNMENT_ENABLED = bool(args.enable_mid_turns)
    SCRAMBLES_ENABLED = not args.no_scrambles
    MIXED_LIGHTING_PROBABILITY = args.mixed_lighting_probability
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
        hard_case=DEFAULT_HARD_CASE,
        nvidia_gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
