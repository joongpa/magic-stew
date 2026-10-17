"""Interactively verify projected simplified articulated-cube pose labels.

Examples:

    .venv/bin/python scripts/verify_cube_pose_labels.py data/mcp_dataset_v5
    .venv/bin/python scripts/verify_cube_pose_labels.py data/mcp_dataset_v5/images
    .venv/bin/python scripts/verify_cube_pose_labels.py data/mcp_dataset_v5 --export-dir pose_overlays
    .venv/bin/python scripts/verify_cube_pose_labels.py data/mcp_dataset_v5 \
        --export-dir pose_overlays --no-display

Controls:
    Left/Right or P/N/A/D  previous/next image
    Q or Esc               quit

The overlay draws a simplified cube made from three 3x3x1 layers. Cyan edges
are stationary; orange edges belong to either outer layer represented by the
pose label. The v5, v6, and direct-head v7 formats are accepted. Corner cuts
are reduced to their dominant single-layer turn.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
LAYER_COORDINATES = (-1.0, 0.0, 1.0)
LAYER_HALF_EXTENT = 1.5


@dataclass(frozen=True)
class Sample:
    stem: str
    image_path: Path
    pose_path: Path | None


def normalize_dataset_dir(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.expanduser()
    if dataset_dir.name == "images" and (dataset_dir.parent / "pose").is_dir():
        return dataset_dir.parent
    return dataset_dir


def discover_samples(dataset_dir: Path) -> list[Sample]:
    dataset_dir = normalize_dataset_dir(dataset_dir)
    images_dir = dataset_dir / "images"
    pose_dir = dataset_dir / "pose"
    if not images_dir.is_dir():
        raise SystemExit(f"Missing images directory: {images_dir}")

    samples = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        pose_path = pose_dir / f"{image_path.stem}.json"
        samples.append(
            Sample(
                stem=image_path.stem,
                image_path=image_path,
                pose_path=pose_path if pose_path.is_file() else None,
            )
        )
    if not samples:
        raise SystemExit(f"No images found in: {images_dir}")
    return samples


def load_pose(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"Could not read pose label {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Pose label must contain a JSON object: {path}")
    return payload


def rotation_matrix(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    rvec = axis * np.deg2rad(float(angle_degrees))
    return cv2.Rodrigues(rvec)[0].astype(np.float32)


def quaternion_matrix(quaternion: object) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64).reshape(4)
    values /= max(float(np.linalg.norm(values)), 1e-12)
    x, y, z, w = values
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )


def head_rotation(
    colatitude_degrees: float,
    azimuth_degrees: float,
    twist_degrees: float,
) -> np.ndarray:
    """Reconstruct the cube rotation from direct v7 head coordinates."""
    u = np.deg2rad(float(colatitude_degrees))
    v = np.deg2rad(float(azimuth_degrees))
    stack = np.asarray(
        (np.sin(u) * np.sin(v), -np.cos(u), np.sin(u) * np.cos(v)),
        dtype=np.float32,
    )
    image_right = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    fallback_right = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    reference = fallback_right if abs(float(image_right @ stack)) > 0.95 else image_right
    local_x = reference - float(reference @ stack) * stack
    local_x /= max(float(np.linalg.norm(local_x)), 1e-12)
    local_y = np.cross(stack, local_x)
    angle = np.deg2rad(float(twist_degrees))
    rotated_x = np.cos(angle) * local_x + np.sin(angle) * local_y
    rotated_y = -np.sin(angle) * local_x + np.cos(angle) * local_y
    return np.stack((rotated_x, rotated_y, stack), axis=1).astype(np.float32)


def pose_layer_angles(pose: dict[str, object]) -> list[float]:
    articulation = pose.get("articulation", {})
    if not isinstance(articulation, dict):
        return [0.0, 0.0]

    layer_angles = articulation.get("layer_angles_degrees")
    if isinstance(layer_angles, list) and len(layer_angles) == 2:
        return [float(layer_angles[0]), float(layer_angles[1])]

    turn = articulation.get("turn")
    if turn is None:
        return [0.0, 0.0]
    if not isinstance(turn, dict):
        raise RuntimeError("Pose articulation.turn must be an object or null")
    layer_index = int(turn.get("layer_index", -1))
    if layer_index not in (0, 2):
        raise RuntimeError("Pose turn must target outer layer index 0 or 2")
    angle = float(turn.get("signed_angle_degrees", 0.0))
    result = [0.0, 0.0]
    result[0 if layer_index == 0 else 1] = angle
    return result


def articulated_layer_line_specs(
    pose: dict[str, object],
) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """Return surface lines for the simplified three-layer cube model."""
    layer_angles = pose_layer_angles(pose)
    layer_angle_by_coordinate = {
        -1.0: float(layer_angles[0]),
        1.0: float(layer_angles[1]),
    }
    axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    axis_index = 2
    line_specs = []
    for layer_coordinate in LAYER_COORDINATES:
        angle = layer_angle_by_coordinate.get(layer_coordinate, 0.0)
        # Pose angles use the turning face's outward normal.  The negative
        # outer slab therefore rotates about -Z, while the positive slab
        # rotates about +Z.  Applying both angles about +Z mirrors every
        # negative-layer turn in the overlay.
        layer_axis = axis if layer_coordinate > 0.0 else -axis
        layer_matrix = (
            rotation_matrix(layer_axis, angle)
            if abs(angle) > 1e-8
            else np.eye(3, dtype=np.float32)
        )
        layer_is_turned = abs(angle) > 1e-8
        for normal_axis in range(3):
            tangent_axes = [index for index in range(3) if index != normal_axis]
            for normal_sign in (-1.0, 1.0):
                # Each slab contributes only its exposed outer face along the
                # stack axis. The other faces contribute their surface strips.
                if normal_axis == axis_index and layer_coordinate != normal_sign:
                    continue
                for varying_axis, fixed_axis in (
                    (tangent_axes[0], tangent_axes[1]),
                    (tangent_axes[1], tangent_axes[0]),
                ):
                    varying_values = (
                        (layer_coordinate - 0.5, layer_coordinate + 0.5)
                        if varying_axis == axis_index
                        else (-LAYER_HALF_EXTENT, LAYER_HALF_EXTENT)
                    )
                    fixed_values = (
                        (layer_coordinate - 0.5, layer_coordinate + 0.5)
                        if fixed_axis == axis_index
                        else (-LAYER_HALF_EXTENT, -0.5, 0.5, LAYER_HALF_EXTENT)
                    )
                    for fixed_value in fixed_values:
                        start = np.zeros(3, dtype=np.float32)
                        end = np.zeros(3, dtype=np.float32)
                        start[normal_axis] = normal_sign * LAYER_HALF_EXTENT
                        end[normal_axis] = normal_sign * LAYER_HALF_EXTENT
                        start[fixed_axis] = fixed_value
                        end[fixed_axis] = fixed_value
                        start[varying_axis] = varying_values[0]
                        end[varying_axis] = varying_values[1]
                        line_specs.append(
                            (
                                layer_matrix @ start,
                                layer_matrix @ end,
                                layer_is_turned,
                            )
                        )
    return line_specs


def projection_parameters(
    pose: dict[str, object],
    *,
    no_perspective: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    rotation_payload = pose.get("cube_rotation_camera", {})
    cube_pose = pose.get("cube_pose", {})
    projection = pose.get("image_projection", {})
    if not isinstance(projection, dict):
        raise RuntimeError("Pose label has no compact image projection")
    if pose.get("format") == "cube_pose_v7":
        if not isinstance(cube_pose, dict):
            raise RuntimeError("v7 pose label has no cube_pose")
        try:
            colatitude = float(cube_pose["u_colatitude_degrees"])
            azimuth = float(cube_pose["v_azimuth_degrees"])
            twist = float(cube_pose["twist_degrees"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("v7 pose label has invalid cube_pose values") from error
        if not 0.0 <= colatitude < 90.0:
            raise RuntimeError("v7 colatitude must be in [0, 90)")
        if not -180.0 <= azimuth < 180.0:
            raise RuntimeError("v7 azimuth must be in [-180, 180)")
        if not -45.0 <= twist < 45.0:
            raise RuntimeError("v7 twist must be in [-45, 45)")
        cube_rotation = head_rotation(colatitude, azimuth, twist)
    else:
        quaternion = (
            rotation_payload.get("quaternion_xyzw")
            if isinstance(rotation_payload, dict)
            else None
        )
        if not isinstance(quaternion, list) or len(quaternion) != 4:
            raise RuntimeError("Pose label has no compact cube rotation quaternion")
        cube_rotation = quaternion_matrix(quaternion)
    center = np.asarray(projection.get("cube_center_normalized"), dtype=np.float32)
    if center.shape != (2,):
        raise RuntimeError("Pose label has no normalized cube center")
    scale = float(projection["image_scale_normalized"])
    perspective_strength = (
        0.0
        if no_perspective
        else float(projection.get("perspective_strength", 0.0))
    )
    return cube_rotation, center, scale, perspective_strength


def project_points(
    points: np.ndarray,
    cube_rotation: np.ndarray,
    center: np.ndarray,
    scale: float,
    perspective_strength: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    camera_points = (cube_rotation @ points.astype(np.float32).T).T
    denominator = 1.0 + (perspective_strength / 3.0) * camera_points[:, 2]
    safe_denominator = np.maximum(denominator, 1e-6)
    normalized = center + scale * camera_points[:, :2] / safe_denominator[:, None]
    pixels = normalized * np.asarray((width, height), dtype=np.float32)
    depths = denominator.copy()
    return pixels, depths


def draw_projected_cubies(
    image: np.ndarray,
    pose: dict[str, object],
    *,
    no_perspective: bool = False,
) -> np.ndarray:
    height, width = image.shape[:2]
    cube_rotation, center, scale, perspective_strength = projection_parameters(
        pose,
        no_perspective=no_perspective,
    )
    overlay = image.copy()

    for start, end, turned in articulated_layer_line_specs(pose):
        pixels, depths = project_points(
            np.asarray((start, end), dtype=np.float32),
            cube_rotation,
            center,
            scale,
            perspective_strength,
            width,
            height,
        )
        color = (0, 130, 255) if turned else (255, 220, 0)
        if depths[0] <= 0.05 or depths[1] <= 0.05:
            continue
        start_pixel = tuple(np.round(pixels[0]).astype(int))
        end_pixel = tuple(np.round(pixels[1]).astype(int))
        cv2.line(overlay, start_pixel, end_pixel, color, 1, cv2.LINE_AA)

    opacity = 0.35
    return cv2.addWeighted(overlay, opacity, image, 1.0 - opacity, 0.0)


def fit_to_window(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale == 1.0:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def key_direction(raw_key: int) -> int:
    key = raw_key & 0xFF
    if raw_key in (81, 2424832, 65361, 123) or key in (ord("p"), ord("a")):
        return -1
    if raw_key in (83, 2555904, 65363, 124) or key in (ord("n"), ord("d")):
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Dataset root or its images/ directory.")
    parser.add_argument("--start", type=int, default=0, help="Initial sample index.")
    parser.add_argument("--max-width", type=int, default=1400)
    parser.add_argument("--max-height", type=int, default=900)
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Also save every projected overlay into this directory.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Export overlays without opening the interactive window.",
    )
    parser.add_argument(
        "--no-perspective",
        action="store_true",
        help="Force scaled-orthographic projection by ignoring perspective strength.",
    )
    args = parser.parse_args()

    dataset_dir = normalize_dataset_dir(args.dataset)
    samples = discover_samples(dataset_dir)
    if args.export_dir is not None:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            if sample.pose_path is None:
                continue
            image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            overlay = draw_projected_cubies(
                image,
                load_pose(sample.pose_path),
                no_perspective=args.no_perspective,
            )
            cv2.imwrite(str(args.export_dir / f"{sample.stem}.png"), overlay)

        if args.no_display:
            return

    sample_index = max(0, min(args.start, len(samples) - 1))
    window_name = "Cube pose label verifier"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            sample = samples[sample_index]
            image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read image: {sample.image_path}")
            if sample.pose_path is None:
                display = image
                status = f"{sample_index + 1}/{len(samples)} {sample.stem} | missing pose label"
            else:
                display = draw_projected_cubies(
                    image,
                    load_pose(sample.pose_path),
                    no_perspective=args.no_perspective,
                )
                status = f"{sample_index + 1}/{len(samples)} {sample.stem} | P/N/A/D or arrows next/previous | Q quit"
            display = fit_to_window(display, args.max_width, args.max_height)
            cv2.setWindowTitle(window_name, status)
            cv2.imshow(window_name, display)
            raw_key = cv2.waitKeyEx(0)
            if (raw_key & 0xFF) in (ord("q"), 27):
                break
            direction = key_direction(raw_key)
            if direction:
                sample_index = (sample_index + direction) % len(samples)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
