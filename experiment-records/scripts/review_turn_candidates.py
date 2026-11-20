#!/usr/bin/env python3
"""Review turn-search candidates frame by frame in a local browser UI.

The reviewer uses the search-time pose and camera inputs stored by
``--turn-candidate-log``.  It does not run any neural models and does not
modify the candidate log. Human selections are written to a separate JSON
file next to the log by default.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


CUBE_SIZE_MM = 55.0
CUBE_HALF_SIZE_MM = CUBE_SIZE_MM / 2.0
CUBE_LAYER_SIZE_MM = CUBE_SIZE_MM / 3.0
MOVING_COLOR_BGR = (0, 0, 255)
STATIC_COLOR_BGR = (0, 0, 0)
HEATMAP_COLOR_BGR = (180, 105, 255)
OBSERVED_COLOR_BGR = (255, 0, 255)
TEXT_COLOR_BGR = (255, 255, 255)
OUTLINE_COLOR_BGR = (0, 0, 0)
FRAME_CACHE_SIZE = 32


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (int, np.integer, str, bool)) or value is None:
        return value
    return str(value)


def format_score(value: object) -> str:
    """Format a possibly-null/non-finite score for the overlay."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.4f}" if np.isfinite(number) else "inf"


def read_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"Invalid JSON on line {line_number} of {path}: {error}") from error
            if not isinstance(record, dict) or "frame_index" not in record:
                raise SystemExit(f"Log line {line_number} is missing a frame_index")
            records.append(record)
    records.sort(key=lambda record: int(record["frame_index"]))
    return records


def review_candidates(record: dict[str, object]) -> list[dict[str, object]]:
    """Return tested trials, or an explicit static-only option."""

    candidates = record.get("candidates", [])
    if isinstance(candidates, list) and candidates:
        return candidates
    if not record.get("has_pose", False) or not isinstance(record.get("search_pose"), dict):
        return []

    static_scores = record.get("static_fit_scores")
    if isinstance(static_scores, list) and len(static_scores) >= 3:
        keypoint_score, heatmap_score, global_score = static_scores[:3]
    else:
        keypoint_score = heatmap_score = global_score = None
    return [
        {
            "label": "static pose (turn search not tested)",
            "is_baseline": True,
            "layer_rotations": record.get("selected_layer_rotations", []),
            "global_mean_error": global_score,
            "global_robust_error": None,
            "keypoint_mean_error": keypoint_score,
            "heatmap_mean_error": heatmap_score,
            "color_error": record.get("turn_fit_color_score"),
            "inlier_count": None,
        }
    ]


def color_likelihood_payload(record: dict[str, object]) -> dict[str, object]:
    """Return direct color matches, or probabilities from a legacy log."""

    if record.get("color_scoring") in ("direct_match_fraction", "directional_confusion_cost") or "turn_color_matches" in record:
        return {
            "color_scoring": record.get('color_scoring', 'direct_match_fraction'),
            "color_identity_labels": record.get("turn_color_identity_labels") or [],
            "color_observation_centers": record.get("turn_color_observation_centers") or [],
            "color_observation_names": record.get("turn_color_observation_names") or [],
            "color_matches": record.get("turn_color_matches") or [],
        }
    likelihoods = record.get("turn_color_matches", record.get("turn_color_likelihoods"))
    labels = record.get("turn_color_identity_labels")
    centers = record.get("turn_color_observation_centers")
    if not isinstance(likelihoods, list) or not likelihoods:
        return {
            "color_identity_labels": [],
            "color_observation_centers": [],
            "color_likelihoods": [],
        }
    if not isinstance(labels, list):
        labels = [f"identity {index}" for index in range(len(likelihoods))]
    if not isinstance(centers, list):
        centers = []
    payload = {
        "color_identity_labels": labels,
        "color_observation_centers": centers,
        "color_likelihoods": likelihoods,
    }
    if "turn_color_matches" in record:
        payload["color_scoring"] = "direct_match_fraction"
    return payload


def selected_candidate_index(record: dict[str, object]) -> int | None:
    """Locate the logged selected configuration, independent of ranking order."""

    candidates = review_candidates(record)
    selected_id = record.get("selected_trial_id")
    if selected_id is not None:
        return next((index for index, candidate in enumerate(candidates)
                     if candidate.get("trial_id") == selected_id), None)
    rotations = record.get("selected_layer_rotations")
    if not isinstance(rotations, list):
        return None
    for index, candidate in enumerate(candidates):
        if candidate.get("layer_rotations") == rotations:
            # Legacy logs may share the same partial geometry across offsets.
            if "state_moves" in candidate and "selected_committed_moves" in record:
                if candidate["state_moves"] != record["selected_committed_moves"]:
                    continue
            if "color_offset" in candidate and "selected_color_offset_degrees" in record:
                if candidate["color_offset"] != record["selected_color_offset_degrees"]:
                    continue
            return index
    return None


def search_debug_payload(record: dict[str, object] | None) -> dict[str, object] | None:
    """Return recorded search evidence; do not infer forecasts for old logs."""

    value = record.get("search") if isinstance(record, dict) else None
    return value if isinstance(value, dict) else None


def pose_for_candidate(
    record: dict[str, object],
    candidate_index: int,
) -> dict[str, object] | None:
    """Use the finalized renderer pose for the selected configuration."""

    selected_index = selected_candidate_index(record)
    if selected_index == candidate_index:
        selected_pose = record.get("selected_pose")
        if isinstance(selected_pose, dict):
            return selected_pose
    search_pose = record.get("search_pose")
    if not isinstance(search_pose, dict):
        return None
    candidates = review_candidates(record)
    symmetry = (candidates[candidate_index].get('orientation_symmetry')
                if 0 <= candidate_index < len(candidates) else None)
    if symmetry is not None:
        rotation = cv2.Rodrigues(np.asarray(search_pose['rvec'], dtype=np.float64))[0]
        return dict(search_pose, rvec=cv2.Rodrigues(
            rotation @ np.asarray(symmetry, dtype=np.float64).T,
        )[0].tolist())
    return search_pose


def rotation_matrix_for_layer(rotation: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    axis = int(rotation["axis"])
    side = int(rotation["side"])
    angle = side * np.deg2rad(float(rotation["angle_degrees"]))
    vector = np.zeros((3, 1), dtype=np.float64)
    vector[axis, 0] = angle
    matrix, _ = cv2.Rodrigues(vector)
    pivot = np.zeros(3, dtype=np.float64)
    if int(rotation.get("layer_count", 1)) < 3:
        pivot[axis] = side * CUBE_HALF_SIZE_MM
    return matrix, pivot


def cubie_in_layer(center: np.ndarray, rotation: dict[str, object]) -> bool:
    axis = int(rotation["axis"])
    side = int(rotation["side"])
    layer_count = int(rotation["layer_count"])
    coordinate = float(center[axis])
    layer_centers = np.asarray(
        (-CUBE_LAYER_SIZE_MM, 0.0, CUBE_LAYER_SIZE_MM),
        dtype=np.float64,
    )
    index = int(np.argmin(np.abs(layer_centers - coordinate)))
    return index >= 3 - layer_count if side > 0 else index < layer_count


def apply_rotations(
    points: np.ndarray,
    center: np.ndarray,
    rotations: list[dict[str, object]],
) -> np.ndarray:
    transformed = np.asarray(points, dtype=np.float64).copy()
    base_center = np.asarray(center, dtype=np.float64).reshape(3)
    for rotation in rotations:
        matrix, pivot = rotation_matrix_for_layer(rotation)
        if not cubie_in_layer(base_center, rotation):
            continue
        transformed = (matrix @ (transformed - pivot).T).T + pivot
    return transformed


def cubie_center(x_index: int, y_index: int, z_index: int) -> np.ndarray:
    return np.asarray(
        (
            (x_index - 1) * CUBE_LAYER_SIZE_MM,
            (y_index - 1) * CUBE_LAYER_SIZE_MM,
            (z_index - 1) * CUBE_LAYER_SIZE_MM,
        ),
        dtype=np.float64,
    )


def face_points(center: np.ndarray, axis: int, side: int) -> np.ndarray:
    other_axes = [index for index in range(3) if index != axis]
    points = []
    for first, second in (
        (-1, -1),
        (1, -1),
        (1, 1),
        (-1, 1),
    ):
        point = np.asarray(center, dtype=np.float64).copy()
        point[axis] = side * CUBE_HALF_SIZE_MM
        point[other_axes[0]] += first * CUBE_LAYER_SIZE_MM / 2.0
        point[other_axes[1]] += second * CUBE_LAYER_SIZE_MM / 2.0
        points.append(point)
    points_array = np.asarray(points, dtype=np.float64)
    expected_normal = np.zeros(3, dtype=np.float64)
    expected_normal[axis] = side
    normal = np.cross(
        points_array[1] - points_array[0],
        points_array[2] - points_array[0],
    )
    if float(np.dot(normal, expected_normal)) < 0.0:
        points_array = points_array[[0, 3, 2, 1]]
    return points_array


def projected_visible_sticker_faces(
    pose: dict[str, object],
    rotations: list[dict[str, object]],
    selected_layer: dict[str, object] | None = None,
) -> list[tuple[np.ndarray, bool]]:
    return [
        (item["polygon"], item["moving"])
        for item in projected_face_regions(pose, rotations, selected_layer)
    ]


def projected_face_regions(
    pose: dict[str, object],
    rotations: list[dict[str, object]],
    selected_layer: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Project visible cubie faces together with their clickable outer edges."""

    camera_matrix = np.asarray(pose["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(pose["dist_coeffs"], dtype=np.float64)
    rvec = np.asarray(pose["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(pose["tvec"], dtype=np.float64).reshape(3, 1)
    camera_rotation, _ = cv2.Rodrigues(rvec)
    geometry_state = pose.get("cube_geometry_state")
    if isinstance(geometry_state, dict) and geometry_state:
        geometry_faces = []
        for value in geometry_state.values():
            try:
                position, axis, side = value
                position = tuple(int(item) for item in position)
                axis = int(axis)
                side = int(side)
            except (TypeError, ValueError):
                continue
            if (
                len(position) == 3
                and axis in (0, 1, 2)
                and side in (-1, 1)
                and position[axis] in (0, 2)
            ):
                geometry_faces.append((position, axis, side))
    else:
        geometry_faces = [
            ((x_index, y_index, z_index), axis, side)
            for x_index in range(3)
            for y_index in range(3)
            for z_index in range(3)
            for axis in range(3)
            for side in (-1, 1)
            if (x_index, y_index, z_index)[axis] == (0 if side < 0 else 2)
        ]
    faces: list[dict[str, object]] = []
    for indices, axis, side in geometry_faces:
        center = cubie_center(*indices)
        base_polygon = face_points(center, axis, side)
        polygon = apply_rotations(base_polygon, center, rotations)
        transformed_center = apply_rotations(
            center.reshape(1, 3),
            center,
            rotations,
        )[0]
        normal = np.cross(
            polygon[1] - polygon[0],
            polygon[2] - polygon[0],
        )
        center_camera = camera_rotation @ transformed_center + tvec.reshape(3)
        normal_camera = camera_rotation @ normal
        if float(np.dot(normal_camera, -center_camera)) <= 0.0:
            continue
        projected, _ = cv2.projectPoints(
            polygon,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        if np.isfinite(projected).all():
            moving = any(cubie_in_layer(center, rotation) for rotation in rotations)
            if not moving and selected_layer is not None:
                moving = cubie_in_layer(center, selected_layer)
            edges = []
            for other_axis in range(3):
                if other_axis == axis or indices[other_axis] == 1:
                    continue
                edge_side = -1 if indices[other_axis] == 0 else 1
                for vertex in range(4):
                    next_vertex = (vertex + 1) % 4
                    if np.isclose(
                        base_polygon[vertex, other_axis],
                        edge_side * CUBE_HALF_SIZE_MM,
                    ) and np.isclose(
                        base_polygon[next_vertex, other_axis],
                        edge_side * CUBE_HALF_SIZE_MM,
                    ):
                        edges.append({
                            "axis": other_axis,
                            "side": edge_side,
                            "start": vertex,
                            "end": next_vertex,
                        })
            faces.append({
                "polygon": projected,
                "moving": moving,
                "cubie_index": list(indices),
                "axis": axis,
                "side": side,
                "depth": float(center_camera[2]),
                "edges": edges,
            })
    return faces


def draw_candidate(
    frame: np.ndarray,
    record: dict[str, object],
    candidate_index: int,
) -> np.ndarray:
    output = frame.copy()
    candidates = review_candidates(record)
    if not 0 <= candidate_index < len(candidates):
        return output
    pose = pose_for_candidate(record, candidate_index)
    if pose is None or any(
        pose.get(name) is None
        for name in ("camera_matrix", "dist_coeffs", "rvec", "tvec")
    ):
        return output
    candidate = candidates[candidate_index]
    rotations = candidate.get("layer_rotations", [])
    if not isinstance(rotations, list):
        rotations = []
    selected_layer = None
    if rotations and all(
        isinstance(rotation, dict)
        and abs(float(rotation.get("angle_degrees", 0.0))) <= 1e-6
        for rotation in rotations
    ):
        selected_layer = rotations[0]
    if selected_layer is None and not rotations:
        record_selected_layer = record.get("selected_layer")
        if isinstance(record_selected_layer, dict):
            selected_layer = record_selected_layer
    try:
        face_regions = projected_face_regions(pose, rotations, selected_layer)
        faces = [
            (item["polygon"], item["moving"])
            for item in face_regions
        ]
    except (cv2.error, KeyError, TypeError, ValueError):
        return output
    overlay = output.copy()
    for polygon, moving in faces:
        points = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
        color = MOVING_COLOR_BGR if moving else STATIC_COLOR_BGR
        cv2.polylines(overlay, [points], True, OUTLINE_COLOR_BGR, 4, cv2.LINE_AA)
        cv2.polylines(overlay, [points], True, color, 2, cv2.LINE_AA)
    output = cv2.addWeighted(output, 0.62, overlay, 0.38, 0.0)

    assignment_points = pose.get("assignment_image_points")
    if assignment_points is not None:
        for point in np.asarray(assignment_points, dtype=np.float64).reshape(-1, 2):
            if np.isfinite(point).all():
                cv2.drawMarker(
                    output,
                    tuple(np.round(point).astype(int)),
                    OBSERVED_COLOR_BGR,
                    cv2.MARKER_CROSS,
                    10,
                    2,
                    cv2.LINE_AA,
                )
    centers = np.asarray(pose.get("sticker_mask_centers") or [], dtype=np.float64)
    color_names = pose.get("sticker_color_names") or record.get("turn_color_observation_names") or []
    color_bgr = {"white": (255, 255, 255), "yellow": (0, 255, 255), "orange": (0, 165, 255),
                 "red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 180, 0)}
    if centers.size == 0:
        centers = np.empty((0, 2), dtype=np.float64)
    for observation_index, center in enumerate(centers.reshape(-1, 2)):
        if np.isfinite(center).all():
            point = tuple(np.round(center).astype(int))
            cv2.circle(output, point, 7, (0, 0, 0), -1, cv2.LINE_AA)
            name = color_names[observation_index] if observation_index < len(color_names) else None
            cv2.circle(output, point, 5, color_bgr.get(name, HEATMAP_COLOR_BGR), -1, cv2.LINE_AA)

    identity_geometry = record.get("turn_color_identity_geometry", [])
    geometry_by_key = {}
    if isinstance(identity_geometry, list):
        for item in identity_geometry:
            if not isinstance(item, dict):
                continue
            cubie_index = item.get("cubie_index")
            if not isinstance(cubie_index, list) or len(cubie_index) != 3:
                continue
            try:
                key = (
                    tuple(int(value) for value in cubie_index),
                    int(item["axis"]),
                    int(item["side"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            geometry_by_key[key] = str(item.get("label", ""))
    candidate_likelihoods = candidate.get("color_likelihoods_by_identity", {})
    if isinstance(candidate_likelihoods, dict) and geometry_by_key:
        for region in face_regions:
            key = (
                tuple(int(value) for value in region["cubie_index"]),
                int(region["axis"]),
                int(region["side"]),
            )
            full_label = geometry_by_key.get(key)
            probability = candidate_likelihoods.get(full_label)
            if full_label is None or probability is None:
                continue
            try:
                probability = float(probability)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(probability):
                continue
            label = full_label.split(" ", 1)[0]
            polygon = np.asarray(region["polygon"], dtype=np.float64).reshape(-1, 2)
            center = np.mean(polygon, axis=0)
            text = f"{label} {probability:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.42
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                text,
                font,
                scale,
                thickness,
            )
            x = int(round(float(center[0]) - text_width / 2.0))
            y = int(round(float(center[1]) + (text_height - baseline) / 2.0))
            cv2.rectangle(
                output,
                (x - 3, y - text_height - 3),
                (x + text_width + 3, y + baseline + 3),
                (0, 0, 0),
                -1,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                text,
                (x, y),
                font,
                scale,
                TEXT_COLOR_BGR,
                thickness,
                cv2.LINE_AA,
            )

    label = str(candidate.get("label", f"candidate {candidate_index}"))
    text = (
        f"{candidate_index}: {label} | "
        f"mean {format_score(candidate.get('global_mean_error'))} | "
        f"robust {format_score(candidate.get('global_robust_error'))} | "
        f"color {format_score(candidate.get('color_error', candidate.get('color_mean_error')))}"
    )
    cv2.rectangle(output, (6, 6), (min(output.shape[1] - 6, 900), 36), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR_BGR,
        1,
        cv2.LINE_AA,
    )
    return output


class ReviewState:
    def __init__(self, video_path: Path, records: list[dict[str, object]], labels_path: Path):
        self.video_path = video_path
        self.records = records
        self.labels_path = labels_path
        self.labels_lock = Lock()
        self.video_lock = Lock()
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise SystemExit(f"Could not open video: {video_path}")
        self.next_video_frame = 0
        self.last_video_frame: np.ndarray | None = None
        self.frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.labels = self._read_labels()

    def _read_labels(self) -> dict[str, dict[str, object]]:
        if not self.labels_path.exists():
            return {}
        try:
            value = json.loads(self.labels_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SystemExit(f"Invalid labels JSON: {self.labels_path}: {error}") from error
        return value if isinstance(value, dict) else {}

    def record_for_frame(self, frame_index: int) -> dict[str, object] | None:
        for record in self.records:
            if int(record["frame_index"]) == frame_index:
                return record
        return None

    def frame(self, frame_index: int) -> np.ndarray:
        if frame_index < 0:
            raise ValueError(f"Invalid video frame {frame_index}")
        with self.video_lock:
            cached = self.frame_cache.get(frame_index)
            if cached is not None:
                self.frame_cache.move_to_end(frame_index)
                self.last_video_frame = cached
                return cached.copy()
            if frame_index < self.next_video_frame:
                self.capture.release()
                self.capture = cv2.VideoCapture(str(self.video_path))
                self.next_video_frame = 0
                self.last_video_frame = None
                if not self.capture.isOpened():
                    raise ValueError(f"Could not reopen video: {self.video_path}")
            while self.next_video_frame <= frame_index:
                ok, frame = self.capture.read()
                if not ok:
                    raise ValueError(
                        f"Could not read video frame {frame_index}; "
                        f"video ended at decoded frame {self.next_video_frame - 1}"
                    )
                self.frame_cache[self.next_video_frame] = frame.copy()
                self.frame_cache.move_to_end(self.next_video_frame)
                while len(self.frame_cache) > FRAME_CACHE_SIZE:
                    self.frame_cache.popitem(last=False)
                self.next_video_frame += 1
                self.last_video_frame = frame
            cached = self.frame_cache.get(frame_index)
            if cached is None:
                raise ValueError(f"Could not cache video frame {frame_index}")
            self.last_video_frame = cached
            return cached.copy()

    def save_label(self, payload: dict[str, object]) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise ValueError("label payload must be a JSON object")
        frame_index = int(payload["frame_index"])
        label = str(payload.get("label", "correct"))
        candidate_index = payload.get("candidate_index")
        if candidate_index is not None:
            candidate_index = int(candidate_index)
            record = self.record_for_frame(frame_index)
            candidates = review_candidates(record) if record is not None else []
            if not 0 <= candidate_index < len(candidates):
                raise ValueError("candidate_index is not valid for this frame")
        with self.labels_lock:
            self.labels[str(frame_index)] = {
                "frame_index": frame_index,
                "candidate_index": candidate_index,
                "label": label,
                "note": str(payload.get("note", "")),
            }
            self.labels_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.labels_path.with_suffix(self.labels_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(self.labels, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(self.labels_path)
            return self.labels[str(frame_index)]


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Turn candidate review</title>
<style>
:root { color-scheme: dark; --bg:#17191d; --panel:#22262d; --line:#3b424d; --accent:#70b7ff; }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:#e8edf3; font:14px system-ui,sans-serif; }
header { padding:12px 18px; border-bottom:1px solid var(--line); display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
button,input,textarea { background:#15181d; color:inherit; border:1px solid var(--line); border-radius:5px; padding:7px 9px; }
button { cursor:pointer; } button:hover, .candidate:hover { border-color:var(--accent); }
main { display:grid; grid-template-columns:minmax(520px, 1fr) 420px; gap:12px; padding:12px; height:calc(100vh - 62px); }
.preview { display:grid; grid-template-columns:1fr 1fr; gap:10px; min-width:0; }
.workspace { min-width:0; overflow:auto; }
.search-panel { background:var(--panel); border:1px solid var(--line); border-radius:6px; margin-top:12px; padding:12px; }
.search-panel h3 { margin:0 0 8px; font-size:15px; }
.search-panel p { margin:6px 0 10px; line-height:1.5; }
.search-panel details { margin-top:12px; }
.search-panel summary { cursor:pointer; padding:5px 0; }
.search-panel select { max-width:100%; background:var(--bg); color:inherit; padding:6px; border:1px solid var(--line); }
.search-table { overflow:auto; margin-top:8px; }
.search-panel table { width:100%; border-collapse:collapse; font-size:12px; white-space:nowrap; }
.search-panel th, .search-panel td { padding:6px 9px; border-bottom:1px solid var(--line); text-align:right; }
.search-panel th:first-child, .search-panel td:first-child { text-align:left; }
.search-panel th { color:#aeb8c4; font-weight:500; }
.search-panel tbody tr:first-child { background:#27313c; }
.search-panel button { padding:3px 6px; }
.candidate.pruned { border-left:3px solid #ba8552; }
.image-panel { background:#0d0f12; border:1px solid var(--line); border-radius:6px; padding:7px; min-width:0; }
.image-panel img { width:100%; height:auto; display:block; object-fit:contain; max-height:46vh; }
#candidateImage { cursor:crosshair; }
.image-panel h3 { margin:2px 0 7px; font-size:13px; color:#b6c0cc; }
.sidebar { background:var(--panel); border:1px solid var(--line); border-radius:6px; display:flex; flex-direction:column; min-height:0; }
.sidebar-top { padding:10px; border-bottom:1px solid var(--line); }
#candidates { overflow:auto; padding:7px; flex:1; min-height:0; }
[hidden] { display:none !important; }
details.group { border:1px solid var(--line); border-radius:5px; margin-bottom:6px; }
details.group summary { cursor:pointer; padding:8px; font-weight:600; color:#cbd5e0; }
details.group summary:hover { color:#fff; }
.group-items { padding:0 7px 1px; }
.candidate { border:1px solid var(--line); border-radius:5px; padding:7px; margin-bottom:6px; cursor:pointer; }
.candidate.selected { border-color:var(--accent); box-shadow:0 0 0 1px var(--accent) inset; }
.candidate.correct { border-color:#57d38c; } .candidate.baseline { background:#292d33; }
.candidate .title { font-weight:600; } .candidate .metrics { color:#aeb8c4; font-size:12px; margin-top:3px; }
.color-panel { border-top:1px solid var(--line); padding:8px; max-height:38vh; overflow:auto; }
.color-panel summary { cursor:pointer; color:#cbd5e0; font-weight:600; }
.color-panel table { border-collapse:collapse; font-size:11px; white-space:nowrap; margin-top:7px; }
.color-panel th, .color-panel td { border:1px solid var(--line); padding:3px 5px; text-align:right; }
.color-panel th:first-child, .color-panel td:first-child { position:sticky; left:0; background:var(--panel); text-align:left; }
.muted { color:#aeb8c4; } .status { margin-left:auto; }
#note { width:100%; min-height:56px; resize:vertical; margin-top:8px; }
.actions { display:flex; gap:6px; margin-top:8px; flex-wrap:wrap; }
.actions button { flex:1; } .primary { background:#245d91; border-color:#70b7ff; }
label { display:flex; gap:5px; align-items:center; }
@media (max-width:1000px) { main { grid-template-columns:1fr; height:auto; } .sidebar { min-height:500px; } }
</style></head><body>
<header>
  <button id="prev">← frame</button><button id="next">frame →</button>
  <label>Frame <input id="frame" type="number" min="0" style="width:90px"></label>
  <button id="source">source</button><button id="selectNext" class="primary">save correct + next</button>
  <span id="status" class="status muted"></span>
</header>
<main>
  <div class="workspace">
  <section class="preview">
    <div class="image-panel"><h3>Source frame</h3><img id="sourceImage"></div>
    <div class="image-panel"><h3 id="candidateTitle">Candidate</h3><img id="candidateImage"></div>
  </section>
  <section class="search-panel" aria-label="Search diagnostics">
    <h3>Search decision</h3>
    <div id="searchSummary" class="muted"></div>
    <p id="searchComparison"></p>
    <label id="searchPathLabel">Forecast <select id="searchPath" aria-label="Forecast"></select></label>
    <p id="searchPathNote" class="muted"></p>
    <div id="searchSteps" class="search-table"></div>
    <details id="searchDepthPanel"><summary>Search expansion and pruning at each frame</summary>
      <p class="muted">Counts describe path expansions. Adaptive search keeps the best hold and top changes per parent using the logged shortlist metric, merges prefixes by frame and cube state, and uses bounds to prune the shortlisted paths. GPU batch size is not a state cutoff. Older logs may use beam or history restrictions. Future choices are forecasts from this frame.</p>
      <div id="searchDepths" class="search-table"></div>
    </details>
    <details id="searchGeometryPanel"><summary>Stage 1: retained geometric fits</summary>
      <p id="searchGeometryNote" class="muted"></p>
      <div id="searchGeometry" class="search-table"></div>
    </details>
  </section>
  </div>
  <aside class="sidebar">
    <div class="sidebar-top">
      <div id="frameInfo" class="muted"></div>
      <div id="layerInfo" class="muted">Click a layer; double-click for two layers. Use , / . to turn.</div>
      <div class="actions"><button id="correct" class="primary">Correct (C)</button><button id="uncertain">Uncertain (U)</button><button id="invalid">No valid pose (N)</button></div>
      <textarea id="note" placeholder="Optional note about the scoring failure..."></textarea>
    </div>
    <div id="candidates"></div>
    <details id="colorPanel" class="color-panel"><summary>Color evidence by sticker identity</summary><div id="colorTable" class="muted">No color evidence recorded for this frame.</div></details>
  </aside>
</main>
<script>
let meta, frameIndex=0, candidateIndex=null, current, candidates=[], faceRegions=[], activeLayer=null, loadVersion=0, clickTimer=null;
const $ = id => document.getElementById(id);
async function getJson(url, options) { const r=await fetch(url, options); if(!r.ok) throw new Error(await r.text()); return r.json(); }
function record() { return meta.frames.find(x => x.frame_index === frameIndex); }
function imageUrl(index) { return `/api/image?frame=${frameIndex}&candidate=${index}&v=${Date.now()}`; }
function candidateGroupKey(candidate) {
  if(candidate.label.includes('turn search not tested')) return 'static-only';
  const axis=Number.isInteger(candidate.layer_axis) ? candidate.layer_axis : null;
  const side=Number.isInteger(candidate.layer_side) ? candidate.layer_side : null;
  const width=Number.isInteger(candidate.layer_count) ? candidate.layer_count : null;
  if(axis!==null && side!==null && width!==null) return `${axis}:${side}:${width}`;
  const rotation=(candidate.layer_rotations||[])[0]||{};
  return `${rotation.axis??'x'}:${rotation.side??'x'}:${rotation.layer_count??'x'}`;
}
function groupTitle(key) {
  if(key==='static-only') return 'Static pose only';
  const [axis,side,width]=key.split(':').map(Number);
  const axisName=['X','Y','Z'][axis]||'?';
  const sideName=side>0?'+':'−';
  return `${axisName}${sideName} · ${width===1?'single':'double'} layer`;
}
function groupSortKey(items) {
  const finite=value=>typeof value==='number' && Number.isFinite(value);
  const sequenceScores=items.map(item=>item.candidate.best_sequence_score).filter(finite);
  if(sequenceScores.length)return [Math.min(...sequenceScores),0];
  if(items.some(item=>item.candidate.search_status))return [Infinity,0];
  const remainderScores=items.map(item=>item.candidate.remainder_score).filter(finite);
  if(remainderScores.length) {
    return [-Math.max(...remainderScores), Math.min(...items.map(item=>item.candidate.global_mean_error).filter(finite))];
  }
  const meanScores=items.map(item=>item.candidate.global_mean_error).filter(finite);
  return [meanScores.length ? Math.min(...meanScores) : Infinity, 0];
}
function compareGroups(first, second, groupsForSort) {
  const a=groupSortKey(groupsForSort.get(first)||[]), b=groupSortKey(groupsForSort.get(second)||[]);
  return a[0]-b[0] || a[1]-b[1] || first.localeCompare(second);
}
function candidateAngle(candidate) {
  if(typeof candidate.color_offset==='number') {
    const rotation=(candidate.layer_rotations||[])[0];
    return (rotation?.angle_degrees||0)+candidate.color_offset;
  }
  if(candidate.is_baseline) return 0;
  const rotations=candidate.layer_rotations||[];
  return rotations.length ? Number(rotations[rotations.length-1].angle_degrees) : null;
}
function selectCandidate(index) { candidateIndex=index; loadFrame(); }
function framePosition() { return meta.frames.findIndex(item => item.frame_index === frameIndex); }
function setFrameFromInput(value) {
  if(!meta.frames.length) return;
  const requested=Number(value);
  const exact=Number.isFinite(requested) ? meta.frames.findIndex(item => item.frame_index === requested) : -1;
  const position=exact>=0 ? exact : meta.frames.reduce((best,item,index)=>
    Math.abs(item.frame_index-requested)<Math.abs(meta.frames[best].frame_index-requested) ? index : best, 0);
  frameIndex=meta.frames[position].frame_index; candidateIndex=null; activeLayer=null; loadFrame();
}
function chooseLayer(axis,side,width) {
  const key=`${axis}:${side}:${width}`;
  const available=candidates.map((candidate,index)=>({candidate,index,angle:candidateAngle(candidate)}))
    .filter(item=>candidateGroupKey(item.candidate)===key && Number.isFinite(item.angle));
  if(!available.length) { $('layerInfo').textContent=`${groupTitle(key)}: no configurations in this log`; return; }
  const old=activeLayer && activeLayer.axis===axis && activeLayer.side===side ? candidateAngle(candidates[candidateIndex]) : 0;
  activeLayer={axis,side,width};
  available.sort((a,b)=>Math.abs(a.angle-(old??0))-Math.abs(b.angle-(old??0)) || a.index-b.index);
  selectCandidate(available[0].index);
}
function stepAngle(direction) {
  if(!activeLayer) { $('layerInfo').textContent='Click a layer first to adjust its angle.'; return; }
  const key=`${activeLayer.axis}:${activeLayer.side}:${activeLayer.width}`;
  const available=candidates.map((candidate,index)=>({candidate,index,angle:candidateAngle(candidate)}))
    .filter(item=>candidateGroupKey(item.candidate)===key && Number.isFinite(item.angle))
    .sort((a,b)=>a.angle-b.angle || a.index-b.index);
  const currentAngle=candidateAngle(candidates[candidateIndex]);
  const next=direction>0 ? available.find(item=>item.angle>currentAngle+1e-6) : [...available].reverse().find(item=>item.angle<currentAngle-1e-6);
  if(next) selectCandidate(next.index);
}
function insidePolygon(point, polygon) {
  let inside=false;
  for(let i=0,j=polygon.length-1;i<polygon.length;j=i++) {
    const a=polygon[i], b=polygon[j];
    if((a[1]>point[1])!==(b[1]>point[1]) && point[0]<(b[0]-a[0])*(point[1]-a[1])/(b[1]-a[1])+a[0]) inside=!inside;
  }
  return inside;
}
function segmentDistance(point,a,b) {
  const dx=b[0]-a[0], dy=b[1]-a[1], length=dx*dx+dy*dy;
  const t=length ? Math.max(0,Math.min(1,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/length)) : 0;
  return Math.hypot(point[0]-a[0]-t*dx,point[1]-a[1]-t*dy);
}
function formatProbability(value) {
  const number=Number(value);
  return Number.isFinite(number) ? number.toExponential(3) : '—';
}
function colorAgreementText(candidate) {
  const total=candidate.color_observed_count??candidate.color_match_count;
  const error=candidate.color_error;
  if(typeof total!=='number' || total<=0 || typeof error!=='number' || !Number.isFinite(error)) return 'not scored';
  const correct=candidate.color_correct_count??Math.round((1-error)*total);
  return `${correct}/${total} correct (${(100*correct/total).toFixed(1)}%)${typeof candidate.color_confusion_count==='number'?` · ${candidate.color_confusion_count} allowed confusions · ${candidate.color_forbidden_count??0} unlisted`:''}`;
}
function renderColorLikelihoods(data, candidate) {
  const root=$('colorTable'); root.innerHTML='';
  const direct=['direct_match_fraction','directional_confusion_cost'].includes(data.color_scoring);
  const formatEvidence=value=>direct ? (value==null?'—':String(Number(value))) : formatProbability(value);
  const perCandidate=candidate?.color_matches_by_identity??candidate?.color_likelihoods_by_identity;
  if(perCandidate && typeof perCandidate==='object' && !Array.isArray(perCandidate) && Object.keys(perCandidate).length) {
    const table=document.createElement('table');
    const head=document.createElement('tr');
    const identityHead=document.createElement('th'); identityHead.textContent='identity'; head.appendChild(identityHead);
    const probabilityHead=document.createElement('th'); probabilityHead.textContent=direct?'color match':'p(observation | prior)'; head.appendChild(probabilityHead);
    table.appendChild(head);
    Object.entries(perCandidate).sort(([first],[second])=>first.localeCompare(second)).forEach(([label,value])=>{
      const row=document.createElement('tr');
      const identity=document.createElement('td'); identity.textContent=String(label); row.appendChild(identity);
      const probability=document.createElement('td'); probability.textContent=formatEvidence(value); row.appendChild(probability);
      table.appendChild(row);
    });
    root.appendChild(table);
    return;
  }
  const labels=data.color_identity_labels||[], matrix=data.color_matches||data.color_likelihoods||[];
  const centers=data.color_observation_centers||[], names=data.color_observation_names||[];
  if(!labels.length || !matrix.length) {
    root.textContent=names.length ? `Observed color keypoints (no color score): ${names.map((name,index)=>`o${index} ${name}`).join(', ')}` : 'No color evidence recorded for this frame.';
    return;
  }
  if(direct) {
    const note=document.createElement('div'); note.className='muted';
    note.textContent='1 = matching color, 0 = different color. One vote per observed projected sticker; any matching duplicate wins. Points outside the projected cube are excluded.';
    root.appendChild(note);
  }
  const columnCount=Math.max(0,...matrix.map(row=>Array.isArray(row)?row.length:0));
  if(!columnCount) { root.textContent='No observed sticker colors recorded for this frame.'; return; }
  const table=document.createElement('table');
  const head=document.createElement('tr');
  const identityHead=document.createElement('th'); identityHead.textContent='identity'; head.appendChild(identityHead);
  for(let column=0; column<columnCount; column++) {
    const cell=document.createElement('th');
    const center=centers[column];
    cell.textContent=`o${column}${names[column]?` ${names[column]}`:''}${Array.isArray(center)?` (${center.map(value=>Math.round(Number(value))).join(',')})`:''}`;
    head.appendChild(cell);
  }
  table.appendChild(head);
  labels.forEach((label,rowIndex)=>{
    const row=document.createElement('tr');
    const identity=document.createElement('td'); identity.textContent=String(label); row.appendChild(identity);
    for(let column=0; column<columnCount; column++) {
      const cell=document.createElement('td'); cell.textContent=formatEvidence(matrix[rowIndex]?.[column]); row.appendChild(cell);
    }
    table.appendChild(row);
  });
  root.appendChild(table);
}
function layerAt(event,width) {
  const img=$('candidateImage'), rect=img.getBoundingClientRect();
  const scale=Math.min(rect.width/img.naturalWidth,rect.height/img.naturalHeight);
  const offsetX=(rect.width-img.naturalWidth*scale)/2, offsetY=(rect.height-img.naturalHeight*scale)/2;
  const point=[(event.clientX-rect.left-offsetX)/scale,(event.clientY-rect.top-offsetY)/scale];
  const face=faceRegions.filter(item=>insidePolygon(point,item.polygon)).sort((a,b)=>a.depth-b.depth)[0];
  if(!face) { $('layerInfo').textContent='Click inside a visible projected cube sticker.'; return; }
  const edge=face.edges.map(item=>({...item,distance:segmentDistance(point,face.polygon[item.start],face.polygon[item.end])})).sort((a,b)=>a.distance-b.distance)[0];
  chooseLayer(edge ? edge.axis : face.axis, edge ? edge.side : face.side, width);
}
function scoreText(value) {
  return typeof value==='number' && Number.isFinite(value) ? value.toFixed(4) : '—';
}
function moveText(trial) { return (trial?.state_moves||[]).join(' ') || 'hold'; }
const searchStatusLabels={invalid_color:'invalid: nonfinite color fit', budget_unfinished:'unfinished: state budget reached', color_ceiling:'rejected: absolute color ceiling', color_pruned:'excluded: color shortlist', shortlist_pruned:'excluded: geometry + color shortlist', complete:'complete path found', bound_pruned:'pruned: cannot beat complete solution', survived:'survived', immediate_undo:'pruned: immediate undo',
  root_return:'pruned: returned to window-start state',
  state_revisit:'pruned: earlier cube state revisited',
  invalid_score:'pruned: invalid fit', merged:'merged into a cheaper equivalent path',
  beam_pruned:'pruned: beam limit', no_legal_continuation:'no surviving continuation', eligible:'eligible'};
function appendSearchTable(root, headers, rows) {
  root.replaceChildren();
  const table=document.createElement('table'), head=document.createElement('thead'), body=document.createElement('tbody');
  const header=document.createElement('tr');
  headers.forEach(text=>{const cell=document.createElement('th');cell.textContent=text;header.appendChild(cell);});
  head.appendChild(header);table.appendChild(head);
  rows.forEach(values=>{
    const row=document.createElement('tr');
    values.forEach(value=>{const cell=document.createElement('td');if(value instanceof Node)cell.appendChild(value);else cell.textContent=String(value??'—');row.appendChild(cell);});
    body.appendChild(row);
  });
  table.appendChild(body);root.appendChild(table);
}
function reviewSearchFrame(frame) {
  if(!meta.frames.some(item=>item.frame_index===frame))return;
  frameIndex=frame; candidateIndex=null; activeLayer=null; loadFrame();
}
function searchForecasts(search) {
  const paths=[...(search.finalists||[])];
  if(search.no_turn_path)paths.push({...search.no_turn_path,comparison_kind:'no_turn'});
  return paths;
}
function renderSearchPath(search,index) {
  const path=searchForecasts(search)[index];
  if(!path){$('searchSteps').replaceChildren();return;}
  const rows=(path.steps||[]).map(step=>{
    const button=document.createElement('button');button.textContent=String(step.frame_index);
    button.title='Review this frame and its own recorded decision';button.onclick=()=>reviewSearchFrame(step.frame_index);
    return [button,moveText(step),(step.layer_rotations||[]).map(r=>`${r.angle_degrees}°`).join(', ')||'0°',
      `${step.color_offset??0}°`,scoreText(step.state_score),scoreText(step.color_error),
      `${(100*(step.weight??0)).toLocaleString(undefined,{maximumFractionDigits:2})}%`,scoreText(step.move_cost??0),scoreText(step.weighted_score)];
  });
  appendSearchTable($('searchSteps'),['Frame','Move','Partial angle','Offset','Frame fit','Color','Weight','Move cost','Contribution'],rows);
  $('searchPathNote').textContent=`Sequence ${scoreText(path.total_score)}${path.observation_score!=null?` (fit ${scoreText(path.observation_score)} + move cost ${scoreText(path.move_cost)})`:''} · current ${scoreText(path.current_score)} · future mean ${scoreText(path.future_mean)}. Forecast recorded at frame ${frameIndex}; frame buttons open that frame’s own decision.`;
}
function renderSearchDiagnostics(search,candidate) {
  const hasSearch=search && typeof search==='object';
  $('searchPathLabel').hidden=!hasSearch || !search.finalists?.length;
  $('searchDepthPanel').hidden=!hasSearch || !search.depths?.length;
  $('searchGeometryPanel').hidden=!hasSearch || !search.geometry?.options?.length;
  $('searchPathNote').textContent='';$('searchSteps').replaceChildren();$('searchDepths').replaceChildren();
  if(!hasSearch){
    $('searchSummary').textContent='No lookahead trace in this record. Cached video recognition with --turn-candidate-log records forecasts and pruning.';
    $('searchComparison').textContent='Frame-level candidates from older logs and live searches remain available.';return;
  }
  if(search.status==='forced_static' && search.reason==='turning_start_frame'){
    $('searchSummary').textContent=`Static hold forced through frame ${search.turning_start_frame}, inclusive (--turning-start-frame).`;
    $('searchComparison').textContent='Move search skipped: no layer offsets or cube-state changes.';return;
  }
  const elapsed=typeof search.elapsed_ms==='number'?`${search.elapsed_ms.toFixed(1)} ms`:'timing unavailable';
  const skipped=search.skipped_frames?.length?` · skipped ${search.skipped_frames.join(', ')}`:'';
  const adaptive=search.algorithm==='adaptive_branch_and_bound';
  const shortlistMetric=search.shortlist_rank==='geometry_plus_color'?'geometry + color':'color';
  const method=adaptive?`adaptive · ${search.state_change_limit!=null?`top ${search.state_change_limit} ${shortlistMetric} changes per parent + hold`:'all state changes'} · GPU batch ${search.state_batch_size??'—'} · no beam cutoff`:`beam ${search.beam_width??'—'}`;
  $('searchSummary').textContent=`Previous move: ${search.previous_move||'none'} · ${method} · ${Math.max(0,(search.frames?.length||1)-1)} future frames · ${elapsed}${skipped}`;
  if(adaptive)$('searchSummary').textContent+=` · ${search.optimality==='proven'?'optimal for scored options':search.optimality==='proven_within_shortlist'?`optimal within ${shortlistMetric} shortlist`:search.optimality||'search incomplete'} · peak queue ${search.peak_queued_paths??'—'}`;
  if(search.state_evaluation_budget!=null)$('searchSummary').textContent+=` · state budget ${search.decision_evaluated_states??search.evaluated_states??'—'}/${search.state_evaluation_budget}`;
  if(search.budget_exhausted)$('searchSummary').textContent+=search.selected_path?' · budget reached; best complete forecast, optimality not proven':' · budget reached before finding a complete forecast';
  if(search.optimality_gap!=null)$('searchSummary').textContent+=` · bound gap ${scoreText(search.optimality_gap)}`;
  if(search.seeds?.length)$('searchSummary').textContent+=` · seeds ${search.seeds.map(seed=>`${seed.source}: ${seed.status}${seed.score!=null?` (${scoreText(seed.score)})`:''}${seed.root_return_pruned?` [${seed.root_return_pruned} root returns excluded]`:''}`).join(', ')}`;
  if(search.projection_cache)$('searchSummary').textContent+=` · assignments ${search.projection_cache.assignment_builds} built / ${search.projection_cache.assignment_hits} reused`;
  if(search.unreliable_color_frames?.length)$('searchSummary').textContent+=` · unreliable future color omitted at ${search.unreliable_color_frames.join(', ')}`;
  if(search.color_error_ceiling_enabled===false)$('searchSummary').textContent+=' · color ceiling disabled';
  else if(search.color_error_ceiling!=null)$('searchSummary').textContent+=` · color ceiling ${scoreText(search.color_error_ceiling)}`;
  if(search.prune_root_state_returns)$('searchSummary').textContent+=' · returns to window-start state pruned';
  else if(search.prune_immediate_undo===false && search.prune_state_revisits===false)$('searchSummary').textContent+=' · inverse moves and state revisits allowed';
  if(search.prune_state_revisits)$('searchSummary').textContent+=` · state-cycle guard (${search.recent_committed_states??0} recent states)`;
  if(search.evaluation_cache){const cache=search.evaluation_cache;
    $('searchSummary').textContent+=` · ${cache.states_scored} states scored / ${cache.cache_hits} cached · ${cache.gpu_batches} GPU batches`;}
  if(search.status==='fallback') {
    const reasons={current_frame_unavailable:'current-frame trials unavailable',no_surviving_path:'no path survived the horizon',state_budget_no_complete_path:'state budget reached before finding a complete forecast'};
    $('searchComparison').textContent=`Fallback: ${reasons[search.fallback_reason]||search.fallback_reason||'search unavailable'}${search.fallback_undo_blocked?' · local undo blocked; held the previous state':''}${search.fallback_state_revisit_blocked?' · state revisit blocked; held the previous state':''}${search.color_evidence_unreliable?' · unreliable color evidence; state held':''}.`;
  } else {
    const roots=search.roots||[], chosen=roots.find(root=>root.trial_id===search.selected_trial_id), local=roots.find(root=>root.trial_id===search.local_best_trial_id);
    $('searchComparison').textContent=`Chosen ${chosen?.trial?.label||search.selected_trial_id||'—'}: sequence ${scoreText(search.selected_path?.total_score)}. ${search.changed_local_choice?`Lookahead changed the current-frame winner (${local?.trial?.label||search.local_best_trial_id||'—'}, fit ${scoreText(local?.trial?.state_score)}).`:'Lookahead kept the current-frame winner.'}`;
  }
  if(search.color_failure_retries?.length)$('searchComparison').textContent+=` Omitted future color evidence: ${search.color_failure_retries.map(item=>`${item.frame_index} (${item.rejected_candidates??'—'} ceiling rejections)`).join(', ')}.`;
  if(search.no_turn_path)$('searchComparison').textContent+=` Complete no-turn forecast ${scoreText(search.no_turn_path.total_score)}; chosen advantage ${scoreText(search.improvement_over_no_turn)} after move costs.`;
  const finals=searchForecasts(search), select=$('searchPath');select.replaceChildren();
  finals.forEach((path,index)=>{const option=document.createElement('option');option.value=index;
    option.textContent=`${path.comparison_kind==='no_turn'?'Complete no-turn continuation':index===0?'Winner':`Alternative ${index}`} · ${moveText(path.steps?.[0])} · ${path.steps?.[0]?.label||path.trial_id||''} · ${scoreText(path.total_score)}`;select.appendChild(option);});
  let pathIndex=finals.findIndex(path=>path.trial_id===candidate?.trial_id);if(pathIndex<0)pathIndex=0;
  select.value=pathIndex;select.onchange=()=>renderSearchPath(search,Number(select.value));
  renderSearchPath(search,pathIndex);
  if(candidate?.search_status && !finals.some(path=>path.trial_id===candidate.trial_id))
    $('searchPathNote').textContent=`Inspected candidate: ${searchStatusLabels[candidate.search_status]||candidate.search_status}. No saved finalist forecast for it. `+$('searchPathNote').textContent;
  if(adaptive) {
    appendSearchTable($('searchDepths'),['Frame','Parents expanded','States scored','Trials','Collapsed holds','Root returns pruned','Shortlist excluded',search.version>=10?'Nonfinite color':'Ceiling rejected','Unreliable states','Merged prefixes','Bound pruned','Completed','Queued','Future lower bound'],
      (search.depths||[]).map(depth=>[depth.frame_index,depth.input_paths,depth.scored_states,depth.attempted_trials,
        depth.collapsed_holds,depth.root_return??0,depth.shortlist_pruned_paths??depth.color_pruned_paths??0,depth.invalid_color??depth.color_ceiling_rejected??0,depth.unreliable_color_states??0,depth.merged_paths,depth.bound_pruned_paths,depth.completed_paths,depth.queued_paths,scoreText(depth.remaining_lower_bound)]));
  } else {
    appendSearchTable($('searchDepths'),['Frame','Input paths','States evaluated','Trials','Undo','State revisit','Invalid','Unavailable','Merged','Beam pruned','Kept','Distinct states','No-turn extra'],
      (search.depths||[]).map(depth=>[depth.frame_index,depth.input_paths,depth.scored_states,depth.attempted_trials,
        depth.immediate_undo,depth.state_revisit??'—',depth.invalid_score,depth.unavailable,depth.merged_paths,depth.beam_pruned_paths,depth.retained_paths,depth.distinct_states,depth.no_turn_extra??'—']));
  }
  const geometry=search.geometry;
  if(geometry){
    $('searchGeometryNote').textContent=`${geometry.tested_configurations} configurations tested; ${geometry.retained_options} partial fits retained. Stage 2 tests three sticker offsets on each retained fit.`;
    appendSearchTable($('searchGeometry'),['Layer','Partial angle','Geometry fit','Keypoints','Centroids','Inliers'],
      (geometry.options||[]).map(trial=>[trial.label,(trial.layer_rotations||[]).map(r=>`${r.angle_degrees}°`).join(', ')||'0°',
        scoreText(trial.global_mean_error),scoreText(trial.keypoint_mean_error),scoreText(trial.heatmap_mean_error),trial.inlier_count]));
  }
}

async function loadFrame() {
  const version=++loadVersion;
  current=record(); if(!current) return;
  const data=await getJson(`/api/candidates?frame=${frameIndex}`);
  if(version!==loadVersion) return;
  candidates=data.candidates;
  const count=data.candidates.length;
  if(candidateIndex===null) candidateIndex=current.selected_index ?? 0;
  candidateIndex=Math.max(0, Math.min(candidateIndex, Math.max(0,count-1)));
  renderColorLikelihoods(data, candidates[candidateIndex]);
  renderSearchDiagnostics(data.search,candidates[candidateIndex]);
  if(data.color_evidence_unreliable && !data.search)$('searchComparison').textContent='Unreliable color evidence; state held.';
  $('frame').value=frameIndex;
  $('sourceImage').onerror=()=>$('status').textContent=`Could not decode video frame ${frameIndex}; check the source video.`;
  $('candidateImage').onerror=()=>$('status').textContent=`Could not decode video frame ${frameIndex}; check the source video.`;
  $('sourceImage').src=imageUrl(-1);
  $('candidateImage').src=imageUrl(candidateIndex);
  faceRegions=[];
  if(count) getJson(`/api/face-regions?frame=${frameIndex}&candidate=${candidateIndex}`).then(data=>{if(version===loadVersion) faceRegions=data.faces;});
  $('frameInfo').textContent=`frame ${frameIndex} · ${count} tested configurations · selected: ${current.selected || 'none'}`;
  const staticOnly=count===1 && data.candidates[0].label.includes('turn search not tested');
  const selectedCandidate=data.candidates[current.selected_index];
  const selectionLabel=selectedCandidate?`${moveText(selectedCandidate)} · ${selectedCandidate.label}`:(current.selected||'none');
  $('frameInfo').textContent=staticOnly
    ? `frame ${frameIndex} · turn search not tested · showing static pose`
    : `frame ${frameIndex} · ${count} tested configurations · ${data.search?'selection uses sequence score':'current-frame scores'} · selected: ${selectionLabel}`;
  if(typeof data.search?.sticker_mismatch_count==='number')
    $('frameInfo').textContent=`frame ${frameIndex} · mismatched stickers ${data.search.sticker_mismatch_count}/54 · ${data.search.sticker_history_qualified_count??0} stickers have 4 observations · selected: ${selectionLabel}`;
  if(typeof data.search?.sticker_history_penalty==='number')
    $('frameInfo').textContent+=` · history penalty ${scoreText(data.search.sticker_history_penalty)} · queue ${scoreText(data.search.queue_priority??data.search.inherited_queue_priority)}`;
  if(data.search?.soft_window_penalty)
    $('frameInfo').textContent+=` · old-fork penalty ${scoreText(data.search.soft_window_penalty)}`;
  if(typeof data.search?.ilds_budget==='number')
    $('frameInfo').textContent+=` · ILDS ${data.search.ilds_discrepancies_used}/${data.search.ilds_budget} discrepancies · choice ${data.search.ilds_choice_index}`;
  if(typeof data.search?.compensatory_beam_id==='number')
    $('frameInfo').textContent+=` · local beam ${data.search.compensatory_beam_id} · frame ${data.search.compensatory_beam_level} (${data.search.compensatory_beam_phase}) · window ${data.search.compensatory_beam_start_frame}–${data.search.compensatory_beam_end_frame}`;
  $('candidateTitle').textContent=staticOnly ? 'Static pose (turn search not tested)' : (count ? `Candidate ${candidateIndex+1}/${count}` : 'No tested candidate');
  $('layerInfo').textContent=activeLayer ? `${groupTitle(`${activeLayer.axis}:${activeLayer.side}:${activeLayer.width}`)} · sticker angle ${candidateAngle(candidates[candidateIndex])}° · , / . to adjust` : 'Click a layer; double-click for two layers. Use , / . to turn.';
  $('correct').disabled=!count; $('selectNext').disabled=!count;
  $('status').textContent=meta.labels[String(frameIndex)] ? `saved: ${meta.labels[String(frameIndex)].label}` : 'unlabeled';
  const list=$('candidates'); list.innerHTML='';
  const groups=new Map();
  data.candidates.forEach((candidate,index)=>{
    const key=candidateGroupKey(candidate);
    if(!groups.has(key)) groups.set(key,[]);
    groups.get(key).push({candidate,index});
  });
  const orderedKeys=[...groups.keys()].sort((first, second)=>compareGroups(first, second, groups));
  orderedKeys.forEach((key)=>{
    const details=document.createElement('details'); details.className='group';
    details.open=(groups.get(key)||[]).some(item=>item.index===candidateIndex);
    const summary=document.createElement('summary'); const items=groups.get(key)||[];
    summary.textContent=`${groupTitle(key)} (${items.length})`; details.appendChild(summary);
    const itemContainer=document.createElement('div'); itemContainer.className='group-items';
    if(!items.length){ const empty=document.createElement('div'); empty.className='muted'; empty.style.padding='5px 0 9px'; empty.textContent='No configuration tested'; itemContainer.appendChild(empty); }
    items.forEach(({candidate:c,index:i})=>{
      const el=document.createElement('div'); el.className='candidate'+(i===candidateIndex?' selected':'')+(c.is_baseline?' baseline':'')+(c.search_status && !['survived','complete','budget_unfinished'].includes(c.search_status)?' pruned':'');
      const title=document.createElement('div');title.className='title';
      title.textContent=`${i}: ${c.label}${i===current.selected_index?' · renderer selection':''}`;
      const metrics=document.createElement('div');metrics.className='metrics';
      metrics.textContent=`frame fit ${scoreText(c.global_mean_error)} · kp ${scoreText(c.keypoint_mean_error)} · sticker points ${scoreText(c.heatmap_mean_error)} · ${['direct_match_fraction','directional_confusion_cost'].includes(data.color_scoring)?`colors ${colorAgreementText(c)} · mismatch ${scoreText(c.color_error)}`:`color ${scoreText(c.color_error??c.color_mean_error)} · matches ${c.color_match_count??'—'}`} · inliers ${c.inlier_count??'—'}`;
      if(typeof c.sticker_mismatch_count==='number')metrics.textContent+=` · mismatched stickers ${c.sticker_mismatch_count}/54`;
      if(typeof c.sticker_history_penalty==='number')metrics.textContent+=` · history penalty ${scoreText(c.sticker_history_penalty)} · queue ${scoreText(c.queue_priority??c.inherited_queue_priority)}`;
      if(c.soft_window_penalty)metrics.textContent+=` · old-fork penalty ${scoreText(c.soft_window_penalty)}`;
      el.append(title,metrics);
      if(c.search_status){const state=document.createElement('div');state.className='metrics';
        state.textContent=`${moveText(c)} · ${searchStatusLabels[c.search_status]||c.search_status}${typeof c.best_sequence_score==='number'?` · best sequence ${scoreText(c.best_sequence_score)}`:''}${c.eliminated_depth!=null?` at depth ${c.eliminated_depth}`:''}`;
        if(c.pruning_counts?.immediate_undo)state.textContent+=` · ${c.pruning_counts.immediate_undo} undo branches pruned`;
        if(c.pruning_counts?.state_revisit)state.textContent+=` · ${c.pruning_counts.state_revisit} state cycles pruned`;
        if(c.pruning_counts?.root_return)state.textContent+=` · ${c.pruning_counts.root_return} returns to window-start state pruned`;
        if(c.root_return_examples?.length){const example=c.root_return_examples[0];
          state.textContent+=` · root-return example at depth ${example.depth}: ${moveText(example)}`;}
        if(c.invalid_color)state.textContent+=' · nonfinite color fit';
        if(c.pruning_counts?.invalid_color)state.textContent+=` · ${c.pruning_counts.invalid_color} nonfinite color fits`;
        if(c.color_error_ceiling_rejected)state.textContent+=' · above color ceiling';
        if(c.pruning_counts?.color_ceiling_rejected)state.textContent+=` · ${c.pruning_counts.color_ceiling_rejected} candidates rejected by color ceiling`;
        if(c.pruning_counts?.shortlist_pruned_paths)state.textContent+=` · ${c.pruning_counts.shortlist_pruned_paths} changes excluded by geometry + color shortlist`;
        if(c.pruning_counts?.color_pruned_paths)state.textContent+=` · ${c.pruning_counts.color_pruned_paths} changes excluded by color shortlist`;
        if(c.pruning_counts?.bound_pruned_paths)state.textContent+=` · ${c.pruning_counts.bound_pruned_paths} prefixes excluded by bounds`;
        if(c.pruning_examples?.length){const example=c.pruning_examples[0];
          state.textContent+=` · example at depth ${example.depth}: lower bound ${scoreText(example.lower_bound)} vs best complete ${scoreText(example.incumbent_score)}`;}
        if(c.pruning_counts?.invalid_score)state.textContent+=` · ${c.pruning_counts.invalid_score} invalid fits pruned`;
        state.title=JSON.stringify(c.pruning_counts||{});el.appendChild(state);}
      el.onclick=()=>{ const parts=key.split(':').map(Number); activeLayer=parts.length===3 && parts.every(Number.isFinite) ? {axis:parts[0],side:parts[1],width:parts[2]} : null; selectCandidate(i); }; itemContainer.appendChild(el);
    });
    details.appendChild(itemContainer); list.appendChild(details);
  });
}
function moveFrame(delta) {
  if(!meta.frames.length) return;
  const currentPosition=framePosition();
  const position=Math.max(0,Math.min(meta.frames.length-1,(currentPosition<0?0:currentPosition)+delta));
  if(position===currentPosition) return;
  frameIndex=meta.frames[position].frame_index; candidateIndex=null; activeLayer=null; loadFrame();
}
async function label(kind, advance=false) {
  const payload={frame_index:frameIndex,candidate_index:kind==='correct'&&current.candidate_count?candidateIndex:null,label:kind,note:$('note').value};
  const saved=await getJson('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  meta.labels[String(frameIndex)]=saved; $('status').textContent=`saved: ${kind}`; if(advance) moveFrame(1);
}
$('prev').onclick=()=>moveFrame(-1); $('next').onclick=()=>moveFrame(1); $('source').onclick=()=>{$('candidateImage').src=imageUrl(-1);};
$('correct').onclick=()=>label('correct'); $('uncertain').onclick=()=>label('uncertain'); $('invalid').onclick=()=>label('no_valid_pose'); $('selectNext').onclick=()=>label('correct',true);
$('frame').onchange=e=>setFrameFromInput(e.target.value);
$('candidateImage').addEventListener('click',e=>{ if(e.detail!==1)return; clearTimeout(clickTimer); clickTimer=setTimeout(()=>layerAt(e,1),250); });
$('candidateImage').addEventListener('dblclick',e=>{ e.preventDefault(); clearTimeout(clickTimer); layerAt(e,2); });
document.addEventListener('keydown',e=>{ if(['INPUT','TEXTAREA'].includes(document.activeElement.tagName)) return; if(e.key===','){e.preventDefault();stepAngle(-1);} if(e.key==='.'){e.preventDefault();stepAngle(1);} if(e.key==='ArrowLeft')moveFrame(-1); if(e.key==='ArrowRight')moveFrame(1); if(e.key==='ArrowUp'){candidateIndex=Math.max(0,candidateIndex-1);activeLayer=null;loadFrame();} if(e.key==='ArrowDown'){candidateIndex++;activeLayer=null;loadFrame();} if(e.key.toLowerCase()==='c')label('correct'); if(e.key.toLowerCase()==='u')label('uncertain'); if(e.key.toLowerCase()==='n')label('no_valid_pose'); });
(async()=>{ meta=await getJson('/api/meta'); frameIndex=meta.frames[0]?.frame_index||0; loadFrame(); })().catch(e=>{$('status').textContent=e;});
</script></body></html>"""


class ReviewHandler(BaseHTTPRequestHandler):
    state: ReviewState

    def _send_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(json_safe(value)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/meta":
            frames = [
                {
                    "frame_index": int(record["frame_index"]),
                    "candidate_count": len(review_candidates(record)),
                    "selected": record.get("selected"),
                    "selected_index": selected_candidate_index(record),
                    "has_pose": bool(record.get("has_pose", False)),
                }
                for record in self.state.records
            ]
            self._send_json(
                {
                    "video": str(self.state.video_path),
                    "labels_path": str(self.state.labels_path),
                    "frames": frames,
                    "labels": self.state.labels,
                }
            )
            return
        if parsed.path == "/api/candidates":
            frame_index = int(query.get("frame", [0])[0])
            record = self.state.record_for_frame(frame_index)
            payload = color_likelihood_payload(record) if record is not None else {
                "color_identity_labels": [],
                "color_observation_centers": [],
                "color_likelihoods": [],
            }
            payload["candidates"] = [] if record is None else review_candidates(record)
            payload["search"] = search_debug_payload(record)
            self._send_json(
                payload
            )
            return
        if parsed.path == "/api/face-regions":
            frame_index = int(query.get("frame", [0])[0])
            candidate_index = int(query.get("candidate", [0])[0])
            record = self.state.record_for_frame(frame_index)
            candidates = review_candidates(record) if record is not None else []
            pose = (
                pose_for_candidate(record, candidate_index)
                if record is not None
                else None
            )
            if not isinstance(pose, dict) or not 0 <= candidate_index < len(candidates):
                self._send_json({"faces": []})
                return
            try:
                faces = projected_face_regions(
                    pose, candidates[candidate_index].get("layer_rotations", [])
                )
            except (cv2.error, KeyError, TypeError, ValueError):
                faces = []
            self._send_json({"faces": faces})
            return
        if parsed.path == "/api/image":
            frame_index = int(query.get("frame", [0])[0])
            candidate_index = int(query.get("candidate", [-1])[0])
            record = self.state.record_for_frame(frame_index)
            if record is None:
                self._send_json({"error": "unknown frame"}, HTTPStatus.NOT_FOUND)
                return
            try:
                frame = self.state.frame(frame_index)
                if candidate_index >= 0:
                    frame = draw_candidate(frame, record, candidate_index)
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            except (ValueError, cv2.error) as error:
                self._send_json({"error": str(error)}, HTTPStatus.NOT_FOUND)
                return
            if not ok:
                self._send_json({"error": "could not encode frame"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_bytes(encoded.tobytes(), "image/jpeg")
            return
        if parsed.path == "/api/labels":
            self._send_json(self.state.labels)
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/label":
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            saved = self.state.save_label(payload)
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(saved)

    def log_message(self, format: str, *args: object) -> None:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Source video used for the candidate log.")
    parser.add_argument("--log", type=Path, required=True, help="turn_candidate_scores.jsonl path or its directory.")
    parser.add_argument("--labels", type=Path, help="Output JSON label file; defaults beside the log.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    log_path = args.log / "turn_candidate_scores.jsonl" if args.log.is_dir() else args.log
    if not log_path.is_file():
        raise SystemExit(f"Could not find candidate log: {log_path}")
    labels_path = args.labels or log_path.with_name(log_path.stem + ".labels.json")
    state = ReviewState(args.video, read_records(log_path), labels_path)
    if not state.records:
        raise SystemExit(f"Candidate log is empty: {log_path}")
    handler = type("BoundReviewHandler", (ReviewHandler,), {"state": state})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Turn candidate review: http://{args.host}:{args.port}", flush=True)
    print(f"Labels will be saved to: {labels_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.capture.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
