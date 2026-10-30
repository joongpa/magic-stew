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
        return "—"
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
            "inlier_count": None,
        }
    ]


def selected_candidate_index(record: dict[str, object]) -> int | None:
    """Locate the logged selected configuration, independent of ranking order."""

    candidates = review_candidates(record)
    rotations = record.get("selected_layer_rotations")
    if not isinstance(rotations, list):
        return None
    for index, candidate in enumerate(candidates):
        if candidate.get("layer_rotations") == rotations:
            return index
    return None


def rotation_matrix_for_layer(rotation: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    axis = int(rotation["axis"])
    side = int(rotation["side"])
    angle = side * np.deg2rad(float(rotation["angle_degrees"]))
    vector = np.zeros((3, 1), dtype=np.float64)
    vector[axis, 0] = angle
    matrix, _ = cv2.Rodrigues(vector)
    pivot = np.zeros(3, dtype=np.float64)
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
    faces: list[dict[str, object]] = []
    for x_index in range(3):
        for y_index in range(3):
            for z_index in range(3):
                center = cubie_center(x_index, y_index, z_index)
                for axis in range(3):
                    for side in (-1, 1):
                        indices = (x_index, y_index, z_index)
                        if indices[axis] != (0 if side < 0 else 2):
                            continue
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
    pose = record.get("search_pose")
    if not isinstance(pose, dict) or any(
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
        faces = projected_visible_sticker_faces(pose, rotations, selected_layer)
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
    heatmap_centers = pose.get("sticker_mask_centers") or []
    centers = np.asarray(heatmap_centers, dtype=np.float64)
    if centers.size == 0:
        centers = np.empty((0, 2), dtype=np.float64)
    for center in centers.reshape(-1, 2):
        if np.isfinite(center).all():
            point = tuple(np.round(center).astype(int))
            cv2.circle(output, point, 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(output, point, 5, HEATMAP_COLOR_BGR, -1, cv2.LINE_AA)

    label = str(candidate.get("label", f"candidate {candidate_index}"))
    text = (
        f"{candidate_index}: {label} | "
        f"mean {format_score(candidate.get('global_mean_error'))} | "
        f"robust {format_score(candidate.get('global_robust_error'))}"
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
            if frame_index == self.next_video_frame - 1 and self.last_video_frame is not None:
                return self.last_video_frame.copy()
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
                self.next_video_frame += 1
                self.last_video_frame = frame
            return self.last_video_frame.copy()

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
.image-panel { background:#0d0f12; border:1px solid var(--line); border-radius:6px; padding:7px; min-width:0; }
.image-panel img { width:100%; height:auto; display:block; object-fit:contain; max-height:calc(100vh - 130px); }
#candidateImage { cursor:crosshair; }
.image-panel h3 { margin:2px 0 7px; font-size:13px; color:#b6c0cc; }
.sidebar { background:var(--panel); border:1px solid var(--line); border-radius:6px; display:flex; flex-direction:column; min-height:0; }
.sidebar-top { padding:10px; border-bottom:1px solid var(--line); }
#candidates { overflow:auto; padding:7px; }
details.group { border:1px solid var(--line); border-radius:5px; margin-bottom:6px; }
details.group summary { cursor:pointer; padding:8px; font-weight:600; color:#cbd5e0; }
details.group summary:hover { color:#fff; }
.group-items { padding:0 7px 1px; }
.candidate { border:1px solid var(--line); border-radius:5px; padding:7px; margin-bottom:6px; cursor:pointer; }
.candidate.selected { border-color:var(--accent); box-shadow:0 0 0 1px var(--accent) inset; }
.candidate.correct { border-color:#57d38c; } .candidate.baseline { background:#292d33; }
.candidate .title { font-weight:600; } .candidate .metrics { color:#aeb8c4; font-size:12px; margin-top:3px; }
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
  <section class="preview">
    <div class="image-panel"><h3>Source frame</h3><img id="sourceImage"></div>
    <div class="image-panel"><h3 id="candidateTitle">Candidate</h3><img id="candidateImage"></div>
  </section>
  <aside class="sidebar">
    <div class="sidebar-top">
      <div id="frameInfo" class="muted"></div>
      <div id="layerInfo" class="muted">Click a layer; double-click for two layers. Use , / . to turn.</div>
      <div class="actions"><button id="correct" class="primary">Correct (C)</button><button id="uncertain">Uncertain (U)</button><button id="invalid">No valid pose (N)</button></div>
      <textarea id="note" placeholder="Optional note about the scoring failure..."></textarea>
    </div>
    <div id="candidates"></div>
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
  const remainderScores=items.map(item=>Number(item.candidate.remainder_score)).filter(Number.isFinite);
  if(remainderScores.length) {
    return [-Math.max(...remainderScores), Math.min(...items.map(item=>Number(item.candidate.global_mean_error)).filter(Number.isFinite))];
  }
  const meanScores=items.map(item=>Number(item.candidate.global_mean_error)).filter(Number.isFinite);
  return [meanScores.length ? Math.min(...meanScores) : Infinity, 0];
}
function compareGroups(first, second, groupsForSort) {
  const a=groupSortKey(groupsForSort.get(first)||[]), b=groupSortKey(groupsForSort.get(second)||[]);
  return a[0]-b[0] || a[1]-b[1] || first.localeCompare(second);
}
function candidateAngle(candidate) {
  if(candidate.is_baseline) return 0;
  const rotations=candidate.layer_rotations||[];
  return rotations.length ? Number(rotations[rotations.length-1].angle_degrees) : null;
}
function selectCandidate(index) { candidateIndex=index; loadFrame(); }
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
async function loadFrame() {
  const version=++loadVersion;
  current=record(); if(!current) return;
  const data=await getJson(`/api/candidates?frame=${frameIndex}`);
  if(version!==loadVersion) return;
  candidates=data.candidates;
  const count=data.candidates.length;
  if(candidateIndex===null) candidateIndex=current.selected_index ?? 0;
  candidateIndex=Math.max(0, Math.min(candidateIndex, Math.max(0,count-1)));
  $('frame').value=frameIndex;
  $('sourceImage').onerror=()=>$('status').textContent=`Could not decode video frame ${frameIndex}; check the source video.`;
  $('candidateImage').onerror=()=>$('status').textContent=`Could not decode video frame ${frameIndex}; check the source video.`;
  $('sourceImage').src=imageUrl(-1);
  $('candidateImage').src=imageUrl(candidateIndex);
  faceRegions=[];
  if(count) getJson(`/api/face-regions?frame=${frameIndex}&candidate=${candidateIndex}`).then(data=>{if(version===loadVersion) faceRegions=data.faces;});
  $('frameInfo').textContent=`frame ${frameIndex} · ${count} tested configurations · selected: ${current.selected || 'none'}`;
  const staticOnly=count===1 && data.candidates[0].label.includes('turn search not tested');
  $('frameInfo').textContent=staticOnly
    ? `frame ${frameIndex} · turn search not tested · showing static pose`
    : `frame ${frameIndex} · ${count} tested configurations · ranked by combined mean · selected: ${current.selected || 'none'}`;
  $('candidateTitle').textContent=staticOnly ? 'Static pose (turn search not tested)' : (count ? `Candidate ${candidateIndex+1}/${count}` : 'No tested candidate');
  $('layerInfo').textContent=activeLayer ? `${groupTitle(`${activeLayer.axis}:${activeLayer.side}:${activeLayer.width}`)} · angle ${candidateAngle(candidates[candidateIndex])}° · , / . to adjust` : 'Click a layer; double-click for two layers. Use , / . to turn.';
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
      const el=document.createElement('div'); el.className='candidate'+(i===candidateIndex?' selected':'')+(c.is_baseline?' baseline':'');
      el.innerHTML=`<div class="title">${i}: ${c.label}${i===current.selected_index?' · renderer selection':''}</div><div class="metrics">rank ${(c.global_mean_error??NaN).toFixed(4)} · accept ${(c.global_robust_error??NaN).toFixed(4)} · kp ${(c.keypoint_mean_error??NaN).toFixed(4)} · heat ${(c.heatmap_mean_error??NaN).toFixed(4)} · inliers ${c.inlier_count??0}</div>`;
      el.onclick=()=>{ const parts=key.split(':').map(Number); activeLayer=parts.length===3 && parts.every(Number.isFinite) ? {axis:parts[0],side:parts[1],width:parts[2]} : null; selectCandidate(i); }; itemContainer.appendChild(el);
    });
    details.appendChild(itemContainer); list.appendChild(details);
  });
}
function moveFrame(delta) { frameIndex=Math.max(0,Math.min(meta.frames.length-1,frameIndex+delta)); candidateIndex=null; activeLayer=null; loadFrame(); }
async function label(kind, advance=false) {
  const payload={frame_index:frameIndex,candidate_index:kind==='correct'&&current.candidate_count?candidateIndex:null,label:kind,note:$('note').value};
  const saved=await getJson('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  meta.labels[String(frameIndex)]=saved; $('status').textContent=`saved: ${kind}`; if(advance) moveFrame(1);
}
$('prev').onclick=()=>moveFrame(-1); $('next').onclick=()=>moveFrame(1); $('source').onclick=()=>{$('candidateImage').src=imageUrl(-1);};
$('correct').onclick=()=>label('correct'); $('uncertain').onclick=()=>label('uncertain'); $('invalid').onclick=()=>label('no_valid_pose'); $('selectNext').onclick=()=>label('correct',true);
$('frame').onchange=e=>{ frameIndex=Math.max(0,Math.min(meta.frames.length-1,Number(e.target.value)||0)); candidateIndex=null; activeLayer=null; loadFrame(); };
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
            self._send_json(
                {"candidates": [] if record is None else review_candidates(record)}
            )
            return
        if parsed.path == "/api/face-regions":
            frame_index = int(query.get("frame", [0])[0])
            candidate_index = int(query.get("candidate", [0])[0])
            record = self.state.record_for_frame(frame_index)
            candidates = review_candidates(record) if record is not None else []
            pose = record.get("search_pose") if record is not None else None
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
