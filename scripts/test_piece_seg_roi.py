"""Test YOLO sticker segmentation on a cube bounding-box ROI.

Pipeline:
1. Detect the whole cube with the existing bbox detector.
2. Crop a padded square ROI around that bbox.
3. Run the center/edge/corner sticker segmentation model on the ROI.
4. Draw the ROI-space masks back onto the original OpenCV frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.cube_net import (  # noqa: E402
    CubeNetState,
    build_cube_net_from_graph,
    draw_cube_net,
)
from live_recognition.detectors import YOLOCubeDetector  # noqa: E402
from live_recognition.piece_graph import (  # noqa: E402
    PieceGraph,
    StickerNode,
    _legal_mask_adjacencies,
    draw_piece_graph as draw_sticker_graph,
)
from live_recognition.visualization import (  # noqa: E402
    _reliable_color_mask,
    save_hsv_samples_interactive_plot,
    save_rgb_samples_interactive_plot,
    save_rgb_samples_plot,
)


DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
DEFAULT_STICKER_MODEL = (
    ROOT
    / "runs/segment/runs/cube_piece_seg_sticker/runs/cube_piece_yolo11n_seg/weights/best.pt"
)
CLASS_COLORS = {
    0: (80, 220, 255),   # center
    1: (80, 255, 120),   # edge
    2: (255, 120, 80),   # corner
}
STICKER_CLASS_NAMES = ["center_sticker", "edge_sticker", "corner_sticker"]
CENTER_CLASS_ID = 0
STICKER_COLOR_SAMPLE_COUNT = 10
DRAW_COLOR_SAMPLE_POINTS = True
DENSITY_CLUSTER_EPS_RADIANS = 0.08
DENSITY_CLUSTER_MIN_POINTS = 4


class ColorDensityClassifier:
    def __init__(
        self,
        *,
        required_frames: int = 30,
        cluster_count: int = 6,
        min_brightness: float = 40.0,
        density_cluster_eps_radians: float = DENSITY_CLUSTER_EPS_RADIANS,
        density_cluster_min_neighbors: int = DENSITY_CLUSTER_MIN_POINTS,
    ) -> None:
        self.required_frames = max(1, required_frames)
        self.cluster_count = max(1, cluster_count)
        self.min_brightness = min_brightness
        self.density_cluster_eps_radians = density_cluster_eps_radians
        self.density_cluster_min_neighbors = max(1, density_cluster_min_neighbors)
        self.detected_frames = 0
        self._rgb_samples: list[np.ndarray] = []
        self.centroids = np.empty((0, 3), dtype=np.float32)
        self.cluster_samples: list[np.ndarray] = []

    @property
    def ready(self) -> bool:
        return len(self.cluster_samples) > 0

    def observe_frame(self, bgr_samples: np.ndarray) -> None:
        if self.ready or len(bgr_samples) == 0:
            return

        bgr_samples = np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 3)
        self._rgb_samples.extend(bgr_samples[:, ::-1])
        self.detected_frames += 1
        if self.detected_frames < self.required_frames:
            return

        self.centroids, self.cluster_samples = density_rgb_clusters(
            np.asarray(self._rgb_samples, dtype=np.uint8),
            target_clusters=self.cluster_count,
            min_brightness=self.min_brightness,
            eps_radians=self.density_cluster_eps_radians,
            min_points=self.density_cluster_min_neighbors,
        )

    def classify_bgr(self, bgr_sample: np.ndarray) -> int | None:
        if not self.ready:
            return None

        rgb = np.asarray(bgr_sample, dtype=np.float32).reshape(3)[::-1]
        if float(np.linalg.norm(rgb)) < self.min_brightness:
            return None

        distances = angular_distance_to_normalized_rgb_clusters(
            rgb,
            self.cluster_samples,
        )
        if len(distances) == 0 or float(distances.min()) > self.density_cluster_eps_radians:
            return None
        return int(distances.argmin())

    def classify_bgr_samples(self, bgr_samples: np.ndarray) -> int | None:
        if not self.ready:
            return None

        samples = np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 3)
        votes = [
            cluster_index
            for sample in samples
            if (cluster_index := self.classify_bgr(sample)) is not None
        ]
        if not votes:
            return None
        return int(np.bincount(np.asarray(votes, dtype=np.int32)).argmax())


def normalized_rgb_points(rgb_samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    brightness = np.maximum(np.linalg.norm(samples, axis=1, keepdims=True), 1.0)
    return samples / brightness


def angular_distance_to_normalized_rgb_clusters(
    rgb_sample: np.ndarray,
    cluster_samples: list[np.ndarray],
) -> np.ndarray:
    point = normalized_rgb_points(np.asarray(rgb_sample, dtype=np.float32).reshape(1, 3))[0]
    distances = []
    for samples in cluster_samples:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
        if len(samples) == 0:
            distances.append(np.inf)
            continue
        normalized_samples = normalized_rgb_points(samples)
        max_cosine = float(np.max(np.clip(normalized_samples @ point, -1.0, 1.0)))
        distances.append(float(np.arccos(max_cosine)))
    return np.asarray(distances, dtype=np.float32)


def density_rgb_clusters(
    rgb_samples: np.ndarray,
    *,
    target_clusters: int,
    min_brightness: float,
    eps_radians: float = DENSITY_CLUSTER_EPS_RADIANS,
    min_points: int = DENSITY_CLUSTER_MIN_POINTS,
) -> tuple[np.ndarray, list[np.ndarray]]:
    samples = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    if len(samples) == 0 or target_clusters <= 0:
        return np.empty((0, 3), dtype=np.float32), []

    brightness = np.linalg.norm(samples, axis=1)
    usable = samples[brightness >= min_brightness]
    if len(usable) < min_points:
        return np.empty((0, 3), dtype=np.float32), []

    labels = density_bin_labels(
        normalized_rgb_points(usable),
        eps_radians=eps_radians,
        min_points=min_points,
    )

    clusters = []
    for label in sorted(set(int(value) for value in labels if value >= 0)):
        members = usable[labels == label]
        if len(members) >= min_points:
            clusters.append(members.astype(np.float32, copy=True))
    clusters = merge_smallest_clusters_to_nearest_neighbor(
        clusters,
        target_count=target_clusters,
    )

    centroids = []
    retained_clusters = []
    for members in clusters:
        centroid = members.mean(axis=0)
        if float(np.linalg.norm(centroid)) <= 1e-6:
            continue
        centroids.append(centroid)
        retained_clusters.append(members)

    return np.asarray(centroids, dtype=np.float32), retained_clusters


def merge_smallest_clusters_to_nearest_neighbor(
    clusters: list[np.ndarray],
    *,
    target_count: int,
) -> list[np.ndarray]:
    clusters = [np.asarray(cluster, dtype=np.float32).reshape(-1, 3) for cluster in clusters]
    clusters = [cluster for cluster in clusters if len(cluster) > 0]
    while len(clusters) > target_count:
        smallest_index = min(range(len(clusters)), key=lambda index: len(clusters[index]))
        nearest_index = nearest_cluster_index(clusters, smallest_index)
        clusters[nearest_index] = np.concatenate(
            [clusters[nearest_index], clusters[smallest_index]],
            axis=0,
        )
        del clusters[smallest_index]
    clusters.sort(key=len, reverse=True)
    return clusters


def nearest_cluster_index(clusters: list[np.ndarray], source_index: int) -> int:
    source_centroid = normalized_rgb_points(clusters[source_index]).mean(axis=0)
    best_index = 0 if source_index != 0 else 1
    best_distance = np.inf
    for index, cluster in enumerate(clusters):
        if index == source_index:
            continue
        centroid = normalized_rgb_points(cluster).mean(axis=0)
        distance = float(np.linalg.norm(source_centroid - centroid))
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def density_bin_labels(
    unit_samples: np.ndarray,
    *,
    eps_radians: float,
    min_points: int,
) -> np.ndarray:
    samples = np.asarray(unit_samples, dtype=np.float32).reshape(-1, 3)
    labels = np.full(len(samples), -1, dtype=np.int32)
    if len(samples) == 0:
        return labels

    eps = max(float(eps_radians), 1e-6)
    chord_radius = float(np.sqrt(max(0.0, 2.0 - 2.0 * np.cos(eps))))
    cell_size = max(chord_radius, 1e-6)
    cell_keys, sample_cell_indexes, cell_counts = np.unique(
        np.floor(samples / cell_size).astype(np.int32),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    cell_lookup = {
        tuple(int(value) for value in cell_key): cell_index
        for cell_index, cell_key in enumerate(cell_keys)
    }
    cell_neighbor_counts = np.zeros(len(cell_keys), dtype=np.int32)
    for cell_index, cell_key in enumerate(cell_keys):
        base_cell = tuple(int(value) for value in cell_key)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor_index = cell_lookup.get(
                        (base_cell[0] + dx, base_cell[1] + dy, base_cell[2] + dz)
                    )
                    if neighbor_index is not None:
                        cell_neighbor_counts[cell_index] += cell_counts[neighbor_index]

    dense_cells = cell_neighbor_counts >= int(min_points)
    cell_labels = np.full(len(cell_keys), -1, dtype=np.int32)
    cluster_id = 0
    for start_cell in np.flatnonzero(dense_cells):
        if cell_labels[start_cell] >= 0:
            continue
        queue = [int(start_cell)]
        cell_labels[start_cell] = cluster_id
        while queue:
            current = queue.pop()
            base_cell = tuple(int(value) for value in cell_keys[current])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor_index = cell_lookup.get(
                            (base_cell[0] + dx, base_cell[1] + dy, base_cell[2] + dz)
                        )
                        if (
                            neighbor_index is None
                            or not dense_cells[neighbor_index]
                            or cell_labels[neighbor_index] >= 0
                        ):
                            continue
                        cell_labels[neighbor_index] = cluster_id
                        queue.append(neighbor_index)
        cluster_id += 1

    for sample_index, cell_index in enumerate(sample_cell_indexes):
        label = cell_labels[cell_index]
        if label >= 0:
            labels[sample_index] = label
    return labels


def parse_source(source: str) -> int | str:
    source = source.strip().strip("\"'").strip()
    return int(source) if source.isdigit() else source


def clipped_square_roi(
    bbox_xyxy: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + padding * 2.0)

    rx1 = int(round(cx - side / 2.0))
    ry1 = int(round(cy - side / 2.0))
    rx2 = int(round(cx + side / 2.0))
    ry2 = int(round(cy + side / 2.0))

    rx1 = max(0, min(frame_width - 1, rx1))
    ry1 = max(0, min(frame_height - 1, ry1))
    rx2 = max(rx1 + 1, min(frame_width, rx2))
    ry2 = max(ry1 + 1, min(frame_height, ry2))
    return rx1, ry1, rx2, ry2


def apply_mask_padding(mask: np.ndarray, padding_px: int) -> np.ndarray:
    if padding_px == 0 or not bool(mask.any()):
        return mask

    radius = abs(padding_px)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8) * 255
    if padding_px < 0:
        padded = cv2.erode(mask_u8, kernel, iterations=1)
    else:
        padded = cv2.dilate(mask_u8, kernel, iterations=1)
    return padded > 0


def sticker_mask_bool(
    mask_data: np.ndarray,
    index: int,
    roi_width: int,
    roi_height: int,
) -> np.ndarray:
    mask = cv2.resize(
        mask_data[index].astype(np.float32),
        (roi_width, roi_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return mask > 0.5


def class_name_for_id(result, sticker_model, class_id: int) -> str:
    names = getattr(result, "names", None) or getattr(sticker_model, "names", {}) or {}
    return str(names.get(class_id, class_id)).lower()


def sticker_piece_type(class_name: str) -> str:
    return class_name.removesuffix("_sticker")


def roi_color_samples(crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reliable_mask = _reliable_color_mask(crop_bgr)
    return crop_bgr, reliable_mask


def random_mask_sample_points(mask: np.ndarray, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or sample_count <= 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
    if len(xs) <= sample_count:
        return ys.astype(np.int32), xs.astype(np.int32)

    seed = int(mask.sum() + ys.sum() * 31 + xs.sum() * 131) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(xs), size=sample_count, replace=False)
    return ys[selected].astype(np.int32), xs[selected].astype(np.int32)


def sticker_color_sample_points_from_roi(
    color_samples: tuple[np.ndarray, np.ndarray],
    sticker_mask: np.ndarray,
    sample_count: int = STICKER_COLOR_SAMPLE_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    _, roi_reliable_mask = color_samples
    if not bool(sticker_mask.any()):
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    sample_mask = roi_reliable_mask & sticker_mask
    if int(sample_mask.sum()) < min(sample_count, int(sticker_mask.sum())):
        sample_mask = sticker_mask

    return random_mask_sample_points(sample_mask, sample_count)


def sticker_color_samples_from_roi(
    color_samples: tuple[np.ndarray, np.ndarray],
    sticker_mask: np.ndarray,
    sample_count: int = STICKER_COLOR_SAMPLE_COUNT,
) -> np.ndarray:
    crop, _ = color_samples
    ys, xs = sticker_color_sample_points_from_roi(
        color_samples,
        sticker_mask,
        sample_count,
    )
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    return crop[ys, xs].astype(np.uint8)


def draw_color_sample_points(
    frame: np.ndarray,
    sample_points: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    if not DRAW_COLOR_SAMPLE_POINTS:
        return

    for xs, ys in sample_points:
        for x, y in zip(xs, ys):
            center = (int(x), int(y))
            cv2.circle(frame, center, 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, center, 1, (255, 255, 255), -1, cv2.LINE_AA)


def sticker_color_samples_at_points(
    color_samples: tuple[np.ndarray, np.ndarray],
    ys: np.ndarray,
    xs: np.ndarray,
) -> np.ndarray:
    crop, _ = color_samples
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    return crop[ys, xs].astype(np.uint8)


def direct_sticker_graph(
    nodes: list[StickerNode],
    *,
    adjacency_radius_px: int,
) -> PieceGraph:
    return PieceGraph(
        nodes=nodes,
        edges=_legal_mask_adjacencies(nodes, adjacency_radius_px=adjacency_radius_px),
    )


def stickers_for_ready_color_classifier(
    frame: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    stickers: list[tuple[np.ndarray, np.ndarray]],
    color_classifier: ColorDensityClassifier | None,
    color_samples: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, int | None]]:
    if color_classifier is None or not color_classifier.ready:
        return [(mask, bgr, None) for mask, bgr in stickers]

    classified_stickers = []
    for sticker_index, (sticker_mask, sticker_bgr) in enumerate(stickers):
        cluster_index = (
            color_classifier.classify_bgr_samples(sticker_bgr)
            if len(sticker_bgr) > 0
            else None
        )
        if color_samples is None:
            x1, y1, x2, y2 = roi_xyxy
            color_samples = roi_color_samples(frame[y1:y2, x1:x2])
        sample_bgr = sticker_color_samples_from_roi(color_samples, sticker_mask)
        if len(sample_bgr) == 0:
            sample_bgr = sticker_bgr
        classified_stickers.append((sticker_mask, sample_bgr, cluster_index))
    return classified_stickers


def classified_net_colors_from_graph(
    graph,
    color_classifier: ColorDensityClassifier,
) -> tuple[dict[int, str], dict[str, tuple[int, int, int]]]:
    colors_by_node_id = {}
    bgr_by_color = {}
    if not color_classifier.ready:
        return colors_by_node_id, bgr_by_color

    for node in graph.nodes:
        cluster_index = color_classifier.classify_bgr(np.asarray(node.mean_bgr, dtype=np.uint8))
        if cluster_index is None or cluster_index >= len(color_classifier.centroids):
            continue
        color_id = f"cluster-{int(cluster_index) + 1}"
        colors_by_node_id[node.node_id] = color_id
        bgr_by_color[color_id] = tuple(
            int(value) for value in color_classifier.centroids[cluster_index][::-1]
        )
    return colors_by_node_id, bgr_by_color


def parse_net_cluster_faces(value: str) -> dict[str, str]:
    face_by_color = {}
    value = value.strip()
    if not value:
        return face_by_color

    for item in value.split(","):
        cluster_text, separator, face = item.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError(
                "Expected comma-separated cluster=face entries, for example 1=R,2=U"
            )
        cluster_text = cluster_text.strip()
        face = face.strip().upper()
        if not cluster_text.isdigit() or face not in {"U", "D", "F", "B", "R", "L"}:
            raise argparse.ArgumentTypeError(
                "Expected cluster numbers and cube faces U,D,F,B,R,L"
            )
        face_by_color[f"cluster-{int(cluster_text)}"] = face
    return face_by_color


def normalized_rgb_html_plot_path(rgb_html_path: Path) -> Path:
    suffix = rgb_html_path.suffix or ".html"
    return rgb_html_path.with_name(f"{rgb_html_path.stem}_normalized{suffix}")


def save_rgb_html_plots(
    bgr_samples: np.ndarray,
    rgb_html_path: Path,
    *,
    title: str,
    normalize_rgb: bool,
    args,
    color_classifier: ColorDensityClassifier | None = None,
) -> None:
    cluster_indexes = None
    cluster_centroids = None
    if color_classifier is not None and color_classifier.ready:
        cluster_indexes = classifier_cluster_indexes_for_bgr_samples(
            color_classifier,
            bgr_samples,
        )
        cluster_centroids = color_classifier.centroids

    save_rgb_samples_interactive_plot(
        bgr_samples,
        str(rgb_html_path),
        title=title,
        normalize_rgb=normalize_rgb,
        point_size=args.plotly_point_size,
        cluster_indexes=cluster_indexes,
        cluster_centroids=cluster_centroids,
    )
    save_rgb_samples_interactive_plot(
        bgr_samples,
        str(normalized_rgb_html_plot_path(rgb_html_path)),
        title=f"{title} normalized to brightness",
        normalize_rgb=True,
        point_size=args.plotly_point_size,
        cluster_indexes=cluster_indexes,
        cluster_centroids=cluster_centroids,
    )


def classifier_cluster_indexes_for_bgr_samples(
    color_classifier: ColorDensityClassifier,
    bgr_samples: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 3)
    return np.asarray(
        [
            -1
            if (cluster_index := color_classifier.classify_bgr(sample)) is None
            else cluster_index
            for sample in samples
        ],
        dtype=np.int32,
    )


def concatenate_samples(
    bgr_batches: list[np.ndarray],
    max_samples: int,
) -> np.ndarray:
    if not bgr_batches:
        return np.empty((0, 3), dtype=np.uint8)

    bgr_samples = np.concatenate(bgr_batches, axis=0)
    if len(bgr_samples) > max_samples:
        indexes = np.linspace(0, len(bgr_samples) - 1, max_samples, dtype=np.int32)
        bgr_samples = bgr_samples[indexes]
    return bgr_samples


def draw_sticker_segments(
    frame: np.ndarray,
    roi: np.ndarray,
    sticker_model,
    roi_xyxy,
    args,
    color_classifier: ColorDensityClassifier | None = None,
    cube_net_state: CubeNetState | None = None,
    prediction_roi: np.ndarray | None = None,
) -> tuple[int, np.ndarray]:
    x1, y1, x2, y2 = roi_xyxy
    results = sticker_model.predict(
        source=prediction_roi if prediction_roi is not None else roi,
        conf=args.sticker_conf,
        imgsz=args.sticker_imgsz,
        verbose=False,
    )
    bgr_batches = []
    if not results:
        return 0, np.empty((0, 3), dtype=np.uint8)

    result = results[0]
    masks = result.masks
    boxes = result.boxes
    if masks is None or boxes is None:
        return 0, np.empty((0, 3), dtype=np.uint8)

    overlay = frame.copy()
    roi_height = y2 - y1
    roi_width = x2 - x1
    count = 0
    confidence_labels = []
    sticker_color_numbers = []
    classifier_bgr_batches = []
    sticker_nodes = []
    color_sample_points = []
    mask_data = result.masks.data.detach().cpu().numpy()
    color_samples = roi_color_samples(roi)
    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item())
        class_name = class_name_for_id(result, sticker_model, class_id)
        if class_name not in STICKER_CLASS_NAMES:
            continue
        piece_type = sticker_piece_type(class_name)
        color = CLASS_COLORS.get(class_id, (255, 255, 255))

        mask_bool = sticker_mask_bool(mask_data, index, roi_width, roi_height)
        if not mask_bool.any():
            continue

        sample_mask = apply_mask_padding(mask_bool, args.segmentation_padding_px)
        if not sample_mask.any():
            sample_mask = mask_bool
        draw_mask = sample_mask
        graph_mask = mask_bool

        ys, xs = np.nonzero(graph_mask)
        label_at = (int(xs.mean()) + x1, int(ys.mean()) + y1)
        confidence = float(boxes.conf[index].item())
        confidence_labels.append((label_at, confidence))
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = graph_mask.astype(np.uint8) * 255

        if args.debug_overlay:
            contours, _ = cv2.findContours(
                draw_mask.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            shifted_contours = []
            for contour in contours:
                contour = contour.copy()
                contour[:, 0, 0] += x1
                contour[:, 0, 1] += y1
                shifted_contours.append(contour)
            for points in shifted_contours:
                if len(points) < 3:
                    continue
                cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)

        sample_ys, sample_xs = sticker_color_sample_points_from_roi(color_samples, sample_mask)
        sample_bgr = sticker_color_samples_at_points(
            color_samples,
            sample_ys,
            sample_xs,
        )
        if len(sample_xs) > 0:
            color_sample_points.append((sample_xs + x1, sample_ys + y1))
        raw_stickers = (
            [(draw_mask, sample_bgr)]
            if len(sample_bgr)
            else []
        )
        if len(sample_bgr) and int(graph_mask.sum()) >= args.graph_min_sticker_area:
            mean_sample_bgr = (
                sample_bgr.astype(np.float32)
                .mean(axis=0)
                .clip(0, 255)
                .astype(np.uint8)
            )
            sticker_nodes.append(
                StickerNode(
                    node_id=len(sticker_nodes),
                    piece_id=len(sticker_nodes),
                    piece_type=piece_type,
                    confidence=confidence,
                    centroid_xy=(float(xs.mean() + x1), float(ys.mean() + y1)),
                    mask=full_mask,
                    mean_bgr=tuple(int(channel) for channel in mean_sample_bgr),
                )
            )
        if not raw_stickers:
            raw_stickers = [
                (
                    draw_mask,
                    np.empty((0, 3), dtype=np.uint8),
                )
            ]
        display_stickers = stickers_for_ready_color_classifier(
            frame,
            roi_xyxy,
            raw_stickers,
            color_classifier,
            color_samples=color_samples,
        )

        roi_overlay = overlay[y1:y2, x1:x2]
        for sticker_mask, sticker_bgr, cluster_index in display_stickers:
            sticker_color = color
            if (
                color_classifier is not None
                and color_classifier.ready
                and getattr(args, "show_sticker_color_classification", True)
                and len(sticker_bgr) > 0
                and cluster_index is not None
                and cluster_index < len(color_classifier.centroids)
            ):
                centroid_rgb = color_classifier.centroids[cluster_index]
                sticker_color = tuple(int(value) for value in centroid_rgb[::-1])
                sticker_ys, sticker_xs = np.nonzero(sticker_mask)
                sticker_label_at = (
                    int(sticker_xs.mean()) + x1,
                    int(sticker_ys.mean()) + y1,
                )
                sticker_color_numbers.append((sticker_label_at, cluster_index + 1))
            roi_overlay[sticker_mask] = sticker_color

        sticker_bgr_batches = []
        for _, sticker_bgr_samples, _ in display_stickers:
            if len(sticker_bgr_samples) > 0:
                sticker_bgr_batches.append(sticker_bgr_samples)
        bgr_samples = concatenate_samples(
            sticker_bgr_batches,
            args.color_plot_samples_per_frame,
        )
        if len(bgr_samples) > 0:
            bgr_batches.append(bgr_samples)
        if color_classifier is not None:
            classifier_sample_batches = [
                sticker_bgr
                for _, sticker_bgr in raw_stickers
                if len(sticker_bgr) > 0
            ]
            classifier_bgr_samples = (
                np.concatenate(classifier_sample_batches, axis=0)
                if classifier_sample_batches
                else np.empty((0, 3), dtype=np.uint8)
            )
            if len(classifier_bgr_samples) > 0:
                classifier_bgr_batches.append(classifier_bgr_samples)
        count += 1

    mask = cv2.addWeighted(frame, 1.0 - args.mask_opacity, overlay, args.mask_opacity, 0)
    frame[:] = mask
    if (args.draw_sticker_graph or cube_net_state is not None) and sticker_nodes:
        graph = direct_sticker_graph(
            sticker_nodes,
            adjacency_radius_px=args.graph_adjacency_radius,
        )
        if cube_net_state is not None and color_classifier is not None and color_classifier.ready:
            classified_colors, classified_bgr = classified_net_colors_from_graph(
                graph,
                color_classifier,
            )
            cube_net_state.replace_with(
                build_cube_net_from_graph(
                    graph,
                    classified_colors_by_node_id=classified_colors,
                    classified_bgr_by_color=classified_bgr,
                    face_by_color=getattr(args, "net_cluster_faces", None),
                )
            )
        if args.draw_sticker_graph:
            frame[:] = draw_sticker_graph(
                frame,
                graph,
                roi_xyxy=roi_xyxy,
                draw_masks=False,
            )
    if args.show_confidence or args.debug_overlay:
        for label_at, confidence in confidence_labels:
            cv2.putText(
                frame,
                f"{confidence:.2f}",
                label_at,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{confidence:.2f}",
                label_at,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
    if args.show_confidence or args.debug_overlay:
        draw_sticker_color_numbers(frame, sticker_color_numbers)
    draw_color_sample_points(frame, color_sample_points)
    bgr_samples = concatenate_samples(
        bgr_batches,
        args.color_plot_samples_per_frame,
    )
    if color_classifier is not None:
        classifier_bgr_samples = (
            np.concatenate(classifier_bgr_batches, axis=0)
            if classifier_bgr_batches
            else np.empty((0, 3), dtype=np.uint8)
        )
        color_classifier.observe_frame(classifier_bgr_samples)
    return count, bgr_samples


def downscale_to_max_size(image: np.ndarray, max_size: int) -> np.ndarray:
    if max_size <= 0:
        return image

    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_size:
        return image

    scale = max_size / float(longest_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def draw_sticker_color_numbers(
    frame: np.ndarray,
    sticker_color_numbers: list[tuple[tuple[int, int], int]],
) -> None:
    for label_at, color_number in sticker_color_numbers:
        text = str(color_number)
        origin = (label_at[0] - 8, label_at[1] + 8)
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def process_frame(
    frame: np.ndarray,
    bbox_detector,
    sticker_model,
    args,
    color_classifier: ColorDensityClassifier | None = None,
    cube_net_state: CubeNetState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    output = frame.copy()
    height, width = frame.shape[:2]
    if getattr(args, "use_bbox_detector", True):
        candidate = bbox_detector.detect(frame)
        if candidate is None:
            return output, np.empty((0, 3), dtype=np.uint8)
        roi_xyxy = clipped_square_roi(candidate.bbox_xyxy, width, height, args.roi_padding)
    else:
        candidate = None
        roi_xyxy = (0, 0, width, height)

    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    prediction_roi = (
        downscale_to_max_size(roi, args.sticker_imgsz)
        if not getattr(args, "use_bbox_detector", True)
        else None
    )
    segment_count, bgr_samples = draw_sticker_segments(
        output,
        roi,
        sticker_model,
        roi_xyxy,
        args,
        color_classifier,
        cube_net_state,
        prediction_roi=prediction_roi,
    )

    if candidate is not None and (args.show_bbox or args.debug_overlay):
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 220, 40), 2)
    if args.debug_overlay:
        debug_label = (
            f"cube {candidate.confidence:.2f} segments {segment_count}"
            if candidate is not None
            else f"full frame segments {segment_count}"
        )
        cv2.putText(
            output,
            debug_label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 220, 40),
            2,
            cv2.LINE_AA,
        )
    return output, bgr_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index, video path, or image path")
    parser.add_argument("--bbox-model", type=Path, default=DEFAULT_BBOX_MODEL)
    parser.add_argument("--sticker-model", type=Path, default=DEFAULT_STICKER_MODEL)
    parser.add_argument("--bbox-conf", type=float, default=0.25)
    parser.add_argument("--sticker-conf", type=float, default=0.25)
    parser.add_argument("--bbox-imgsz", type=int, default=640)
    parser.add_argument("--sticker-imgsz", type=int, default=256)
    parser.add_argument(
        "--skip-bbox",
        dest="use_bbox_detector",
        action="store_false",
        help="Run sticker segmentation on the full frame instead of first detecting a cube ROI",
    )
    parser.add_argument("--roi-padding", type=float, default=0.08)
    parser.add_argument(
        "--segmentation-padding-px",
        type=int,
        default=-2,
        help="Pixel padding applied to each segmentation mask before sampling; negative values erode",
    )
    parser.add_argument(
        "--draw-sticker-graph",
        dest="draw_sticker_graph",
        action="store_true",
        help="Draw sticker graph nodes and edges on top of the segmentation overlay",
    )
    parser.add_argument(
        "--adjacency-radius",
        "--graph-adjacency-radius",
        dest="graph_adjacency_radius",
        type=int,
        default=1,
        help="Pixel distance threshold used when connecting nearby graph nodes",
    )
    parser.add_argument(
        "--graph-min-sticker-area",
        type=int,
        default=12,
        help="Minimum sticker area kept as a graph node",
    )
    parser.add_argument("--mask-opacity", type=float, default=1.0)
    parser.add_argument("--rgb-color-plot", type=Path, help="Output PNG path for 3D RGB samples")
    parser.add_argument(
        "--rgb-html-plot",
        type=Path,
        help="Output HTML path for interactive 3D RGB samples",
    )
    parser.add_argument(
        "--hsv-html-plot",
        type=Path,
        help="Output HTML path for interactive 3D HSV samples",
    )
    parser.add_argument(
        "--normalize-rgb-plot",
        action="store_true",
        help="Plot RGB coordinates normalized by RGB vector length",
    )
    parser.add_argument(
        "--hide-sticker-color-classification",
        dest="show_sticker_color_classification",
        action="store_false",
        help="Do not recolor sticker masks using density-cluster color classification",
    )
    parser.add_argument(
        "--sticker-color-frames",
        type=int,
        default=90,
        help="Detected frames of sticker color samples to collect before classifying stickers",
    )
    parser.add_argument(
        "--sticker-color-min-brightness",
        type=float,
        default=40.0,
        help="Minimum raw RGB vector length used while fitting density color clusters",
    )
    parser.add_argument(
        "--color-cluster-count",
        "--rgb-regression-clusters",
        dest="color_cluster_count",
        type=int,
        default=6,
        help="Target number of density color clusters to keep",
    )
    parser.add_argument(
        "--color-classifier-mode",
        choices=("density-cluster",),
        default="density-cluster",
        help="Color classification mode. Only density-cluster is supported.",
    )
    parser.add_argument(
        "--density-cluster-eps-radians",
        "--cluster-eps-radians",
        dest="density_cluster_eps_radians",
        type=float,
        default=DENSITY_CLUSTER_EPS_RADIANS,
        help="Angular radius for normalized RGB density clustering",
    )
    parser.add_argument(
        "--density-cluster-min-neighbors",
        type=int,
        default=DENSITY_CLUSTER_MIN_POINTS,
        help="Minimum normalized RGB points in the epsilon neighborhood for a density core point",
    )
    parser.add_argument("--color-plot-frames", type=int, default=90)
    parser.add_argument("--color-plot-samples-per-frame", type=int, default=800)
    parser.add_argument("--average-plot-point-size", type=float, default=36.0)
    parser.add_argument(
        "--plotly-point-size",
        type=float,
        default=1.0,
        help="Marker size for interactive Plotly RGB/HSV plots",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--save", type=Path)
    parser.add_argument(
        "--save-net",
        type=Path,
        help="Output PNG path for the latest cube net built from sticker ROI graphs",
    )
    parser.add_argument(
        "--net-cluster-faces",
        type=parse_net_cluster_faces,
        default={},
        help="Map learned color clusters to cube faces, for example 1=R,2=U",
    )
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--hide-bbox", dest="show_bbox", action="store_false")
    parser.add_argument("--hide-confidence", dest="show_confidence", action="store_false")
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Draw ROI box, polygon outlines, labels, and FPS",
    )
    parser.set_defaults(
        show_bbox=True,
        show_confidence=True,
        show_sticker_color_classification=True,
        use_bbox_detector=True,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    from ultralytics import YOLO

    bbox_detector = (
        YOLOCubeDetector(
            model_name=str(args.bbox_model),
            confidence_threshold=args.bbox_conf,
            image_size=args.bbox_imgsz,
        )
        if args.use_bbox_detector
        else None
    )
    sticker_model = YOLO(str(args.sticker_model))
    source = parse_source(args.source)
    plot_point_size = args.average_plot_point_size
    color_classifier = ColorDensityClassifier(
        required_frames=args.sticker_color_frames,
        cluster_count=args.color_cluster_count,
        min_brightness=args.sticker_color_min_brightness,
        density_cluster_eps_radians=args.density_cluster_eps_radians,
        density_cluster_min_neighbors=args.density_cluster_min_neighbors,
    )
    color_plot_bgr_batches = []
    color_plot_detected_frames = 0
    cube_net_state = CubeNetState() if args.save_net else None

    if isinstance(source, str) and Path(source).suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }:
        frame = cv2.imread(source)
        if frame is None:
            raise SystemExit(f"Could not read image: {source}")
        output, bgr_samples = process_frame(
            frame,
            bbox_detector,
            sticker_model,
            args,
            color_classifier,
            cube_net_state,
        )
        if args.rgb_color_plot or args.rgb_html_plot or args.hsv_html_plot:
            if len(bgr_samples) == 0:
                raise SystemExit(
                    "Could not collect color plot samples because no stickers were segmented"
                )
            if args.rgb_color_plot:
                save_rgb_samples_plot(
                    bgr_samples,
                    str(args.rgb_color_plot),
                    title="Sticker segmentation reliable pixels in RGB",
                    point_size=plot_point_size,
                    normalize_rgb=args.normalize_rgb_plot,
                )
            if args.rgb_html_plot:
                save_rgb_html_plots(
                    bgr_samples,
                    args.rgb_html_plot,
                    title="Sticker segmentation reliable pixels in RGB",
                    normalize_rgb=args.normalize_rgb_plot,
                    args=args,
                    color_classifier=color_classifier,
                )
            if args.hsv_html_plot:
                save_hsv_samples_interactive_plot(
                    bgr_samples,
                    str(args.hsv_html_plot),
                    title="Sticker segmentation reliable pixels in HSV",
                    point_size=args.plotly_point_size,
                )
        if args.save:
            cv2.imwrite(str(args.save), output)
        if args.save_net and cube_net_state is not None:
            args.save_net.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save_net), draw_cube_net(cube_net_state))
        if args.display:
            cv2.imshow("cube sticker segmentation", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    previous = perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        output, bgr_samples = process_frame(
            frame,
            bbox_detector,
            sticker_model,
            args,
            color_classifier,
            cube_net_state,
        )
        now = perf_counter()
        fps = 1.0 / max(now - previous, 1e-6)
        previous = now
        if args.debug_overlay:
            cv2.putText(
                output,
                f"FPS {fps:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                3,
            )
            cv2.putText(
                output,
                f"FPS {fps:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 20, 20),
                1,
            )

        if args.save and args.once:
            cv2.imwrite(str(args.save), output)
        if args.rgb_color_plot or args.rgb_html_plot or args.hsv_html_plot:
            if color_plot_detected_frames < args.color_plot_frames and len(bgr_samples) > 0:
                color_plot_bgr_batches.append(bgr_samples)
                color_plot_detected_frames += 1
                if color_plot_detected_frames >= args.color_plot_frames and not args.display:
                    break
        if args.display:
            cv2.imshow("cube sticker segmentation", output)
            key = cv2.waitKey(0 if args.once else 1) & 0xFF
            if key == ord("q"):
                break
        if args.once:
            break

    if args.display:
        cv2.destroyAllWindows()
    capture.release()
    if args.save_net and cube_net_state is not None:
        args.save_net.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_net), draw_cube_net(cube_net_state))
    if args.rgb_color_plot or args.rgb_html_plot or args.hsv_html_plot:
        if not color_plot_bgr_batches:
            raise SystemExit("Could not collect color plot samples from any segmented frames")
        bgr_samples = np.concatenate(color_plot_bgr_batches, axis=0)
        if args.rgb_color_plot:
            save_rgb_samples_plot(
                bgr_samples,
                str(args.rgb_color_plot),
                title=f"Sticker segmentation RGB samples across {color_plot_detected_frames} frames",
                point_size=plot_point_size,
                normalize_rgb=args.normalize_rgb_plot,
            )
        if args.rgb_html_plot:
            save_rgb_html_plots(
                bgr_samples,
                args.rgb_html_plot,
                title=f"Sticker segmentation RGB samples across {color_plot_detected_frames} frames",
                normalize_rgb=args.normalize_rgb_plot,
                args=args,
                color_classifier=color_classifier,
            )
        if args.hsv_html_plot:
            save_hsv_samples_interactive_plot(
                bgr_samples,
                str(args.hsv_html_plot),
                title=f"Sticker segmentation HSV samples across {color_plot_detected_frames} frames",
                point_size=args.plotly_point_size,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
