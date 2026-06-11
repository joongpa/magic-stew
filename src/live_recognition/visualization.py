from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from live_recognition.types import FrameResult


COLOR_DEBUG_MIN_SATURATION = 8
COLOR_DEBUG_MIN_VALUE = 18
COLOR_DEBUG_MAX_VALUE = 254
COLOR_DEBUG_MIN_CHANNEL_DELTA = 0
COLOR_DEBUG_LIGHTNESS = 220
COLOR_DEBUG_CHROMA_CONTRAST = 1.55
NORMALIZED_RGB_DENSITY_EPS_RADIANS = 0.08
RAW_RGB_DENSITY_EPS = 24.0
NORMALIZED_RGB_DENSITY_MIN_POINTS = 4


def _normalize_rgb_brightness(rgb_samples: NDArray[np.uint8] | NDArray[np.float32]) -> NDArray[np.float32]:
    samples = np.asarray(rgb_samples, dtype=np.float32)
    brightness = np.maximum(np.linalg.norm(samples, axis=1, keepdims=True), 1.0)
    return samples / brightness


def draw_result(
    frame_bgr: NDArray[np.uint8],
    result: FrameResult,
    *,
    draw_mask_overlay: bool = False,
    fps: float | None = None,
    draw_color_debug: bool = False,
) -> NDArray[np.uint8]:
    output = build_luminance_debug_view(frame_bgr) if draw_color_debug else frame_bgr.copy()
    if fps is not None:
        cv2.putText(
            output,
            f"FPS {fps:.1f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            output,
            f"FPS {fps:.1f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            1,
        )

    if result.candidate is not None:
        if draw_mask_overlay and result.candidate.mask is not None:
            overlay = np.zeros_like(output)
            overlay[:, :] = (40, 220, 40)
            mask = result.candidate.mask > 0
            output[mask] = cv2.addWeighted(output, 0.45, overlay, 0.55, 0)[mask]

        x1, y1, x2, y2 = result.candidate.bbox_xyxy
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 220, 40), 2)
        label = f"cube {result.candidate.confidence:.2f}"
        cv2.putText(output, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 40), 2)

        if draw_mask_overlay and result.candidate.keypoints_xy is not None:
            for point in result.candidate.keypoints_xy:
                cv2.circle(output, tuple(point.astype(int)), 4, (0, 220, 255), -1)

        if draw_mask_overlay and result.candidate.contour is not None:
            contour = result.candidate.contour.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(output, [contour], isClosed=True, color=(0, 255, 255), thickness=2)

    return output


def build_lab_chroma_debug_view(crop_bgr: NDArray[np.uint8]) -> NDArray[np.uint8] | None:
    if crop_bgr.size == 0:
        return None

    reliable_mask = _reliable_color_mask(crop_bgr)
    if int(reliable_mask.sum()) < 10:
        reliable_mask = np.ones(crop_bgr.shape[:2], dtype=bool)

    balanced = _gray_world_white_balance(crop_bgr, reliable_mask)
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = COLOR_DEBUG_LIGHTNESS
    lab[:, :, 1:] = np.clip(
        128.0 + (lab[:, :, 1:].astype(np.float32) - 128.0) * COLOR_DEBUG_CHROMA_CONTRAST,
        0,
        255,
    ).astype(np.uint8)

    chroma_view = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rejected_mask = ~reliable_mask
    chroma_view[rejected_mask] = (chroma_view[rejected_mask] * 0.25).astype(np.uint8)
    return chroma_view


def build_luminance_debug_view(frame_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0]
    return cv2.cvtColor(luminance, cv2.COLOR_GRAY2BGR)


def lab_ab_samples_from_bbox(
    frame_bgr: NDArray[np.uint8],
    bbox_xyxy: tuple[int, int, int, int],
    *,
    sample_mask: NDArray[np.uint8] | None = None,
    max_samples: int = 6000,
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    x1, y1, x2, y2 = _clip_bbox(bbox_xyxy, frame_bgr.shape[1], frame_bgr.shape[0])
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    crop = frame_bgr[y1:y2, x1:x2]
    crop_sample_mask = _crop_sample_mask(sample_mask, (x1, y1, x2, y2), crop.shape[:2])
    if crop_sample_mask is not None and not bool(crop_sample_mask.any()):
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    reliable_mask = _reliable_color_mask(crop)
    if crop_sample_mask is not None:
        reliable_mask = reliable_mask & crop_sample_mask
    if int(reliable_mask.sum()) < 10:
        reliable_mask = (
            crop_sample_mask.copy()
            if crop_sample_mask is not None
            else np.ones(crop.shape[:2], dtype=bool)
        )
    if not bool(reliable_mask.any()):
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    balanced = _gray_world_white_balance(crop, reliable_mask)
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    ab_samples = lab[:, :, 1:3][reliable_mask].astype(np.float32)
    bgr_samples = balanced[reliable_mask]
    if len(ab_samples) > max_samples:
        indexes = np.linspace(0, len(ab_samples) - 1, max_samples, dtype=np.int32)
        ab_samples = ab_samples[indexes]
        bgr_samples = bgr_samples[indexes]
    return ab_samples, bgr_samples


def save_lab_ab_samples_plot(
    ab_samples: NDArray[np.float32],
    bgr_samples: NDArray[np.uint8],
    output_path: str,
    *,
    title: str = "Cube bbox reliable pixels in LAB a/b",
    point_size: float = 4,
) -> None:
    if len(ab_samples) == 0:
        raise ValueError("No LAB a/b samples available for plotting")

    cache_dir = Path(".matplotlib-cache")
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb_samples = cv2.cvtColor(bgr_samples.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    colors = rgb_samples.astype(np.float32) / 255.0
    figure, axes = plt.subplots(figsize=(7, 7), dpi=140)
    axes.scatter(
        ab_samples[:, 0],
        ab_samples[:, 1],
        c=colors,
        s=point_size,
        alpha=0.35,
        linewidths=0,
    )
    axes.axvline(128, color="0.75", linewidth=1)
    axes.axhline(128, color="0.75", linewidth=1)
    axes.set_xlim(0, 255)
    axes.set_ylim(0, 255)
    axes.set_xlabel("LAB a")
    axes.set_ylabel("LAB b")
    axes.set_title(title)
    axes.grid(True, color="0.9", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_rgb_samples_plot(
    bgr_samples: NDArray[np.uint8],
    output_path: str,
    *,
    title: str = "Reliable pixels in RGB",
    point_size: float = 4,
    normalize_rgb: bool = False,
    color_by_cluster: bool = False,
    cluster_count: int = 6,
    cluster_min_brightness: float = 40.0,
    raw_density_filter_radius: float = 0.0,
    raw_density_filter_min_neighbors: int = 0,
    density_cluster_raw_eps: float = RAW_RGB_DENSITY_EPS,
    density_cluster_min_neighbors: int = NORMALIZED_RGB_DENSITY_MIN_POINTS,
    cluster_display_mode: str = "raw",
) -> list[str]:
    if len(bgr_samples) == 0:
        raise ValueError("No RGB samples available for plotting")

    cache_dir = Path(".matplotlib-cache")
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb_samples = cv2.cvtColor(bgr_samples.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    use_raw_density_clusters = cluster_display_mode == "raw-density"
    if color_by_cluster or use_raw_density_clusters:
        if use_raw_density_clusters:
            cluster_indexes, cluster_centroids = _raw_density_rgb_sample_cluster_indexes(
                rgb_samples,
                cluster_count=cluster_count,
                min_brightness=cluster_min_brightness,
                eps=density_cluster_raw_eps,
                min_points=density_cluster_min_neighbors,
            )
        else:
            cluster_indexes = np.full(len(rgb_samples), -1, dtype=np.int32)
            cluster_centroids = np.empty((0, 3), dtype=np.float32)
        unknown_color = np.asarray([0.55, 0.55, 0.55], dtype=np.float32)
        colors = np.asarray(
            [
                (
                    _cluster_rgb_color(cluster_index, cluster_centroids) / 255.0
                    if cluster_index >= 0
                    else unknown_color
                )
                for cluster_index in cluster_indexes
            ],
            dtype=np.float32,
        )
    else:
        colors = rgb_samples.astype(np.float32) / 255.0
    plot_samples = rgb_samples.astype(np.float32)
    axis_limit = 255
    axis_label_prefix = ""
    if normalize_rgb:
        plot_samples = _normalize_rgb_brightness(plot_samples)
        axis_limit = 1.0
        axis_label_prefix = "normalized "

    output_paths = _rgb_plot_output_paths(output_path)
    views = [
        ("rg", 18, -90),
        ("rb", 18, 0),
        ("gb", 18, 90),
    ]
    saved_paths = []
    for output_path_for_view, (view_name, elev, azim) in zip(output_paths, views):
        figure = plt.figure(figsize=(8, 7), dpi=140)
        axes = figure.add_subplot(111, projection="3d")
        axes.scatter(
            plot_samples[:, 0],
            plot_samples[:, 1],
            plot_samples[:, 2],
            c=colors,
            s=point_size,
            alpha=0.35,
            linewidths=0,
            depthshade=False,
        )
        axes.set_xlim(0, axis_limit)
        axes.set_ylim(0, axis_limit)
        axes.set_zlim(0, axis_limit)
        axes.set_xlabel(f"{axis_label_prefix}R")
        axes.set_ylabel(f"{axis_label_prefix}G")
        axes.set_zlabel(f"{axis_label_prefix}B")
        axes.set_title(f"{title} ({view_name.upper()} view)")
        axes.view_init(elev=elev, azim=azim)
        figure.tight_layout()
        figure.savefig(output_path_for_view)
        plt.close(figure)
        saved_paths.append(str(output_path_for_view))
    return saved_paths


def save_rgb_samples_interactive_plot(
    bgr_samples: NDArray[np.uint8],
    output_path: str,
    *,
    title: str = "Reliable pixels in RGB",
    normalize_rgb: bool = False,
    point_size: float = 2,
    cluster_count: int = 6,
    cluster_min_brightness: float = 40.0,
    color_by_cluster: bool = False,
    raw_density_filter_radius: float = 0.0,
    raw_density_filter_min_neighbors: int = 0,
    density_cluster_eps_radians: float = NORMALIZED_RGB_DENSITY_EPS_RADIANS,
    density_cluster_raw_eps: float = RAW_RGB_DENSITY_EPS,
    density_cluster_min_neighbors: int = NORMALIZED_RGB_DENSITY_MIN_POINTS,
    cluster_display_mode: str = "raw",
    cluster_indexes: NDArray[np.int32] | None = None,
    cluster_centroids: NDArray[np.float32] | None = None,
) -> str:
    if len(bgr_samples) == 0:
        raise ValueError("No RGB samples available for plotting")

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required for --rgb-html-plot. Install it in the Python "
            "environment used to run this script."
        ) from exc

    rgb_samples = cv2.cvtColor(bgr_samples.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    plot_samples = rgb_samples.astype(np.float32)
    axis_range = [0, 255]
    axis_prefix = ""
    if normalize_rgb:
        plot_samples = _normalize_rgb_brightness(plot_samples)
        axis_range = [0, 1]
        axis_prefix = "normalized "

    has_external_clusters = cluster_indexes is not None and cluster_centroids is not None
    use_raw_density_clusters = cluster_display_mode == "raw-density"
    if has_external_clusters:
        cluster_indexes = np.asarray(cluster_indexes, dtype=np.int32).reshape(-1)
        if len(cluster_indexes) != len(rgb_samples):
            raise ValueError("cluster_indexes must have one entry per RGB sample")
        cluster_centroids = np.asarray(cluster_centroids, dtype=np.float32).reshape(-1, 3)
    elif use_raw_density_clusters:
        cluster_indexes, cluster_centroids = _raw_density_rgb_sample_cluster_indexes(
            rgb_samples,
            cluster_count=cluster_count,
            min_brightness=cluster_min_brightness,
            eps=density_cluster_raw_eps,
            min_points=density_cluster_min_neighbors,
        )
    else:
        cluster_indexes = np.full(len(rgb_samples), -1, dtype=np.int32)
        cluster_centroids = np.empty((0, 3), dtype=np.float32)
    traces = _interactive_sample_traces(
        go,
        plot_samples,
        rgb_samples,
        cluster_indexes,
        cluster_centroids,
        point_size=point_size,
        color_by_cluster=color_by_cluster or use_raw_density_clusters,
        hovertemplate=(
            f"{axis_prefix}R=%{{x:.3f}}<br>"
            f"{axis_prefix}G=%{{y:.3f}}<br>"
            f"{axis_prefix}B=%{{z:.3f}}<br>"
            "raw RGB=(%{customdata[0]}, %{customdata[1]}, %{customdata[2]})<br>"
            "cluster=%{customdata[3]}<extra></extra>"
        ),
    )
    if normalize_rgb and has_external_clusters:
        traces.extend(
            _normalized_rgb_cluster_perimeter_traces(
                go,
                plot_samples,
                cluster_indexes,
                cluster_centroids,
            )
        )
    elif use_raw_density_clusters:
        traces.extend(
            _normalized_rgb_cluster_perimeter_traces(
                go,
                plot_samples,
                cluster_indexes,
                cluster_centroids,
            )
        )
    elif normalize_rgb:
        density_cluster_indexes, density_cluster_centroids = _density_rgb_sample_cluster_indexes(
            rgb_samples,
            cluster_count=cluster_count,
            min_brightness=cluster_min_brightness,
            eps_radians=density_cluster_eps_radians,
            min_points=density_cluster_min_neighbors,
        )
        traces.extend(
            _normalized_rgb_cluster_perimeter_traces(
                go,
                plot_samples,
                density_cluster_indexes,
                density_cluster_centroids,
            )
        )
    figure = go.Figure(data=traces)
    figure.update_layout(
        title=title,
        scene={
            "xaxis": {"title": f"{axis_prefix}R", "range": axis_range},
            "yaxis": {"title": f"{axis_prefix}G", "range": axis_range},
            "zaxis": {"title": f"{axis_prefix}B", "range": axis_range},
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 48},
    )
    figure.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path


def save_hsv_samples_interactive_plot(
    bgr_samples: NDArray[np.uint8],
    output_path: str,
    *,
    title: str = "Reliable pixels in HSV",
    point_size: float = 2,
    color_by_cluster: bool = False,
    cluster_count: int = 6,
    cluster_min_brightness: float = 40.0,
    raw_density_filter_radius: float = 0.0,
    raw_density_filter_min_neighbors: int = 0,
) -> str:
    if len(bgr_samples) == 0:
        raise ValueError("No HSV samples available for plotting")

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required for --hsv-html-plot. Install it in the Python "
            "environment used to run this script."
        ) from exc

    rgb_samples = cv2.cvtColor(bgr_samples.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    hsv_samples = cv2.cvtColor(bgr_samples.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    cluster_indexes = np.full(len(rgb_samples), -1, dtype=np.int32)
    cluster_centroids = np.empty((0, 3), dtype=np.float32)
    figure = go.Figure(
        data=_interactive_sample_traces(
            go,
            hsv_samples.astype(np.float32),
            rgb_samples,
            cluster_indexes,
            cluster_centroids,
            point_size=point_size,
            color_by_cluster=color_by_cluster,
            hovertemplate=(
                "H=%{x:.0f}<br>"
                "S=%{y:.0f}<br>"
                "V=%{z:.0f}<br>"
                "raw RGB=(%{customdata[0]}, %{customdata[1]}, %{customdata[2]})<br>"
                "cluster=%{customdata[3]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=title,
        scene={
            "xaxis": {"title": "H", "range": [0, 179]},
            "yaxis": {"title": "S", "range": [0, 255]},
            "zaxis": {"title": "V", "range": [0, 255]},
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 48},
    )
    figure.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path

def _interactive_sample_traces(
    go,
    plot_samples: NDArray[np.float32],
    rgb_samples: NDArray[np.uint8],
    cluster_indexes: NDArray[np.int32],
    cluster_centroids: NDArray[np.float32],
    *,
    point_size: float,
    color_by_cluster: bool,
    hovertemplate: str,
) -> list:
    customdata = np.column_stack([rgb_samples, cluster_indexes + 1])
    if not color_by_cluster:
        return _raw_color_sample_traces(
            go,
            plot_samples,
            rgb_samples,
            cluster_indexes,
            customdata,
            point_size=point_size,
            hovertemplate=hovertemplate,
        )

    traces = []
    for cluster_index in sorted(set(int(index) for index in cluster_indexes)):
        member_mask = cluster_indexes == cluster_index
        marker_color = _cluster_hex_color(cluster_index, cluster_centroids)
        trace_name = (
            f"cluster {cluster_index + 1}"
            if cluster_index >= 0
            else "cluster unknown"
        )
        traces.append(
            go.Scatter3d(
                x=plot_samples[member_mask, 0],
                y=plot_samples[member_mask, 1],
                z=plot_samples[member_mask, 2],
                mode="markers",
                marker={
                    "size": point_size,
                    "color": marker_color,
                    "opacity": 0.75,
                },
                customdata=customdata[member_mask],
                hovertemplate=hovertemplate,
                name=trace_name,
            )
        )
    return traces


def _normalized_rgb_cluster_perimeter_traces(
    go,
    plot_samples: NDArray[np.float32],
    cluster_indexes: NDArray[np.int32],
    cluster_centroids: NDArray[np.float32],
) -> list:
    traces = []
    for cluster_index in sorted(set(int(index) for index in cluster_indexes)):
        if cluster_index < 0:
            continue
        member_points = plot_samples[cluster_indexes == cluster_index]
        if len(member_points) < 3:
            continue

        boundary_points = _projected_form_fit_boundary_points(member_points)
        if len(boundary_points) < 3:
            continue

        boundary_points = np.vstack([boundary_points, boundary_points[0]])
        traces.append(
            go.Scatter3d(
                x=boundary_points[:, 0],
                y=boundary_points[:, 1],
                z=boundary_points[:, 2],
                mode="lines",
                line={
                    "color": _cluster_hex_color(cluster_index, cluster_centroids),
                    "width": 7,
                },
                name=f"cluster {cluster_index + 1} perimeter",
                hovertemplate=f"cluster {cluster_index + 1} perimeter<extra></extra>",
            )
        )
    return traces


def _density_rgb_sample_cluster_indexes(
    rgb_samples: NDArray[np.uint8],
    *,
    cluster_count: int,
    min_brightness: float,
    eps_radians: float = NORMALIZED_RGB_DENSITY_EPS_RADIANS,
    min_points: int = NORMALIZED_RGB_DENSITY_MIN_POINTS,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    samples = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    labels = np.full(len(samples), -1, dtype=np.int32)
    if len(samples) == 0 or cluster_count <= 0:
        return labels, np.empty((0, 3), dtype=np.float32)

    brightness = np.linalg.norm(samples, axis=1)
    usable_indexes = np.flatnonzero(brightness >= min_brightness).astype(np.int32)
    usable = samples[usable_indexes]
    if len(usable) < min_points:
        return labels, np.empty((0, 3), dtype=np.float32)

    unit_samples = _normalize_rgb_brightness(usable)
    usable_labels = _density_bin_labels(
        unit_samples,
        eps_radians=eps_radians,
        min_points=min_points,
    )

    clusters = [
        np.flatnonzero(usable_labels == label).astype(np.int32)
        for label in sorted(set(int(value) for value in usable_labels if value >= 0))
    ]
    clusters = [cluster for cluster in clusters if len(cluster) >= min_points]
    clusters = _merge_smallest_index_clusters_to_nearest_neighbor(
        clusters,
        samples_for_distance=unit_samples,
        target_count=cluster_count,
        normalize_centroids=True,
    )

    centroids = []
    for next_label, cluster in enumerate(clusters):
        labels[usable_indexes[cluster]] = next_label
        centroids.append(samples[usable_indexes[cluster]].mean(axis=0))

    return labels, np.asarray(centroids, dtype=np.float32)


def _raw_density_rgb_sample_cluster_indexes(
    rgb_samples: NDArray[np.uint8],
    *,
    cluster_count: int,
    min_brightness: float,
    eps: float = RAW_RGB_DENSITY_EPS,
    min_points: int = NORMALIZED_RGB_DENSITY_MIN_POINTS,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    samples = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    labels = np.full(len(samples), -1, dtype=np.int32)
    if len(samples) == 0 or cluster_count <= 0:
        return labels, np.empty((0, 3), dtype=np.float32)

    brightness = np.linalg.norm(samples, axis=1)
    usable_indexes = np.flatnonzero(brightness >= min_brightness).astype(np.int32)
    usable = samples[usable_indexes]
    if len(usable) < min_points:
        return labels, np.empty((0, 3), dtype=np.float32)

    usable_labels = _raw_density_bin_labels(
        usable,
        eps=eps,
        min_points=min_points,
    )
    clusters = [
        np.flatnonzero(usable_labels == label).astype(np.int32)
        for label in sorted(set(int(value) for value in usable_labels if value >= 0))
    ]
    clusters = [cluster for cluster in clusters if len(cluster) >= min_points]
    clusters = _merge_smallest_index_clusters_to_nearest_neighbor(
        clusters,
        samples_for_distance=usable,
        target_count=cluster_count,
        normalize_centroids=False,
    )

    centroids = []
    for next_label, cluster in enumerate(clusters):
        labels[usable_indexes[cluster]] = next_label
        centroids.append(samples[usable_indexes[cluster]].mean(axis=0))

    return labels, np.asarray(centroids, dtype=np.float32)


def _merge_smallest_index_clusters_to_nearest_neighbor(
    clusters: list[NDArray[np.int32]],
    *,
    samples_for_distance: NDArray[np.float32],
    target_count: int,
    normalize_centroids: bool,
) -> list[NDArray[np.int32]]:
    clusters = [np.asarray(cluster, dtype=np.int32).reshape(-1) for cluster in clusters]
    clusters = [cluster for cluster in clusters if len(cluster) > 0]
    while len(clusters) > target_count:
        smallest_index = min(range(len(clusters)), key=lambda index: len(clusters[index]))
        nearest_index = _nearest_index_cluster_index(
            clusters,
            smallest_index,
            samples_for_distance=samples_for_distance,
            normalize_centroids=normalize_centroids,
        )
        clusters[nearest_index] = np.concatenate(
            [clusters[nearest_index], clusters[smallest_index]],
            axis=0,
        )
        del clusters[smallest_index]
    clusters.sort(key=len, reverse=True)
    return clusters


def _nearest_index_cluster_index(
    clusters: list[NDArray[np.int32]],
    source_index: int,
    *,
    samples_for_distance: NDArray[np.float32],
    normalize_centroids: bool,
) -> int:
    source_centroid = _index_cluster_distance_centroid(
        clusters[source_index],
        samples_for_distance=samples_for_distance,
        normalize_centroids=normalize_centroids,
    )
    best_index = 0 if source_index != 0 else 1
    best_distance = np.inf
    for index, cluster in enumerate(clusters):
        if index == source_index:
            continue
        centroid = _index_cluster_distance_centroid(
            cluster,
            samples_for_distance=samples_for_distance,
            normalize_centroids=normalize_centroids,
        )
        distance = float(np.linalg.norm(source_centroid - centroid))
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def _index_cluster_distance_centroid(
    cluster: NDArray[np.int32],
    *,
    samples_for_distance: NDArray[np.float32],
    normalize_centroids: bool,
) -> NDArray[np.float32]:
    centroid = samples_for_distance[np.asarray(cluster, dtype=np.int32)].mean(axis=0)
    if normalize_centroids:
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-6:
            centroid = centroid / norm
    return centroid


def _density_bin_labels(
    unit_samples: NDArray[np.float32],
    *,
    eps_radians: float,
    min_points: int,
) -> NDArray[np.int32]:
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


def _raw_density_bin_labels(
    rgb_samples: NDArray[np.float32],
    *,
    eps: float,
    min_points: int,
) -> NDArray[np.int32]:
    samples = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    labels = np.full(len(samples), -1, dtype=np.int32)
    if len(samples) == 0:
        return labels

    cell_size = max(float(eps), 1e-6)
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


def _projected_form_fit_boundary_points(points_3d: NDArray[np.float32]) -> NDArray[np.float32]:
    points = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
    if len(points) < 3:
        return np.empty((0, 3), dtype=np.float32)

    centered = points - points.mean(axis=0, keepdims=True)
    _, singular_values, vectors_t = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) == 0 or float(singular_values[0]) <= 1e-6:
        return np.empty((0, 3), dtype=np.float32)
    if len(singular_values) < 2 or float(singular_values[1]) <= 1e-6:
        return _projected_line_boundary_points(points, centered, vectors_t)

    points_2d = centered @ vectors_t[:2].T
    boundary_2d = _form_fit_boundary_2d(points_2d)
    if len(boundary_2d) < 3:
        hull_indexes = _convex_hull_indexes_2d(points_2d)
        if len(hull_indexes) < 3:
            return np.empty((0, 3), dtype=np.float32)
        return points[hull_indexes]

    return points.mean(axis=0, keepdims=True) + boundary_2d @ vectors_t[:2]


def _projected_line_boundary_points(
    points: NDArray[np.float32],
    centered: NDArray[np.float32],
    vectors_t: NDArray[np.float32],
) -> NDArray[np.float32]:
    axis = vectors_t[0]
    projections = centered @ axis
    min_projection = float(projections.min())
    max_projection = float(projections.max())
    if max_projection - min_projection <= 1e-6:
        return np.empty((0, 3), dtype=np.float32)

    perpendicular = _stable_perpendicular_axis(axis, vectors_t)
    closest_line_points = np.outer(projections, axis)
    residual_distances = np.linalg.norm(centered - closest_line_points, axis=1)
    finite_residuals = residual_distances[np.isfinite(residual_distances)]
    line_length = max_projection - min_projection
    if len(finite_residuals) > 0:
        radius = max(float(np.percentile(finite_residuals, 90)), line_length * 0.035)
    else:
        radius = line_length * 0.035
    radius = max(radius, 1e-4)

    center = points.mean(axis=0)
    start = center + axis * min_projection
    end = center + axis * max_projection
    return np.asarray(
        [
            start - perpendicular * radius,
            end - perpendicular * radius,
            end + perpendicular * radius,
            start + perpendicular * radius,
        ],
        dtype=np.float32,
    )


def _stable_perpendicular_axis(
    axis: NDArray[np.float32],
    vectors_t: NDArray[np.float32],
) -> NDArray[np.float32]:
    if len(vectors_t) >= 2:
        perpendicular = vectors_t[1]
        norm = float(np.linalg.norm(perpendicular))
        if norm > 1e-6:
            return perpendicular / norm

    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(axis, fallback))) > 0.9:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    perpendicular = np.cross(axis, fallback)
    norm = max(float(np.linalg.norm(perpendicular)), 1e-6)
    return perpendicular / norm


def _form_fit_boundary_2d(points_2d: NDArray[np.float32]) -> NDArray[np.float32]:
    points = np.asarray(points_2d, dtype=np.float32).reshape(-1, 2)
    unique_points = np.unique(np.round(points, decimals=7), axis=0).astype(np.float32)
    if len(unique_points) < 8:
        hull_indexes = _convex_hull_indexes_2d(points)
        return points[hull_indexes] if len(hull_indexes) >= 3 else np.empty((0, 2), dtype=np.float32)

    min_xy = unique_points.min(axis=0)
    max_xy = unique_points.max(axis=0)
    span = max_xy - min_xy
    max_span = float(max(span))
    if max_span <= 1e-6:
        return np.empty((0, 2), dtype=np.float32)

    canvas_size = 220
    margin = 12
    scale = (canvas_size - margin * 2) / max_span
    pixels = np.round((unique_points - min_xy) * scale + margin).astype(np.int32)
    nearest_spacing = _median_nearest_neighbor_distance_2d(unique_points)
    radius = int(np.clip(round(nearest_spacing * scale * 1.6), 2, 14))

    mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    for x, y in pixels:
        cv2.circle(mask, (int(x), int(y)), radius, 255, -1, cv2.LINE_AA)

    kernel_radius = max(1, radius // 2)
    kernel = np.ones((kernel_radius * 2 + 1, kernel_radius * 2 + 1), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return np.empty((0, 2), dtype=np.float32)

    epsilon = max(1.0, radius * 0.45)
    contour = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)
    if len(contour) < 3:
        return np.empty((0, 2), dtype=np.float32)
    return ((contour.astype(np.float32) - margin) / scale) + min_xy


def _median_nearest_neighbor_distance_2d(points_2d: NDArray[np.float32]) -> float:
    points = np.asarray(points_2d, dtype=np.float32).reshape(-1, 2)
    if len(points) < 2:
        return 1.0

    max_samples = 400
    if len(points) > max_samples:
        indexes = np.linspace(0, len(points) - 1, max_samples, dtype=np.int32)
        points = points[indexes]

    deltas = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    distances[distances <= 1e-9] = np.inf
    nearest = distances.min(axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if len(nearest) == 0:
        return 1.0
    return max(float(np.median(nearest)), 1e-6)


def _projected_convex_hull_indexes(points_3d: NDArray[np.float32]) -> NDArray[np.int32]:
    points = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
    centered = points - points.mean(axis=0, keepdims=True)
    _, singular_values, vectors_t = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 2 or float(singular_values[1]) <= 1e-6:
        return np.empty((0,), dtype=np.int32)

    points_2d = centered @ vectors_t[:2].T
    return _convex_hull_indexes_2d(points_2d)


def _convex_hull_indexes_2d(points_2d: NDArray[np.float32]) -> NDArray[np.int32]:
    points = np.asarray(points_2d, dtype=np.float32).reshape(-1, 2)
    order = sorted(range(len(points)), key=lambda index: (points[index, 0], points[index, 1]))

    unique_order = []
    seen = set()
    for index in order:
        key = (round(float(points[index, 0]), 7), round(float(points[index, 1]), 7))
        if key in seen:
            continue
        seen.add(key)
        unique_order.append(index)
    if len(unique_order) < 3:
        return np.asarray(unique_order, dtype=np.int32)

    def cross(origin: int, point_a: int, point_b: int) -> float:
        oa = points[point_a] - points[origin]
        ob = points[point_b] - points[origin]
        return float(oa[0] * ob[1] - oa[1] * ob[0])

    lower = []
    for index in unique_order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], index) <= 0:
            lower.pop()
        lower.append(index)

    upper = []
    for index in reversed(unique_order):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], index) <= 0:
            upper.pop()
        upper.append(index)

    return np.asarray(lower[:-1] + upper[:-1], dtype=np.int32)


def _raw_color_sample_traces(
    go,
    plot_samples: NDArray[np.float32],
    rgb_samples: NDArray[np.uint8],
    cluster_indexes: NDArray[np.int32],
    customdata: NDArray[np.int32],
    *,
    point_size: float,
    hovertemplate: str,
) -> list:
    traces = []
    for trace_name, member_mask in (
        ("samples (raw color)", cluster_indexes >= 0),
        ("cluster unknown", cluster_indexes < 0),
    ):
        if not bool(member_mask.any()):
            continue
        member_rgb_samples = rgb_samples[member_mask]
        hex_colors = [
            f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
            for rgb in member_rgb_samples
        ]
        traces.append(
            go.Scatter3d(
                x=plot_samples[member_mask, 0],
                y=plot_samples[member_mask, 1],
                z=plot_samples[member_mask, 2],
                mode="markers",
                marker={
                    "size": point_size,
                    "color": hex_colors,
                    "opacity": 0.75,
                },
                customdata=customdata[member_mask],
                hovertemplate=hovertemplate,
                name=trace_name,
            )
        )
    return traces


def _cluster_hex_colors(
    cluster_indexes: NDArray[np.int32],
    centroids: NDArray[np.float32],
) -> list[str]:
    colors = []
    for cluster_index in cluster_indexes:
        colors.append(_cluster_hex_color(int(cluster_index), centroids))
    return colors


def _cluster_hex_color(
    cluster_index: int,
    centroids: NDArray[np.float32],
) -> str:
    if cluster_index < 0 or cluster_index >= len(centroids):
        return "rgb(140,140,140)"
    centroid = _cluster_rgb_color(cluster_index, centroids)
    return f"rgb({int(centroid[0])},{int(centroid[1])},{int(centroid[2])})"


def _cluster_rgb_color(
    cluster_index: int,
    centroids: NDArray[np.float32],
) -> NDArray[np.float32]:
    if cluster_index < 0 or cluster_index >= len(centroids):
        return np.array([140, 140, 140], dtype=np.float32)
    return np.clip(centroids[cluster_index], 0, 255)


def _rgb_plot_output_paths(output_path: str) -> list[Path]:
    path = Path(output_path)
    suffix = path.suffix or ".png"
    return [
        path.with_name(f"{path.stem}_{view}{suffix}")
        for view in ("rg", "rb", "gb")
    ]


def save_lab_ab_plot(
    frame_bgr: NDArray[np.uint8],
    bbox_xyxy: tuple[int, int, int, int],
    output_path: str,
    *,
    sample_mask: NDArray[np.uint8] | None = None,
) -> None:
    ab_samples, bgr_samples = lab_ab_samples_from_bbox(
        frame_bgr,
        bbox_xyxy,
        sample_mask=sample_mask,
    )
    save_lab_ab_samples_plot(ab_samples, bgr_samples, output_path)


def _reliable_color_mask(crop_bgr: NDArray[np.uint8]) -> NDArray[np.bool_]:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    hue = hsv[:, :, 0]
    value = hsv[:, :, 2]
    max_channel = crop_bgr.max(axis=2)
    min_channel = crop_bgr.min(axis=2)

    orange_candidate = (hue <= 30) & (saturation >= 20) & (value >= 35)
    bright_white_candidate = value >= 180
    colorful_enough = (
        (saturation >= COLOR_DEBUG_MIN_SATURATION) | bright_white_candidate | orange_candidate
    )
    not_too_dark = value >= COLOR_DEBUG_MIN_VALUE
    not_clipped = (max_channel <= COLOR_DEBUG_MAX_VALUE) | orange_candidate
    not_flat_gray = (max_channel - min_channel) >= COLOR_DEBUG_MIN_CHANNEL_DELTA
    return colorful_enough & not_too_dark & not_clipped & not_flat_gray


def _gray_world_white_balance(
    crop_bgr: NDArray[np.uint8],
    reliable_mask: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    pixels = crop_bgr[reliable_mask].astype(np.float32)
    if pixels.size == 0:
        return crop_bgr.copy()

    channel_means = pixels.mean(axis=0)
    target = float(channel_means.mean())
    scales = target / np.maximum(channel_means, 1.0)
    balanced = crop_bgr.astype(np.float32) * scales.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)


def _clip_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def _crop_sample_mask(
    sample_mask: NDArray[np.uint8] | None,
    bbox_xyxy: tuple[int, int, int, int],
    crop_shape: tuple[int, int],
) -> NDArray[np.bool_] | None:
    if sample_mask is None:
        return None

    x1, y1, x2, y2 = bbox_xyxy
    mask_crop = sample_mask[y1:y2, x1:x2]
    crop_height, crop_width = crop_shape
    if mask_crop.shape[:2] != (crop_height, crop_width):
        mask_crop = cv2.resize(
            mask_crop.astype(np.uint8),
            (crop_width, crop_height),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask_crop > 0
