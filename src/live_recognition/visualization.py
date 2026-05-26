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
    colors = rgb_samples.astype(np.float32) / 255.0
    plot_samples = rgb_samples.astype(np.float32)
    axis_limit = 255
    axis_label_prefix = ""
    if normalize_rgb:
        channel_sum = np.maximum(plot_samples.sum(axis=1, keepdims=True), 1.0)
        plot_samples = plot_samples / channel_sum
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
    point_size: float = 4,
    draw_regression_lines: bool = True,
    regression_cluster_count: int = 6,
    regression_min_brightness: float = 40.0,
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
        channel_sum = np.maximum(plot_samples.sum(axis=1, keepdims=True), 1.0)
        plot_samples = plot_samples / channel_sum
        axis_range = [0, 1]
        axis_prefix = "normalized "

    hex_colors = [
        f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
        for rgb in rgb_samples
    ]
    traces = [
        go.Scatter3d(
            x=plot_samples[:, 0],
            y=plot_samples[:, 1],
            z=plot_samples[:, 2],
            mode="markers",
            marker={
                "size": point_size,
                "color": hex_colors,
                "opacity": 0.75,
            },
            customdata=rgb_samples,
            hovertemplate=(
                f"{axis_prefix}R=%{{x:.3f}}<br>"
                f"{axis_prefix}G=%{{y:.3f}}<br>"
                f"{axis_prefix}B=%{{z:.3f}}<br>"
                "raw RGB=%{customdata}<extra></extra>"
            ),
            name="samples",
        )
    ]
    if draw_regression_lines and not normalize_rgb:
        traces.extend(
            _rgb_origin_regression_line_traces(
                go,
                rgb_samples,
                cluster_count=regression_cluster_count,
                min_brightness=regression_min_brightness,
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


def _rgb_origin_regression_line_traces(
    go,
    rgb_samples: NDArray[np.uint8],
    *,
    cluster_count: int,
    min_brightness: float,
) -> list:
    directions, centroids = rgb_origin_regression_rays(
        rgb_samples,
        cluster_count=cluster_count,
        min_brightness=min_brightness,
    )
    traces = []
    axis_max = 255.0
    for index, (direction, centroid) in enumerate(zip(directions, centroids)):
        endpoint_scale = axis_max / max(float(direction.max()), 1e-6)
        endpoint = direction * endpoint_scale
        color = f"rgb({int(centroid[0])},{int(centroid[1])},{int(centroid[2])})"
        traces.append(
            go.Scatter3d(
                x=[0, endpoint[0]],
                y=[0, endpoint[1]],
                z=[0, endpoint[2]],
                mode="lines",
                line={"color": color, "width": 7},
                name=f"RGB ray {index + 1}",
                hovertemplate=(
                    "origin ray<br>"
                    f"centroid RGB=({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})"
                    "<extra></extra>"
                ),
            )
        )
    return traces


def rgb_origin_regression_rays(
    rgb_samples: NDArray[np.uint8] | NDArray[np.float32],
    *,
    cluster_count: int = 6,
    min_brightness: float = 40.0,
    max_iterations: int = 50,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Fit origin rays by clustering RGB directions and using raw RGB centroids."""
    samples = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    if len(samples) == 0 or cluster_count <= 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )

    brightness = np.linalg.norm(samples, axis=1)
    usable = samples[brightness >= min_brightness]
    if len(usable) == 0:
        usable = samples[brightness > 0]
    if len(usable) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )

    usable_brightness = np.maximum(np.linalg.norm(usable, axis=1, keepdims=True), 1e-6)
    unit_samples = usable / usable_brightness
    k = min(cluster_count, len(unit_samples))
    centers = _initialize_direction_centers(unit_samples, k)

    labels = np.zeros(len(unit_samples), dtype=np.int32)
    for _ in range(max_iterations):
        distances = np.linalg.norm(unit_samples[:, None, :] - centers[None, :, :], axis=2)
        next_labels = distances.argmin(axis=1)
        next_centers = centers.copy()
        for cluster_index in range(k):
            members = unit_samples[next_labels == cluster_index]
            if len(members) > 0:
                center = members.mean(axis=0)
                next_centers[cluster_index] = center / max(float(np.linalg.norm(center)), 1e-6)
        if np.array_equal(labels, next_labels) and np.allclose(centers, next_centers):
            centers = next_centers
            labels = next_labels
            break
        centers = next_centers
        labels = next_labels

    centroids = []
    directions = []
    for cluster_index in range(k):
        members = usable[labels == cluster_index]
        if len(members) == 0:
            continue
        centroid = members.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm <= 1e-6:
            continue
        centroids.append(centroid)
        directions.append(centroid / norm)

    if not centroids:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )

    order = np.argsort([_rgb_hue_sort_key(centroid) for centroid in centroids])
    return (
        np.asarray(directions, dtype=np.float32)[order],
        np.asarray(centroids, dtype=np.float32)[order],
    )


def _initialize_direction_centers(
    unit_samples: NDArray[np.float32],
    cluster_count: int,
) -> NDArray[np.float32]:
    first_index = int(np.argmax(np.linalg.norm(unit_samples, axis=1)))
    centers = [unit_samples[first_index]]
    while len(centers) < cluster_count:
        distances = np.min(
            np.linalg.norm(unit_samples[:, None, :] - np.asarray(centers)[None, :, :], axis=2),
            axis=1,
        )
        centers.append(unit_samples[int(np.argmax(distances))])
    return np.asarray(centers, dtype=np.float32)


def _rgb_hue_sort_key(rgb: NDArray[np.float32]) -> float:
    rgb_u8 = np.clip(np.asarray(rgb).reshape(1, 1, 3), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    return float(hsv[0, 0, 0])


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
