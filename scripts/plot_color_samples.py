"""Plot a cached color-sample file using HDBSCAN clustering."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from live_recognition.color_clustering import (
    hdbscan_cluster_memberships,
    recursive_density_plane_splits,
)

try:
    from test_piece_seg_roi import bgr_samples_to_oklab
except ModuleNotFoundError:
    from scripts.test_piece_seg_roi import bgr_samples_to_oklab
from live_recognition.visualization import save_oklab_samples_interactive_plot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Oklab cross-section cluster plots from cached samples."
    )
    parser.add_argument("samples_file", type=Path, help="NPZ file from test_piece_seg_roi.py")
    parser.add_argument(
        "--oklab-html-plot", type=Path, required=True,
        help="Output HTML path for the 3D Oklab plot",
    )
    parser.add_argument("--plotly-point-size", type=float, default=1.0)
    parser.add_argument(
        "--cluster-count", type=int, default=6,
        help="Maximum number of HDBSCAN clusters to retain",
    )
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=10)
    parser.add_argument(
        "--hdbscan-min-samples", type=int, default=None,
        help="HDBSCAN core-point conservatism; defaults to min-cluster-size",
    )
    parser.add_argument(
        "--hdbscan-cluster-selection-method", choices=("eom", "leaf"), default="eom",
    )
    parser.add_argument("--hdbscan-cluster-selection-epsilon", type=float, default=0.0)
    parser.add_argument("--hdbscan-alpha", type=float, default=1.0)
    parser.add_argument("--hdbscan-metric", default="euclidean")
    parser.add_argument("--hdbscan-core-dist-n-jobs", type=int, default=1)
    parser.add_argument(
        "--hdbscan-allow-single-cluster", action="store_true",
        help="Allow HDBSCAN to return one cluster",
    )
    parser.add_argument(
        "--hdbscan-leave-noise",
        action="store_true",
        help="Keep HDBSCAN noise points as an unassigned trace instead of attaching them",
    )
    parser.add_argument(
        "--hdbscan-hierarchy-plot", type=Path,
        help="Optional PNG/PDF path for the HDBSCAN condensed-tree hierarchy graph",
    )
    parser.add_argument("--plane-angle-count", type=int, default=36)
    parser.add_argument("--plane-tilt-count", type=int, default=3)
    parser.add_argument("--plane-max-tilt-degrees", type=float, default=15.0)
    parser.add_argument("--plane-offset-count", type=int, default=25)
    parser.add_argument("--plane-density-grid-size", type=int, default=48)
    parser.add_argument("--plane-density-smoothing", type=float, default=1.5)
    parser.add_argument("--plane-bandwidth", type=float, default=0.0)
    parser.add_argument("--plane-min-child-size", type=int, default=10)
    parser.add_argument("--plane-min-child-fraction", type=float, default=0.15)
    parser.add_argument("--plane-balance-penalty", type=float, default=0.25)
    parser.add_argument("--plane-relative-valley-weight", type=float, default=1.0)
    parser.add_argument("--plane-min-side-mass", type=float, default=0.12)
    parser.add_argument("--plane-cut-mass-weight", type=float, default=0.5)
    parser.add_argument("--plane-cluster-size-weight", type=float, default=0.25)
    return parser


def recursive_axial_density_splits(
    oklab_samples: np.ndarray,
    labels: np.ndarray,
    *,
    target_clusters: int,
    angle_count: int = 180,
    min_child_size: int = 10,
    density_bandwidth: float = 0.0,
    balance_penalty: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Split merged clusters at low-density planes containing the L axis."""
    samples = np.asarray(oklab_samples, dtype=np.float32).reshape(-1, 3)
    output = np.asarray(labels, dtype=np.int32).reshape(-1).copy()
    if len(output) != len(samples):
        raise ValueError("labels must align with Oklab samples")
    target = max(1, int(target_clusters))
    child_size = max(2, int(min_child_size))
    if not np.any(output >= 0):
        output[:] = 0

    next_label = int(output[output >= 0].max() + 1) if np.any(output >= 0) else 0
    angles = np.arange(max(8, int(angle_count)), dtype=np.float32) * np.pi / max(8, int(angle_count))
    directions = np.column_stack([np.cos(angles), np.sin(angles)])

    while len(np.unique(output[output >= 0])) < target:
        best: tuple[float, int, np.ndarray] | None = None
        for cluster_id in np.unique(output[output >= 0]):
            indexes = np.flatnonzero(output == cluster_id)
            if len(indexes) < 2 * child_size:
                continue
            chroma = samples[indexes, 1:3]
            for direction in directions:
                signed = chroma @ direction
                left = signed < 0.0
                right = ~left
                if int(left.sum()) < child_size or int(right.sum()) < child_size:
                    continue
                if density_bandwidth > 0.0:
                    bandwidth = float(density_bandwidth)
                else:
                    bandwidth = max(float(np.quantile(np.abs(signed), 0.25)) * 0.5, 0.002)
                valley_density = float(
                    np.mean(np.exp(-0.5 * (signed / bandwidth) ** 2))
                )
                imbalance = abs(float(left.sum() - right.sum())) / len(indexes)
                score = valley_density + max(0.0, float(balance_penalty)) * imbalance
                if best is None or score < best[0]:
                    best = (score, int(cluster_id), left)
        if best is None:
            break
        _score, cluster_id, left = best
        indexes = np.flatnonzero(output == cluster_id)
        output[indexes[left]] = next_label
        next_label += 1

    retained_ids = np.unique(output[output >= 0])
    remapped = np.full(len(output), -1, dtype=np.int32)
    for new_index, old_index in enumerate(retained_ids):
        remapped[output == old_index] = new_index
    centroids = np.asarray(
        [
            samples[remapped == index].mean(axis=0)
            if np.any(remapped == index) else np.zeros(3, dtype=np.float32)
            for index in range(len(retained_ids))
        ],
        dtype=np.float32,
    )
    return remapped, centroids


def recursive_watershed_splits(
    oklab_samples: np.ndarray,
    labels: np.ndarray,
    *,
    target_clusters: int,
    grid_size: int = 128,
    smoothing: float = 2.0,
    max_peaks: int = 6,
    min_peak_separation: float = 0.015,
    min_child_size: int = 10,
    min_child_fraction: float = 0.15,
    balance_penalty: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Recursively split clusters along low-density ridges in flattened a/b."""
    import cv2
    from scipy import ndimage

    samples = np.asarray(oklab_samples, dtype=np.float32).reshape(-1, 3)
    output = np.asarray(labels, dtype=np.int32).reshape(-1).copy()
    child_size = max(2, int(min_child_size))
    next_label = int(output[output >= 0].max() + 1) if np.any(output >= 0) else 0
    split_boundaries: list[np.ndarray] = []
    candidate_boundaries: list[np.ndarray] = []

    while True:
        best: tuple[float, int, np.ndarray, np.ndarray] | None = None
        best_diagnostic: tuple[float, np.ndarray] | None = None
        for cluster_id in np.unique(output[output >= 0]):
            indexes = np.flatnonzero(output == cluster_id)
            required_child_size = max(
                child_size,
                int(np.ceil(len(indexes) * max(0.0, float(min_child_fraction)))),
            )
            if len(indexes) < 2 * required_child_size:
                continue
            chroma = samples[indexes, 1:3]
            lower = chroma.min(axis=0)
            upper = chroma.max(axis=0)
            span = np.maximum(upper - lower, 1e-5)
            lower -= span * 0.03
            upper += span * 0.03
            size = max(32, int(grid_size))
            histogram, _, _ = np.histogram2d(
                chroma[:, 0], chroma[:, 1], bins=size,
                range=((float(lower[0]), float(upper[0])), (float(lower[1]), float(upper[1]))),
            )
            density = cv2.GaussianBlur(
                histogram.astype(np.float32), (0, 0), max(0.0, float(smoothing))
            )
            peak_mask = density == cv2.dilate(density, np.ones((5, 5), np.uint8))
            # Do not apply an absolute density threshold here. A secondary
            # mode may be globally dense but still sit in a meaningful local
            # valley relative to its neighboring peaks.
            peak_mask &= density > 0.0
            peak_count, peak_labels, peak_stats, peak_centers = cv2.connectedComponentsWithStats(
                peak_mask.astype(np.uint8), 8
            )
            peaks = []
            for peak_id in range(1, peak_count):
                pixels = peak_labels == peak_id
                if not np.any(pixels):
                    continue
                yx = np.argwhere(pixels)
                best_pixel = yx[np.argmax(density[pixels])]
                x, y = best_pixel
                peaks.append((float(density[x, y]), int(x), int(y)))
            peaks.sort(reverse=True)
            selected = []
            for peak in peaks:
                point = np.asarray((
                    lower[0] + (peak[1] + 0.5) / size * (upper[0] - lower[0]),
                    lower[1] + (peak[2] + 0.5) / size * (upper[1] - lower[1]),
                ))
                if all(np.linalg.norm(point - other[1]) >= min_peak_separation for other in selected):
                    selected.append((peak[0], point, peak[1], peak[2]))
                if len(selected) >= max(2, int(max_peaks)):
                    break
            if len(selected) < 2:
                continue
            density_image = density.T
            normalized = density_image / max(float(density_image.max()), 1e-6)
            watershed_image = np.clip(255.0 * (1.0 - normalized), 0, 255).astype(np.uint8)
            coordinates = (chroma - lower) / np.maximum(upper - lower, 1e-8)
            x_indexes = np.clip((coordinates[:, 0] * size).astype(np.int32), 0, size - 1)
            y_indexes = np.clip((coordinates[:, 1] * size).astype(np.int32), 0, size - 1)
            for first_peak_index in range(len(selected) - 1):
                for second_peak_index in range(first_peak_index + 1, len(selected)):
                    first_peak = selected[first_peak_index]
                    second_peak = selected[second_peak_index]
                    markers = np.zeros((size, size), dtype=np.int32)
                    markers[first_peak[3], first_peak[2]] = 1
                    markers[second_peak[3], second_peak[2]] = 2
                    watershed_markers = ndimage.watershed_ift(watershed_image, markers)
                    basin_ids = watershed_markers[y_indexes, x_indexes]
                    first_mask = basin_ids == 1
                    second_mask = basin_ids == 2
                    first_grid = watershed_markers == 1
                    second_grid = watershed_markers == 2
                    contours, _ = cv2.findContours(
                        first_grid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                    )
                    contour = max(contours, key=cv2.contourArea, default=None)
                    if contour is None or len(contour) < 3:
                        continue
                    contour_xy = contour.reshape(-1, 2)
                    boundary_yx = np.column_stack([contour_xy[:, 1], contour_xy[:, 0]])
                    boundary_points = np.column_stack([
                        lower[0] + (boundary_yx[:, 1] + 0.5) / size * (upper[0] - lower[0]),
                        lower[1] + (boundary_yx[:, 0] + 0.5) / size * (upper[1] - lower[1]),
                    ]).astype(np.float32)
                    boundary_mask = np.zeros_like(first_grid, dtype=np.uint8)
                    cv2.drawContours(boundary_mask, [contour], -1, 1, 1)
                    valley_density = float(np.median(density_image[boundary_mask > 0]))
                    peak_density = min(first_peak[0], second_peak[0])
                    relative_contrast = np.log(
                        max(peak_density, 1e-6) / max(valley_density, 1e-6)
                    )
                    imbalance = abs(int(first_mask.sum()) - int(second_mask.sum())) / len(indexes)
                    score = -relative_contrast + max(0.0, float(balance_penalty)) * imbalance
                    if best_diagnostic is None or score < best_diagnostic[0]:
                        best_diagnostic = (score, boundary_points)
                    if (
                        int(first_mask.sum()) < required_child_size
                        or int(second_mask.sum()) < required_child_size
                    ):
                        continue
                    if best is None or score < best[0]:
                        best = (score, int(cluster_id), first_mask, boundary_points)
        if best is None:
            if best_diagnostic is not None:
                candidate_boundaries.append(best_diagnostic[1])
            break
        _score, cluster_id, first_mask, boundary_points = best
        if len(np.unique(output[output >= 0])) >= max(1, int(target_clusters)):
            candidate_boundaries.append(boundary_points)
            break
        indexes = np.flatnonzero(output == cluster_id)
        output[indexes[~first_mask]] = next_label
        next_label += 1
        split_boundaries.append(boundary_points)

    retained_ids = np.unique(output[output >= 0])
    remapped = np.full(len(output), -1, dtype=np.int32)
    for new_index, old_index in enumerate(retained_ids):
        remapped[output == old_index] = new_index
    centroids = np.asarray(
        [samples[remapped == index].mean(axis=0) for index in range(len(retained_ids))],
        dtype=np.float32,
    ) if len(retained_ids) else np.empty((0, 3), dtype=np.float32)
    return remapped, centroids, split_boundaries, candidate_boundaries


def save_ab_density_plot(
    oklab_samples: np.ndarray,
    output_path: Path,
    *,
    cluster_labels: np.ndarray | None = None,
    split_boundaries: list[np.ndarray] | None = None,
    candidate_boundaries: list[np.ndarray] | None = None,
    grid_size: int = 256,
    smoothing: float = 2.0,
    point_size: float = 5.0,
) -> None:
    """Save all a/b samples colored by bounded 2D histogram density."""
    try:
        import cv2
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --ab-density-plot") from exc

    samples = np.asarray(oklab_samples, dtype=np.float32).reshape(-1, 3)
    if len(samples) == 0:
        raise ValueError("At least one Oklab sample is required for a density plot")
    chroma = samples[:, 1:3]
    size = max(16, int(grid_size))
    lower = chroma.min(axis=0)
    upper = chroma.max(axis=0)
    span = np.maximum(upper - lower, 1e-5)
    lower = lower - span * 0.02
    upper = upper + span * 0.02
    histogram, _, _ = np.histogram2d(
        chroma[:, 0], chroma[:, 1], bins=size,
        range=((float(lower[0]), float(upper[0])), (float(lower[1]), float(upper[1]))),
    )
    blurred = cv2.GaussianBlur(
        histogram.astype(np.float32), (0, 0), max(0.0, float(smoothing))
    )
    coordinates = (chroma - lower) / np.maximum(upper - lower, 1e-8)
    x_indexes = np.clip((coordinates[:, 0] * size).astype(np.int32), 0, size - 1)
    y_indexes = np.clip((coordinates[:, 1] * size).astype(np.int32), 0, size - 1)
    density = np.log1p(blurred[x_indexes, y_indexes])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 7), dpi=160)
    scatter = axis.scatter(
        chroma[:, 0], chroma[:, 1], c=density, s=max(1.0, float(point_size)),
        cmap="viridis", alpha=0.8, linewidths=0,
    )
    if split_boundaries:
        for boundary in split_boundaries:
            boundary = np.asarray(boundary, dtype=np.float32).reshape(-1, 2)
            if len(boundary):
                axis.plot(boundary[:, 0], boundary[:, 1], color="white", linewidth=1.5, alpha=0.95)
    if candidate_boundaries:
        for boundary in candidate_boundaries:
            boundary = np.asarray(boundary, dtype=np.float32).reshape(-1, 2)
            if len(boundary):
                axis.plot(
                    boundary[:, 0], boundary[:, 1], color="orange", linewidth=1.5,
                    alpha=0.95, label="best unselected watershed candidate",
                )
    figure.colorbar(scatter, ax=axis, label="log smoothed grid density")
    axis.set_xlabel("Oklab a")
    axis.set_ylabel("Oklab b")
    axis.set_title(f"Flattened Oklab a/b density (grid={size})")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, color="0.88", linewidth=0.7)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = build_parser().parse_args()
    if not args.samples_file.exists():
        raise SystemExit(f"Sample file does not exist: {args.samples_file}")
    with np.load(args.samples_file, allow_pickle=False) as cached:
        if "bgr_samples" not in cached:
            raise SystemExit("Sample file must contain a bgr_samples array")
        bgr_samples = np.asarray(cached["bgr_samples"], dtype=np.uint8).reshape(-1, 3)
    if len(bgr_samples) == 0:
        raise SystemExit("Sample file contains no color samples")

    oklab_samples = bgr_samples_to_oklab(bgr_samples)
    cluster_indexes, cluster_centroids, hdbscan_noise = hdbscan_cluster_memberships(
        oklab_samples,
        cluster_count=args.cluster_count,
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        cluster_selection_method=args.hdbscan_cluster_selection_method,
        cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
        alpha=args.hdbscan_alpha,
        metric=args.hdbscan_metric,
        allow_single_cluster=args.hdbscan_allow_single_cluster,
        core_dist_n_jobs=args.hdbscan_core_dist_n_jobs,
        assign_noise=False,
        hierarchy_plot=args.hdbscan_hierarchy_plot,
    )
    print(
        "HDBSCAN clusters: "
        f"{len(np.unique(cluster_indexes[cluster_indexes >= 0]))}; "
        f"outliers: {int(hdbscan_noise.sum())}"
    )
    cluster_indexes, cluster_centroids, split_planes = recursive_density_plane_splits(
        oklab_samples,
        cluster_indexes,
        target_clusters=args.cluster_count,
        angle_count=args.plane_angle_count,
        tilt_count=args.plane_tilt_count,
        max_tilt_degrees=args.plane_max_tilt_degrees,
        offset_count=args.plane_offset_count,
        density_grid_size=args.plane_density_grid_size,
        density_smoothing=args.plane_density_smoothing,
        plane_bandwidth=args.plane_bandwidth,
        min_child_size=args.plane_min_child_size,
        min_child_fraction=args.plane_min_child_fraction,
        balance_penalty=args.plane_balance_penalty,
        relative_valley_weight=args.plane_relative_valley_weight,
        min_side_mass=args.plane_min_side_mass,
        cut_mass_weight=args.plane_cut_mass_weight,
        cluster_size_weight=args.plane_cluster_size_weight,
    )
    print(
        "Clusters after density-plane splitting: "
        f"{len(np.unique(cluster_indexes[cluster_indexes >= 0]))}"
    )
    print(f"Density-plane splits applied: {len(split_planes)}")
    if not args.hdbscan_leave_noise and np.any(hdbscan_noise) and len(cluster_centroids):
        noise_indexes = np.flatnonzero(hdbscan_noise)
        distances = np.linalg.norm(
            oklab_samples[noise_indexes, None, :] - cluster_centroids[None, :, :],
            axis=2,
        )
        cluster_indexes[noise_indexes] = distances.argmin(axis=1).astype(np.int32)
        cluster_centroids = np.asarray(
            [
                oklab_samples[cluster_indexes == index].mean(axis=0)
                if np.any(cluster_indexes == index) else np.zeros(3, dtype=np.float32)
                for index in range(len(cluster_centroids))
            ],
            dtype=np.float32,
        )
    save_oklab_samples_interactive_plot(
        bgr_samples, str(args.oklab_html_plot),
        title=f"Cached Oklab samples from {args.samples_file.name}",
        point_size=args.plotly_point_size,
        oklab_samples=oklab_samples,
        color_by_cluster=True,
        cluster_trace_original_colors=True,
        cluster_indexes=cluster_indexes,
        cluster_centroids=cluster_centroids,
        split_planes=split_planes,
    )
    print(f"Saved Oklab 3D HTML plot: {args.oklab_html_plot.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
