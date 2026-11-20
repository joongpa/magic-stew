from __future__ import annotations

from pathlib import Path

import numpy as np

def hdbscan_cluster_memberships(
    oklab_samples: np.ndarray,
    *,
    cluster_count: int,
    min_cluster_size: int,
    min_samples: int | None,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    alpha: float,
    metric: str,
    allow_single_cluster: bool,
    core_dist_n_jobs: int,
    assign_noise: bool,
    hierarchy_plot: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit HDBSCAN and retain at most the requested number of persistent clusters."""
    try:
        import hdbscan
    except ImportError as exc:
        raise RuntimeError(
            "HDBSCAN is required for plot_color_samples.py. Install it with `pip install hdbscan`."
        ) from exc

    samples = np.asarray(oklab_samples, dtype=np.float32).reshape(-1, 3)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, int(min_cluster_size)),
        min_samples=(max(1, int(min_samples)) if min_samples is not None else None),
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=max(0.0, float(cluster_selection_epsilon)),
        alpha=max(1e-6, float(alpha)),
        metric=metric,
        allow_single_cluster=allow_single_cluster,
        core_dist_n_jobs=int(core_dist_n_jobs),
        prediction_data=False,
    ).fit(samples)
    if hierarchy_plot is not None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required for --hdbscan-hierarchy-plot"
            ) from exc
        hierarchy_plot.parent.mkdir(parents=True, exist_ok=True)
        # The selected-cluster overlay in hdbscan's plotter cannot render
        # infinite lambda values caused by duplicate samples. The unselected
        # condensed tree remains valid and shows the complete hierarchy.
        axis = clusterer.condensed_tree_.plot(select_clusters=False)
        figure = axis.get_figure() if hasattr(axis, "get_figure") else axis.figure
        figure.savefig(hierarchy_plot, dpi=180, bbox_inches="tight")
        plt.close(figure)
    labels = np.asarray(clusterer.labels_, dtype=np.int32)
    cluster_ids = np.unique(labels[labels >= 0])
    max_clusters = max(1, int(cluster_count))
    if len(cluster_ids) > max_clusters:
        persistence = np.asarray(clusterer.cluster_persistence_, dtype=np.float32).reshape(-1)
        ranked = sorted(
            cluster_ids.tolist(),
            key=lambda label: (
                float(persistence[int(label)]) if int(label) < len(persistence) else 0.0,
                int(np.sum(labels == label)),
            ),
            reverse=True,
        )
        retained = np.asarray(ranked[:max_clusters], dtype=np.int32)
        labels[~np.isin(labels, retained)] = -1

    noise_mask = labels < 0
    retained_ids = np.unique(labels[labels >= 0])
    if assign_noise and len(retained_ids):
        retained_centroids = np.asarray(
            [samples[labels == cluster_id].mean(axis=0) for cluster_id in retained_ids],
            dtype=np.float32,
        )
        noise_indexes = np.flatnonzero(labels < 0)
        if len(noise_indexes):
            distances = np.linalg.norm(
                samples[noise_indexes, None, :] - retained_centroids[None, :, :],
                axis=2,
            )
            labels[noise_indexes] = retained_ids[distances.argmin(axis=1)]
    remapped = np.full(len(labels), -1, dtype=np.int32)
    for new_index, old_index in enumerate(retained_ids):
        remapped[labels == old_index] = new_index
    centroids = np.asarray(
        [
            samples[remapped == index].mean(axis=0)
            if np.any(remapped == index) else np.zeros(3, dtype=np.float32)
            for index in range(len(retained_ids))
        ],
        dtype=np.float32,
    )
    return remapped, centroids, noise_mask


def recursive_density_plane_splits(
    oklab_samples: np.ndarray,
    labels: np.ndarray,
    *,
    target_clusters: int,
    angle_count: int,
    tilt_count: int,
    max_tilt_degrees: float,
    offset_count: int,
    density_grid_size: int,
    density_smoothing: float,
    plane_bandwidth: float,
    min_child_size: int,
    min_child_fraction: float,
    balance_penalty: float,
    relative_valley_weight: float = 1.0,
    min_side_mass: float = 0.12,
    cut_mass_weight: float = 0.5,
    cluster_size_weight: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, float, np.ndarray]]]:
    """Recursively split clusters with low-density, near-L-axis planes."""
    from scipy.ndimage import gaussian_filter

    samples = np.asarray(oklab_samples, dtype=np.float32).reshape(-1, 3)
    output = np.asarray(labels, dtype=np.int32).reshape(-1).copy()
    next_label = int(output[output >= 0].max() + 1) if np.any(output >= 0) else 0
    split_planes: list[tuple[np.ndarray, float, np.ndarray]] = []

    def point_density(points: np.ndarray) -> np.ndarray:
        size = max(16, int(density_grid_size))
        lower = points.min(axis=0)
        upper = points.max(axis=0)
        span = np.maximum(upper - lower, 1e-5)
        lower -= span * 0.03
        upper += span * 0.03
        grid, _ = np.histogramdd(
            points, bins=size,
            range=tuple((float(lo), float(hi)) for lo, hi in zip(lower, upper)),
        )
        grid = gaussian_filter(grid.astype(np.float32), sigma=max(0.0, float(density_smoothing)))
        coordinates = (points - lower) / np.maximum(upper - lower, 1e-8)
        indexes = np.clip((coordinates * size).astype(np.int32), 0, size - 1)
        density = grid[indexes[:, 0], indexes[:, 1], indexes[:, 2]]
        return density / max(float(np.median(density[density > 0])) if np.any(density > 0) else 1.0, 1e-6)

    angles = np.arange(max(8, int(angle_count)), dtype=np.float32) * np.pi / max(8, int(angle_count))
    tilts = np.linspace(0.0, np.deg2rad(max(0.0, float(max_tilt_degrees))), max(1, int(tilt_count)))
    target = max(1, int(target_clusters))
    while len(np.unique(output[output >= 0])) < target:
        best: tuple[float, int, np.ndarray, np.ndarray, float, np.ndarray] | None = None
        active_ids = np.unique(output[output >= 0])
        largest_cluster_size = max(
            (int(np.sum(output == cluster_id)) for cluster_id in active_ids),
            default=1,
        )
        for cluster_id in active_ids:
            indexes = np.flatnonzero(output == cluster_id)
            required = max(
                max(2, int(min_child_size)),
                int(np.ceil(len(indexes) * max(0.0, float(min_child_fraction)))),
            )
            if len(indexes) < 2 * required:
                continue
            points = samples[indexes]
            weights = point_density(points)
            for tilt in tilts:
                for angle in angles:
                    normal = np.asarray(
                        [
                            np.sin(tilt),
                            np.cos(tilt) * np.cos(angle),
                            np.cos(tilt) * np.sin(angle),
                        ],
                        dtype=np.float32,
                    )
                    projection = points @ normal
                    offsets = np.linspace(
                        np.quantile(projection, 0.10),
                        np.quantile(projection, 0.90),
                        max(3, int(offset_count)),
                    )
                    bandwidth = (
                        float(plane_bandwidth)
                        if plane_bandwidth > 0.0
                        else max(float(np.quantile(np.abs(projection - np.median(projection)), 0.25)) * 0.35, 0.002)
                    )
                    for offset in offsets:
                        left = projection < offset
                        right = ~left
                        if int(left.sum()) < required or int(right.sum()) < required:
                            continue
                        cut_kernel = np.exp(-0.5 * ((projection - offset) / bandwidth) ** 2)
                        total_mass = max(float(np.sum(weights)), 1e-6)
                        density_cut_cost = float(np.sum(weights * cut_kernel) / total_mass)
                        geometric_cut_mass = float(np.mean(cut_kernel))
                        cut_cost = (
                            (1.0 - np.clip(float(cut_mass_weight), 0.0, 1.0))
                            * density_cut_cost
                            + np.clip(float(cut_mass_weight), 0.0, 1.0)
                            * geometric_cut_mass
                        )
                        side_mass = min(
                            float(np.sum(weights[left])),
                            float(np.sum(weights[right])),
                        ) / total_mass
                        # A perimeter cut can have a very low absolute cost
                        # simply because one side contains a sparse fringe.
                        # Compare the valley against the supported mass on
                        # both sides so a small bridge between two peaks wins.
                        if side_mass < max(0.0, float(min_side_mass)):
                            continue
                        relative_cost = cut_cost / max(side_mass, 1e-6)
                        imbalance = abs(int(left.sum()) - int(right.sum())) / len(points)
                        score = (
                            max(0.0, float(relative_valley_weight)) * relative_cost
                            + max(0.0, float(balance_penalty)) * imbalance
                        )
                        # Prefer cutting larger unresolved clusters, but only
                        # as a soft prior so a clearly better split in a
                        # smaller cluster can still win.
                        size_ratio = len(points) / max(largest_cluster_size, 1)
                        score *= 1.0 + max(0.0, float(cluster_size_weight)) * (1.0 - size_ratio)
                        if best is None or score < best[0]:
                            best = (score, int(cluster_id), left, normal, float(offset), points)
        if best is None:
            break
        _score, cluster_id, left, normal, offset, plane_points = best
        indexes = np.flatnonzero(output == cluster_id)
        output[indexes[~left]] = next_label
        next_label += 1
        # Keep the points from the cluster being split so visualizers can
        # bound the rendered plane to that cluster rather than the full plot.
        split_planes.append((normal, offset, plane_points.copy()))

    retained_ids = np.unique(output[output >= 0])
    retained_ids = np.unique(output[output >= 0])
    remapped = np.full(len(output), -1, dtype=np.int32)
    for new_index, old_index in enumerate(retained_ids):
        remapped[output == old_index] = new_index
    centroids = np.asarray(
        [samples[remapped == index].mean(axis=0) for index in range(len(retained_ids))],
        dtype=np.float32,
    ) if len(retained_ids) else np.empty((0, 3), dtype=np.float32)
    return remapped, centroids, split_planes
