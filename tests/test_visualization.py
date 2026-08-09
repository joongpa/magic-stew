import numpy as np
import pytest

import live_recognition.visualization as visualization
from live_recognition.types import CubeCandidate, FrameResult
from live_recognition.visualization import (
    _convex_hull_indexes_2d,
    _density_rgb_sample_cluster_indexes,
    _form_fit_boundary_2d,
    _normalize_rgb_brightness,
    _projected_form_fit_boundary_points,
    _reliable_color_mask,
    build_lab_chroma_debug_view,
    build_luminance_debug_view,
    draw_result,
    lab_ab_samples_from_bbox,
    save_lab_ab_plot,
    save_hsv_samples_interactive_plot,
    save_rgb_samples_interactive_plot,
    save_rgb_samples_plot,
)


def test_normalize_rgb_brightness_projects_to_unit_sphere():
    rgb_samples = np.array(
        [
            [10, 10, 10],
            [3, 4, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    normalized = _normalize_rgb_brightness(rgb_samples)

    assert np.allclose(normalized[0], [1 / np.sqrt(3)] * 3)
    assert np.allclose(normalized[1], [0.6, 0.8, 0.0])
    assert np.allclose(normalized[2], [0.0, 0.0, 0.0])


def test_convex_hull_indexes_2d_wrap_outer_points():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )

    hull_indexes = _convex_hull_indexes_2d(points)

    assert set(hull_indexes.tolist()) == {0, 1, 2, 3}


def test_form_fit_boundary_2d_follows_concave_point_cloud():
    points = []
    for x in np.linspace(0.0, 1.0, 12):
        points.append([x, 0.0])
    for y in np.linspace(0.0, 1.0, 12):
        points.append([0.0, y])
    for x in np.linspace(0.0, 0.35, 6):
        for y in np.linspace(0.0, 0.35, 6):
            points.append([x, y])
    points = np.asarray(points, dtype=np.float32)

    boundary = _form_fit_boundary_2d(points)

    assert len(boundary) >= 3
    assert boundary[:, 0].max() > 0.9
    assert boundary[:, 1].max() > 0.9
    assert not np.any((boundary[:, 0] > 0.7) & (boundary[:, 1] > 0.7))


def test_projected_form_fit_boundary_handles_line_like_clusters():
    points = np.array(
        [
            [40.0, 10.0, 10.0],
            [80.0, 20.0, 20.0],
            [120.0, 30.0, 30.0],
            [160.0, 40.0, 40.0],
        ],
        dtype=np.float32,
    )

    boundary = _projected_form_fit_boundary_points(points)

    assert boundary.shape == (4, 3)
    assert np.ptp(boundary[:, 0]) > 100


def test_raw_color_sample_traces_group_by_cluster_with_original_colors():
    class FakeGo:
        @staticmethod
        def Scatter3d(**kwargs):
            return kwargs

    plot_samples = np.asarray(
        [
            [220.0, 20.0, 20.0],
            [210.0, 25.0, 20.0],
            [20.0, 220.0, 20.0],
            [6.0, 6.0, 6.0],
        ],
        dtype=np.float32,
    )
    rgb_samples = plot_samples.astype(np.uint8)
    cluster_indexes = np.asarray([1, 1, 0, -1], dtype=np.int32)
    customdata = np.column_stack([rgb_samples, cluster_indexes + 1])

    traces = visualization._raw_color_sample_traces(
        FakeGo,
        plot_samples,
        rgb_samples,
        cluster_indexes,
        customdata,
        point_size=2,
        hovertemplate="hover",
    )

    assert [trace["name"] for trace in traces] == [
        "cluster 1",
        "cluster 2",
        "cluster unknown",
    ]
    assert traces[0]["marker"]["color"] == ["rgb(20,220,20)"]
    assert traces[1]["marker"]["color"] == [
        "rgb(220,20,20)",
        "rgb(210,25,20)",
    ]


def test_rgb_regression_ray_traces_extend_to_rgb_cube_edge():
    class FakeGo:
        @staticmethod
        def Scatter3d(**kwargs):
            return kwargs

    traces = visualization._rgb_regression_ray_traces(
        FakeGo,
        np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
        np.array([[200.0, 50.0, 0.0]], dtype=np.float32),
        normalize_rgb=False,
    )

    assert len(traces) == 1
    assert traces[0]["name"] == "ray 1"
    assert traces[0]["mode"] == "lines"
    assert np.allclose(traces[0]["x"], [0.0, 255.0])
    assert np.allclose(traces[0]["y"], [0.0, 255.0])
    assert np.allclose(traces[0]["z"], [0.0, 0.0])
    assert traces[0]["line"]["color"] == "rgb(200,50,0)"


def test_normalized_rgb_regression_ray_traces_use_unit_endpoint():
    class FakeGo:
        @staticmethod
        def Scatter3d(**kwargs):
            return kwargs

    traces = visualization._rgb_regression_ray_traces(
        FakeGo,
        np.array([[3.0, 4.0, 0.0]], dtype=np.float32),
        np.array([[20.0, 220.0, 20.0]], dtype=np.float32),
        normalize_rgb=True,
    )

    assert np.allclose(traces[0]["x"], [0.0, 0.6])
    assert np.allclose(traces[0]["y"], [0.0, 0.8])
    assert np.allclose(traces[0]["z"], [0.0, 0.0])


def test_index_cluster_merge_weights_origin_ray_component_less_than_other_components():
    samples = np.asarray(
        [
            [100.0, 28.0, 0.0],
            [102.0, 30.0, 0.0],
            [98.0, 20.0, 0.0],
            [102.0, 20.0, 0.0],
            [112.0, 22.4, 0.0],
            [114.0, 22.8, 0.0],
        ],
        dtype=np.float32,
    )
    clusters = [
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([2, 3], dtype=np.int32),
        np.asarray([4, 5], dtype=np.int32),
    ]

    nearest_index = visualization._nearest_index_cluster_index(
        clusters,
        source_index=1,
        samples_for_distance=samples,
        normalize_centroids=False,
        ray_distance_divisor=4.0,
    )

    assert nearest_index == 2

    unweighted_nearest_index = visualization._nearest_index_cluster_index(
        clusters,
        source_index=1,
        samples_for_distance=samples,
        normalize_centroids=False,
        ray_distance_divisor=1.0,
    )

    assert unweighted_nearest_index == 0


def test_index_cluster_merging_compares_against_original_accepted_clusters():
    samples = np.asarray(
        [
            [100.0, 0.0, 0.0],
            [101.0, 0.0, 0.0],
            [102.0, 0.0, 0.0],
            [200.0, 0.0, 0.0],
            [201.0, 0.0, 0.0],
            [202.0, 0.0, 0.0],
            [130.0, 0.0, 0.0],
            [160.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    clusters = [
        np.asarray([0, 1, 2], dtype=np.int32),
        np.asarray([3, 4, 5], dtype=np.int32),
        np.asarray([6], dtype=np.int32),
        np.asarray([7], dtype=np.int32),
    ]

    merged_clusters = visualization._merge_smallest_index_clusters_to_nearest_neighbor(
        clusters,
        samples_for_distance=samples,
        target_count=2,
        normalize_centroids=False,
        ray_distance_divisor=1.0,
    )

    cluster_with_a = next(cluster for cluster in merged_clusters if 0 in cluster)
    cluster_with_b = next(cluster for cluster in merged_clusters if 3 in cluster)
    assert 6 in cluster_with_a
    assert 7 not in cluster_with_a
    assert 7 in cluster_with_b


def test_density_rgb_sample_cluster_indexes_merges_smallest_clusters_to_target_count():
    rgb_samples = []
    for angle in np.linspace(0.0, 0.5, 8):
        direction = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
        rgb_samples.extend(direction * value for value in (100.0, 110.0, 120.0, 130.0))
    rgb_samples = np.asarray(rgb_samples, dtype=np.uint8)

    labels, centroids = _density_rgb_sample_cluster_indexes(
        rgb_samples,
        cluster_count=6,
        min_brightness=1,
        eps_radians=0.02,
        min_points=2,
    )

    assert len(set(labels[labels >= 0].tolist())) == 6
    assert centroids.shape == (6, 3)


def test_draw_result_defaults_to_bounding_box_only_when_candidate_has_mask():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[6:14, 6:14] = 255
    result = FrameResult(
        candidate=CubeCandidate(
            bbox_xyxy=(5, 5, 15, 15),
            confidence=0.9,
            mask=mask,
        ),
        orientations=[],
    )

    output = draw_result(frame, result)

    assert (output[10, 10] == [0, 0, 0]).all()
    assert (output[5, 5] == [40, 220, 40]).all()


def test_draw_result_can_opt_into_mask_overlay():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[6:14, 6:14] = 255
    result = FrameResult(
        candidate=CubeCandidate(
            bbox_xyxy=(5, 5, 15, 15),
            confidence=0.9,
            mask=mask,
        ),
        orientations=[],
    )

    output = draw_result(frame, result, draw_mask_overlay=True)

    assert (output[10, 10] == [22, 121, 22]).all()


def test_draw_result_can_overlay_fps_counter():
    frame = np.zeros((60, 160, 3), dtype=np.uint8)
    result = FrameResult(candidate=None, orientations=[])

    output = draw_result(frame, result, fps=12.3)

    assert output.sum() > 0
    assert (output[:36, :120] != 0).any()


def test_reliable_color_mask_rejects_clipped_glare_and_dark_pixels():
    crop = np.zeros((2, 3, 3), dtype=np.uint8)
    crop[0, 0] = [0, 0, 220]
    crop[0, 1] = [255, 255, 255]
    crop[0, 2] = [5, 5, 5]

    mask = _reliable_color_mask(crop)

    assert mask[0, 0]
    assert not mask[0, 1]
    assert not mask[0, 2]


def test_reliable_color_mask_keeps_low_saturation_white_and_yellow_samples():
    crop = np.zeros((1, 2, 3), dtype=np.uint8)
    crop[0, 0] = [225, 225, 230]
    crop[0, 1] = [0, 220, 230]

    mask = _reliable_color_mask(crop)

    assert mask[0, 0]
    assert mask[0, 1]


def test_reliable_color_mask_keeps_orange_samples():
    crop = np.zeros((1, 2, 3), dtype=np.uint8)
    crop[0, 0] = [0, 115, 230]
    crop[0, 1] = [10, 95, 180]

    mask = _reliable_color_mask(crop)

    assert mask[0, 0]
    assert mask[0, 1]


def test_draw_result_can_apply_full_frame_color_debug_effect():
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    frame[30:90, 40:100] = [0, 0, 220]
    result = FrameResult(
        candidate=CubeCandidate(bbox_xyxy=(40, 30, 100, 90), confidence=0.9),
        orientations=[],
    )

    output = draw_result(frame, result, draw_color_debug=True)

    assert output.mean() > frame.mean()
    assert (output[40:80, 50:90, 0] == output[40:80, 50:90, 1]).all()
    assert (output[40:80, 50:90, 1] == output[40:80, 50:90, 2]).all()


def test_luminance_debug_view_is_grayscale():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = [0, 0, 220]
    frame[0, 1] = [0, 220, 0]
    frame[1, 0] = [220, 0, 0]
    frame[1, 1] = [225, 225, 225]

    output = build_luminance_debug_view(frame)

    assert output.shape == frame.shape
    assert (output[:, :, 0] == output[:, :, 1]).all()
    assert (output[:, :, 1] == output[:, :, 2]).all()


def test_lab_chroma_debug_view_preserves_crop_shape():
    crop = np.zeros((20, 30, 3), dtype=np.uint8)
    crop[:, :10] = [0, 0, 220]
    crop[:, 10:20] = [0, 180, 0]
    crop[:, 20:] = [220, 0, 0]

    debug_view = build_lab_chroma_debug_view(crop)

    assert debug_view is not None
    assert debug_view.shape == crop.shape


def test_lab_chroma_debug_view_boosts_chroma_contrast():
    crop = np.zeros((20, 40, 3), dtype=np.uint8)
    crop[:, :20] = [40, 40, 180]
    crop[:, 20:] = [70, 70, 130]

    debug_view = build_lab_chroma_debug_view(crop)

    assert debug_view is not None
    assert debug_view.mean() > crop.mean()


def test_lab_ab_samples_from_bbox_returns_ab_pairs_and_colors():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[:, :10] = [0, 0, 220]
    frame[:, 10:20] = [0, 220, 220]
    frame[:, 20:] = [220, 0, 0]

    ab_samples, bgr_samples = lab_ab_samples_from_bbox(frame, (0, 0, 30, 20), max_samples=50)

    assert ab_samples.shape == (50, 2)
    assert bgr_samples.shape == (50, 3)


def test_lab_ab_samples_from_bbox_can_sample_inside_mask_only():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[:, :15] = [0, 0, 220]
    frame[:, 15:] = [220, 0, 0]
    sample_mask = np.zeros((20, 30), dtype=np.uint8)
    sample_mask[:, :15] = 255

    ab_samples, bgr_samples = lab_ab_samples_from_bbox(
        frame,
        (0, 0, 30, 20),
        sample_mask=sample_mask,
        max_samples=1000,
    )

    assert len(ab_samples) == 300
    assert (bgr_samples[:, 2] > bgr_samples[:, 0]).all()


def test_save_lab_ab_plot_writes_png(tmp_path):
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[:, :10] = [0, 0, 220]
    frame[:, 10:20] = [0, 220, 220]
    frame[:, 20:] = [220, 0, 0]
    output_path = tmp_path / "plot.png"

    save_lab_ab_plot(frame, (0, 0, 30, 20), str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_rgb_samples_plot_writes_three_angles(tmp_path):
    bgr_samples = np.array(
        [
            [0, 0, 220],
            [0, 220, 0],
            [220, 0, 0],
            [0, 220, 220],
        ],
        dtype=np.uint8,
    )
    output_path = tmp_path / "rgb.png"

    saved_paths = save_rgb_samples_plot(bgr_samples, str(output_path))

    assert len(saved_paths) == 3
    for suffix in ("rg", "rb", "gb"):
        angle_path = tmp_path / f"rgb_{suffix}.png"
        assert str(angle_path) in saved_paths
        assert angle_path.exists()
        assert angle_path.stat().st_size > 0


def test_save_rgb_samples_plot_can_normalize_rgb_coordinates(tmp_path):
    bgr_samples = np.array(
        [
            [0, 0, 220],
            [0, 220, 0],
            [220, 0, 0],
            [40, 80, 160],
        ],
        dtype=np.uint8,
    )
    output_path = tmp_path / "rgb_normalized.png"

    saved_paths = save_rgb_samples_plot(bgr_samples, str(output_path), normalize_rgb=True)

    assert len(saved_paths) == 3
    for suffix in ("rg", "rb", "gb"):
        angle_path = tmp_path / f"rgb_normalized_{suffix}.png"
        assert str(angle_path) in saved_paths
        assert angle_path.exists()
        assert angle_path.stat().st_size > 0


def test_save_rgb_samples_plot_can_color_by_cluster(tmp_path):
    bgr_samples = np.array(
        [
            [20, 20, 220],
            [18, 18, 210],
            [16, 16, 200],
            [14, 14, 190],
            [20, 220, 20],
            [18, 210, 18],
            [16, 200, 16],
            [14, 190, 14],
            [220, 40, 40],
        ],
        dtype=np.uint8,
    )
    output_path = tmp_path / "rgb_cluster.png"

    saved_paths = save_rgb_samples_plot(
        bgr_samples,
        str(output_path),
        color_by_cluster=True,
        cluster_count=2,
        cluster_min_brightness=20,
    )

    assert len(saved_paths) == 3
    for suffix in ("rg", "rb", "gb"):
        angle_path = tmp_path / f"rgb_cluster_{suffix}.png"
        assert str(angle_path) in saved_paths
        assert angle_path.exists()
        assert angle_path.stat().st_size > 0


def test_raw_density_rgb_static_plot_uses_raw_density_clusters(tmp_path, monkeypatch):
    calls = []

    def fake_raw_density_clusters(rgb_samples, *, cluster_count, min_brightness, eps, min_points):
        calls.append((len(rgb_samples), cluster_count, min_brightness, eps, min_points))
        return (
            np.zeros(len(rgb_samples), dtype=np.int32),
            np.array([[220.0, 20.0, 20.0]], dtype=np.float32),
        )

    monkeypatch.setattr(
        visualization,
        "_raw_density_rgb_sample_cluster_indexes",
        fake_raw_density_clusters,
    )
    bgr_samples = np.array(
        [
            [20, 20, 220],
            [25, 25, 210],
            [30, 30, 205],
        ],
        dtype=np.uint8,
    )
    output_path = tmp_path / "rgb_raw_density.png"

    save_rgb_samples_plot(
        bgr_samples,
        str(output_path),
        cluster_min_brightness=20,
        density_cluster_raw_eps=18,
        density_cluster_min_neighbors=3,
        cluster_display_mode="raw-density",
    )

    assert calls == [(3, 6, 20, 18, 3)]


def test_normalized_rgb_interactive_plot_omits_cluster_perimeters(tmp_path):
    pytest.importorskip("plotly")
    rgb_samples = np.array(
        [
            [220, 20, 20],
            [210, 30, 20],
            [220, 25, 35],
            [205, 18, 32],
            [20, 220, 20],
            [30, 210, 20],
            [25, 220, 35],
            [18, 205, 32],
            [20, 20, 220],
            [30, 20, 210],
            [25, 35, 220],
            [18, 32, 205],
        ],
        dtype=np.uint8,
    )
    output_path = tmp_path / "rgb_normalized.html"

    save_rgb_samples_interactive_plot(
        rgb_samples[:, ::-1],
        str(output_path),
        normalize_rgb=True,
        cluster_count=3,
        cluster_min_brightness=20,
    )

    html = output_path.read_text()
    assert "cluster 1 perimeter" not in html
    assert "cluster 2 perimeter" not in html
    assert "cluster 3 perimeter" not in html


def test_normalized_rgb_interactive_plot_can_toggle_external_clusters_with_raw_colors(tmp_path):
    pytest.importorskip("plotly")
    bgr_samples = np.array(
        [
            [20, 20, 220],
            [22, 18, 210],
            [18, 22, 200],
            [20, 220, 20],
            [18, 210, 22],
            [22, 200, 18],
        ],
        dtype=np.uint8,
    )
    output_path = tmp_path / "rgb_external.html"

    save_rgb_samples_interactive_plot(
        bgr_samples,
        str(output_path),
        normalize_rgb=True,
        cluster_indexes=np.array([1, 1, 1, 0, 0, 0], dtype=np.int32),
        cluster_centroids=np.array(
            [
                [20.0, 210.0, 20.0],
                [210.0, 20.0, 20.0],
            ],
            dtype=np.float32,
        ),
    )

    html = output_path.read_text()
    assert "samples (raw color)" not in html
    assert "cluster 1" in html
    assert "cluster 2" in html
    assert "cluster 1 perimeter" not in html
    assert "cluster 2 perimeter" not in html
    assert "rgb(220,20,20)" in html
    assert "rgb(20,220,20)" in html


def test_raw_density_rgb_interactive_plot_draws_toggleable_raw_color_clusters(tmp_path):
    pytest.importorskip("plotly")
    rgb_samples = np.array(
        [
            [220, 20, 20],
            [210, 30, 20],
            [220, 25, 35],
            [205, 18, 32],
            [20, 220, 20],
            [30, 210, 20],
            [25, 220, 35],
            [18, 205, 32],
        ],
        dtype=np.uint8,
    )
    output_path = tmp_path / "rgb_raw_density.html"

    save_rgb_samples_interactive_plot(
        rgb_samples[:, ::-1],
        str(output_path),
        cluster_count=2,
        cluster_min_brightness=20,
        density_cluster_raw_eps=35,
        density_cluster_min_neighbors=3,
        cluster_display_mode="raw-density",
    )

    html = output_path.read_text()
    assert "cluster 1" in html
    assert "cluster 2" in html
    assert "cluster 1 perimeter" not in html
    assert "cluster 2 perimeter" not in html
    assert "rgb(220,20,20)" in html
    assert "rgb(20,220,20)" in html


def test_save_hsv_samples_interactive_plot_writes_html(tmp_path):
    pytest.importorskip("plotly")
    bgr_samples = np.array(
        [
            [0, 0, 220],
            [0, 220, 0],
            [220, 0, 0],
            [0, 220, 220],
        ],
        dtype=np.uint8,
    )
    output_path = tmp_path / "hsv.html"

    save_hsv_samples_interactive_plot(bgr_samples, str(output_path))

    html = output_path.read_text()
    assert "Reliable pixels in HSV" in html
    assert "live-recognition-hsv-html: cylinder raw-point-colors" in html
    assert "S cos(H)" in html
    assert "S sin(H)" in html
    assert "raw RGB" in html
    assert "H=%{customdata[4]:.0f} deg" in html


def test_hsv_cylinder_interactive_plot_accepts_external_clusters_with_raw_colors(tmp_path):
    pytest.importorskip("plotly")
    bgr_samples = np.array(
        [
            [20, 20, 220],
            [22, 18, 210],
            [20, 220, 20],
            [18, 210, 22],
        ],
        dtype=np.uint8,
    )
    output_path = tmp_path / "hsv_cylinder_external.html"

    save_hsv_samples_interactive_plot(
        bgr_samples,
        str(output_path),
        cluster_indexes=np.array([1, 1, 0, 0], dtype=np.int32),
        cluster_centroids=np.array(
            [
                [20.0, 210.0, 20.0],
                [210.0, 20.0, 20.0],
            ],
            dtype=np.float32,
        ),
    )

    html = output_path.read_text()
    assert "cluster 1" in html
    assert "cluster 2" in html
    assert "rgb(220,20,20)" in html
    assert "rgb(20,220,20)" in html


def test_raw_color_plot_shows_low_brightness_samples_as_unknown(tmp_path):
    pytest.importorskip("plotly")
    rgb_samples = np.array(
        [
            [220, 20, 20],
            [210, 18, 18],
            [200, 16, 16],
            [190, 14, 14],
            [20, 220, 20],
            [18, 210, 18],
            [16, 200, 16],
            [14, 190, 14],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )
    rgb_output_path = tmp_path / "rgb_raw_unknown.html"
    hsv_output_path = tmp_path / "hsv_raw_unknown.html"

    save_rgb_samples_interactive_plot(
        rgb_samples[:, ::-1],
        str(rgb_output_path),
        cluster_count=2,
        cluster_min_brightness=20,
    )
    save_hsv_samples_interactive_plot(
        rgb_samples[:, ::-1],
        str(hsv_output_path),
        cluster_count=2,
        cluster_min_brightness=20,
    )

    rgb_html = rgb_output_path.read_text()
    hsv_html = hsv_output_path.read_text()
    assert "samples (raw color)" in rgb_html
    assert "cluster unknown" in rgb_html
    assert "samples (raw color)" in hsv_html
    assert "cluster unknown" in hsv_html
