import numpy as np
import pytest

from live_recognition.types import CubeCandidate, FrameResult
from live_recognition.visualization import (
    _reliable_color_mask,
    build_lab_chroma_debug_view,
    build_luminance_debug_view,
    draw_result,
    lab_ab_samples_from_bbox,
    rgb_origin_regression_rays,
    save_lab_ab_plot,
    save_rgb_samples_interactive_plot,
    save_rgb_samples_plot,
)


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


def test_rgb_origin_regression_rays_finds_six_directions():
    base_colors = np.array(
        [
            [220, 20, 20],
            [230, 120, 20],
            [220, 220, 30],
            [30, 180, 40],
            [30, 70, 220],
            [220, 220, 220],
        ],
        dtype=np.float32,
    )
    samples = []
    for color in base_colors:
        for scale in (0.35, 0.55, 0.8, 1.0):
            samples.append(np.clip(color * scale, 0, 255))

    directions, centroids = rgb_origin_regression_rays(
        np.asarray(samples, dtype=np.uint8),
        cluster_count=6,
        min_brightness=20,
    )

    assert directions.shape == (6, 3)
    assert centroids.shape == (6, 3)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)


def test_save_rgb_samples_interactive_plot_can_draw_regression_lines(tmp_path):
    pytest.importorskip("plotly")
    rgb_samples = np.array(
        [
            [220, 20, 20],
            [120, 10, 10],
            [20, 220, 20],
            [10, 120, 10],
            [20, 20, 220],
            [10, 10, 120],
        ],
        dtype=np.uint8,
    )
    bgr_samples = rgb_samples[:, ::-1]
    output_path = tmp_path / "rgb.html"

    save_rgb_samples_interactive_plot(
        bgr_samples,
        str(output_path),
        regression_cluster_count=3,
        regression_min_brightness=20,
    )

    html = output_path.read_text()
    assert "RGB ray 1" in html
    assert "RGB ray 2" in html
    assert "RGB ray 3" in html
