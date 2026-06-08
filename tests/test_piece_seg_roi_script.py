import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "test_piece_seg_roi.py"
SPEC = importlib.util.spec_from_file_location("test_piece_seg_roi_script", SCRIPT_PATH)
piece_seg_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(piece_seg_script)


class StubBboxDetector:
    def detect(self, frame):
        return SimpleNamespace(bbox_xyxy=(5, 5, 25, 25), confidence=0.9)


class StubBoxes:
    cls = [SimpleNamespace(item=lambda: 0)]
    conf = [SimpleNamespace(item=lambda: 0.9)]

    def __len__(self):
        return 1


class StubMixedBoxes:
    cls = [SimpleNamespace(item=lambda: 0), SimpleNamespace(item=lambda: 1)]
    conf = [SimpleNamespace(item=lambda: 0.9), SimpleNamespace(item=lambda: 0.8)]

    def __len__(self):
        return 2


class StubPieceModel:
    def predict(self, source, conf, imgsz, verbose):
        mask = np.zeros((1, source.shape[0], source.shape[1]), dtype=np.float32)
        mask[:, 5:15, 5:15] = 1.0
        masks = SimpleNamespace(
            data=SimpleNamespace(
                detach=lambda: SimpleNamespace(
                    cpu=lambda: SimpleNamespace(numpy=lambda: mask)
                )
            )
        )
        return [SimpleNamespace(masks=masks, boxes=StubBoxes())]


class StubMixedPieceModel:
    def predict(self, source, conf, imgsz, verbose):
        mask = np.zeros((2, source.shape[0], source.shape[1]), dtype=np.float32)
        mask[0, 5:15, 5:15] = 1.0
        mask[1, 5:15, 12:20] = 1.0
        masks = SimpleNamespace(
            data=SimpleNamespace(
                detach=lambda: SimpleNamespace(
                    cpu=lambda: SimpleNamespace(numpy=lambda: mask)
                )
            )
        )
        return [SimpleNamespace(masks=masks, boxes=StubMixedBoxes())]


def default_args(**overrides):
    values = {
        "piece_conf": 0.25,
        "piece_imgsz": 256,
        "roi_padding": 0.0,
        "segmentation_padding_px": 0,
        "debug_overlay": False,
        "show_bbox": False,
        "show_confidence": False,
        "mask_opacity": 1.0,
        "color_plot_samples_per_frame": 800,
        "average_mask_color": False,
        "show_center_color_classification": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_process_frame_uses_same_output_when_collecting_plot_samples():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    args = default_args(debug_overlay=True, show_bbox=True)

    output, ab_samples, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubPieceModel(),
        args,
    )
    plot_output, plot_ab_samples, plot_bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubPieceModel(),
        args,
    )

    assert np.array_equal(output, plot_output)
    assert np.array_equal(ab_samples, plot_ab_samples)
    assert np.array_equal(bgr_samples, plot_bgr_samples)
    assert len(ab_samples) > 1


def test_piece_sampling_uses_same_piece_mask_as_rendering():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:20] = [0, 0, 220]
    frame[24:30, 24:30] = [220, 0, 0]
    args = default_args()

    _, ab_samples, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubPieceModel(),
        args,
    )

    assert len(ab_samples) > 1
    assert (bgr_samples[:, 2] > bgr_samples[:, 0]).all()
    assert not (bgr_samples == [220, 0, 0]).all(axis=1).any()


def test_segmentation_padding_changes_rendered_and_sampled_mask():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[12:18, 12:18] = [0, 0, 220]
    args = default_args(segmentation_padding_px=-2)

    output, ab_samples, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubPieceModel(),
        args,
    )

    assert len(ab_samples) > 1
    assert (bgr_samples[:, 2] > bgr_samples[:, 0]).all()
    assert (output[10, 10] == [40, 40, 40]).all()
    assert (output[15, 15] == [80, 220, 255]).all()


def test_average_mask_color_returns_one_sample_per_piece():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:20] = [0, 0, 220]
    args = default_args(average_mask_color=True)

    _, ab_samples, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubPieceModel(),
        args,
    )

    assert ab_samples.shape == (1, 2)
    assert bgr_samples.shape == (1, 3)
    assert bgr_samples[0, 2] > bgr_samples[0, 0]


def test_rgbxy_sticker_segmentation_splits_edge_into_two_color_regions():
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[5:20, 5:12] = [0, 0, 220]
    frame[5:20, 12:20] = [220, 0, 0]
    piece_mask = np.zeros((30, 30), dtype=bool)
    piece_mask[5:20, 5:20] = True

    stickers = piece_seg_script.sticker_segments_from_piece(
        frame,
        (0, 0, 30, 30),
        piece_mask,
        class_id=1,
    )
    sticker_bgrs = np.asarray([sticker_bgr for _, sticker_bgr, _ in stickers])

    assert len(stickers) == 2
    assert (sticker_bgrs[:, 2] > sticker_bgrs[:, 0]).any()
    assert (sticker_bgrs[:, 0] > sticker_bgrs[:, 2]).any()


def test_rgbxy_sticker_segmentation_collapses_same_color_edge_region():
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[5:20, 5:20] = [220, 0, 0]
    piece_mask = np.zeros((30, 30), dtype=bool)
    piece_mask[5:20, 5:20] = True

    stickers = piece_seg_script.sticker_segments_from_piece(
        frame,
        (0, 0, 30, 30),
        piece_mask,
        class_id=1,
    )

    assert len(stickers) == 1
    assert np.array_equal(stickers[0][1], [220, 0, 0])


def test_color_classifier_collects_mean_samples_from_sticker_segments():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:17] = [0, 0, 220]
    frame[10:20, 17:25] = [220, 0, 0]
    classifier = piece_seg_script.ColorRayClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=20,
    )

    _, _, _ = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubMixedPieceModel(),
        default_args(),
        classifier,
    )

    observed = np.asarray(classifier._rgb_samples)
    assert classifier.detected_frames == 1
    assert len(observed) == 2
    assert (observed[:, 0] > observed[:, 2]).any()
    assert (observed[:, 2] > observed[:, 0]).any()


def test_learned_color_recolors_each_sticker_after_classifier_ready():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:17] = [0, 0, 220]
    frame[10:20, 17:25] = [220, 0, 0]
    classifier = piece_seg_script.ColorRayClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=20,
    )

    piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubMixedPieceModel(),
        default_args(),
        classifier,
    )
    output, _, _ = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubMixedPieceModel(),
        default_args(),
        classifier,
    )

    assert classifier.ready
    assert not np.array_equal(output[15, 15], piece_seg_script.CLASS_COLORS[0])
    assert not np.array_equal(output[15, 22], piece_seg_script.CLASS_COLORS[1])
    assert np.array_equal(output[15, 22], [220, 0, 0])


def test_dim_center_masks_are_not_recolored_by_classifier():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:17] = [0, 0, 220]
    frame[10:20, 17:25] = [220, 0, 0]
    classifier = piece_seg_script.ColorRayClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=120,
    )

    piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubMixedPieceModel(),
        default_args(),
        classifier,
    )
    dim_frame = np.zeros((40, 40, 3), dtype=np.uint8)
    dim_frame[:, :] = [5, 5, 5]
    dim_frame[10:20, 10:17] = [0, 0, 25]
    dim_frame[10:20, 17:25] = [25, 0, 0]

    output, _, _ = piece_seg_script.process_frame(
        dim_frame,
        StubBboxDetector(),
        StubMixedPieceModel(),
        default_args(),
        classifier,
    )

    assert classifier.ready
    assert np.array_equal(output[15, 15], piece_seg_script.CLASS_COLORS[0])


def test_classifier_uses_nearest_regression_line():
    classifier = piece_seg_script.ColorRayClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=1,
    )
    classifier.directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    classifier.centroids = np.array(
        [
            [200.0, 0.0, 0.0],
            [4.0, 20.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert classifier.classify_bgr(np.array([0, 5, 5], dtype=np.uint8)) == 0


def test_classifier_uses_nearest_cluster_sample_before_centroid():
    classifier = piece_seg_script.ColorRayClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=1,
    )
    classifier.centroids = np.array(
        [
            [50.0, 0.0, 0.0],
            [60.0, 10.0, 0.0],
        ],
        dtype=np.float32,
    )
    classifier.cluster_samples = [
        np.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        np.array([[60.0, 10.0, 0.0]], dtype=np.float32),
    ]

    assert classifier.classify_bgr(np.array([0, 1, 100], dtype=np.uint8)) == 0


def test_classifier_linear_regression_mode_ignores_cluster_samples():
    classifier = piece_seg_script.ColorRayClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=1,
        mode="linear-regression",
    )
    classifier.directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    classifier.centroids = np.array(
        [
            [200.0, 0.0, 0.0],
            [4.0, 20.0, 0.0],
        ],
        dtype=np.float32,
    )
    classifier.cluster_samples = [
        np.array([[5.0, 5.0, 0.0]], dtype=np.float32),
        np.array([[100.0, 0.0, 0.0]], dtype=np.float32),
    ]

    assert classifier.classify_bgr(np.array([0, 5, 5], dtype=np.uint8)) == 0


def test_parser_exposes_color_classifier_mode():
    args = piece_seg_script.build_parser().parse_args(
        ["--color-classifier-mode", "linear-regression"]
    )

    assert args.color_classifier_mode == "linear-regression"
