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


class RaisingBboxDetector:
    def detect(self, frame):
        raise AssertionError("bbox detector should not be called")


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


class StubPoseBoxes:
    cls = [SimpleNamespace(item=lambda: 1)]
    conf = [SimpleNamespace(item=lambda: 0.9)]

    def __len__(self):
        return 1


class StubFacePoseBoxes:
    cls = [SimpleNamespace(item=lambda: 0)]
    conf = [SimpleNamespace(item=lambda: 0.9)]

    def __len__(self):
        return 1


class TensorLike:
    def __init__(self, data):
        self.data = data

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class StubStickerModel:
    names = {0: "U_sticker", 1: "F_sticker", 2: "R_sticker"}

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


class StubPoseStickerModel:
    names = {0: "U_sticker"}

    def predict(self, source, conf, imgsz, verbose):
        keypoints = np.array(
            [
                [
                    [5.0, 5.0],
                    [15.0, 5.0],
                    [15.0, 15.0],
                    [5.0, 15.0],
                ]
            ],
            dtype=np.float32,
        )
        return [
            SimpleNamespace(
                keypoints=SimpleNamespace(xy=TensorLike(keypoints)),
                boxes=StubPoseBoxes(),
            )
        ]


class StubVisibleFacePoseModel:
    names = {0: "U_sticker"}

    def predict(self, source, conf, imgsz, verbose):
        keypoints = np.array(
            [
                [
                    [5.0, 5.0],
                    [15.0, 5.0],
                    [15.0, 15.0],
                    [5.0, 15.0],
                ]
            ],
            dtype=np.float32,
        )
        return [
            SimpleNamespace(
                keypoints=SimpleNamespace(xy=TensorLike(keypoints)),
                boxes=StubFacePoseBoxes(),
            )
        ]


class RecordingStickerModel:
    names = {0: "U_sticker", 1: "F_sticker", 2: "R_sticker"}

    def __init__(self):
        self.source_shape = None
        self.source = None

    def predict(self, source, conf, imgsz, verbose):
        self.source_shape = source.shape[:2]
        self.source = source.copy()
        mask = np.zeros((1, source.shape[0], source.shape[1]), dtype=np.float32)
        mask[:, 2:8, 5:15] = 1.0
        masks = SimpleNamespace(
            data=SimpleNamespace(
                detach=lambda: SimpleNamespace(
                    cpu=lambda: SimpleNamespace(numpy=lambda: mask)
                )
            )
        )
        return [SimpleNamespace(masks=masks, boxes=StubBoxes())]


class StubMixedStickerModel:
    names = {0: "U_sticker", 1: "F_sticker", 2: "R_sticker"}

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
        "sticker_conf": 0.25,
        "sticker_imgsz": 256,
        "use_bbox_detector": True,
        "roi_padding": 0.0,
        "segmentation_padding_px": 0,
        "debug_overlay": False,
        "show_bbox": False,
        "show_confidence": False,
        "mask_opacity": 1.0,
        "color_plot_samples_per_frame": 800,
        "draw_sticker_graph": False,
        "graph_adjacency_radius": 6,
        "graph_min_sticker_area": 12,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_process_frame_uses_same_output_when_collecting_plot_samples():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    args = default_args(debug_overlay=True, show_bbox=True)

    output, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        args,
    )
    plot_output, plot_bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        args,
    )

    assert np.array_equal(output, plot_output)
    assert np.array_equal(bgr_samples, plot_bgr_samples)
    assert len(bgr_samples) > 1


def test_process_frame_can_use_sticker_only_viewport_without_bbox_detector():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[5:15, 5:15] = [0, 0, 220]

    output, bgr_samples = piece_seg_script.process_frame(
        frame,
        RaisingBboxDetector(),
        StubStickerModel(),
        default_args(use_bbox_detector=False, sticker_imgsz=40),
    )

    assert bgr_samples.shape == (10, 3)
    assert (bgr_samples[:, 2] > bgr_samples[:, 0]).all()
    assert np.any(np.all(output == piece_seg_script.CLASS_COLORS[0], axis=2))


def test_process_frame_requires_segmentation_masks():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:20] = [0, 0, 220]

    output, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubPoseStickerModel(),
        default_args(),
    )

    assert bgr_samples.shape == (0, 3)
    assert np.array_equal(output, frame)


def test_process_frame_ignores_pose_only_outputs():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    output, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubVisibleFacePoseModel(),
        default_args(),
    )

    assert bgr_samples.shape == (0, 3)
    assert np.array_equal(output, frame)


def test_sticker_only_scales_square_viewport_to_model_size():
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[8:16, 30:50] = [0, 0, 220]
    sticker_model = RecordingStickerModel()

    output, bgr_samples = piece_seg_script.process_frame(
        frame,
        RaisingBboxDetector(),
        sticker_model,
        default_args(use_bbox_detector=False, sticker_imgsz=20),
    )

    assert sticker_model.source_shape == (20, 20)
    assert sticker_model.source is not None
    assert np.any(sticker_model.source[:, :, 2] > sticker_model.source[:, :, 0])
    assert output.shape == (20, 20, 3)
    assert bgr_samples.shape == (10, 3)
    assert (bgr_samples[:, 2] > bgr_samples[:, 0]).all()


def test_sticker_only_viewport_uses_same_roi_overlay_flow_as_bbox_mode():
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    args = default_args(use_bbox_detector=False, sticker_imgsz=20, show_bbox=True)

    output, _ = piece_seg_script.process_frame(
        frame,
        RaisingBboxDetector(),
        RecordingStickerModel(),
        args,
    )

    assert output.shape == (20, 20, 3)
    assert output[0, 0].tolist() == [40, 220, 40]
    assert output[19, 19].tolist() == [40, 220, 40]


def test_centered_square_viewport_uses_largest_square_from_frame():
    assert piece_seg_script.centered_square_viewport(640, 480) == (80, 0, 560, 480)


def test_resize_to_square_size_scales_viewport_to_model_input_size():
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    resized = piece_seg_script.resize_to_square_size(image, 20)

    assert resized.shape == (20, 20, 3)


def test_frame_for_processing_scales_sticker_only_viewbox():
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[:, 0:20] = [220, 0, 0]
    frame[:, 20:60] = [0, 0, 220]
    frame[:, 60:80] = [0, 220, 0]

    working_frame = piece_seg_script.frame_for_processing(
        frame,
        default_args(use_bbox_detector=False, sticker_imgsz=20),
    )

    assert working_frame.shape == (20, 20, 3)
    assert np.all(working_frame[:, :, 2] > working_frame[:, :, 0])


def test_sticker_sampling_uses_same_mask_as_rendering():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:20] = [0, 0, 220]
    frame[24:30, 24:30] = [220, 0, 0]
    args = default_args()

    _, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        args,
    )

    assert len(bgr_samples) > 1
    assert (bgr_samples[:, 2] > bgr_samples[:, 0]).all()
    assert not (bgr_samples == [220, 0, 0]).all(axis=1).any()


def test_segmentation_padding_changes_displayed_and_sampled_mask_not_graph_mask():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[12:18, 12:18] = [0, 0, 220]
    args = default_args(segmentation_padding_px=-2)

    output, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        args,
    )

    assert len(bgr_samples) > 1
    assert (bgr_samples[:, 2] > bgr_samples[:, 0]).all()
    assert (output[10, 10] == [0, 0, 0]).all()
    assert not (output[15, 15] == [0, 0, 0]).all()


def test_sticker_color_sampling_returns_ten_samples_per_sticker():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:20] = [0, 0, 220]

    _, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        default_args(),
    )

    assert bgr_samples.shape == (10, 3)
    assert (bgr_samples[:, 2] > bgr_samples[:, 0]).all()


def test_sticker_color_sample_points_are_random_inside_mask():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 1:9] = True

    ys, xs = piece_seg_script.random_mask_sample_points(mask, 6)
    repeat_ys, repeat_xs = piece_seg_script.random_mask_sample_points(mask, 6)

    assert len(xs) == 6
    assert len(set(zip(xs, ys))) == 6
    assert mask[ys, xs].all()
    assert np.array_equal(xs, repeat_xs)
    assert np.array_equal(ys, repeat_ys)


def test_process_frame_can_draw_sticker_graph_overlay():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:20] = [0, 0, 220]
    args = default_args(draw_sticker_graph=True, graph_min_sticker_area=10_000)

    output, _ = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        args,
    )

    assert np.any(np.all(output == piece_seg_script.CLASS_COLORS[0], axis=2))


def test_process_frame_graph_uses_unpadded_mask_when_sampling_is_padded():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[12:18, 12:18] = [0, 0, 220]
    args = default_args(
        draw_sticker_graph=True,
        graph_min_sticker_area=1,
        segmentation_padding_px=-2,
    )
    captured = {}
    original_draw_sticker_graph = piece_seg_script.draw_sticker_graph

    def capture_graph(frame, graph, **kwargs):
        captured["graph"] = graph
        return frame

    piece_seg_script.draw_sticker_graph = capture_graph
    try:
        piece_seg_script.process_frame(
            frame,
            StubBboxDetector(),
            StubStickerModel(),
            args,
        )
    finally:
        piece_seg_script.draw_sticker_graph = original_draw_sticker_graph

    graph = captured["graph"]
    assert len(graph.nodes) == 1
    assert graph.nodes[0].mask.sum() == 100 * 255


def test_process_frame_does_not_accumulate_cube_net_without_classifier():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:20] = [0, 0, 220]
    args = default_args(graph_min_sticker_area=1)
    net_state = piece_seg_script.CubeNetState()

    piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        args,
        cube_net_state=net_state,
    )

    assert net_state.cells == {}


def test_parser_exposes_sticker_only_option():
    args = piece_seg_script.build_parser().parse_args(["--sticker-only"])

    assert args.use_bbox_detector is False


def test_parser_keeps_skip_bbox_as_hidden_compatibility_alias():
    parser = piece_seg_script.build_parser()
    args = parser.parse_args(["--skip-bbox"])

    assert args.use_bbox_detector is False
    assert "--skip-bbox" not in parser.format_help()


def test_parser_exposes_sticker_graph_options():
    args = piece_seg_script.build_parser().parse_args(
        [
            "--draw-sticker-graph",
            "--adjacency-radius",
            "9",
            "--graph-min-sticker-area",
            "5",
        ]
    )

    assert args.draw_sticker_graph is True
    assert args.graph_adjacency_radius == 9
    assert args.graph_min_sticker_area == 5


def test_parser_exposes_cube_net_output_option():
    args = piece_seg_script.build_parser().parse_args(["--save-net", "runs/net.png"])

    assert str(args.save_net) == "runs/net.png"


def test_parser_exposes_density_color_classifier_options():
    args = piece_seg_script.build_parser().parse_args(
        [
            "--color-classifier-mode",
            "density-cluster",
            "--cluster-eps-radians",
            "0.02",
            "--density-cluster-min-neighbors",
            "70",
            "--sticker-color-min-brightness",
            "50",
        ]
    )

    assert args.color_classifier_mode == "density-cluster"
    assert args.density_cluster_eps_radians == 0.02
    assert args.density_cluster_min_neighbors == 70
    assert args.sticker_color_min_brightness == 50


def test_color_plot_frames_zero_disables_color_features():
    args = piece_seg_script.build_parser().parse_args(
        [
            "--color-plot-frames",
            "0",
            "--rgb-color-plot",
            "runs/colors.png",
        ]
    )

    assert piece_seg_script.color_features_enabled(args) is False
    assert piece_seg_script.color_plot_outputs_requested(args) is False


def test_classifier_cluster_indexes_match_display_classifier():
    classifier = piece_seg_script.ColorDensityClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=1,
        density_cluster_eps_radians=0.2,
        density_cluster_min_neighbors=2,
    )
    classifier.centroids = np.array(
        [
            [220.0, 0.0, 0.0],
            [0.0, 220.0, 0.0],
        ],
        dtype=np.float32,
    )
    classifier.cluster_samples = [
        np.array([[220.0, 0.0, 0.0], [210.0, 5.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 220.0, 0.0], [5.0, 210.0, 0.0]], dtype=np.float32),
    ]
    bgr_samples = np.array(
        [
            [0, 0, 220],
            [0, 220, 0],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )

    indexes = piece_seg_script.classifier_cluster_indexes_for_bgr_samples(
        classifier,
        bgr_samples,
    )

    assert indexes.tolist() == [0, 1, -1]


def test_normalized_rgb_html_plot_path_uses_rgb_html_stem():
    path = piece_seg_script.normalized_rgb_html_plot_path(Path("runs/rgb.html"))

    assert path == Path("runs/rgb_normalized.html")


def test_normalized_rgb_html_plot_path_adds_html_suffix():
    path = piece_seg_script.normalized_rgb_html_plot_path(Path("runs/rgb"))

    assert path == Path("runs/rgb_normalized.html")
