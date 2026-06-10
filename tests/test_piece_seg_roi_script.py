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


class StubStickerModel:
    names = {0: "center_sticker", 1: "edge_sticker", 2: "corner_sticker"}

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


class RecordingStickerModel:
    names = {0: "center_sticker", 1: "edge_sticker", 2: "corner_sticker"}

    def __init__(self):
        self.source_shape = None

    def predict(self, source, conf, imgsz, verbose):
        self.source_shape = source.shape[:2]
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
    names = {0: "center_sticker", 1: "edge_sticker", 2: "corner_sticker"}

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
        "show_sticker_color_classification": True,
        "draw_sticker_graph": False,
        "graph_adjacency_radius": 6,
        "graph_min_sticker_area": 12,
        "net_cluster_faces": {},
        "density_cluster_eps_radians": piece_seg_script.DENSITY_CLUSTER_EPS_RADIANS,
        "density_cluster_raw_eps": piece_seg_script.DENSITY_CLUSTER_RAW_EPS,
        "density_cluster_min_neighbors": piece_seg_script.DENSITY_CLUSTER_MIN_POINTS,
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


def test_process_frame_can_skip_bbox_detector_and_segment_full_frame():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[5:15, 5:15] = [0, 0, 220]

    output, bgr_samples = piece_seg_script.process_frame(
        frame,
        RaisingBboxDetector(),
        StubStickerModel(),
        default_args(use_bbox_detector=False),
    )

    assert bgr_samples.shape == (10, 3)
    assert (bgr_samples[:, 2] > bgr_samples[:, 0]).all()
    assert np.any(np.all(output == piece_seg_script.CLASS_COLORS[0], axis=2))


def test_skip_bbox_downscales_segmentation_input():
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[8:32, 20:60] = [0, 0, 220]
    sticker_model = RecordingStickerModel()

    _, bgr_samples = piece_seg_script.process_frame(
        frame,
        RaisingBboxDetector(),
        sticker_model,
        default_args(use_bbox_detector=False, sticker_imgsz=20),
    )

    assert sticker_model.source_shape == (10, 20)
    assert bgr_samples.shape == (10, 3)
    assert (bgr_samples[:, 2] > bgr_samples[:, 0]).all()


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
    assert (output[10, 10] == [40, 40, 40]).all()
    assert not (output[15, 15] == [40, 40, 40]).all()


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
    assert len(set(zip(xs, ys, strict=True))) == 6
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


def test_process_frame_can_accumulate_cube_net_from_roi_graph():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:20] = [0, 0, 220]
    args = default_args(graph_min_sticker_area=1, net_cluster_faces={"cluster-1": "R"})
    net_state = piece_seg_script.CubeNetState()
    classifier = piece_seg_script.ColorRayClassifier(required_frames=1)
    classifier.centroids = np.array([[220.0, 0.0, 0.0]], dtype=np.float32)
    classifier.cluster_samples = [np.array([[220.0, 0.0, 0.0]], dtype=np.float32)]

    piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        args,
        classifier,
        cube_net_state=net_state,
    )

    assert net_state.cells[("R", 1, 1)].color == "cluster-1"
    assert net_state.cells[("R", 1, 1)].bgr == (0, 0, 220)
    assert net_state.cells[("R", 1, 1)].source == "observed"


def test_process_frame_waits_for_color_sampling_before_cube_net_accumulation():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:20] = [0, 0, 220]
    args = default_args(graph_min_sticker_area=1)
    net_state = piece_seg_script.CubeNetState()
    classifier = piece_seg_script.ColorRayClassifier(required_frames=2)

    piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        args,
        classifier,
        cube_net_state=net_state,
    )

    assert classifier.ready is False
    assert net_state.cells == {}


def test_color_classifier_collects_regular_samples_from_sticker_detections():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:17] = [0, 0, 220]
    frame[10:20, 17:25] = [220, 0, 0]
    classifier = piece_seg_script.ColorRayClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=20,
    )

    _, _ = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubMixedStickerModel(),
        default_args(),
        classifier,
    )

    observed = np.asarray(classifier._rgb_samples)
    assert classifier.detected_frames == 1
    assert len(observed) == 20
    assert (observed[:, 0] > observed[:, 2]).any()
    assert (observed[:, 2] > observed[:, 0]).any()


def test_ready_color_classifier_merges_same_color_sticker_segments():
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[5:20, 5:12] = [0, 0, 220]
    frame[5:20, 12:20] = [0, 0, 180]
    left_mask = np.zeros((30, 30), dtype=bool)
    left_mask[5:20, 5:12] = True
    right_mask = np.zeros((30, 30), dtype=bool)
    right_mask[5:20, 12:20] = True
    classifier = piece_seg_script.ColorRayClassifier(
        required_frames=1,
        cluster_count=1,
        min_brightness=20,
    )
    classifier.centroids = np.array([[220.0, 0.0, 0.0]], dtype=np.float32)
    classifier.cluster_samples = [np.array([[220.0, 0.0, 0.0]], dtype=np.float32)]
    stickers = [
        (left_mask, np.array([[0, 0, 220]], dtype=np.uint8)),
        (right_mask, np.array([[0, 0, 180]], dtype=np.uint8)),
    ]

    merged = piece_seg_script.stickers_for_ready_color_classifier(
        frame,
        (0, 0, 30, 30),
        stickers,
        classifier,
    )

    assert len(merged) == 1
    assert merged[0][2] == 0
    assert int(merged[0][0].sum()) == int(left_mask.sum() + right_mask.sum())


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
        StubMixedStickerModel(),
        default_args(),
        classifier,
    )
    output, _ = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubMixedStickerModel(),
        default_args(),
        classifier,
    )

    assert classifier.ready
    assert not np.array_equal(output[15, 15], piece_seg_script.CLASS_COLORS[0])
    assert not np.array_equal(output[15, 22], piece_seg_script.CLASS_COLORS[1])
    assert np.any(np.all(output == [220, 0, 0], axis=2))


def test_dim_sticker_masks_are_not_recolored_by_classifier():
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
        StubMixedStickerModel(),
        default_args(),
        classifier,
    )
    dim_frame = np.zeros((40, 40, 3), dtype=np.uint8)
    dim_frame[:, :] = [5, 5, 5]
    dim_frame[10:20, 10:17] = [0, 0, 25]
    dim_frame[10:20, 17:25] = [25, 0, 0]

    output, _ = piece_seg_script.process_frame(
        dim_frame,
        StubBboxDetector(),
        StubMixedStickerModel(),
        default_args(),
        classifier,
    )

    assert classifier.ready
    assert not np.any(np.all(output == [0, 0, 220], axis=2))
    assert not np.any(np.all(output == [220, 0, 0], axis=2))


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


def test_normalized_rgb_cluster_distance_ignores_brightness():
    distances = piece_seg_script.distance_to_normalized_rgb_clusters(
        np.array([100.0, 0.0, 0.0], dtype=np.float32),
        [
            np.array([[10.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[70.0, 70.0, 0.0]], dtype=np.float32),
        ],
    )

    assert distances[0] == 0
    assert distances[1] > 0


def test_classifier_can_use_nearest_brightness_normalized_cluster_sample():
    classifier = piece_seg_script.ColorRayClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=1,
        mode="closest-cluster-normalized",
    )
    classifier.centroids = np.array(
        [
            [10.0, 0.0, 0.0],
            [100.0, 30.0, 0.0],
        ],
        dtype=np.float32,
    )
    classifier.cluster_samples = [
        np.array([[10.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[100.0, 30.0, 0.0]], dtype=np.float32),
    ]

    assert classifier.classify_bgr(np.array([0, 0, 100], dtype=np.uint8)) == 0


def test_normalized_cluster_centroid_directions_average_normalized_members():
    directions = piece_seg_script.normalized_cluster_centroid_directions(
        [
            np.array(
                [
                    [100.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0.0, 100.0, 0.0],
                    [0.0, 10.0, 0.0],
                ],
                dtype=np.float32,
            ),
        ]
    )

    assert np.allclose(directions[0], [1.0, 0.0, 0.0])
    assert np.allclose(directions[1], [0.0, 1.0, 0.0])


def test_classifier_can_use_normalized_cluster_centroid_regression():
    classifier = piece_seg_script.ColorRayClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=1,
        mode="normalized-centroid-regression",
    )
    classifier.centroids = np.array(
        [
            [30.0, 0.0, 0.0],
            [100.0, 80.0, 0.0],
        ],
        dtype=np.float32,
    )
    classifier.cluster_samples = [
        np.array([[30.0, 0.0, 0.0], [80.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[100.0, 80.0, 0.0], [120.0, 96.0, 0.0]], dtype=np.float32),
    ]

    assert classifier.classify_bgr(np.array([0, 0, 180], dtype=np.uint8)) == 0


def test_density_rgb_clusters_merges_smallest_clusters_to_target_count():
    rgb_samples = []
    for angle in np.linspace(0.0, 0.5, 8):
        direction = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
        rgb_samples.extend(direction * value for value in (100.0, 110.0, 120.0, 130.0))
    rgb_samples = np.asarray(rgb_samples, dtype=np.float32)

    _, _, clusters = piece_seg_script.density_rgb_clusters(
        rgb_samples,
        target_clusters=6,
        min_brightness=1,
        eps_radians=0.02,
        min_points=2,
    )

    assert len(clusters) == 6


def test_raw_density_rgb_clusters_uses_raw_rgb_distance():
    rgb_samples = np.array(
        [
            [100.0, 0.0, 0.0],
            [105.0, 0.0, 0.0],
            [110.0, 2.0, 0.0],
            [0.0, 0.0, 100.0],
            [0.0, 5.0, 105.0],
            [2.0, 0.0, 110.0],
        ],
        dtype=np.float32,
    )

    _, centroids, clusters = piece_seg_script.raw_density_rgb_clusters(
        rgb_samples,
        target_clusters=6,
        min_brightness=1,
        eps=12,
        min_points=2,
    )

    assert len(clusters) == 2
    assert centroids.shape == (2, 3)


def test_density_classifier_rejects_without_regression_fallback():
    classifier = piece_seg_script.ColorRayClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=1,
        mode="density-cluster",
    )
    classifier.directions = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    classifier.centroids = np.array([[0.0, 100.0, 0.0]], dtype=np.float32)
    classifier.cluster_samples = [
        np.array([[100.0, 0.0, 0.0]], dtype=np.float32),
    ]

    assert classifier.classify_bgr(np.array([0, 100, 0], dtype=np.uint8)) is None


def test_raw_density_classifier_rejects_outside_raw_epsilon():
    classifier = piece_seg_script.ColorRayClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=1,
        density_cluster_raw_eps=12,
        mode="raw-density-cluster",
    )
    classifier.centroids = np.array([[100.0, 0.0, 0.0]], dtype=np.float32)
    classifier.cluster_samples = [
        np.array([[100.0, 0.0, 0.0]], dtype=np.float32),
    ]

    assert classifier.classify_bgr(np.array([0, 0, 110], dtype=np.uint8)) == 0
    assert classifier.classify_bgr(np.array([0, 0, 140], dtype=np.uint8)) is None


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


def test_parser_exposes_skip_bbox_option():
    args = piece_seg_script.build_parser().parse_args(["--skip-bbox"])

    assert args.use_bbox_detector is False


def test_parser_exposes_normalized_cluster_color_classifier_mode():
    args = piece_seg_script.build_parser().parse_args(
        ["--color-classifier-mode", "closest-cluster-normalized"]
    )

    assert args.color_classifier_mode == "closest-cluster-normalized"


def test_parser_exposes_density_cluster_color_classifier_mode():
    args = piece_seg_script.build_parser().parse_args(
        ["--color-classifier-mode", "density-cluster"]
    )

    assert args.color_classifier_mode == "density-cluster"


def test_parser_exposes_raw_density_cluster_color_classifier_mode():
    args = piece_seg_script.build_parser().parse_args(
        ["--color-classifier-mode", "raw-density-cluster"]
    )

    assert args.color_classifier_mode == "raw-density-cluster"


def test_parser_exposes_density_cluster_options():
    args = piece_seg_script.build_parser().parse_args(
        [
            "--density-cluster-eps-radians",
            "0.12",
            "--density-cluster-raw-eps",
            "18",
            "--density-cluster-min-neighbors",
            "7",
        ]
    )

    assert args.density_cluster_eps_radians == 0.12
    assert args.density_cluster_raw_eps == 18
    assert args.density_cluster_min_neighbors == 7


def test_parser_exposes_normalized_centroid_regression_color_classifier_mode():
    args = piece_seg_script.build_parser().parse_args(
        ["--color-classifier-mode", "normalized-centroid-regression"]
    )

    assert args.color_classifier_mode == "normalized-centroid-regression"


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


def test_parser_exposes_net_cluster_face_mapping():
    args = piece_seg_script.build_parser().parse_args(
        ["--net-cluster-faces", "1=R,2=U"]
    )

    assert args.net_cluster_faces == {"cluster-1": "R", "cluster-2": "U"}


def test_color_cluster_legend_draws_classifier_centroids():
    classifier = piece_seg_script.ColorRayClassifier(required_frames=1)
    classifier.centroids = np.array(
        [
            [220.0, 0.0, 0.0],
            [0.0, 220.0, 0.0],
        ],
        dtype=np.float32,
    )
    classifier.cluster_samples = [np.array([[220.0, 0.0, 0.0]], dtype=np.float32)]

    legend = piece_seg_script.draw_color_cluster_legend(classifier)

    assert legend.shape[0] >= 128
    assert (legend[16, 16] == [0, 0, 220]).all()
    assert (legend[80, 16] == [0, 220, 0]).all()


def test_cluster_legend_path_uses_net_path_stem():
    path = piece_seg_script.cluster_legend_path(Path("runs/net.png"))

    assert path == Path("runs/net_clusters.png")


def test_normalized_rgb_html_plot_path_uses_rgb_html_stem():
    path = piece_seg_script.normalized_rgb_html_plot_path(Path("runs/rgb.html"))

    assert path == Path("runs/rgb_normalized.html")


def test_normalized_rgb_html_plot_path_adds_html_suffix():
    path = piece_seg_script.normalized_rgb_html_plot_path(Path("runs/rgb"))

    assert path == Path("runs/rgb_normalized.html")
