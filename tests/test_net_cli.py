from types import SimpleNamespace

import numpy as np

from live_recognition import net_cli
from live_recognition.net_cli import (
    StickerMask,
    StickerSegmentation,
    build_parser,
    draw_sticker_segmentation_graph,
    segment_stickers,
)
from live_recognition.piece_graph import PieceGraph as StickerGraph


def test_net_command_exposes_raw_net_outputs():
    args = build_parser().parse_args(
        [
            "--source",
            "frame.jpg",
            "--save-net",
            "runs/net.png",
            "--save-graph",
            "runs/graph.jpg",
        ]
    )

    assert args.source == "frame.jpg"
    assert str(args.save_net) == "runs/net.png"
    assert str(args.save_graph) == "runs/graph.jpg"


def test_net_command_defaults_match_sticker_graph_command():
    args = build_parser().parse_args(["--source", "frame.jpg"])

    assert args.sticker_conf == 0.25
    assert args.sticker_imgsz == 256
    assert args.roi_padding == 0.0
    assert args.segmentation_padding_px == -2


def test_net_command_uses_sticker_model_argument():
    args = build_parser().parse_args(
        [
            "--source",
            "frame.jpg",
            "--sticker-model",
            "runs/stickers.pt",
            "--sticker-conf",
            "0.9",
            "--sticker-imgsz",
            "320",
        ]
    )

    assert args.sticker_model == "runs/stickers.pt"
    assert args.sticker_conf == 0.9
    assert args.sticker_imgsz == 320


def test_net_command_uses_bbox_roi_by_default(monkeypatch, tmp_path):
    source = tmp_path / "frame.png"
    source.write_bytes(b"not a real image")
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    sticker_masks = [object()]
    sticker_nodes = [object()]
    captured = {}
    graph_calls = []

    class StubCubeDetector:
        def __init__(self, **kwargs):
            captured["bbox_loaded"] = True

        def detect(self, frame):
            return SimpleNamespace(bbox_xyxy=(2, 2, 18, 18), confidence=0.9)

    class StubYOLO:
        def __init__(self, model_name):
            captured["sticker_model"] = model_name

    def fake_draw_sticker_segmentation_graph(frame, masks, graph, **kwargs):
        captured["masks"] = masks
        captured["roi_xyxy"] = kwargs.get("roi_xyxy")
        captured["cube_candidate"] = kwargs.get("cube_candidate")
        return frame

    monkeypatch.setattr(net_cli.cv2, "imread", lambda path: frame.copy())
    monkeypatch.setattr(net_cli.cv2, "imwrite", lambda path, image: True)
    monkeypatch.setattr(net_cli, "YOLOCubeDetector", StubCubeDetector)
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", SimpleNamespace(YOLO=StubYOLO))
    monkeypatch.setattr(
        net_cli,
        "segment_stickers",
        lambda *args, **kwargs: StickerSegmentation(masks=sticker_masks, nodes=sticker_nodes),
    )
    monkeypatch.setattr(
        net_cli,
        "build_sticker_adjacency_graph",
        lambda nodes, **kwargs: (
            graph_calls.append(kwargs) or SimpleNamespace(nodes=nodes, edges=[])
        ),
    )
    monkeypatch.setattr(
        net_cli,
        "build_observed_cube_net_from_graph",
        lambda graph: SimpleNamespace(cells={}),
    )
    monkeypatch.setattr(net_cli, "draw_cube_net", lambda state: frame.copy())
    monkeypatch.setattr(
        net_cli,
        "draw_sticker_segmentation_graph",
        fake_draw_sticker_segmentation_graph,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "cube-net",
            "--source",
            str(source),
            "--save-graph",
            str(tmp_path / "graph.png"),
        ],
    )

    assert net_cli.main() == 0
    assert captured["bbox_loaded"] is True
    assert captured["sticker_model"].endswith("best.pt")
    assert captured["masks"] is sticker_masks
    assert captured["roi_xyxy"] == (2, 2, 18, 18)
    assert captured["cube_candidate"].bbox_xyxy == (2, 2, 18, 18)
    assert graph_calls == [
        {"adjacency_radius_px": 0, "infer_piece_ids": False},
        {"adjacency_radius_px": 0, "infer_piece_ids": False},
    ]


def test_net_command_passes_adjacency_radius_to_sticker_graph(monkeypatch, tmp_path):
    source = tmp_path / "frame.png"
    source.write_bytes(b"not a real image")
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    captured = {}

    class StubCubeDetector:
        def __init__(self, **kwargs):
            pass

        def detect(self, frame):
            return SimpleNamespace(bbox_xyxy=(2, 2, 18, 18), confidence=0.9)

    class StubYOLO:
        def __init__(self, model_name):
            pass

    monkeypatch.setattr(net_cli.cv2, "imread", lambda path: frame.copy())
    monkeypatch.setattr(net_cli.cv2, "imwrite", lambda path, image: True)
    monkeypatch.setattr(net_cli, "YOLOCubeDetector", StubCubeDetector)
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", SimpleNamespace(YOLO=StubYOLO))
    monkeypatch.setattr(
        net_cli,
        "segment_stickers",
        lambda *args, **kwargs: StickerSegmentation(masks=[], nodes=[]),
    )

    def fake_build_sticker_adjacency_graph(nodes, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(nodes=[], edges=[])

    monkeypatch.setattr(net_cli, "build_sticker_adjacency_graph", fake_build_sticker_adjacency_graph)
    monkeypatch.setattr(
        net_cli,
        "build_observed_cube_net_from_graph",
        lambda graph: SimpleNamespace(cells={}),
    )
    monkeypatch.setattr(net_cli, "draw_cube_net", lambda state: frame.copy())
    monkeypatch.setattr(
        "sys.argv",
        [
            "cube-net",
            "--source",
            str(source),
            "--adjacency-radius",
            "37",
        ],
    )

    assert net_cli.main() == 0
    assert captured["adjacency_radius_px"] == 37
    assert captured["infer_piece_ids"] is False


def test_sticker_segmentation_graph_draws_roi_masks_directly():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    roi_mask = np.zeros((6, 6), dtype=bool)
    roi_mask[2:4, 1:5] = True
    sticker_mask = StickerMask(
        sticker_id=0,
        class_id=0,
        sticker_type="center",
        confidence=0.99,
        mask=np.zeros((20, 20), dtype=np.uint8),
        roi_mask=roi_mask,
        roi_xyxy=(5, 6, 11, 12),
    )

    output = draw_sticker_segmentation_graph(
        frame,
        [sticker_mask],
        StickerGraph(nodes=[], edges=[]),
        cube_candidate=SimpleNamespace(bbox_xyxy=(5, 6, 11, 12)),
        roi_xyxy=(5, 6, 11, 12),
        show_confidence=False,
    )

    assert np.any(output[8:10, 6:10] != frame[8:10, 6:10])


def test_sticker_segmentation_graph_colors_masks_by_sticker_type():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    roi_mask = np.zeros((10, 10), dtype=bool)
    roi_mask[4:6, 4:6] = True
    sticker_mask = StickerMask(
        sticker_id=0,
        class_id=0,
        sticker_type="corner",
        confidence=0.99,
        mask=np.zeros((20, 20), dtype=np.uint8),
        roi_mask=roi_mask,
        roi_xyxy=(4, 5, 14, 15),
    )

    output = draw_sticker_segmentation_graph(
        frame,
        [sticker_mask],
        StickerGraph(nodes=[], edges=[]),
        cube_candidate=SimpleNamespace(bbox_xyxy=(4, 5, 14, 15)),
        roi_xyxy=(4, 5, 14, 15),
        show_confidence=False,
    )

    assert output[9, 8].tolist() == [255, 120, 80]


def test_sticker_segmentation_graph_ignores_padded_sample_mask_for_display():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    roi_mask = np.zeros((8, 8), dtype=bool)
    roi_mask[2:6, 2:6] = True
    sample_roi_mask = np.zeros((8, 8), dtype=bool)
    sample_roi_mask[3:5, 3:5] = True
    sticker_mask = StickerMask(
        sticker_id=0,
        class_id=0,
        sticker_type="center",
        confidence=0.99,
        mask=np.zeros((20, 20), dtype=np.uint8),
        roi_mask=roi_mask,
        roi_xyxy=(4, 5, 12, 13),
    )

    output = draw_sticker_segmentation_graph(
        frame,
        [sticker_mask],
        StickerGraph(nodes=[], edges=[]),
        roi_xyxy=(4, 5, 12, 13),
        show_confidence=False,
    )

    assert output[7, 6].tolist() == [80, 220, 255]


def test_sticker_segmentation_graph_draws_roi_style_bbox_on_top():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    output = draw_sticker_segmentation_graph(
        frame,
        [],
        StickerGraph(nodes=[], edges=[]),
        cube_candidate=SimpleNamespace(bbox_xyxy=(4, 5, 14, 15)),
        roi_xyxy=(4, 5, 14, 15),
    )

    assert output[5, 4].tolist() == [40, 220, 40]


def test_segment_stickers_accepts_roi_command_class_names():
    class Item:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class Boxes:
        cls = [Item(0)]
        conf = [Item(0.95)]

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

    class StickerModel:
        names = {0: "center_sticker"}

        def predict(self, source, conf, imgsz, verbose):
            mask = np.zeros((1, source.shape[0], source.shape[1]), dtype=np.float32)
            mask[0, 1:4, 2:5] = 1.0
            return [SimpleNamespace(masks=SimpleNamespace(data=TensorLike(mask)), boxes=Boxes())]

    roi = np.zeros((6, 8, 3), dtype=np.uint8)

    segmentation = segment_stickers(
        StickerModel(),
        roi,
        frame_shape=(20, 20),
        roi_xyxy=(5, 6, 13, 12),
        confidence_threshold=0.25,
        image_size=256,
        segmentation_padding_px=0,
        min_sticker_area_px=1,
    )

    stickers = segmentation.masks
    assert len(stickers) == 1
    assert stickers[0].sticker_type == "center"
    assert stickers[0].roi_mask.any()
    assert stickers[0].mask[7:10, 7:10].any()
    assert len(segmentation.nodes) == 1
    assert segmentation.nodes[0].mean_bgr == (0, 0, 0)
