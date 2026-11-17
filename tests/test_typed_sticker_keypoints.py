from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ultralytics.cfg import get_cfg

from live_recognition.typed_sticker_keypoints import CLASS_NAMES, sticker_class_id
from live_recognition.typed_sticker_training import TypedStickerTrainer
from scripts.relabel_typed_sticker_keypoints import relabel_rows


def test_relabel_changes_only_sticker_id_and_preserves_landmark_bytes():
    text = "0\t0.2 .3 .1 .1 .2 .3 2\r\n1 .4 .4 .1 .1 .4 .4 2\r\n2 .6 .6 .1 .1 .6 .6 2\r\n7\t.5 .6 .2 .2 .5 .6 1\r\n"
    target = {
        "image_size": [100, 100],
        "sticker_observations": [
            {
                "source_object": "StickerlessMaterialRadialPart.cubie.001.Blue",
                "class": "blue",
                "class_id": 7,
                "pixel_point": [50, 60],
                "visibility": 1,
            }
        ],
    }
    result, metadata = relabel_rows(text, target, {"cubie.001": "edge"})
    assert result == text.replace("7\t", "16\t")
    assert metadata[0]["class"] == "blue edge" and metadata[0]["class_id"] == 16
    assert metadata[0]["pixel_point"] == [50, 60] and metadata[0]["visibility"] == 1
    assert metadata[0]["source_class_id"] == 7


def test_no_reprojection_needed_even_without_pose_or_camera():
    text = "3 .5 .5 .2 .2 .5 .5 2\n"
    target = {
        "image_size": [100, 100],
        "sticker_observations": [
            {
                "source_object": "cubie.005:White",
                "source_cubie": "cubie.005",
                "class": "white",
                "class_id": 3,
                "pixel_point": [50, 50],
                "visibility": 2,
            }
        ],
    }
    result, _ = relabel_rows(text, target, {"cubie.005": "corner"})
    assert result == text
    assert relabel_rows("", {"sticker_observations": []}, {}) == ("", [])


def test_missing_metadata_fails_instead_of_dropping_sticker_rows():
    with pytest.raises(ValueError, match="count differs"):
        relabel_rows("3 .5 .5 .2 .2 .5 .5 2\n", {"sticker_observations": []}, {})


def test_schema_keeps_original_three_landmark_ids():
    assert CLASS_NAMES[:3] == ("corner", "corner-edge", "corner-edge-center")
    assert len(CLASS_NAMES) == 21
    assert sticker_class_id("white", "corner") == 3
    assert sticker_class_id("green", "center") == 20


def test_native_21_class_one_point_loss_backward():
    trainer = TypedStickerTrainer.__new__(TypedStickerTrainer)
    trainer.args = get_cfg()
    trainer.data = {"names": dict(enumerate(CLASS_NAMES)), "channels": 3, "kpt_shape": [1, 3]}
    torch.set_num_threads(2)
    model = trainer.get_model("yolo26n-pose.yaml", verbose=False)
    criterion = model.init_criterion()
    branch = getattr(criterion, "one2one", criterion)
    assert type(branch).__name__ == "PoseLoss26"
    assert model.model[-1].kpt_shape == [1, 3]
    batch = {
        "img": torch.rand(2, 3, 128, 128),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [20.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]] * 2),
        "keypoints": torch.tensor([[[0.5, 0.5, 2.0]], [[0.5, 0.5, 1.0]]]),
    }
    model.train()
    loss, _ = criterion(model(batch["img"]), batch)
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())


def test_renderer_maps_single_points_and_filters_object_scores():
    from live_recognition.roi_tracking import RoiTracker
    from scripts import render_yolo_typed_sticker_keypoints as renderer

    args = renderer.build_parser().parse_args(
        ["--source", "0", "--no-show-fps", "--roi-padding", "0"]
    )
    args.pose_conf = 0.25
    args.pose_class_conf = dict(renderer.DEFAULT_CLASS_CONF)
    result = SimpleNamespace(
        names=dict(enumerate(CLASS_NAMES)),
        boxes=SimpleNamespace(cls=np.asarray([1, 16]), conf=np.asarray([0.3, 0.95])),
        keypoints=SimpleNamespace(
            xy=np.asarray([[[5.0, 5.0]], [[15.0, 15.0]]]), conf=np.asarray([[0.999], [0.999]])
        ),
    )
    model = SimpleNamespace(predict=lambda **kwargs: [result])
    detector = SimpleNamespace(
        detect=lambda frame: SimpleNamespace(bbox_xyxy=(10, 10, 40, 40), confidence=0.85)
    )
    output, report = renderer.render_frame(
        np.zeros((100, 100, 3), np.uint8), detector, model, RoiTracker(), args
    )
    assert output.shape == (100, 100, 3)
    assert len(report["detections"]) == 1
    assert report["detections"][0]["class_name"] == "blue edge"
    np.testing.assert_allclose(report["detections"][0]["keypoints_full_frame"], [[25, 25]])
    assert renderer.parse_class_conf("corner-edge=0.3,corner_stickers=0.8")[1] == 0.3
