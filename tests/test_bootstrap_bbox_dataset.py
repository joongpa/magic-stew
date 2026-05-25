from pathlib import Path

from live_recognition.types import CubeCandidate
from scripts.bootstrap_bbox_dataset import clip_bbox, sample_images, write_yolo_label


def test_sample_images_is_repeatable():
    paths = [Path(f"frame_{index}.jpg") for index in range(20)]

    assert sample_images(paths, sample_size=5, seed=123) == sample_images(
        paths, sample_size=5, seed=123
    )


def test_clip_bbox_clamps_to_image_bounds():
    candidate = CubeCandidate(bbox_xyxy=(-10, 5, 110, 75), confidence=0.8)

    assert clip_bbox(candidate, width=100, height=50) == (0.0, 5.0, 100.0, 45.0)


def test_clip_bbox_drops_empty_boxes():
    candidate = CubeCandidate(bbox_xyxy=(20, 20, 10, 40), confidence=0.8)

    assert clip_bbox(candidate, width=100, height=50) is None


def test_write_yolo_label_normalizes_xywh(tmp_path):
    label_path = tmp_path / "frame.txt"

    write_yolo_label(label_path, [(10.0, 20.0, 30.0, 40.0)], width=100, height=200)

    assert label_path.read_text(encoding="utf-8") == "0 0.250000 0.200000 0.300000 0.200000\n"
