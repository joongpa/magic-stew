import json
from pathlib import Path

from scripts.train_yolo11n_whole_cube_seg_local import write_dataset


def _write_positive_dataset(source_dir: Path) -> None:
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels" / "whole_cube"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (labels_dir / "classes.json").write_text('{"0": "cube"}', encoding="utf-8")
    (images_dir / "positive.png").write_bytes(b"positive")
    (labels_dir / "positive.txt").write_text(
        "0 0.1 0.1 0.9 0.1 0.5 0.9\n",
        encoding="utf-8",
    )


def test_write_dataset_includes_negative_images_with_empty_labels(tmp_path):
    source_dir = tmp_path / "mcp_dataset_v4"
    negative_dir = tmp_path / "negative_ROI_backgrounds"
    dataset_dir = tmp_path / "prepared"
    _write_positive_dataset(source_dir)
    negative_dir.mkdir()
    (negative_dir / "background.JPG").write_bytes(b"background")
    (negative_dir / "background.png").write_bytes(b"background")

    data_yaml = write_dataset(
        source_dir=source_dir,
        dataset_dir=dataset_dir,
        val_fraction=0.0,
        seed=17,
        negative_dir=negative_dir,
    )

    manifest = json.loads((dataset_dir / "conversion_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 3
    assert sum(item["negative"] for item in manifest) == 2
    assert sum(item["training_instances"] for item in manifest) == 1
    assert data_yaml.read_text(encoding="utf-8").startswith(f"path: {dataset_dir.resolve()}\n")

    negative_items = [item for item in manifest if item["negative"]]
    assert all(item["source_label"] is None for item in negative_items)
    assert all(Path(item["label"]).read_text(encoding="utf-8") == "" for item in negative_items)
