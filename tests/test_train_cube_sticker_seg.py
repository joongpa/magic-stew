import json
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_yolo11n_cube_sticker_seg_local.py"


def load_training_module():
    spec = importlib.util.spec_from_file_location("cube_sticker_seg_training", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training script: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_dataset(source_dir: Path, stem: str) -> None:
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (images_dir / f"{stem}.png").write_bytes(b"image")
    (labels_dir / f"{stem}.txt").write_text(
        "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
        "1 0.2 0.2 0.3 0.2 0.3 0.3 0.2 0.3\n",
        encoding="utf-8",
    )


def test_write_dataset_combines_sources_and_negative_images(tmp_path):
    training = load_training_module()
    source_a = tmp_path / "dataset_a"
    source_b = tmp_path / "dataset_b"
    negatives = tmp_path / "negatives"
    output = tmp_path / "prepared"
    _write_dataset(source_a, "a")
    _write_dataset(source_b, "b")
    negatives.mkdir()
    (negatives / "background.jpg").write_bytes(b"background")

    data_yaml = training.write_dataset(
        source_dirs=[source_a, source_b],
        dataset_dir=output,
        val_fraction=0.0,
        seed=17,
        class_names=["cube", "sticker"],
        extra_negative_dir=negatives,
    )

    manifest = json.loads((output / "conversion_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 3
    assert sum(item["negative"] for item in manifest) == 1
    negative_item = next(item for item in manifest if item["negative"])
    assert negative_item["source_label"] is None
    assert Path(negative_item["label"]).read_text(encoding="utf-8") == ""
    assert data_yaml.read_text(encoding="utf-8").endswith("  0: cube\n  1: sticker\n")
