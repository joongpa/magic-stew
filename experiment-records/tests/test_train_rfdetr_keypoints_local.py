from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_rfdetr_keypoints_local.py"


@pytest.fixture
def training():
    spec = importlib.util.spec_from_file_location("rfdetr_keypoint_training", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_source(root, names, *, manifest="sticker_keypoint_manifest.json", count=4):
    (root / "images").mkdir(parents=True)
    (root / "pose").mkdir()
    (root / manifest).write_text(json.dumps({"class_names": names, "keypoint_shape": [1, 3]}))
    for index in range(count):
        (root / "images" / f"{index}.png").write_bytes(b"image")
        (root / "pose" / f"{index}.txt").write_text(
            f"{len(names) - 1} 0.5 0.5 0.1 0.1 0.51 0.49 2\n"
        )
    return root


def test_combined_sources_preserve_color_order_and_negative_labels(training, tmp_path):
    # Generator color order differs from the illustrative order in the chat.
    names = [
        "corner",
        "corner-edge",
        "corner-edge-center",
        "white",
        "yellow",
        "orange",
        "red",
        "blue",
        "green",
    ]
    sources = [make_source(tmp_path / name, names) for name in ("a", "b")]
    negatives = tmp_path / "negatives"
    negatives.mkdir()
    for name in ("background.JPG", "background.webp", "background.png"):
        (negatives / name).write_bytes(b"background")
    data_yaml = training.write_dataset(
        sources, tmp_path / "out", 0.25, 17, extra_negative_dir=negatives
    )
    metadata = yaml.safe_load(data_yaml.read_text())
    assert list(metadata["names"].values()) == names
    assert metadata["kpt_shape"] == [1, 3]
    manifest_path = data_yaml.parent / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest) == 11
    assert sum(item["negative"] for item in manifest) == 3
    for item in manifest:
        if item["negative"]:
            assert Path(item["label"]).read_text() == ""
        else:
            assert Path(item["label"]).read_text().startswith("8 ")
    assert {item["split"] for item in manifest if not item["negative"]} == {"train", "val"}
    assert len({item["source_image"] for item in manifest}) == 11
    # Repeating preparation with the same seed gives exactly the same split.
    training.write_dataset(sources, data_yaml.parent, 0.25, 17, extra_negative_dir=negatives)
    assert json.loads(manifest_path.read_text()) == manifest


def test_legacy_manifest_overrides_segmentation_names(training, tmp_path):
    names = ["corner", "corner-edge", "corner-edge-center"]
    source = make_source(tmp_path / "source", names, manifest="independent_keypoint_manifest.json")
    (source / "data.yaml").write_text("names: [cube, sticker]\nkpt_shape: [1, 3]\n")
    assert training.source_class_names(source) == names
    prepared = training.write_dataset([source], tmp_path / "out", 0.25, 17)
    assert list(yaml.safe_load(prepared.read_text())["names"].values()) == names


def test_yaml_schema_and_explicit_override(training, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.yaml").write_text("names: {0: landmark, 1: red}\nkpt_shape: [1, 3]\n")
    assert training.source_class_names(source) == ["landmark", "red"]
    assert training.resolve_class_names([source], ["point"]) == ["point"]


def test_mismatched_class_orders_are_rejected(training, tmp_path):
    a = make_source(tmp_path / "a", ["red", "orange"])
    b = make_source(tmp_path / "b", ["orange", "red"])
    with pytest.raises(ValueError, match="Class order differs"):
        training.write_dataset([a, b], tmp_path / "out", 0.2, 17)


@pytest.mark.parametrize(
    "label",
    [
        "0.5 0.5 0.5 0.1 0.1 0.5 0.5 2",
        "1 0.5 0.5 0.1 0.1 0.5 0.5 2",
        "0 0.5 0.5 0 0.1 0.5 0.5 2",
        "0 0.5 0.5 0.1 0.1 1.1 0.5 2",
        "0 0.5 0.5 0.1 0.1 0.5 0.5 1.5",
        "0 nan 0.5 0.1 0.1 0.5 0.5 2",
        "0 0.5 0.5 0.1 0.1 0.5 0.5",
    ],
)
def test_invalid_labels_rejected_before_replacing_dataset(training, tmp_path, label):
    source = make_source(tmp_path / "source", ["point"])
    output = tmp_path / "out"
    training.write_dataset([source], output, 0.2, 17)
    original = (output / "conversion_manifest.json").read_text()
    (source / "pose/0.txt").write_text(label)
    with pytest.raises(ValueError):
        training.write_dataset([source], output, 0.2, 17)
    assert (output / "conversion_manifest.json").read_text() == original


def test_missing_label_is_not_silently_treated_as_background(training, tmp_path):
    source = make_source(tmp_path / "source", ["point"])
    (source / "pose/0.txt").unlink()
    with pytest.raises(FileNotFoundError, match="Missing pose label"):
        training.write_dataset([source], tmp_path / "out", 0.2, 17)


def test_skipping_missing_labels_requires_at_least_one_labeled_image(training, tmp_path):
    source = make_source(tmp_path / "source", ["point"], count=1)
    (source / "pose/0.txt").unlink()
    with pytest.raises(ValueError, match="No images with pose labels"):
        training.write_dataset([source], tmp_path / "out", 0.2, 17, skip_missing_labels=True)


def test_output_cannot_replace_source_or_unrecognized_directory(training, tmp_path):
    source = make_source(tmp_path / "source", ["point"])
    with pytest.raises(ValueError, match="separate"):
        training.write_dataset([source], source, 0.2, 17)
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="unrecognized"):
        training.write_dataset([source], output, 0.2, 17)
    assert (output / "keep.txt").read_text() == "keep"


def test_training_routes_schema_and_options_to_rfdetr(training, tmp_path, monkeypatch):
    names = ["corner", "red", "orange"]
    source = make_source(tmp_path / "source", names)
    prepared = training.write_dataset([source], tmp_path / "dataset", 0.25, 17)
    captured = {}

    class FakeRFDETR:
        def __init__(self, **kwargs):
            captured["model"] = kwargs

        def train(self, **kwargs):
            captured["train"] = kwargs
            output = Path(kwargs["output_dir"])
            output.mkdir(parents=True)
            (output / "checkpoint_best_total.pth").write_bytes(b"checkpoint")

    fake_module = types.ModuleType("rfdetr")
    fake_module.RFDETRKeypointPreview = FakeRFDETR
    monkeypatch.setitem(sys.modules, "rfdetr", fake_module)
    args = training.parse_args(
        [
            "--device",
            "0",
            "--epochs",
            "2",
            "--batch",
            "2",
            "--resolution",
            "144",
            "--model",
            "custom.pth",
            "--no-ema",
            "--patience",
            "0",
        ]
    )
    output = training.train(prepared, tmp_path / "runs", args)
    assert output.is_file()
    assert captured["model"] == {
        "num_classes": 3,
        "num_keypoints_per_class": [1, 1, 1],
        "resolution": 144,
        "device": "cuda:0",
        "pretrain_weights": "custom.pth",
    }
    options = captured["train"]
    assert options["dataset_file"] == "yolo"
    assert options["class_names"] == names
    assert options["batch_size"] == 2
    assert options["progress_bar"] == "tqdm"
    assert options["skip_best_epochs"] == 1
    assert options["use_ema"] is False
    assert options["early_stopping"] is False
    assert options["run_test"] is False
    assert options["multi_scale"] is False
    assert options["scale_jitter"] is False
    assert options["aug_config"] == {}
    second = training.train(prepared, tmp_path / "runs", args)
    assert second.parent.name == f"{args.name}-2"


def test_training_requires_positive_validation_data(training, tmp_path):
    source = make_source(tmp_path / "source", ["point"], count=1)
    prepared = training.write_dataset([source], tmp_path / "dataset", 0.2, 17)
    with pytest.raises(ValueError, match="at least two positives"):
        training.train(prepared, tmp_path / "runs", training.parse_args([]))


def test_auto_resource_defaults_and_resolution_scaling(training, monkeypatch):
    args = training.parse_args([])
    assert args.batch == "auto"
    assert args.workers == "auto"
    assert training.resolve_num_workers(3) == 3
    assert training.resolve_batch_size(2, "cuda:0", 288) == 2

    class FakeCuda:
        def is_available(self):
            return True

        def current_device(self):
            return 0

        def get_device_properties(self, index):
            assert index == 0
            return types.SimpleNamespace(total_memory=24 * 2**30)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert training.resolve_batch_size("auto", "cuda:0", 288) == 4
    assert training.resolve_batch_size("auto", "cuda:0", 144) == 16


@pytest.mark.parametrize(
    "arguments",
    [
        ["--imgsz", "256"],
        ["--batch", "-1"],
        ["--val-fraction", "1"],
        ["--name", "../outside"],
        ["--lr", "nan"],
        ["--device", "0,1"],
    ],
)
def test_invalid_cli_arguments(training, arguments):
    with pytest.raises(SystemExit):
        training.parse_args(arguments)
