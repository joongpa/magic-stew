from __future__ import annotations

import ast
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/train_rfdetr_keypoints_colab.ipynb"


def notebook_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return notebook, {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}


def test_notebook_code_is_valid_and_bundles_current_trainer():
    notebook, cells = notebook_cells()
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len(cells) == len(notebook["cells"])
    for item in notebook["cells"]:
        if item["cell_type"] == "code":
            ast.parse(cells[item["id"]])
            assert item["outputs"] == []
            assert item["execution_count"] is None
    assignment = ast.parse(cells["rfdetr-09"]).body[0]
    assert ast.literal_eval(assignment.value) == (
        ROOT / "scripts/train_rfdetr_keypoints_local.py"
    ).read_text(encoding="utf-8")


def test_zips_prepare_train_and_download_without_external_services(tmp_path, monkeypatch):
    _, cells = notebook_cells()
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
    dataset_zip = tmp_path / "generated.zip"
    with zipfile.ZipFile(dataset_zip, "w") as archive:
        for dataset in ("generated_a", "generated_b"):
            archive.writestr(
                f"{dataset}/sticker_keypoint_manifest.json",
                json.dumps({"class_names": names, "keypoint_shape": [1, 3]}),
            )
            for index in range(2):
                archive.writestr(f"{dataset}/images/{index}.png", b"test image")
                archive.writestr(f"{dataset}/pose/{index}.txt", "6 0.5 0.5 0.2 0.2 0.5 0.5 2\n")
        archive.writestr("generated_a/images/missing_label.png", b"unfinished image")
    negative_zip = tmp_path / "negatives.zip"
    with zipfile.ZipFile(negative_zip, "w") as archive:
        archive.writestr("one/background.jpg", b"negative image one")
        archive.writestr("two/background.jpg", b"negative image two")

    captured = {}

    class FakeRFDETR:
        def __init__(self, **kwargs):
            captured["model"] = kwargs

        def train(self, **kwargs):
            captured["train"] = kwargs
            output = Path(kwargs["output_dir"])
            output.mkdir(parents=True)
            (output / "checkpoint_best_total.pth").write_bytes(b"test checkpoint")

    fake_rfdetr = types.ModuleType("rfdetr")
    fake_rfdetr.RFDETRKeypointPreview = FakeRFDETR
    monkeypatch.setitem(sys.modules, "rfdetr", fake_rfdetr)

    downloaded = []
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(download=downloaded.append)
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)

    ns = {"__name__": "__main__"}
    exec(cells["rfdetr-02"], ns)
    ns.update({"DATASET_ZIP": str(dataset_zip), "NEGATIVE_ZIP": str(negative_zip), "EPOCHS": 2})
    # Colab uses /content; keep this local smoke test within its temporary directory.
    input_cell = cells["rfdetr-07"].replace("dir='/content'", f"dir={str(tmp_path)!r}")
    exec(input_cell, ns)
    exec(cells["rfdetr-09"], ns)
    exec(cells["rfdetr-11"], ns)
    exec(cells["rfdetr-13"], ns)
    exec(cells["rfdetr-15"], ns)

    assert captured["model"]["num_classes"] == 9
    assert captured["model"]["num_keypoints_per_class"] == [1] * 9
    assert captured["train"]["class_names"] == names
    assert captured["train"]["progress_bar"] == "tqdm"
    assert captured["train"]["epochs"] == 2
    manifest = json.loads((ns["PREPARED_DIR"] / "conversion_manifest.json").read_text())
    assert len(ns["SOURCE_DIRS"]) == 2
    assert len(manifest) == 6
    assert sum(item["negative"] for item in manifest) == 2
    assert all(Path(item["label"]).read_text() == "" for item in manifest if item["negative"])
    assert downloaded == [str(ns["BEST_CHECKPOINT"])]

    unsafe_zip = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe_zip, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe ZIP entry"):
        ns["extract_zip"](unsafe_zip, tmp_path / "extract")
    assert not (tmp_path / "escape.txt").exists()


def test_multiple_datasets_reject_incompatible_class_orders(tmp_path):
    _, cells = notebook_cells()
    dataset_zip = tmp_path / "datasets.zip"
    with zipfile.ZipFile(dataset_zip, "w") as archive:
        for dataset, names in (("a", ["corner", "red"]), ("b", ["red", "corner"])):
            archive.writestr(
                f"{dataset}/sticker_keypoint_manifest.json",
                json.dumps({"class_names": names, "keypoint_shape": [1, 3]}),
            )
            archive.writestr(f"{dataset}/images/0.png", b"image")
            archive.writestr(f"{dataset}/pose/0.txt", "0 0.5 0.5 0.2 0.2 0.5 0.5 2\n")
    ns = {"__name__": "__main__"}
    exec(cells["rfdetr-02"], ns)
    ns["DATASET_ZIP"] = str(dataset_zip)
    exec(cells["rfdetr-07"].replace("dir='/content'", f"dir={str(tmp_path)!r}"), ns)
    exec(cells["rfdetr-09"], ns)
    with pytest.raises(ValueError, match="Class order differs"):
        exec(cells["rfdetr-11"], ns)
