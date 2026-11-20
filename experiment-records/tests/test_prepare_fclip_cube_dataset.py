import json
from pathlib import Path

import cv2
import numpy as np

from scripts.prepare_fclip_cube_dataset import prepare_dataset


def test_prepare_dataset_writes_empty_targets_for_negative_roi(tmp_path):
    source_dir = tmp_path / "mcp_dataset_v4"
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels" / "wireframe"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    positive = np.zeros((16, 16, 3), dtype=np.uint8)
    assert cv2.imwrite(str(images_dir / "000000.png"), positive)
    (labels_dir / "000000.txt").write_text(
        "0 0.1 0.1 0.9 0.9\n",
        encoding="utf-8",
    )

    negative_dir = tmp_path / "negative_ROI_backgrounds"
    negative_dir.mkdir()
    assert cv2.imwrite(str(negative_dir / "background.png"), positive)

    output_dir = tmp_path / "fclip"
    prepare_dataset(source_dir, negative_dir, output_dir, val_fraction=0.0, seed=17)

    manifest = json.loads((output_dir / "conversion_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 2
    negative = next(item for item in manifest if item["negative"])
    with np.load(negative["line"]) as line_targets:
        assert float(line_targets["lcmap"].sum()) == 0.0
    with np.load(negative["label"]) as label_targets:
        assert label_targets["lpos"].shape == (0, 2, 3)

