from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_three_face_correspondence_dataset.py"


def load_script():
    spec = importlib.util.spec_from_file_location("three_face_correspondence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_three_face_layout_has_37_unique_points():
    module = load_script()

    slots = module.build_point_slots()

    assert len(slots) == 37
    assert len({slot.coordinate for slot in slots}) == 37
    assert slots[0].coordinate == (0, 3, 3)
    assert slots[3].faces == ("U", "R")


def test_partial_turn_is_undone_before_lattice_assignment():
    module = load_script()
    angle = 30.0
    source = np.asarray((1.5, 0.5, 1.5), dtype=float)
    rotation = module._rotation_matrix(module.FACE_AXES["U"], angle)
    turned = rotation @ source
    layer_state = {
        "mode": "mid_turn",
        "face": "U",
        "angle_degrees": angle,
        "direction": 1,
        "objects": ["cubie"],
    }

    restored, error = module.canonicalize_world_point(
        turned,
        np.eye(4),
        "cubie",
        layer_state,
    )

    np.testing.assert_allclose(restored, source, atol=1e-10)
    assert error < 1e-10


def test_metadata_iterator_keeps_complete_records_before_truncated_tail(tmp_path):
    module = load_script()
    path = tmp_path / "metadata.json"
    path.write_text('[{"image": "a"}, {"image": "b"}, {"image": "c"', encoding="utf-8")

    records, diagnostics = module.iter_metadata_records(path)

    assert [record["image"] for record in records] == ["a", "b"]
    assert diagnostics["complete_records"] == 2
    assert diagnostics["truncated_tail"] == 1
