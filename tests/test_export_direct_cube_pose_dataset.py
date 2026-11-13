import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import export_direct_cube_pose_dataset as exporter  # noqa: E402


RECORDS = json.loads((Path(__file__).parent / "fixtures/direct_cube_pose_metadata.json").read_text())


def test_streaming_aggregate_handles_chunk_boundaries_and_truncated_last_record(tmp_path):
    path = tmp_path / "metadata.json"
    records = [{"image": "a.png", "text": 'braces } and an escaped "'}, {"image": "b.png"}]
    path.write_text(json.dumps(records)[:-1] + ', {"image": "incomplete')
    diagnostics = {}
    assert list(exporter.stream_array(path, diagnostics, chunk_size=7)) == records
    assert diagnostics == {"complete_records": 2, "truncated_tail": True}


@pytest.mark.parametrize("text", ['[{"image":"a"} {"image":"b"}]', '[{"image":"a"},]', '[1]'])
def test_malformed_aggregate_is_rejected(tmp_path, text):
    path = tmp_path / "metadata.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        list(exporter.stream_array(path, {}, chunk_size=3))


def test_per_file_metadata_wins_and_aggregate_fills_missing_images(tmp_path):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "a.json").write_text(json.dumps({"image": "a.png", "version": "new"}))
    (tmp_path / "metadata.json").write_text(json.dumps([
        {"image": "a.png", "version": "old"}, {"image": "b.png"},
    ]))
    records = list(exporter.metadata_records(tmp_path, {}))
    assert [(stem, value.get("version")) for stem, value, _ in records] == [("a", "new"), ("b", None)]
    assert records[0][2]["storage"] == "per_file"
    assert records[1][2]["record_index"] == 1


@pytest.mark.parametrize("colors,expected", [([], 24), (["white"], 4), (["white", "green"], 1)])
def test_centers_are_ordered_by_color_and_visibility_one_constrains_symmetries(colors, expected):
    record = copy.deepcopy(RECORDS[0])
    record["sticker_keypoints"]["keypoints"] = [
        {"class": color, "source_cubie": "center", "visibility": 1, "pixel_point": [10, 20]}
        for color in colors
    ]
    original = copy.deepcopy(record)
    target = exporter.geometry.build_target(record)
    target = exporter.add_center_targets(target, record, {"center"}, exporter.symmetry_manifest())
    assert target["pose_anchor_color_order"] == ["white", "yellow", "green", "blue", "red", "orange"]
    assert [item["name"] for item in target["virtual_core_anchors"]] == list(exporter.geometry.FACE_ORDER)
    assert len(target["symmetry"]["allowed_ids"]) == expected
    for point in target["virtual_core_anchors"]:
        assert point["source_label_visibility"] == (1 if point["color"] in colors else None)
        assert point["physical_visibility"] is None
    # All compact symmetry references reconstruct the SAME physical geometry,
    # including slice/corner-cut operations in the separate real-fixture tests.
    manifest = exporter.symmetry_manifest()
    for index in target["symmetry"]["allowed_ids"]:
        matrix = np.asarray(manifest["candidates"][index]["candidate_to_reference"])
        alternative = exporter.symmetry.transform_target(target, matrix)
        metrics = exporter.symmetry.verify_candidate(target, alternative, matrix, target["symmetry"]["retained_center_faces"])
        assert max(metrics.values()) < 1e-9
    assert record == original


def test_unobserved_geometry_is_included_without_claiming_verification():
    record = copy.deepcopy(RECORDS[0])
    record["sticker_keypoints"]["landmark_keypoints"] = []
    target = exporter.geometry.build_target(record, allow_unobserved=True)
    assert exporter.check_verification(target)
    assert target["verification"]["reference_passed"] is None
    assert target["verification"]["status"] == "unavailable_no_saved_landmarks"
    assert target["verification"]["source_camera_error_px"]["max"] is None
    assert len(target["virtual_core_anchors"]) == 14
    assert len(target["landmark_anchors"]) == 56


def test_empty_scene_masks_pose_even_when_saved_transform_is_invalid():
    record = copy.deepcopy(RECORDS[0])
    record["sticker_keypoints"]["landmark_keypoints"] = []
    record["sticker_keypoints"]["keypoints"] = []
    record["cube_pose"] = {}  # No pose should be reconstructed or supervised.
    target = exporter.build_export_target(record, set(), exporter.symmetry_manifest())
    assert not target["pose_supervision"]["enabled"]
    assert target["rotation_matrix"] is None
    assert target["projection"] is None
    assert target["virtual_core_anchors"] == []
    assert target["verification"]["status"] == "pose_supervision_masked"


def test_sticker_only_image_gets_six_center_targets_without_fabricated_verification():
    record = copy.deepcopy(RECORDS[0])
    record["sticker_keypoints"]["landmark_keypoints"] = []
    record["sticker_keypoints"]["keypoints"] = [
        {"class": "white", "source_cubie": "center", "visibility": 1, "pixel_point": [10, 20]},
    ]
    target = exporter.build_export_target(record, {"center"}, exporter.symmetry_manifest())
    assert target["pose_supervision"]["enabled"]
    assert len(target["virtual_core_anchors"]) == 6
    assert len(target["symmetry"]["allowed_ids"]) == 4
    assert target["verification"]["reference_passed"] is None


def test_bulk_export_omits_only_image_without_metadata_and_keeps_original_labels(tmp_path):
    root = tmp_path / "part1"
    for directory in ("images", "pose"):
        (root / directory).mkdir(parents=True)
    record = copy.deepcopy(RECORDS[0])
    record["image"] = "images/000000.png"
    # Six distinct center-owner tags are sufficient for the inventory pass;
    # retain the fixture's exact world/image observations for geometry checks.
    for index, point in enumerate(record["sticker_keypoints"]["landmark_keypoints"][:6]):
        point.update(source_piece_type="center", source_object=f"cubie.{index:03}")
    negative = copy.deepcopy(record)
    negative["image"] = "images/000001.png"
    negative["sticker_keypoints"]["landmark_keypoints"] = []
    negative["sticker_keypoints"]["keypoints"] = []
    negative["cube_pose"] = {}
    (root / "metadata.json").write_text(json.dumps([record, negative]))
    for stem in ("000000", "000001", "000002"):
        cv2.imwrite(str(root / "images" / f"{stem}.png"), np.zeros((144, 144, 3), np.uint8))
        (root / "pose" / f"{stem}.txt").write_text("" if stem == "000001" else "3 0.5 0.5 0.1 0.1 0.5 0.5 2\n")
    original_files = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    output = tmp_path / "export"
    report = exporter.export_dataset([root], output)
    assert report["exported"] == 2
    assert report["sources"][0]["images_without_metadata"] == ["000002"]
    assert report["sources"][0]["pose_supervision_masked"] == 1
    entries = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    entry = entries[0]
    assert (output / entry["image"]).read_bytes() == (root / "images" / "000000.png").read_bytes()
    assert (output / entry["pose"]).read_bytes() == (root / "pose" / "000000.txt").read_bytes()
    target = json.loads((output / entry["target"]).read_text())
    assert len(target["virtual_core_anchors"]) == 6
    assert target["source"]["metadata"]["storage"] == "aggregate"
    assert len(list((output / "images").iterdir())) == 2
    assert (output / entries[1]["pose"]).read_bytes() == b""
    assert not json.loads((output / entries[1]["target"]).read_text())["pose_supervision"]["enabled"]
    assert original_files == {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
