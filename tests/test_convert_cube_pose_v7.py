import json

import torch

from scripts.convert_cube_pose_v5_v6_to_v7 import convert_payload, convert_directory
from scripts.train_yolo_cube_pose_local import pose_target_from_json


def projection():
    return {
        "model": "scaled_perspective",
        "cube_center_normalized": [0.5, 0.5],
        "image_scale_normalized": 0.1,
        "perspective_strength": 0.2,
    }


def test_v6_conversion_emits_direct_head_fields():
    payload = {
        "format": "cube_pose_v6",
        "cube_rotation_camera": {
            "quaternion_xyzw": [0.087155743, 0.0, 0.0, 0.996194698],
        },
        "image_projection": projection(),
        "articulation": {"layer_angles_degrees": [12.0, 0.0]},
    }

    converted = convert_payload(payload, "test.json")

    assert converted["format"] == "cube_pose_v7"
    assert set(converted["cube_pose"]) == {
        "u_colatitude_degrees",
        "v_azimuth_degrees",
        "twist_degrees",
    }
    assert converted["articulation"]["layer_angles_degrees"] == [12.0, 0.0]
    assert "cube_rotation_camera" not in converted

    target = pose_target_from_json(converted)
    assert target["rotation_quaternion"].shape == (4,)


def test_v5_conversion_uses_bounded_colatitude_and_preserves_rotation():
    payload = {
        "format": "cube_pose_v5",
        "cube_rotation_camera": {
            # 30 degrees about camera X: stack points 30 degrees below
            # image zenith, so colatitude is 60 degrees.
            "quaternion_xyzw": [
                0.2588190451,
                0.0,
                0.0,
                0.9659258263,
            ],
        },
        "image_projection": projection(),
        "articulation": {
            "turn": {"face": "", "layer_index": 2, "signed_angle_degrees": 0.0}
        },
    }

    converted = convert_payload(payload, "test.json")

    assert 0.0 < converted["cube_pose"]["u_colatitude_degrees"] < 90.0
    target = pose_target_from_json(converted)
    assert torch.allclose(
        target["rotation_quaternion"].abs(),
        torch.tensor([0.25881905, 0.0, 0.0, 0.9659258]),
        atol=1e-5,
    )


def test_v6_turn_conversion_selects_visible_stack_sign():
    payload = {
        "format": "cube_pose_v6",
        "cube_rotation_camera": {
            # The stack points below image horizon; v6's layer index gives us
            # the axis but not the face, so canonicalization must choose -Z.
            "quaternion_xyzw": [
                0.9659258263,
                0.0,
                0.0,
                -0.2588190451,
            ],
        },
        "image_projection": projection(),
        "articulation": {"layer_angles_degrees": [0.0, 20.0]},
    }

    converted = convert_payload(payload, "test.json")

    assert 0.0 <= converted["cube_pose"]["u_colatitude_degrees"] < 90.0
    assert converted["articulation"]["layer_angles_degrees"] == [20.0, 0.0]


def test_v5_face_turn_remaps_the_active_layer_channel():
    payload = {
        "format": "cube_pose_v5",
        "cube_rotation_camera": {
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "image_projection": projection(),
        "articulation": {
            "turn": {"face": "F", "layer_index": 0, "signed_angle_degrees": 12.0}
        },
    }

    converted = convert_payload(payload, "test.json")

    assert converted["articulation"]["layer_angles_degrees"] == [0.0, 12.0]


def test_v5_no_turn_uses_image_zenith_canonicalization():
    payload = {
        "format": "cube_pose_v5",
        "cube_rotation_camera": {
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "image_projection": projection(),
        "articulation": {"turn": None},
    }

    converted = convert_payload(payload, "test.json")

    assert abs(converted["cube_pose"]["u_colatitude_degrees"]) < 1e-6
    assert converted["articulation"]["layer_angles_degrees"] == [0.0, 0.0]


def test_directory_conversion_writes_new_directory_only(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    source_payload = {
        "format": "cube_pose_v6",
        "cube_rotation_camera": {"quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "image_projection": projection(),
        "articulation": {"layer_angles_degrees": [0.0, -20.0]},
    }
    source_path = source / "000001.json"
    source_path.write_text(json.dumps(source_payload), encoding="utf-8")

    assert convert_directory(source, output) == 1
    assert json.loads(source_path.read_text(encoding="utf-8")) == source_payload
    assert (output / "000001.json").is_file()
    assert (output / "schema.json").is_file()
