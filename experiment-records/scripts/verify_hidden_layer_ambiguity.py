#!/usr/bin/env python3
"""Expose hidden-articulation ambiguity in the current single-state loss.

The synthetic camera crop is completely covered by an opaque F face. There
is no silhouette/side/back evidence outside that crop. All 16 F-face landmark
positions remain identical as a hidden B layer changes, or as a Bw turn is
encoded by a rotated core frame and compensating F turn. This does NOT infer
hidden layers from absent labels in the historical datasets.

    .venv/bin/python scripts/verify_hidden_layer_ambiguity.py

A successful diagnostic reports ambiguity_handled_by_current_loss=false:
it demonstrates a known supervision limitation, not a passing training gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from live_recognition.direct_cube_pose import (  # noqa: E402
    FACE_AXES, FACE_COLORS, FACE_ORDER, SymmetryPoseLoss, articulated_landmarks, project_points,
)
from live_recognition.direct_cube_pose_data import landmark_template, target_tensors  # noqa: E402
from scripts.export_direct_cube_pose_dataset import symmetry_manifest  # noqa: E402
from scripts.prepare_direct_cube_pose_targets import rotation  # noqa: E402


def make_state(back_angle=20., wide=False):
    # Camera right=+X, down=-Z, forward=+Y: F (-Y) faces the camera.
    rotation_matrix = np.asarray(((1., 0, 0), (0, 0, -1.), (0, 1., 0)))
    angles = torch.zeros(1, 6)
    if wide:
        rotation_matrix = rotation_matrix @ rotation(FACE_AXES[3], -back_angle)
        angles[0, 2] = back_angle  # R_core * F(+angle) leaves the F layer fixed.
    else:
        angles[0, 3] = back_angle
    state = {"rotation_matrix": torch.tensor(rotation_matrix, dtype=torch.float32)[None],
             "center": torch.full((1, 2), .5), "scale": torch.tensor([.3]),
             "perspective": torch.tensor([.2]), "principal": torch.full((1, 2), .5),
             "angles": angles, "quality_logit": torch.zeros(1), "quality": torch.full((1,), .5)}
    centers = torch.tensor(FACE_AXES, dtype=torch.float32)[None] * 1.5
    state["projected_centers"] = project_points(centers, state)
    state["center_keypoints"] = state["projected_centers"].clone()
    return state


def front_only_target(state, image_size=160):
    rest, owners = landmark_template()
    points = articulated_landmarks(torch.tensor(rest), torch.tensor(owners), state["angles"])
    pixels = project_points(points, state)[0] * image_size
    names = [f"anchor_{index}" for index in range(len(rest))]
    visible = np.flatnonzero(rest[:, 1] == -1.5)
    manifest = symmetry_manifest()
    allowed = [candidate["id"] for candidate in manifest["candidates"]
               if candidate["reference_face_to_candidate_face"]["F"] == "F"]
    target = {"format": "direct_cube_centers_projection_v3", "image_size": [image_size] * 2,
              "pose_supervision": {"enabled": True}, "pose_anchor_order": list(FACE_ORDER),
              "pose_anchor_color_order": list(FACE_COLORS),
              "rotation_matrix": state["rotation_matrix"][0].tolist(),
              "projection": {"center_xy_normalized": [.5, .5], "image_scale_normalized": .3,
                             "perspective_strength": .2, "principal_point_normalized": [.5, .5]},
              "layer_angles_degrees": state["angles"][0].tolist(),
              "virtual_core_anchors": [{"name": face, "pixel_point": pixel.tolist()}
                                       for face, pixel in zip(FACE_ORDER, state["projected_centers"][0] * image_size)],
              "landmark_anchors": [{"name": name, "pixel_point": point.tolist()} for name, point in zip(names, pixels)],
              "matched_observations": [{"anchor": names[index], "visibility": 2,
                                        "observed_pixel": pixels[index].tolist()} for index in visible],
              "symmetry": {"allowed_ids": allowed}}
    return target, visible


def diagnose():
    torch.set_num_threads(2)
    reference = make_state()
    target, visible = front_only_target(reference)
    tensors = {key: value[None] for key, value in target_tensors(target).items()}
    criterion = SymmetryPoseLoss(symmetry_manifest(), *landmark_template())
    cases = []
    for name, state in (("B +20 (reference)", reference), ("B -20 (hidden angle changes)", make_state(-20)),
                        ("B 0 (hidden angle changes)", make_state(0)), ("Bw +20 (joint frame and F compensation)", make_state(20, wide=True))):
        losses, metrics, selected = criterion(state, tensors, image_size=160)
        cases.append({"name": name, "angles": dict(zip(FACE_ORDER, state["angles"][0].tolist())),
                      "rotation_matrix": state["rotation_matrix"][0].tolist(),
                      "matched_front_landmarks": len(visible),
                      "observed_front_rmse_px": float(metrics["observed_landmark_rmse_px"]),
                      "full_target_landmark_rmse_px": float(metrics["landmark_rmse_px"]),
                      "geometry_loss_excluding_quality": float(sum(criterion.weights[key] * losses[key]
                                                                   for key in ("rotation", "projection", "angles", "centers", "geometry"))),
                      "current_quality_target": float(metrics["quality_target"]),
                      "selected_cube_symmetry": selected.item()})
    equivalent_observations = all(case["observed_front_rmse_px"] < 1e-3 for case in cases)
    alternatives_penalized = all(case["geometry_loss_excluding_quality"] > .01 for case in cases[1:])
    return {"diagnostic_passed": equivalent_observations and alternatives_penalized,
            "scope": "synthetic crop filled by opaque front face; outside silhouette and hidden surfaces are not observations",
            "ambiguity_handled_by_current_loss": not alternatives_penalized,
            "required_training_gate": "Do not treat successful single-target overfitting as verification of articulation ambiguity support",
            "cases": cases}


def render_diagnostic(report, output):
    # The identical observation is deliberately separate from the differing
    # hidden state descriptions. No existing dataset image/label is masked.
    tile = 240
    canvas = np.full((tile + 110, tile * len(report["cases"]), 3), 24, np.uint8)
    colors = [(40, 180, 40), (20, 140, 240), (245, 245, 245), (20, 220, 245), (40, 180, 40),
              (240, 80, 40), (20, 140, 240), (40, 180, 40), (30, 40, 220)]
    for index, case in enumerate(report["cases"]):
        face = np.full((tile, tile, 3), 35, np.uint8)
        for cell, color in enumerate(colors):
            x, y = cell % 3 * 80, cell // 3 * 80
            cv2.rectangle(face, (x + 3, y + 3), (x + 76, y + 76), color, -1)
        canvas[:tile, index * tile:(index + 1) * tile] = face
        for row, value in enumerate((case["name"].split(" (")[0], "Same F-face observations",
                                     f"front RMSE {case['observed_front_rmse_px']:.6f}px",
                                     f"current loss {case['geometry_loss_excluding_quality']:.3f}")):
            cv2.putText(canvas, value, (index * tile + 7, tile + 22 + row * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .43, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hidden_layer_ambiguity_verification"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = diagnose()
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    render_diagnostic(report, args.output_dir / "identical_front_observations.png")
    print(json.dumps(report, indent=2))
    if not report["diagnostic_passed"]:
        raise SystemExit("Diagnostic did not reproduce the expected ambiguity; inspect report")


if __name__ == "__main__":
    main()
