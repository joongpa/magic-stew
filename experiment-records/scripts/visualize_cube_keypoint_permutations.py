"""Write a visual HTML report for the 56-keypoint cube orientation permutations.

The report uses the same keypoint names, x/y/z rotation operations, and
orientation permutation builder as train_yolo11n_piece_seg_local.py.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_yolo11n_piece_seg_local.py"
DEFAULT_OUTPUT = ROOT / "runs" / "cube_56_keypoint_pose_local" / "permutation_debug.html"
DEFAULT_ORIENTATIONS = ("no-op", "x", "y", "z")


def load_training_module():
    spec = importlib.util.spec_from_file_location("cube_pose_train_debug", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training script: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def orientation_label(operations: tuple[str, ...]) -> str:
    if not operations:
        return "no-op"
    parts = []
    index = 0
    while index < len(operations):
        axis = operations[index]
        count = 1
        index += 1
        while index < len(operations) and operations[index] == axis:
            count += 1
            index += 1
        parts.append(axis if count == 1 else f"{axis}{count}")
    return " ".join(parts)


FACE_LAYOUT = {
    "U": (1, 0),
    "L": (0, 1),
    "F": (1, 1),
    "R": (2, 1),
    "B": (3, 1),
    "D": (1, 2),
}


def face_color(face: str) -> str:
    return {
        "U": "#e2e8f0",
        "D": "#facc15",
        "F": "#22c55e",
        "B": "#3b82f6",
        "R": "#ef4444",
        "L": "#f97316",
    }[face]


def point_face_uv(point: tuple[float, float, float], face: str) -> tuple[float, float] | None:
    x, y, z = point
    if face == "U" and y == 1.5:
        return x, z
    if face == "D" and y == -1.5:
        return x, -z
    if face == "F" and z == -1.5:
        return x, y
    if face == "B" and z == 1.5:
        return -x, y
    if face == "R" and x == 1.5:
        return z, y
    if face == "L" and x == -1.5:
        return -z, y
    return None


def svg_net(
    *,
    output_coords: list[tuple[float, float, float]],
    source_names_by_output: list[str],
    output_names: tuple[str, ...],
    face_size: int = 150,
) -> str:
    padding = 18
    label_band = 18
    cols = 4
    rows = 3
    size_x = padding * 2 + cols * face_size
    size_y = padding * 2 + label_band + rows * face_size
    point_scale = face_size / 4.0
    pieces = [
        f'<svg viewBox="0 0 {size_x} {size_y}" class="net" role="img" aria-label="Cube net permutation view">',
    ]
    for face, (col, row) in FACE_LAYOUT.items():
        x0 = padding + col * face_size
        y0 = padding + label_band + row * face_size
        face_fill = face_color(face)
        pieces.extend(
            [
                f'<rect class="face" x="{x0}" y="{y0}" width="{face_size}" height="{face_size}" fill="{face_fill}" />',
                f'<text class="face-label" x="{x0 + face_size / 2}" y="{y0 + 14}" text-anchor="middle">{face}</text>',
            ]
        )
        for grid_index in range(1, 4):
            grid_x = x0 + grid_index * face_size / 4.0
            grid_y = y0 + grid_index * face_size / 4.0
            pieces.append(
                f'<line class="grid" x1="{grid_x:.2f}" y1="{y0}" x2="{grid_x:.2f}" y2="{y0 + face_size}" />'
            )
            pieces.append(
                f'<line class="grid" x1="{x0}" y1="{grid_y:.2f}" x2="{x0 + face_size}" y2="{grid_y:.2f}" />'
            )

    for output_index, point in enumerate(output_coords):
        source_name = source_names_by_output[output_index]
        output_name = output_names[output_index]
        changed = source_name != output_name
        for face, (col, row) in FACE_LAYOUT.items():
            uv = point_face_uv(point, face)
            if uv is None:
                continue
            u, v = uv
            x0 = padding + col * face_size
            y0 = padding + label_band + row * face_size
            cx = x0 + face_size / 2.0 + u * point_scale
            cy = y0 + face_size / 2.0 - v * point_scale
            title_text = html.escape(
                f"face {face}: output {output_index:02d} {output_name} <= source {source_name} at {point}"
            )
            pieces.append(
                f'<g class="net-pt{" changed" if changed else ""}">'
                f"<title>{title_text}</title>"
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="7" />'
                f'<text class="net-point-label" x="{cx:.2f}" y="{cy + 2.8:.2f}" text-anchor="middle">{html.escape(source_name)}</text>'
                "</g>"
            )
    pieces.append("</svg>")
    return "\n".join(pieces)


def mapping_table(source_names_by_output: list[str], output_names: tuple[str, ...]) -> str:
    rows = []
    for index, (output_name, source_name) in enumerate(zip(output_names, source_names_by_output)):
        changed = source_name != output_name
        rows.append(
            '<tr class="{}"><td>{:02d}</td><td>{}</td><td>{}</td></tr>'.format(
                "changed" if changed else "",
                index,
                html.escape(output_name),
                html.escape(source_name),
            )
        )
    return (
        '<table><thead><tr><th>#</th><th>Output slot</th><th>Source after rotation</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def build_report() -> str:
    training = load_training_module()
    names = training.CUBE_KEYPOINT_NAMES
    coords = [training.cube_keypoint_xyz(name) for name in names]
    permutations = training.cube_orientation_permutations()
    operations_list = training.CUBE_ORIENTATION_OPERATIONS

    cards = []
    for orientation_index, (operations, permutation) in enumerate(zip(operations_list, permutations), start=1):
        label = orientation_label(operations)
        if label not in DEFAULT_ORIENTATIONS:
            continue
        source_names_by_output = [names[source_index] for source_index in permutation]
        changed_count = sum(source != output for source, output in zip(source_names_by_output, names))
        cards.append(
            f"""
            <section class="card">
              <h2>{orientation_index}. {html.escape(label)}</h2>
              <p class="meta">{changed_count} of {len(names)} output slots receive a different source label.</p>
              {svg_net(output_coords=coords, source_names_by_output=source_names_by_output, output_names=names)}
              <details>
                <summary>Permutation table</summary>
                {mapping_table(source_names_by_output, names)}
              </details>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Cube 56-Keypoint Orientation Permutations</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      background: #101418;
      color: #edf2f7;
    }}
    h1 {{ margin-bottom: 4px; }}
    .note {{ color: #a8b3c2; max-width: 960px; line-height: 1.45; }}
    .controls {{ margin: 18px 0; }}
    .card {{
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 16px;
      margin: 18px 0;
      background: #161d26;
    }}
    .meta {{ color: #94a3b8; }}
    .net {{ width: min(100%, 690px); height: auto; display: block; margin-top: 12px; background: #0b1016; border-radius: 6px; }}
    .face {{ fill-opacity: 0.24; stroke: #e2e8f0; stroke-width: 1.2; }}
    .face-label {{ fill: #e2e8f0; font-size: 13px; font-weight: 700; paint-order: stroke; stroke: #020617; stroke-width: 3px; }}
    .grid {{ stroke: #64748b; stroke-width: 0.7; stroke-opacity: 0.7; }}
    .net-pt circle {{ fill: #0f172a; stroke: #e2e8f0; stroke-width: 1.1; }}
    .net-pt.changed circle {{ fill: #172554; stroke: #f8fafc; stroke-width: 1.8; }}
    .net-point-label {{
      fill: #f8fafc;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 7.1px;
      font-weight: 700;
      paint-order: stroke;
      stroke: #020617;
      stroke-width: 2px;
    }}
    body.hide-labels .net-point-label {{ display: none; }}
    table {{ border-collapse: collapse; margin-top: 12px; font-size: 13px; }}
    th, td {{ border: 1px solid #334155; padding: 4px 8px; }}
    tr.changed td {{ background: #172554; }}
    summary {{ cursor: pointer; color: #bfdbfe; margin-top: 12px; }}
    code {{ background: #0b1016; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Cube 56-Keypoint Orientation Permutations</h1>
  <p class="note">
    This report shows only <code>no-op</code>, <code>x</code>, <code>y</code>, and <code>z</code>.
    Each card uses the exact keypoint list and rotation/permutation functions from
    <code>scripts/train_yolo11n_piece_seg_local.py</code>. The net is unfolded as
    U above F, L-F-R-B across the middle, and D below F. Text inside each face position is
    the source keypoint matched to that fixed output slot for the orientation.
  </p>
  <p class="note">
    White-outlined dots are slots whose source label changed from no-op. Hover a dot for
    face, output index, output slot, source label, and 3D coordinate.
  </p>
  <div class="controls">
    <label><input id="toggle-labels" type="checkbox" checked /> Show labels</label>
  </div>
  {''.join(cards)}
  <script>
    const toggle = document.getElementById('toggle-labels');
    toggle.addEventListener('change', () => {{
      document.body.classList.toggle('hide-labels', !toggle.checked);
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
