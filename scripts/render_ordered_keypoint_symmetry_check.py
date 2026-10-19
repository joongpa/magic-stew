#!/usr/bin/env python3
"""Render a static check of the ordered-keypoint symmetry manifest.

The manifest is produced by ``apply_blender_cube_moves.py`` and uses the same
convention as the ordered-pose loss:

    permutation[output_index] = source_index

In particular, the canonical top and front center outputs are keypoints 20
and 22.  This script intentionally does not rebuild cube rotations itself;
it reads the generated permutations and derives the displayed colors from
those entries directly.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


EXPECTED_NAMES = [
    "UBL", "UBR", "UFR", "UFL",
    "DBL", "DBR", "DFR", "DFL",
    "UB", "UR", "UF", "UL",
    "DB", "DR", "DF", "DL",
    "BR", "BL", "FR", "FL",
    "U", "D", "F", "B", "R", "L",
]
FACE_ORDER = ["U", "D", "F", "B", "R", "L"]
FACE_INFO = {
    "U": {"color": "#f5f5f5", "text": "#202020", "name": "White"},
    "D": {"color": "#f0dc00", "text": "#202020", "name": "Yellow"},
    "F": {"color": "#00a651", "text": "#ffffff", "name": "Green"},
    "B": {"color": "#1671c8", "text": "#ffffff", "name": "Blue"},
    "R": {"color": "#d92920", "text": "#ffffff", "name": "Red"},
    "L": {"color": "#f07d18", "text": "#202020", "name": "Orange"},
}


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "ordered_cube_keypoint_symmetry_v1":
        raise ValueError(f"unsupported manifest format: {manifest.get('format')!r}")
    if manifest.get("permutation_convention") != "permutation[output_index] = source_index":
        raise ValueError("manifest does not use the expected permutation convention")
    if manifest.get("keypoint_count") != 26:
        raise ValueError(f"expected 26 keypoints, got {manifest.get('keypoint_count')!r}")
    if manifest.get("keypoint_names") != EXPECTED_NAMES:
        raise ValueError("manifest keypoint names do not match the ordered-pose convention")

    permutations = manifest.get("permutations")
    if not isinstance(permutations, list) or len(permutations) != 24:
        raise ValueError(f"expected 24 permutations, got {len(permutations or [])}")
    expected_indices = set(range(26))
    if any(set(permutation) != expected_indices or len(permutation) != 26 for permutation in permutations):
        raise ValueError("every symmetry permutation must be a bijection over 26 keypoints")
    if len({tuple(permutation) for permutation in permutations}) != 24:
        raise ValueError("symmetry manifest contains duplicate permutations")
    if list(range(26)) not in permutations:
        raise ValueError("symmetry manifest is missing the identity permutation")
    return manifest


def build_orientations(manifest: dict) -> list[dict]:
    names = manifest["keypoint_names"]
    orientations = []
    for index, permutation in enumerate(manifest["permutations"]):
        # This is deliberately the same gather convention used by the loss:
        # candidate_targets = gt_kpt[:, permutations, :].
        source_top = names[permutation[20]]
        source_front = names[permutation[22]]
        if source_top not in FACE_INFO or source_front not in FACE_INFO:
            raise ValueError(
                f"permutation {index} maps canonical centers to "
                f"{source_top!r}/{source_front!r}, not face centers"
            )
        orientations.append({
            "index": index,
            "source_top": source_top,
            "source_front": source_front,
            "top": FACE_INFO[source_top],
            "front": FACE_INFO[source_front],
        })
    return orientations


def render_html(manifest_path: Path, manifest: dict) -> str:
    orientations = build_orientations(manifest)
    grouped = sorted(
        orientations,
        key=lambda item: (FACE_ORDER.index(item["source_top"]), item["index"]),
    )
    pair_counts = {}
    for orientation in orientations:
        pair = (orientation["source_top"], orientation["source_front"])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    duplicate_pairs = {pair for pair, count in pair_counts.items() if count > 1}

    tile_width = 148
    tile_height = 138
    gap_x = 10
    gap_y = 10
    left = 36
    top = 18
    columns = 4
    rows = (len(grouped) + columns - 1) // columns
    view_height = 48 + rows * (tile_height + gap_y)

    def esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    svg = [
        f'<svg id="ordered-symmetry-svg" viewBox="0 0 660 {view_height}" '
        'preserveAspectRatio="xMidYMin meet">',
        '<title>All 24 proper cube rotations from the Blender permutation manifest</title>',
        '<desc>Each tile shows the standard color on the canonical top and front center after applying one generated permutation.</desc>',
    ]
    for layout_index, orientation in enumerate(grouped):
        column = layout_index % columns
        row = layout_index // columns
        x = left + column * (tile_width + gap_x)
        y = top + row * (tile_height + gap_y)
        if column == 0:
            up_face = orientation["source_top"]
            svg.append(
                f'<text x="4" y="{y + 76}" class="group-label">'
                f'up center: {esc(FACE_INFO[up_face]["name"])} ({esc(up_face)})</text>'
            )
        pair = (orientation["source_top"], orientation["source_front"])
        duplicate_class = " duplicate" if pair in duplicate_pairs else ""
        svg.append(
            f'<rect x="{x}" y="{y}" width="{tile_width}" height="{tile_height}" '
            f'rx="6" class="tile{duplicate_class}"/>'
        )
        svg.append(
            f'<text x="{x + 10}" y="{y + 18}" class="tile-index">'
            f'symmetry {orientation["index"]}</text>'
        )

        center_x = x + tile_width / 2
        face_top = y + 32
        face_width = 60
        face_height = 48
        slant = 12
        top_points = (
            f'{center_x - face_width / 2},{face_top + slant} '
            f'{center_x},{face_top} '
            f'{center_x + face_width / 2},{face_top + slant} '
            f'{center_x},{face_top + face_height * 0.55}'
        )
        front_points = (
            f'{center_x - face_width / 2},{face_top + slant} '
            f'{center_x},{face_top + face_height * 0.55} '
            f'{center_x},{face_top + face_height} '
            f'{center_x - face_width / 2},{face_top + face_height - slant}'
        )
        svg.append(
            f'<polygon points="{top_points}" fill="{orientation["top"]["color"]}" '
            'stroke="var(--line)" stroke-width="1"/>'
        )
        svg.append(
            f'<polygon points="{front_points}" fill="{orientation["front"]["color"]}" '
            'stroke="var(--line)" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{x + 10}" y="{y + 104}" class="color-label">'
            f'top: {esc(orientation["source_top"])} · '
            f'{esc(orientation["top"]["name"])}</text>'
        )
        svg.append(
            f'<text x="{x + 10}" y="{y + 118}" class="color-label">'
            f'front: {esc(orientation["source_front"])} · '
            f'{esc(orientation["front"]["name"])}</text>'
        )
        svg.append(
            f'<text x="{x + 10}" y="{y + 131}" class="mapping">'
            f'perm[{20}]→{esc(orientation["source_top"])}  '
            f'perm[{22}]→{esc(orientation["source_front"])}</text>'
        )
    svg.append("</svg>")

    source_text = esc(manifest_path.as_posix())
    status = f"{len(orientations)} rotations · {len(pair_counts)} unique top/front pairs · source: {source_text}"
    return f'''<div id="ordered-symmetry-color-check" role="img" aria-label="Twenty-four cube symmetry orientations read from the Blender-generated ordered keypoint permutation manifest.">
  <style>
    #ordered-symmetry-color-check {{
      --surface: var(--background);
      --ink: var(--foreground);
      --muted-ink: var(--muted-foreground);
      --line: var(--border);
      color: var(--ink);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 980px;
      margin: 0 auto;
    }}
    #ordered-symmetry-color-check .heading {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
      margin-bottom: 12px;
    }}
    #ordered-symmetry-color-check h2 {{
      font-size: 18px;
      font-weight: 500;
      margin: 0;
    }}
    #ordered-symmetry-color-check .status {{
      color: var(--muted-ink);
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    #ordered-symmetry-color-check svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    #ordered-symmetry-color-check .tile {{
      fill: color-mix(in srgb, var(--surface) 94%, var(--ink));
      stroke: var(--line);
      stroke-width: 1;
    }}
    #ordered-symmetry-color-check .tile.duplicate {{
      stroke: var(--red);
      stroke-width: 2;
    }}
    #ordered-symmetry-color-check .tile-index,
    #ordered-symmetry-color-check .color-label,
    #ordered-symmetry-color-check .mapping {{
      fill: var(--ink);
      font-size: 11px;
    }}
    #ordered-symmetry-color-check .tile-index {{ font-weight: 500; }}
    #ordered-symmetry-color-check .color-label {{ fill: var(--muted-ink); font-size: 10px; }}
    #ordered-symmetry-color-check .mapping {{ fill: var(--muted-ink); font-size: 9px; }}
    #ordered-symmetry-color-check .group-label {{ fill: var(--muted-ink); font-size: 10px; font-weight: 500; }}
  </style>
  <div class="heading">
    <h2>Ordered cube symmetries · top/front center colors</h2>
    <div class="status" aria-live="polite">{status}</div>
  </div>
  {''.join(svg)}
</div>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/synthetic_cube_dataset/ordered_keypoint_symmetries.json"),
        help="Blender-generated ordered_keypoint_symmetries.json",
    )
    parser.add_argument("--output", type=Path, required=True, help="HTML fragment output path")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(args.manifest, manifest))
    print(f"wrote {args.output} from {args.manifest}: 24 permutations")


if __name__ == "__main__":
    main()
