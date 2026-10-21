#!/usr/bin/env python3
"""Build an inline viewer for real ordered-keypoint v2 metadata samples."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from verify_ordered_keypoint_layer_metadata import validate_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/ordered_pose_dataset_v2"))
    parser.add_argument("--metadata", type=Path, help="Override source-dir/metadata.json")
    parser.add_argument("--output", type=Path, required=True, help="Output HTML fragment path")
    parser.add_argument("--samples-per-face", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    return parser.parse_args()


def choose_records(records: list[dict], samples_per_face: int) -> list[dict]:
    mid_turns = [
        record for record in records
        if (record.get("layer_state") or {}).get("mode") == "mid_turn"
    ]
    by_face: dict[str, list[dict]] = {}
    for record in mid_turns:
        face = str((record.get("layer_state") or {}).get("face"))
        by_face.setdefault(face, []).append(record)

    chosen: list[dict] = []
    for face in ("U", "D", "F", "B", "R", "L"):
        face_records = sorted(
            by_face.get(face, []),
            key=lambda record: float((record.get("layer_state") or {}).get("angle_degrees", 0.0)),
        )
        if not face_records:
            continue
        count = min(samples_per_face, len(face_records))
        indices = [round(i * (len(face_records) - 1) / max(1, count - 1)) for i in range(count)]
        chosen.extend(face_records[index] for index in indices)
    return chosen


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def compact_record(record: dict, source_dir: Path, tolerance: float) -> dict:
    result = validate_record(record, tolerance)
    image_path = source_dir / "images" / Path(str(record["image"])).name
    ordered = record["ordered_keypoints"]
    keypoints = []
    for keypoint in ordered["keypoints"]:
        keypoints.append({
            "index": keypoint["index"],
            "name": keypoint["name"],
            "kind": keypoint["kind"],
            "local": keypoint.get("local_point"),
            "pixel": keypoint.get("pixel_point"),
            "normalized": keypoint.get("normalized_point"),
            "visibility": keypoint.get("visibility", 0),
            "hand_occluded": keypoint.get("hand_occluded", False),
            "occluded_by": keypoint.get("occluded_by"),
        })
    return {
        "image": str(record["image"]),
        "image_uri": image_data_uri(image_path),
        "image_size": record.get("image_size", [256, 256]),
        "layer_state": record.get("layer_state") or {"mode": "none"},
        "keypoints": keypoints,
        "validation": {
            "ok": result["ok"],
            "max_geometry_error": result["max_geometry_error"],
            "geometry_errors": result["geometry_errors"],
            "expected_moved": result["expected_moved"],
            "expected_displaced": result["expected_displaced"],
        },
    }


def render_html(samples: list[dict]) -> str:
    data = json.dumps(samples, separators=(",", ":"), ensure_ascii=False)
    return f'''<div id="ordered-keypoint-metadata-viewer">
  <style>
    #ordered-keypoint-metadata-viewer {{
      color: var(--foreground);
      font-family: var(--font-family, ui-sans-serif, system-ui, sans-serif);
      width: 100%;
    }}
    #ordered-keypoint-metadata-viewer .controls {{
      display: flex; flex-wrap: wrap; gap: 8px; align-items: end; margin-bottom: 10px;
    }}
    #ordered-keypoint-metadata-viewer label {{ display: grid; gap: 3px; font-size: 12px; }}
    #ordered-keypoint-metadata-viewer select,
    #ordered-keypoint-metadata-viewer button {{ font: inherit; color: inherit; }}
    #ordered-keypoint-metadata-viewer select {{ min-width: 260px; }}
    #ordered-keypoint-metadata-viewer .main {{
      display: grid; grid-template-columns: minmax(300px, 1.35fr) minmax(280px, 1fr); gap: 14px;
      align-items: start;
    }}
    #ordered-keypoint-metadata-viewer .image-wrap {{ position: relative; min-width: 0; }}
    #ordered-keypoint-metadata-viewer canvas {{ width: 100%; height: auto; display: block; }}
    #ordered-keypoint-metadata-viewer .summary {{ display: grid; gap: 8px; }}
    #ordered-keypoint-metadata-viewer .summary-line {{ font-size: 13px; line-height: 1.35; }}
    #ordered-keypoint-metadata-viewer .summary-line code {{ font-size: 12px; }}
    #ordered-keypoint-metadata-viewer .keypoint-table {{
      width: 100%; border-collapse: collapse; font-size: 11px; font-variant-numeric: tabular-nums;
    }}
    #ordered-keypoint-metadata-viewer .keypoint-table th,
    #ordered-keypoint-metadata-viewer .keypoint-table td {{ padding: 3px 4px; text-align: left; border-bottom: 1px solid var(--border); }}
    #ordered-keypoint-metadata-viewer .keypoint-table th {{ color: var(--muted-foreground); font-weight: 500; }}
    #ordered-keypoint-metadata-viewer .moved {{ color: var(--orange); font-weight: 500; }}
    #ordered-keypoint-metadata-viewer .pass {{ color: var(--green); }}
    #ordered-keypoint-metadata-viewer .fail {{ color: var(--red); }}
    #ordered-keypoint-metadata-viewer .legend {{ display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px; }}
    #ordered-keypoint-metadata-viewer .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: -1px; }}
    @media (max-width: 700px) {{
      #ordered-keypoint-metadata-viewer .main {{ grid-template-columns: 1fr; }}
      #ordered-keypoint-metadata-viewer select {{ min-width: min(100%, 260px); }}
    }}
  </style>
  <div class="controls">
    <label>Real mid-turn sample
      <select id="sample-select" aria-label="Select a real mid-turn metadata sample"></select>
    </label>
    <button type="button" id="previous-sample">Previous</button>
    <button type="button" id="next-sample">Next</button>
    <label><input id="show-all" type="checkbox" checked> show all labeled points</label>
  </div>
  <div class="legend" aria-label="Point legend">
    <span><span class="swatch" style="background:var(--orange)"></span>turned layer</span>
    <span><span class="swatch" style="background:var(--blue)"></span>other point</span>
    <span>solid = visible</span><span>hollow = occluded</span>
  </div>
  <div class="main">
    <div class="image-wrap">
      <canvas id="sample-canvas" width="256" height="256" role="img" aria-label="Real dataset image with ordered keypoints"></canvas>
    </div>
    <div class="summary">
      <div id="sample-summary" class="summary-line" aria-live="polite"></div>
      <table class="keypoint-table">
        <thead><tr><th>idx</th><th>name</th><th>layer</th><th>vis</th><th>local error</th></tr></thead>
        <tbody id="keypoint-rows"></tbody>
      </table>
    </div>
  </div>
  <script>
    (() => {{
      const samples = {data};
      const root = document.getElementById('ordered-keypoint-metadata-viewer');
      const select = root.querySelector('#sample-select');
      const canvas = root.querySelector('#sample-canvas');
      const context = canvas.getContext('2d');
      const summary = root.querySelector('#sample-summary');
      const rows = root.querySelector('#keypoint-rows');
      const showAll = root.querySelector('#show-all');
      let current = 0;
      let image = null;

      const movedColor = 'rgb(238 151 60)';
      const otherColor = 'rgb(77 156 228)';
      const visibilityName = {{0: 'unlabeled', 1: 'occluded', 2: 'visible'}};

      function isMoved(sample, keypoint) {{
        return sample.validation.expected_moved.includes(keypoint.name);
      }}

      function draw() {{
        const sample = samples[current];
        const [width, height] = sample.image_size;
        canvas.width = width; canvas.height = height;
        context.clearRect(0, 0, width, height);
        context.drawImage(image, 0, 0, width, height);
        const scale = Math.max(1, width / 256);
        const visiblePoints = sample.keypoints.filter(point => point.visibility !== 0);
        for (const point of visiblePoints) {{
          if (!showAll.checked && !isMoved(sample, point)) continue;
          const x = point.pixel[0]; const y = point.pixel[1];
          const moved = isMoved(sample, point);
          const color = moved ? movedColor : otherColor;
          const radius = (moved ? 5 : 4) * scale;
          context.beginPath(); context.arc(x, y, radius, 0, Math.PI * 2);
          context.lineWidth = 1.5 * scale;
          context.strokeStyle = color;
          context.fillStyle = color;
          if (point.visibility === 2) context.fill(); else context.stroke();
          context.fillStyle = color;
          context.font = `${{Math.max(9, 10 * scale)}}px ui-monospace, monospace`;
          context.fillText(`${{point.index}} ${{point.name}}`, x + radius + 1, y - radius - 1);
        }}
        const state = sample.layer_state;
        const validation = sample.validation;
        const status = validation.ok ? '<span class="pass">PASS</span>' : '<span class="fail">FAIL</span>';
        summary.innerHTML = `${{status}} · <code>${{sample.image}}</code><br>` +
          `mode <b>${{state.mode}}</b> · turned face <b>${{state.face}}</b> · ` +
          `angle <b>${{Number(state.angle_degrees).toFixed(2)}}°</b> · direction <b>${{state.direction}}</b><br>` +
          `max local-point derivation error <b>${{validation.max_geometry_error.toExponential(2)}}</b><br>` +
          `turned-layer points: <span class="moved">${{validation.expected_moved.join(', ')}}</span>`;
        rows.innerHTML = sample.keypoints.map(point => {{
          const moved = isMoved(sample, point);
          const errorEntry = validation.geometry_errors.find(error => error.name === point.name);
          const error = errorEntry ? errorEntry.error : 0;
          return `<tr class="${{moved ? 'moved' : ''}}"><td>${{point.index}}</td><td>${{point.name}}</td>` +
            `<td>${{moved ? 'yes' : 'no'}}</td><td>${{visibilityName[point.visibility]}}</td>` +
            `<td>${{error.toExponential(2)}}</td></tr>`;
        }}).join('');
      }}

      function load(index) {{
        current = (index + samples.length) % samples.length;
        const sample = samples[current];
        select.value = String(current);
        image = new Image();
        image.onload = draw;
        image.src = sample.image_uri;
      }}

      samples.forEach((sample, index) => {{
        const option = document.createElement('option');
        option.value = String(index);
        const state = sample.layer_state;
        option.textContent = `${{String(index + 1).padStart(2, '0')}} · ${{sample.image.split('/').pop()}} · ${{state.face}} ${{Number(state.angle_degrees).toFixed(1)}}°`;
        select.appendChild(option);
      }});
      select.addEventListener('change', event => load(Number(event.target.value)));
      root.querySelector('#previous-sample').addEventListener('click', () => load(current - 1));
      root.querySelector('#next-sample').addEventListener('click', () => load(current + 1));
      showAll.addEventListener('change', draw);
      load(0);
    }})();
  </script>
</div>
'''


def main() -> int:
    args = parse_args()
    metadata_path = args.metadata or args.source_dir / "metadata.json"
    with metadata_path.open() as handle:
        records = json.load(handle)
    chosen = choose_records(records, max(1, args.samples_per_face))
    samples = [compact_record(record, args.source_dir, args.tolerance) for record in chosen]
    if not samples:
        raise SystemExit("No mid_turn records were found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(samples), encoding="utf-8")
    print(f"Wrote {len(samples)} real-data samples to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Could not build viewer: {exc}", file=sys.stderr)
        raise SystemExit(2)
