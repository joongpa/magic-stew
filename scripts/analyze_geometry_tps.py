"""Offline geometry-only TPS experiments on saved successful reconstructions.

Predictions read axis, residual angles, existing layer support and geometry flags.
Selected paths are evaluation targets only, never estimator inputs. No model,
projection, fitting, search or production TPS configuration is changed.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {"0.25s": 0.25, "0.5s": 0.5, "1s": 1.0, "2s": 2.0}
THRESHOLDS = (0, 5, 6, 10)


def relative_angles(fit):
    angles = fit["angles"]
    # Distance to the closest aligned quarter-turn, independent of middle gauge.
    return tuple((angles[i] - angles[1] + 45) % 90 - 45 for i in (0, 2))


def active_layers(fit, threshold):
    return sum(abs(a) > threshold for a in relative_angles(fit))


def travel(delta):
    return max(0., *delta) - min(0., *delta)


def geometry_features(frames, durations):
    """Amounts per processed interval, not rates; no selected-path access."""
    features = {f"primary_gt{t}": [] for t in THRESHOLDS}
    features.update({name: [] for name in [
        "primary_layers_gt6", "top3_gt6", "top3_layers_gt6", "primary_gt6_edge",
        "primary_gt6_episodes", "primary_gt6_unknown54",
        "primary_wrapped_travel", "primary_raw_travel", "primary_raw_travel_gt6",
        "primary_snapped6_travel", "primary_commit6_events", "primary_commit6_layers",
    ]})
    previous_active = False
    previous = None
    previous_axis = None
    for frame, dt in zip(frames, durations):
        bank = frame["bank"]
        primary = bank[0] if bank else None
        for threshold in THRESHOLDS:
            features[f"primary_gt{threshold}"].append(float(primary is not None and active_layers(primary, threshold) > 0))
        count = active_layers(primary, 6) if primary else 0
        active = count > 0
        # Match search's existing usable-articulated-fit gate; exclude its reserved flat slot.
        usable = [fit for fit in bank if not fit["flat_fallback"] and (
            not any(fit["angles"]) or fit.get("layer_inliers", 0) > 0)][:3]
        counts = [active_layers(fit, 6) for fit in usable]
        features["primary_layers_gt6"].append(float(count))
        features["top3_gt6"].append(float(any(counts)))
        features["top3_layers_gt6"].append(float(max(counts, default=0)))
        features["primary_gt6_edge"].append(float(active or previous_active))
        features["primary_gt6_episodes"].append(float(active and not previous_active))
        features["primary_gt6_unknown54"].append(54 * dt if not bank or frame["geometry_flagged"] else float(active))
        current = relative_angles(primary) if primary else (0., 0.)
        axis = primary["axis"] if primary else None
        old = previous or (0., 0.)
        same_axis = axis == previous_axis
        def difference(a, b, wrap=False):
            if not same_axis:
                return travel(tuple(-v for v in a)) + travel(b)
            delta = tuple(y - x for x, y in zip(a, b))
            if wrap:
                delta = tuple((v + 45) % 90 - 45 for v in delta)
            return travel(delta)
        features["primary_wrapped_travel"].append(difference(old, current, True) / 90.)
        features["primary_raw_travel"].append(difference(old, current) / 90.)
        clean_old = tuple(v if abs(v) > 6 else 0. for v in old)
        clean_new = tuple(v if abs(v) > 6 else 0. for v in current)
        features["primary_raw_travel_gt6"].append(difference(clean_old, clean_new) / 90.)
        snapped_old = tuple(math.copysign(45., v) if abs(v) > 6 else 0. for v in old)
        snapped_new = tuple(math.copysign(45., v) if abs(v) > 6 else 0. for v in current)
        features["primary_snapped6_travel"].append(difference(snapped_old, snapped_new) / 90.)
        commits = [abs(b) > 6 and (not same_axis or abs(a) <= 6 or
                   (a * b < 0 and abs(b - a) < 45)) for a, b in zip(old, current)]
        features["primary_commit6_events"].append(float(any(commits)))
        features["primary_commit6_layers"].append(float(sum(commits)))
        previous, previous_axis = current, axis
        previous_active = active
    return features


class Integral:
    """Piecewise-constant rates, including fractional window-boundary intervals."""
    def __init__(self, boundaries, amounts):
        assert len(boundaries) == len(amounts) + 1
        self.t = boundaries
        self.amounts = amounts
        self.prefix = [0.0]
        for amount in amounts:
            self.prefix.append(self.prefix[-1] + amount)

    def at(self, timestamp):
        if timestamp <= self.t[0]:
            return 0.0
        if timestamp >= self.t[-1]:
            return self.prefix[-1]
        i = bisect.bisect_right(self.t, timestamp) - 1
        return self.prefix[i] + self.amounts[i] * (timestamp - self.t[i]) / (self.t[i + 1] - self.t[i])

    def rate(self, start, end):
        return (self.at(end) - self.at(start)) / (end - start)


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = math.floor(position)
    return values[lo] + (values[math.ceil(position)] - values[lo]) * (position - lo)


def distribution(values):
    return {"mean": statistics.mean(values), "p50": quantile(values, .5),
            "p95": quantile(values, .95), "max": max(values)} if values else None


def pearson(x, y):
    if len(x) < 3:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    denominator = sum((v - mx) ** 2 for v in x) * sum((v - my) ** 2 for v in y)
    return sum((v - mx) * (w - my) for v, w in zip(x, y)) / math.sqrt(denominator) if denominator > 1e-12 else None


def verify_solution(data, tables):
    """Independent permutation application, followed by uniform/distinct faces."""
    colors = [tables["solved_colors"][i] for i in tables["identity"]]
    for raw in data["config"]["scramble"].replace("’", "'").split():
        token = raw[:-1] if raw.endswith("2'") else raw
        colors = [colors[i] for i in tables["moves"][token]]
    labels = {label: colors[i] for i, label in enumerate(tables["identity"])}
    state = list(tables["identity"])
    for step in data["result"]["steps"]:
        for token in step["state_moves"]:
            state = [state[i] for i in tables["moves"][token]]
    faces = [[labels[state[i]] for i in group] for group in tables["face_groups"]]
    assert all(len(set(face)) == 1 for face in faces) and len({face[0] for face in faces}) == 6, data["name"]


def token_quarters(moves):
    # A secondary diagnostic only: equivalent wide/slice spellings can change this count.
    return sum(2 if "2" in token else 1 for token in moves if token[0] not in "xyz")


def analyze_case(data, tables):
    verify_solution(data, tables)
    fps = data["config"]["fps"]
    steps = data["result"]["steps"]
    by_frame = {frame["frame"]: frame for frame in data["frames"]}
    hold = data["config"].get("turning_start_frame")
    start = max(steps[0]["frame_index"] - 1, hold if hold is not None else -math.inf) / fps
    selected = [step for step in steps if step["frame_index"] / fps > start]
    frames = [by_frame[step["frame_index"]] for step in selected]
    boundaries = [start] + [frame["frame"] / fps for frame in frames]
    durations = [b - a for a, b in zip(boundaries, boundaries[1:])]
    assert all(dt > 0 for dt in durations)
    amounts = geometry_features(frames, durations)
    # Do not compress invisible/held intervals into the next single frame.
    reference = [0.0] * len(frames)
    prior_active = start
    for step, frame in zip(selected, frames):
        if not frame["bank"]:
            continue
        end = step["frame_index"] / fps
        quarters = step["angular_travel_degrees"] / 90.0
        assert end > prior_active
        for j, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
            overlap = max(0., min(right, end) - max(left, prior_active))
            reference[j] += quarters * overlap / (end - prior_active)
        prior_active = end
    amounts["reference_angular"] = reference
    amounts["reference_tokens"] = [token_quarters(step["state_moves"]) for step in selected]
    integrals = {name: Integral(boundaries, values) for name, values in amounts.items()}
    rows = []
    for i, end in enumerate(boundaries[1:]):
        windows = [(name, max(start, end - length), end - start + 1e-9 >= length) for name, length in WINDOWS.items()]
        windows += [("8frames", boundaries[max(0, i - 7)], i >= 7), ("prefix", start, True)]
        for window, left, full in windows:
            row = {"case": data["name"], "family": data["family"], "frame": frames[i]["frame"],
                   "end_seconds": end, "start_seconds": left, "duration_seconds": end - left,
                   "window": window, "full_window": full,
                   "geometry_flagged": frames[i]["geometry_flagged"]}
            row.update({name: integral.rate(left, end) for name, integral in integrals.items()})
            rows.append(row)
    whole = {name: integral.rate(start, boundaries[-1]) for name, integral in integrals.items()}
    summary = {"case": data["name"], "family": data["family"], "source": data["source"],
               "frames": len(frames), "duration_seconds": boundaries[-1] - start,
               "configured_opening_hold_excluded": hold, "goal_verified": True,
               "whole_solve": whole, "windows": {}}
    for window in [*WINDOWS, "8frames"]:
        sample = [row for row in rows if row["window"] == window and row["full_window"]]
        target = [row for row in sample if row["reference_angular"] >= 1.]
        summary["windows"][window] = {"samples": len(sample), "active_samples": len(target),
            "reference_angular": distribution([r["reference_angular"] for r in sample]),
            "reference_tokens": distribution([r["reference_tokens"] for r in sample]), "estimators": {}}
        for feature in amounts:
            if feature.startswith("reference_"):
                continue
            summary["windows"][window]["estimators"][feature] = {
                **(distribution([r[feature] for r in sample]) or {}),
                "pearson_r_zero_lag": pearson([r[feature] for r in sample], [r["reference_angular"] for r in sample]),
                "mean_absolute_error_tps": statistics.mean(abs(r[feature] - r["reference_angular"]) for r in sample) if sample else None,
                "underestimate_fraction_active": statistics.mean(r[feature] + 1e-9 < r["reference_angular"] for r in target) if target else None,
                "p05_estimate_to_reference": quantile([r[feature] / r["reference_angular"] for r in target], .05),
            }
    # Feature extraction alone: no JSON parsing, reference calculations or logging.
    begin = time.perf_counter()
    for _ in range(100):
        geometry_features(frames, durations)
    summary["python_feature_microseconds_per_frame"] = (time.perf_counter() - begin) * 1e6 / (100 * len(frames))
    # A synthetic diagnostic using the real pose sequence: hold each observed
    # pose for 2x/4x as many processed frames, at the original sampling cadence.
    # This is not an additional real slow-solver capture.
    summary["synthetic_slowdown"] = {}
    for factor in (2, 4):
        slow_frames = [frame for frame in frames for _ in range(factor)]
        slow_durations = [dt for dt in durations for _ in range(factor)]
        slow_features = geometry_features(slow_frames, slow_durations)
        duration = sum(slow_durations)
        summary["synthetic_slowdown"][str(factor)] = {
            "reference_angular": whole["reference_angular"] / factor,
            **{name: sum(values) / duration for name, values in slow_features.items()},
        }
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    tables = json.loads((ROOT / "pose-fitter/tables/move-tables.json").read_text())
    report = {"scope": __doc__, "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "reference": "Selected successful-path angular travel / 90 / elapsed source time; inferred endpoint travel, not hand-labeled physical ground truth. Token TPS is secondary and spelling-dependent.",
              "window_semantics": "Trailing source-time windows with fractional interval overlap; full-window aggregates exclude warmup. 8frames uses actual elapsed time; prefix is causal since the recorded opening hold/end of admission. Whole-solve excludes configured opening hold only.",
              "assumption": "primary_gt6 counts one assumed committed quarter turn EACH processed frame with a strict >6-degree misalignment to the nearest quarter-turn alignment; episodes counts only entry into the condition.",
              "cases": []}
    all_rows = []
    for case in manifest["cases"]:
        path = ROOT / case["input"]
        summary, rows = analyze_case(json.loads(path.read_text()), tables)
        summary["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["cases"].append(summary)
        all_rows.extend(rows)
        whole = summary["whole_solve"]
        print(case["name"],f"{summary['duration_seconds']:.2f}s", "reference/gt0/gt6/layers6/top3:", *(round(whole[k],2) for k in ["reference_angular", "primary_gt0", "primary_gt6", "primary_layers_gt6", "top3_gt6"]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)


if __name__ == "__main__":
    main()
