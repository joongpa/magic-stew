"""Curve-focused, grouped cross-validation and plots for geometry TPS experiments."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_geometry_tps as core

FEATURES = ['primary_gt6', 'primary_wrapped_travel', 'primary_snapped6_travel', 'primary_commit6_layers']
DISPLAY = {
    'primary_gt0': 'Any nonzero offset / frame',
    'primary_gt6': '>6 deg = one turn / frame',
    'primary_gt6_edge': '>6 deg in current or previous frame',
    'primary_wrapped_travel': 'Wrapped angle change',
    'primary_snapped6_travel': '>6 deg ternary phase change',
    'phase6_mild': '1.1 x phase-change TPS + 1',
    'phase6_upper': '1.25 x phase-change TPS + 2',
    'primary_commit6_layers': '>6 deg anticipated commit events',
    'top3_gt6': 'Any of top three fits >6 deg',
    'cv_blend': 'Held-solve-out geometry blend',
    'cv_blend_upper90': 'Held-solve-out blend + training p90 margin',
}


def fit_model(rows):
    # Demeaning each solve puts emphasis on curve changes, rather than learning
    # differences in mean TPS between solves. No held family's labels are used.
    design, target = [], []
    families = sorted({r['family'] for r in rows})
    for family in families:
        family_rows = [r for r in rows if r['family'] == family]
        for case in sorted({r['case'] for r in family_rows}):
            for window in ['0.5s', '1s']:
                sample = [r for r in family_rows if r['case'] == case and r['window'] == window and r['full_window']]
                if not sample:
                    continue
                x = np.array([[r[f] for f in FEATURES] for r in sample])
                y = np.array([r['reference_angular'] for r in sample])
                weight = 1 / math.sqrt(len(sample) * len({r['case'] for r in family_rows}))
                design.extend((x - x.mean(axis=0)) * weight)
                target.extend((y - y.mean()) * weight)
    coef, _ = nnls(np.asarray(design), np.asarray(target))
    # Equal family weights for the intercept and upper margin as well.
    residuals = []
    for family in families:
        values = [r['reference_angular'] - sum(r[f] * c for f, c in zip(FEATURES, coef))
                  for r in rows if r['family'] == family and r['window'] in ['0.5s', '1s'] and r['full_window']]
        residuals.append(values)
    intercept = float(statistics.mean(statistics.mean(v) for v in residuals))
    margin = float(max(0., max(core.quantile([v - intercept for v in values], .90) for values in residuals)))
    return {'coefficients': dict(zip(FEATURES, map(float, coef))), 'intercept_tps': intercept,
            'upper90_margin_tps': margin,
            'margin_definition': 'Maximum within-training-family p90 positive residual on full 0.5s/1s windows; no held-family labels.'}


def predict(row, model):
    return max(0., model['intercept_tps'] + sum(row[f] * c for f, c in model['coefficients'].items()))


def metrics(sample, feature):
    x = [r[feature] for r in sample]
    y = [r['reference_angular'] for r in sample]
    active = [r for r in sample if r['reference_angular'] >= 1]
    return {'pearson_r': core.pearson(x, y), 'mae_tps': statistics.mean(abs(a - b) for a, b in zip(x, y)),
            'pearson_r_committed_token_reference': core.pearson(x, [r['reference_tokens'] for r in sample]),
            'bias_tps': statistics.mean(a - b for a, b in zip(x, y)),
            'mean_estimate_tps': statistics.mean(x), 'mean_reference_tps': statistics.mean(y),
            'overestimate_fraction_active': statistics.mean(r[feature] >= r['reference_angular'] for r in active) if active else None}


def lag_metrics(sample, feature):
    times = np.array([r['end_seconds'] for r in sample])
    x = np.array([r[feature] for r in sample])
    y = np.array([r['reference_angular'] for r in sample])
    # Fixed common support across shifts; interpolation is evaluation-only.
    valid = (times >= times[0] + .2) & (times <= times[-1] - .2)
    if valid.sum() < 5:
        return None
    candidates = []
    for shift in np.arange(-.20, .201, .025):
        rho = core.pearson(np.interp(times[valid] + shift, times, x).tolist(), y[valid].tolist())
        if rho is not None:
            candidates.append((rho, float(shift)))
    if not candidates:
        return None
    rho, shift = max(candidates, key=lambda v: (v[0], -abs(v[1])))
    return {'best_r': rho, 'prediction_time_shift_seconds': shift,
            'definition': 'Compare estimate(t + shift) with reference(t) on common support trimmed 0.2s at both ends, diagnostic only; plots and main metrics use zero shift.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    original = json.loads(args.report.read_text())
    rows = list(csv.DictReader(args.report.with_suffix('.csv').open()))
    strings = {'case', 'family', 'window'}
    booleans = {'full_window', 'geometry_flagged'}
    rows = [{k: v if k in strings else v == 'True' if k in booleans else float(v) for k, v in r.items()} for r in rows]
    models = {}
    for family in sorted({r['family'] for r in rows}):
        models[family] = fit_model([r for r in rows if r['family'] != family])
    for row in rows:
        model = models[row['family']]
        row['cv_blend'] = predict(row, model)
        row['cv_blend_upper90'] = row['cv_blend'] + model['upper90_margin_tps']
        row['phase6_upper'] = 1.25 * row['primary_snapped6_travel'] + 2.
        row['phase6_mild'] = 1.1 * row['primary_snapped6_travel'] + 1.
    result = {'scope': 'Geometry-only curve variants; all learned coefficients/intercepts/margins are held-solve-family-out. Crop variants share one family. No production changes.',
              'reference': original['reference'], 'features': FEATURES, 'fold_models': models,
              'all_data_prototype_model': fit_model(rows), 'display_names': DISPLAY, 'cases': [], 'aggregate': {}}
    for case in original['cases']:
        out = {'case': case['case'], 'family': case['family'], 'windows': {}}
        for window in [*core.WINDOWS, '8frames']:
            sample = [r for r in rows if r['case'] == case['case'] and r['window'] == window and r['full_window']]
            out['windows'][window] = {f: {**metrics(sample, f), 'lag_diagnostic': lag_metrics(sample, f)} for f in DISPLAY}
        result['cases'].append(out)
    for window in [*core.WINDOWS, '8frames']:
        result['aggregate'][window] = {}
        for feature in DISPLAY:
            family_metrics = []
            for family in models:
                cases = [c for c in result['cases'] if c['family'] == family]
                corr = [c['windows'][window][feature]['pearson_r'] for c in cases]
                family_metrics.append({'pearson_r': statistics.mean(v for v in corr if v is not None) if any(v is not None for v in corr) else None,
                    'mae_tps': statistics.mean(c['windows'][window][feature]['mae_tps'] for c in cases),
                    'overestimate_fraction_active': statistics.mean(c['windows'][window][feature]['overestimate_fraction_active'] for c in cases)})
            result['aggregate'][window][feature] = {
                'families_total': len(family_metrics),
                'families_with_defined_correlation': sum(v['pearson_r'] is not None for v in family_metrics),
                'median_family_r': statistics.median(v['pearson_r'] for v in family_metrics if v['pearson_r'] is not None),
                'worst_family_r': min(v['pearson_r'] for v in family_metrics if v['pearson_r'] is not None),
                'mean_family_mae_tps': statistics.mean(v['mae_tps'] for v in family_metrics),
                'mean_family_overestimate_fraction_active': statistics.mean(v['overestimate_fraction_active'] for v in family_metrics)}
    output = args.report.with_name('curves.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    with output.with_suffix('.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for window in ['0.25s', '0.5s', '1s', '2s', '8frames']:
        print(window)
        for feature in DISPLAY:
            m = result['aggregate'][window][feature]
            print(feature, 'median r', round(m['median_family_r'], 3), 'worst r', round(m['worst_family_r'], 3), 'MAE', round(m['mean_family_mae_tps'], 2), 'over%', round(m['mean_family_overestimate_fraction_active'] * 100, 1))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for window in ['0.25s', '0.5s', '1s', '2s', '8frames']:
        fig, axes = plt.subplots(5, 2, figsize=(14, 16), layout='constrained')
        for ax, case in zip(axes.flat, original['cases']):
            sample = [r for r in rows if r['case'] == case['case'] and r['window'] == window and r['full_window']]
            start = min(r['start_seconds'] for r in rows if r['case'] == case['case'])
            times = [r['end_seconds'] - start for r in sample]
            for feature, label, color, style in [
                ('reference_angular', 'Reconstructed angular TPS', '#111827', '-'),
                ('primary_gt6', 'One turn per >6° frame', '#d97706', '--'),
                ('primary_snapped6_travel', '>6° phase-change proxy', '#2563eb', '-'),
                ('phase6_upper', '1.25 × phase proxy + 2 TPS', '#059669', '-'),
            ]:
                ax.plot(times, [r[feature] for r in sample], label=label, color=color, linestyle=style, linewidth=1.5, alpha=.85)
            m = next(c for c in result['cases'] if c['case'] == case['case'])['windows'][window]
            ax.set_title(f"{case['case']} · {window} · r: count {m['primary_gt6']['pearson_r']:.2f}, phase {m['primary_snapped6_travel']['pearson_r']:.2f}")
            ax.set_ylim(bottom=0); ax.grid(alpha=.2); ax.set_ylabel('Quarter-turn equivalents / s'); ax.set_xlabel('Seconds since evaluation start')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=2)
        fig.suptitle(f'Geometry-only TPS estimates · trailing {window} windows\nReference is successful reconstructed travel, not hand-labeled physical speed', fontsize=14)
        fig.savefig(args.report.with_name(f'curves-{window}.png'), dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, name in zip(axes.flat, ['nahm', 'charlie', 'matty', 'kyle']):
        sample = [r for r in rows if r['case'] == name and r['window'] == '1s' and r['full_window']]
        start = min(r['start_seconds'] for r in rows if r['case'] == name)
        times = [r['end_seconds'] - start for r in sample]
        for feature, label, color, style in [
            ('reference_angular', 'Successful-path angular TPS', '#111827', '-'),
            ('primary_gt6', 'One turn per >6° frame', '#d97706', '--'),
            ('primary_snapped6_travel', '>6° phase-change proxy', '#2563eb', '-'),
            ('phase6_upper', '1.25 × phase proxy + 2 TPS', '#059669', '-'),
        ]:
            ax.plot(times, [r[feature] for r in sample], label=label, color=color, linestyle=style, linewidth=1.8)
        m = next(c for c in result['cases'] if c['case'] == name)['windows']['1s']
        ax.set_title(f"{name.title()} · correlation: count {m['primary_gt6']['pearson_r']:.2f}, phase {m['primary_snapped6_travel']['pearson_r']:.2f}")
        ax.set_ylim(bottom=0); ax.grid(alpha=.2); ax.set_ylabel('Quarter-turn equivalents / s'); ax.set_xlabel('Seconds since evaluation start')
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='outside lower center', ncol=2)
    fig.suptitle('Geometry-only estimates · trailing 1-second windows\nReference is reconstructed travel; not hand-labeled physical speed', fontsize=13)
    fig.savefig(args.report.with_name('curves-highlight.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
