"""
build_simulator.py
Builds the interactive "ABS Print Lab" simulator (simulator/ABS_Print_Lab.html).

1. Re-fits the best model (selected by run_project.py) on all 383 samples.
2. Exports its decision trees, scalers, the dataset, the optimiser settings, the
   precomputed design-space cloud / Pareto front and the recommended recipes to JSON.
3. Inlines that JSON and model_runtime.js into simulator_template.html, producing a
   single self-contained HTML file that runs the exact model in the browser (offline).
4. Writes test vectors (Python predictions) that test_runtime.js uses to verify that the
   JavaScript model reproduces the Python model exactly.

Run from the project root, after run_project.py:  python simulator/build_simulator.py
Then (optional check):                              node simulator/test_runtime.js
"""

import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.append(os.path.join(ROOT, 'src'))

from sklearn.multioutput import MultiOutputRegressor
from data_loader import load_raw_data, get_param_bounds, get_param_levels, FEATURE_COLUMNS, TARGET_COLUMNS
from baseline_models import get_baseline_models
from advanced_models import get_advanced_models
from model_utils import unwrap_model
from optimizer import (worst_case_predict, estimate_relative_print_time, composite_strength,
                       TOLERANCE, SHELL_FRACTION)

TAB = os.path.join(ROOT, 'outputs', 'tables')


def export_hgb_target(hgb):
    """Flatten all trees of one HistGradientBoostingRegressor into parallel arrays."""
    feat, thr, left, right, val, roots = [], [], [], [], [], []
    for iteration in hgb._predictors:
        nodes = iteration[0].nodes
        offset = len(feat)
        roots.append(offset)
        for n in nodes:
            leaf = bool(n['is_leaf'])
            feat.append(-1 if leaf else int(n['feature_idx']))
            thr.append(0.0 if leaf else float(n['num_threshold']))
            left.append(0 if leaf else offset + int(n['left']))
            right.append(0 if leaf else offset + int(n['right']))
            val.append(float(n['value']) if leaf else 0.0)
    return {'base': float(np.ravel(hgb._baseline_prediction)[0]), 'roots': roots, 'feat': feat,
            'thr': [float(f'{t:.12g}') for t in thr], 'left': left, 'right': right,
            'val': [float(f'{v:.12g}') for v in val]}


def main():
    summary = json.load(open(os.path.join(TAB, 'summary_key_results.json'), encoding='utf-8'))
    best = summary['best_model']
    if best != 'Hist_Gradient_Boosting':
        raise SystemExit(f'Simulator export supports Hist_Gradient_Boosting; best model is {best}')

    df = load_raw_data()
    X, y = df[FEATURE_COLUMNS].values, df[TARGET_COLUMNS].values
    model = {**get_baseline_models(), **get_advanced_models()}[best]
    model.fit(X, y)
    xs, est, ys = unwrap_model(model)
    assert isinstance(est, MultiOutputRegressor)

    bounds = get_param_bounds(df)
    levels = get_param_levels(df)

    # Design-space cloud (worst-case, same definition as the Pareto front in the report)
    rng = np.random.default_rng(7)
    n = 2500
    cloud_x = np.column_stack([
        rng.uniform(*bounds['Nozzle_Temp_C'], n).round(0),
        rng.uniform(*bounds['Bed_Temp_C'], n).round(0),
        rng.uniform(*bounds['Print_Speed_mm_per_s'], n).round(0),
        rng.choice(levels['Layer_Height_mm'], n),
        rng.uniform(*bounds['Infill_Density_percent'], n).round(0),
    ])
    wc = worst_case_predict(model, cloud_x)
    cloud = [[*map(float, cloud_x[i]), round(float(composite_strength(*wc[i])), 3),
              round(float(estimate_relative_print_time(cloud_x[i, 2], cloud_x[i, 3], cloud_x[i, 4])), 2)]
             for i in range(n)]

    pareto = pd.read_csv(os.path.join(TAB, 'pareto_front_solutions.csv'))
    optimal = pd.read_csv(os.path.join(TAB, 'optimal_parameters.csv'))
    cases = pd.read_csv(os.path.join(TAB, 'automotive_case_study_recommendations.csv'))
    ranking = pd.read_csv(os.path.join(TAB, 'model_ranking.csv'), index_col=0)

    comp = optimal[optimal['Objective'] == 'Composite'].iloc[0]
    rec = lambda r: [float(r[c]) for c in FEATURE_COLUMNS]
    pedal = cases[cases['Component'].str.contains('Brake')].iloc[0]
    handle = cases[cases['Component'].str.contains('Door')].iloc[0]
    weakest = df.loc[df['Tensile_Strength_Mpa'].idxmin()]

    data = {
        'modelName': summary['best_model_pretty'].split(' (')[0],
        'cvR2': float(ranking.loc[best, 'CV_R2']),
        'cvRmse': [summary['best_cv'][t]['RMSE_Mean'] for t in TARGET_COLUMNS],
        'nSamples': int(len(df)),
        'features': FEATURE_COLUMNS,
        'bounds': [list(bounds[c]) for c in FEATURE_COLUMNS],
        'levels': [levels[c] for c in FEATURE_COLUMNS],
        'layerLevels': levels['Layer_Height_mm'],
        'tolerance': [TOLERANCE[c] for c in FEATURE_COLUMNS],
        'shellFraction': SHELL_FRACTION,
        'safetyMargins': summary['safety_margins'],
        'xScale': xs.scale_.tolist(), 'xMin': xs.min_.tolist(),
        'yScale': ys.scale_.tolist(), 'yMin': ys.min_.tolist(),
        'targets': [export_hgb_target(e) for e in est.estimators_],
        'background': X.tolist(),
        'measured': y.tolist(),
        'cloud': cloud,
        'pareto': pareto[FEATURE_COLUMNS + ['WorstCase_Composite_MPa', 'Print_Time_Index']].values.tolist(),
        'presets': {
            'optimum': {'label': 'Max strength', 'x': rec(comp)},
            'pedal': {'label': 'Brake pedal', 'x': rec(pedal)},
            'handle': {'label': 'Door handle', 'x': rec(handle)},
            'weak': {'label': 'Weakest specimen', 'x': [float(weakest[c]) for c in FEATURE_COLUMNS]},
        },
        'maxStrengthTau': float(comp['Print_Time_Index']),
        'cases': {
            'pedal': {'requirement': 45.0, 'target': 1, 'margin': float(pedal['Safety_Margin_MPa'])},
            'handle': {'requirement': 30.0, 'target': 0, 'margin': float(handle['Safety_Margin_MPa'])},
        },
    }

    payload = json.dumps(data, separators=(',', ':'))
    runtime = open(os.path.join(HERE, 'model_runtime.js'), encoding='utf-8').read()
    template = open(os.path.join(HERE, 'simulator_template.html'), encoding='utf-8').read()
    body = (template.replace('/*__MODEL_DATA__*/null', payload)
                    .replace('/*__MODEL_RUNTIME__*/', runtime))
    assert '__MODEL_DATA__' not in body and '__MODEL_RUNTIME__' not in body

    # Content-only version (for publishing as an artifact) and a standalone offline file
    with open(os.path.join(HERE, 'ABS_Print_Lab_content.html'), 'w', encoding='utf-8') as f:
        f.write(body)
    standalone = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                  '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                  '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
    with open(os.path.join(HERE, 'ABS_Print_Lab.html'), 'w', encoding='utf-8') as f:
        f.write(standalone)

    # Test vectors for the JavaScript runtime
    rng = np.random.default_rng(1)
    tx = np.column_stack([rng.uniform(*bounds[c], 400) for c in FEATURE_COLUMNS])
    tx[:, 3] = rng.choice(levels['Layer_Height_mm'], 400)
    tx = np.vstack([tx, X[:100]])
    tests = {'x': tx.tolist(), 'pred': model.predict(tx).tolist(),
             'wc_x': tx[:40].tolist(), 'wc': worst_case_predict(model, tx[:40]).tolist()}
    with open(os.path.join(HERE, 'test_vectors.json'), 'w', encoding='utf-8') as f:
        json.dump(tests, f)

    print(f'Saved: {os.path.join(HERE, "ABS_Print_Lab.html")} ({len(standalone) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
