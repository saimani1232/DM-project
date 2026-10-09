"""
optimizer.py
Formal process-parameter optimisation (Review 1 feedback item #3) and automotive
case studies.

Design space
------------
The search is restricted to the window covered by the training data (Table 2 of
Munshi et al.): nozzle 200-250 °C, bed 50-110 °C, speed 10-70 mm/s,
infill 20-100 %, layer height 0.2-0.8 mm. Layer height only takes two levels in the
data (0.2 and 0.8 mm), so it is treated as a categorical factor and optimised by
enumeration; the other four parameters are optimised continuously.

Methods
-------
1. optimize_single_objective   - scipy.optimize.differential_evolution (Storn & Price,
   1997) maximising tensile, compressive or composite strength. The optimisation is
   ROBUST: each recipe is scored by its worst predicted strength within a process
   tolerance band (nozzle ±5 °C, bed ±5 °C, speed ±5 mm/s, infill ±5 %), which keeps
   recommendations away from the step edges of the tree surrogate. Gradient-based
   polishing is disabled because tree ensembles are piecewise constant. The result
   is cross-checked against an exhaustive search over all 960 tested combinations.
2. optimize_constrained        - minimise print-time index subject to a minimum
   worst-case strength (+ model-error safety margin), used for the brake-pedal and
   door-handle case studies.
3. sensitivity_analysis        - one-at-a-time (OAT) local (±10 %) and full-range
   sensitivity around the optimum.
4. evaluate_pareto_front       - strength vs print-time Pareto frontier on a dense grid.

Print-time index (relative, dimensionless)
------------------------------------------
Print time is proportional to deposited volume divided by volumetric flow rate
(flow rate ~ speed x layer height x road width). Deposited volume is modelled as a
fixed shell fraction plus the infilled core:
    tau = 1000 * (phi_s + (1 - phi_s) * infill/100) / (speed * layer_height),
with an assumed shell (perimeter/top/bottom) volume fraction phi_s = 0.30.
Lower tau = faster production.
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from data_loader import FEATURE_COLUMNS, FEATURE_DISPLAY_NAMES, get_param_bounds, get_param_levels

OUTPUT_FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'outputs',
    'figures'
)

OUTPUT_TAB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'outputs',
    'tables'
)

SHELL_FRACTION = 0.30
TIE_BREAK = 1e-6
LAYER_IDX = FEATURE_COLUMNS.index('Layer_Height_mm')
CONTINUOUS_IDX = [i for i in range(len(FEATURE_COLUMNS)) if i != LAYER_IDX]

# Process-parameter window (from the dataset) and tested levels
PARAM_BOUNDS = get_param_bounds()
PARAM_LEVELS = get_param_levels()
LAYER_LEVELS = PARAM_LEVELS['Layer_Height_mm']

# Process tolerance band used for robust optimisation (assumed machine/process scatter).
# A recipe is scored by its WORST predicted strength over this band, which keeps the
# recommendations away from the step edges of the tree-ensemble surrogate (those edges
# lie half-way between tested levels, where the model has no data).
TOLERANCE = {'Nozzle_Temp_C': 5.0, 'Bed_Temp_C': 5.0, 'Print_Speed_mm_per_s': 5.0,
             'Layer_Height_mm': 0.0, 'Infill_Density_percent': 5.0}
_STENCIL = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=len(FEATURE_COLUMNS))))
_STENCIL = np.unique(_STENCIL * np.array([TOLERANCE[c] for c in FEATURE_COLUMNS]), axis=0)
_LOWER = np.array([PARAM_BOUNDS[c][0] for c in FEATURE_COLUMNS])
_UPPER = np.array([PARAM_BOUNDS[c][1] for c in FEATURE_COLUMNS])


def estimate_relative_print_time(speed, layer_height, infill):
    """Relative print-time index tau (lower = faster). See module docstring."""
    speed = np.asarray(speed, dtype=float)
    layer_height = np.asarray(layer_height, dtype=float)
    infill = np.asarray(infill, dtype=float)
    volume = SHELL_FRACTION + (1.0 - SHELL_FRACTION) * infill / 100.0
    return 1000.0 * volume / (speed * layer_height)


def composite_strength(tensile, compressive):
    """Equal-weight composite strength score (MPa) used throughout the optimisation."""
    return 0.5 * np.asarray(tensile) + 0.5 * np.asarray(compressive)


def worst_case_predict(model, X):
    """
    Worst-case (minimum) predicted tensile and compressive strength of each recipe in X
    over the tolerance band (3^4 stencil, clipped to the tested window). Shape (n, 2).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    pts = np.clip(X[:, None, :] + _STENCIL[None, :, :], _LOWER, _UPPER)
    pred = model.predict(pts.reshape(-1, X.shape[1])).reshape(X.shape[0], len(_STENCIL), -1)
    return pred.min(axis=1)


def _score(pred, objective):
    if objective == 'tensile':
        return pred[:, 0]
    if objective == 'compressive':
        return pred[:, 1]
    if objective == 'composite':
        return composite_strength(pred[:, 0], pred[:, 1])
    raise ValueError(f"Unknown objective '{objective}'")


def _rounding_candidates(x):
    """All floor/ceil combinations of the continuous parameters (machine resolution 1 unit)."""
    cands = []
    for combo in itertools.product([np.floor, np.ceil], repeat=len(CONTINUOUS_IDX)):
        c = np.asarray(x, dtype=float).copy()
        for fn, i in zip(combo, CONTINUOUS_IDX):
            lo, hi = PARAM_BOUNDS[FEATURE_COLUMNS[i]]
            c[i] = float(np.clip(fn(c[i]), lo, hi))
        cands.append(c)
    return np.unique(np.array(cands), axis=0)


def _full_x(x_cont, layer):
    """Insert the fixed layer height into continuous-variable vectors (shape (n,4) or (4,))."""
    x_cont = np.atleast_2d(x_cont)
    full = np.empty((x_cont.shape[0], len(FEATURE_COLUMNS)))
    full[:, CONTINUOUS_IDX] = x_cont
    full[:, LAYER_IDX] = layer
    return full


def _run_de(func_of_full_X, seed, maxiter=150, popsize=25):
    """
    Run differential evolution once per layer-height level; returns best (x_full, f, result).
    func_of_full_X maps an (n,5) array to n objective values (to be minimised).
    """
    bounds = [PARAM_BOUNDS[FEATURE_COLUMNS[i]] for i in CONTINUOUS_IDX]
    best = None
    for layer in LAYER_LEVELS:
        def f(xc, layer=layer):
            # vectorized=True -> xc has shape (4, S)
            return func_of_full_X(_full_x(np.asarray(xc).T, layer))
        res = differential_evolution(
            f, bounds, seed=seed, maxiter=maxiter, popsize=popsize,
            init='sobol', mutation=(0.5, 1.0), recombination=0.7,
            # tol = atol = 0: run until the whole population agrees (or maxiter); a
            # piecewise-constant surrogate gives no gradient to polish with
            tol=0, atol=0, polish=False, vectorized=True, updating='deferred'
        )
        x_full = _full_x(res.x, layer)[0]
        if best is None or res.fun < best[1]:
            best = (x_full, float(res.fun), res)
    return best


def experimental_grid():
    """All combinations of the tested levels (6 x 4 x 4 x 2 x 5 = 960 recipes)."""
    return np.array(list(itertools.product(*[PARAM_LEVELS[c] for c in FEATURE_COLUMNS])), dtype=float)


def optimize_single_objective(model, objective='composite', seed=42):
    """
    Robust maximisation of tensile, compressive or composite strength with
    differential evolution: the objective is the WORST-case strength over the
    tolerance band; ties (flat plateaus) are broken in favour of shorter print time.

    Returns a dict with the recommended (machine-rounded) recipe, its nominal and
    worst-case predicted strengths, and the exhaustive-grid cross-check.
    """
    def f(X):
        return (-_score(worst_case_predict(model, X), objective)
                + TIE_BREAK * estimate_relative_print_time(X[:, 2], X[:, 3], X[:, 4]))

    best_x, best_f, res = _run_de(f, seed)

    # Cross-check: exhaustive robust search over the 960 tested level combinations
    grid = experimental_grid()
    grid_f = f(grid)
    g = int(np.argmin(grid_f))

    cands = _rounding_candidates(best_x)
    x_rec = cands[int(np.argmin(f(cands)))]
    if f(x_rec.reshape(1, -1))[0] > grid_f[g] + 1e-9:
        # Never report a recipe worse than the best tested combination
        x_rec = grid[g].copy()

    pred = model.predict(x_rec.reshape(1, -1))[0]
    wc = worst_case_predict(model, x_rec)[0]
    grid_wc = worst_case_predict(model, grid[g])[0]

    out = {'Objective': objective.capitalize(), 'Method': 'Robust differential evolution',
           'DE_Generations': int(res.nit)}
    for i, name in enumerate(FEATURE_COLUMNS):
        out[name] = x_rec[i]
    out['Predicted_Tensile_MPa'] = round(float(pred[0]), 2)
    out['Predicted_Compressive_MPa'] = round(float(pred[1]), 2)
    out['Composite_Score_MPa'] = round(float(composite_strength(pred[0], pred[1])), 2)
    out['WorstCase_Tensile_MPa'] = round(float(wc[0]), 2)
    out['WorstCase_Compressive_MPa'] = round(float(wc[1]), 2)
    out['Objective_Score_WorstCase_MPa'] = round(float(_score(wc.reshape(1, -1), objective)[0]), 2)
    out['Print_Time_Index'] = round(float(estimate_relative_print_time(x_rec[2], x_rec[3], x_rec[4])), 1)
    out['Best_Tested_Combination'] = ', '.join(f'{v:g}' for v in grid[g])
    out['Best_Tested_Combination_Score_MPa'] = round(float(_score(grid_wc.reshape(1, -1), objective)[0]), 2)
    out['Best_Tested_WorstCase_Tensile_MPa'] = round(float(grid_wc[0]), 2)
    out['Best_Tested_WorstCase_Compressive_MPa'] = round(float(grid_wc[1]), 2)
    return out


def optimize_constrained(model, target_idx, min_strength, safety_margin=0.0, seed=42):
    """
    Minimise print-time index subject to WORST-CASE predicted strength (over the tolerance
    band) >= min_strength + safety_margin.  target_idx: 0 = tensile, 1 = compressive.
    """
    required = min_strength + safety_margin

    def penalised(X):
        wc = worst_case_predict(model, X)
        tau = estimate_relative_print_time(X[:, 2], X[:, 3], X[:, 4])
        violation = np.maximum(0.0, required - wc[:, target_idx])
        return tau + 1e4 * violation

    best_x, best_f, res = _run_de(penalised, seed)

    # Machine-rounded recipe: feasible floor/ceil combination with the lowest print time
    candidates = _rounding_candidates(best_x)
    wc = worst_case_predict(model, candidates)
    taus = estimate_relative_print_time(candidates[:, 2], candidates[:, 3], candidates[:, 4])
    ok = wc[:, target_idx] >= required
    x_rec = candidates[int(np.argmin(np.where(ok, taus, np.inf)))] if np.any(ok) else best_x
    pred = model.predict(x_rec.reshape(1, -1))[0]
    wc_rec = worst_case_predict(model, x_rec)[0]
    feasible = bool(wc_rec[target_idx] >= required)
    return x_rec, pred, wc_rec, feasible, res


def run_case_studies(model, safety_margins=(0.0, 0.0), reference_recipe=None, seed=42):
    """
    Automotive case studies (Munshi et al. Section 3.3):
      Brake pedal : compressive strength >= 45 MPa, minimum print time
      Door handle : tensile strength >= 30 MPa, minimum print time
    The requirement must hold for the worst case over the tolerance band, plus a
    model-error safety margin: safety_margins = (tensile, compressive) in MPa.
    reference_recipe: recipe whose print time is the 100 % reference (max-strength recipe).
    """
    cases = [
        ('Automotive Brake Pedal', 'Compressive strength >= 45 MPa, minimum print time', 1, 45.0),
        ('Automotive Door Handle', 'Tensile strength >= 30 MPa, minimum print time', 0, 30.0),
    ]
    if reference_recipe is None:
        reference_recipe = [np.nan, np.nan, PARAM_BOUNDS['Print_Speed_mm_per_s'][0], LAYER_LEVELS[0], 100.0]
    reference_tau = float(estimate_relative_print_time(reference_recipe[2], reference_recipe[3],
                                                       reference_recipe[4]))
    rows = []
    for name, req, t_idx, threshold in cases:
        margin = safety_margins[t_idx]
        x, pred, wc, feasible, res = optimize_constrained(model, t_idx, threshold, margin, seed)
        tau = float(estimate_relative_print_time(x[2], x[3], x[4]))
        rows.append({
            'Component': name,
            'Requirement': req,
            'Safety_Margin_MPa': round(margin, 2),
            'Required_WorstCase_MPa': round(threshold + margin, 2),
            'Nozzle_Temp_C': x[0], 'Bed_Temp_C': x[1], 'Print_Speed_mm_per_s': x[2],
            'Layer_Height_mm': x[3], 'Infill_Density_percent': x[4],
            'Predicted_Tensile_MPa': round(float(pred[0]), 2),
            'Predicted_Compressive_MPa': round(float(pred[1]), 2),
            'WorstCase_Tensile_MPa': round(float(wc[0]), 2),
            'WorstCase_Compressive_MPa': round(float(wc[1]), 2),
            'Print_Time_Index': round(tau, 1),
            'Print_Time_Saving_vs_Max_Strength_Recipe_%': round(100 * (1 - tau / reference_tau), 1),
            'Feasible': feasible
        })
    return pd.DataFrame(rows)


def _predict_single(model, x):
    return model.predict(np.asarray(x, dtype=float).reshape(1, -1))[0]


def sensitivity_analysis(model, base_x, perturbation_pct=10, n_curve=161):
    """
    One-at-a-time sensitivity around base_x (usually the composite optimum):
      * local   : each continuous parameter moved ±perturbation_pct % (clipped to the
                  tested window); layer height is switched to its other tested level
      * global  : each parameter swept over its full tested range
    Returns (df_sensitivity, curves) where curves[param] = (values, tensile, compressive).
    """
    base_x = np.asarray(base_x, dtype=float)
    base_pred = _predict_single(model, base_x)
    rows, curves = [], {}

    for i, (pname, pdisp) in enumerate(zip(FEATURE_COLUMNS, FEATURE_DISPLAY_NAMES)):
        lo_b, hi_b = PARAM_BOUNDS[pname]
        if i == LAYER_IDX:
            other = [l for l in LAYER_LEVELS if l != base_x[i]]
            low_val, high_val = min(LAYER_LEVELS), max(LAYER_LEVELS)
            local_note = f'switch to {other[0]:g} mm' if other else 'single level'
        else:
            delta = abs(base_x[i]) * perturbation_pct / 100.0
            low_val = max(base_x[i] - delta, lo_b)
            high_val = min(base_x[i] + delta, hi_b)
            local_note = f'±{perturbation_pct}% (clipped to tested range)'

        x_low, x_high = base_x.copy(), base_x.copy()
        x_low[i], x_high[i] = low_val, high_val
        p_low, p_high = _predict_single(model, x_low), _predict_single(model, x_high)

        # Full-range sweep
        if i == LAYER_IDX:
            vals = np.array(LAYER_LEVELS, dtype=float)
        else:
            vals = np.linspace(lo_b, hi_b, n_curve)
        X_sweep = np.tile(base_x, (len(vals), 1))
        X_sweep[:, i] = vals
        p_sweep = model.predict(X_sweep)
        curves[pname] = (vals, p_sweep[:, 0], p_sweep[:, 1])
        comp_sweep = composite_strength(p_sweep[:, 0], p_sweep[:, 1])
        near = vals[comp_sweep >= 0.99 * composite_strength(base_pred[0], base_pred[1])]
        window = (f'{near.min():.1f} - {near.max():.1f}' if i != LAYER_IDX
                  else ', '.join(f'{v:g}' for v in near))

        rows.append({
            'Parameter': pdisp,
            'Optimal_Value': round(float(base_x[i]), 2),
            'Local_Perturbation': local_note,
            'Low_Value': round(float(low_val), 2),
            'High_Value': round(float(high_val), 2),
            'Tensile_at_Low': round(float(p_low[0]), 2),
            'Tensile_at_High': round(float(p_high[0]), 2),
            'Local_Tensile_Change_MPa': round(float(abs(p_high[0] - p_low[0])), 3),
            'Compressive_at_Low': round(float(p_low[1]), 2),
            'Compressive_at_High': round(float(p_high[1]), 2),
            'Local_Compressive_Change_MPa': round(float(abs(p_high[1] - p_low[1])), 3),
            'FullRange_Tensile_Swing_MPa': round(float(p_sweep[:, 0].max() - p_sweep[:, 0].min()), 3),
            'FullRange_Compressive_Swing_MPa': round(float(p_sweep[:, 1].max() - p_sweep[:, 1].min()), 3),
            'Tensile_Drop_From_Optimum_Worst_MPa': round(float(base_pred[0] - p_sweep[:, 0].min()), 3),
            'Near_Optimal_Window_(>=99%_composite)': window,
        })

    return pd.DataFrame(rows), curves


def plot_sensitivity_tornado(df_sens, save_dir=OUTPUT_FIG_DIR):
    """Tornado chart of local (±10 %) and full-range strength changes around the optimum."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    for ax, (label, local_col, full_col) in zip(axes, [
        ('Tensile Strength', 'Local_Tensile_Change_MPa', 'FullRange_Tensile_Swing_MPa'),
        ('Compressive Strength', 'Local_Compressive_Change_MPa', 'FullRange_Compressive_Swing_MPa')
    ]):
        d = df_sens.sort_values(full_col, ascending=True)
        y = np.arange(len(d))
        ax.barh(y + 0.2, d[full_col], height=0.38, color='#FF7043', edgecolor='black',
                label='Full tested range sweep')
        ax.barh(y - 0.2, d[local_col], height=0.38, color='#42A5F5', edgecolor='black',
                label='Local ±10 % (layer: other level)')
        ax.set_yticks(y)
        ax.set_yticklabels(d['Parameter'])
        xmax = d[full_col].max()
        for j, (_, r) in enumerate(d.iterrows()):
            ax.text(r[full_col] + xmax * 0.01, j + 0.2, f'{r[full_col]:.2f}', va='center', fontsize=8)
            ax.text(r[local_col] + xmax * 0.01, j - 0.2, f'{r[local_col]:.2f}', va='center', fontsize=8)
        ax.set_xlim(0, xmax * 1.15)
        ax.set_xlabel('Change in predicted strength (MPa)', fontsize=10)
        ax.set_title(f'Sensitivity around the composite optimum — {label}', fontsize=11, fontweight='bold')
        ax.grid(axis='x', linestyle=':', alpha=0.4)
        ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'optimization_sensitivity.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


def plot_sensitivity_curves(curves, base_x, save_dir=OUTPUT_FIG_DIR):
    """One-at-a-time response curves through the optimum for every parameter."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6), sharey=True)
    for i, (ax, pname) in enumerate(zip(axes, FEATURE_COLUMNS)):
        vals, t, c = curves[pname]
        style = 'o-' if len(vals) <= 3 else '-'
        ax.plot(vals, t, style, color='#1f77b4', lw=2, label='Tensile')
        ax.plot(vals, c, style, color='#d62728', lw=2, label='Compressive')
        ax.axvline(base_x[i], color='gray', ls='--', lw=1, label='Optimum')
        ax.set_xlabel(FEATURE_DISPLAY_NAMES[i])
        ax.grid(True, ls=':', alpha=0.5)
        if i == 0:
            ax.set_ylabel('Predicted strength (MPa)')
            ax.legend(fontsize=8)
    fig.suptitle('One-at-a-time response curves through the composite optimum '
                 '(other parameters held at their optimal values)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'sensitivity_response_curves.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


def evaluate_pareto_front(model, df_cases=None, save_dir=OUTPUT_FIG_DIR, tab_dir=OUTPUT_TAB_DIR):
    """
    Bi-objective trade-off: maximise WORST-CASE composite strength (over the tolerance
    band, as in the optimiser), minimise print-time index. Evaluated on a dense grid of
    the tested window (layer height at its tested levels).
    """
    grids = [
        np.linspace(*PARAM_BOUNDS['Nozzle_Temp_C'], 11),
        np.linspace(*PARAM_BOUNDS['Bed_Temp_C'], 7),
        np.linspace(*PARAM_BOUNDS['Print_Speed_mm_per_s'], 13),
        np.array(LAYER_LEVELS, dtype=float),
        np.linspace(*PARAM_BOUNDS['Infill_Density_percent'], 17),
    ]
    X = np.array(list(itertools.product(*grids)))
    pred = worst_case_predict(model, X)
    strength = composite_strength(pred[:, 0], pred[:, 1])
    tau = estimate_relative_print_time(X[:, 2], X[:, 3], X[:, 4])

    # Non-dominated filter: sort by time (then strength desc), keep strictly improving strength
    order = np.lexsort((-strength, tau))
    pareto_idx, best_s = [], -np.inf
    for i in order:
        if strength[i] > best_s + 1e-9:
            pareto_idx.append(i)
            best_s = strength[i]
    pareto_idx = np.array(pareto_idx)

    df_pareto = pd.DataFrame(X[pareto_idx], columns=FEATURE_COLUMNS)
    df_pareto['WorstCase_Tensile_MPa'] = pred[pareto_idx, 0].round(2)
    df_pareto['WorstCase_Compressive_MPa'] = pred[pareto_idx, 1].round(2)
    df_pareto['WorstCase_Composite_MPa'] = strength[pareto_idx].round(2)
    df_pareto['Print_Time_Index'] = tau[pareto_idx].round(1)
    os.makedirs(tab_dir, exist_ok=True)
    df_pareto.to_csv(os.path.join(tab_dir, 'pareto_front_solutions.csv'), index=False)

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6.2))
    rng = np.random.default_rng(42)
    sub = rng.choice(len(X), size=min(6000, len(X)), replace=False)
    plt.scatter(tau[sub], strength[sub], color='lightgray', s=8, alpha=0.5,
                label=f'Feasible design space ({len(X):,} evaluated recipes)')
    plt.plot(tau[pareto_idx], strength[pareto_idx], '-o', color='#d62728', ms=5,
             mec='black', lw=1.5, label=f'Pareto-optimal frontier ({len(pareto_idx)} recipes)')
    if df_cases is not None:
        markers = ['s', '^']
        for k, (_, r) in enumerate(df_cases.iterrows()):
            s_case = composite_strength(r['WorstCase_Tensile_MPa'], r['WorstCase_Compressive_MPa'])
            plt.scatter(r['Print_Time_Index'], s_case, marker=markers[k % 2], s=180,
                        color=['#1565C0', '#2E7D32'][k % 2], edgecolors='black', zorder=5,
                        label=f"{r['Component']} recipe")
    plt.xscale('log')
    plt.title("Pareto Frontier: Composite Strength vs Relative Print Time", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Relative print-time index τ (log scale, lower = faster)", fontsize=10)
    plt.ylabel("Worst-case composite strength 0.5·σt + 0.5·σc (MPa)", fontsize=10)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "pareto_frontier_automotive.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")
    return df_pareto


def run_full_optimization(model, safety_margins=(0.0, 0.0), save_dir_fig=OUTPUT_FIG_DIR,
                          save_dir_tab=OUTPUT_TAB_DIR):
    """
    Master function: single-objective optima, sensitivity, case studies and Pareto front.
    Returns (df_optimal, df_sens, df_cases, df_pareto).
    """
    os.makedirs(save_dir_tab, exist_ok=True)

    results = []
    for obj in ['tensile', 'compressive', 'composite']:
        print(f"  [Optimization] Differential evolution: maximise {obj} strength...")
        r = optimize_single_objective(model, objective=obj)
        print(f"    -> T={r['Predicted_Tensile_MPa']} MPa, C={r['Predicted_Compressive_MPa']} MPa "
              f"(best tested combination score {r['Best_Tested_Combination_Score_MPa']} MPa)")
        results.append(r)
    df_optimal = pd.DataFrame(results)
    df_optimal.to_csv(os.path.join(save_dir_tab, 'optimal_parameters.csv'), index=False)
    print(f"  Saved: {os.path.join(save_dir_tab, 'optimal_parameters.csv')}")

    composite = df_optimal[df_optimal['Objective'] == 'Composite'].iloc[0]
    base_x = np.array([composite[c] for c in FEATURE_COLUMNS], dtype=float)

    print("  [Optimization] Sensitivity analysis around the composite optimum...")
    df_sens, curves = sensitivity_analysis(model, base_x)
    df_sens.to_csv(os.path.join(save_dir_tab, 'sensitivity_analysis.csv'), index=False)
    print(f"  Saved: {os.path.join(save_dir_tab, 'sensitivity_analysis.csv')}")
    plot_sensitivity_tornado(df_sens, save_dir_fig)
    plot_sensitivity_curves(curves, base_x, save_dir_fig)

    print("  [Optimization] Constrained case studies (brake pedal, door handle)...")
    df_cases = run_case_studies(model, safety_margins, reference_recipe=base_x)
    df_cases.to_csv(os.path.join(save_dir_tab, 'automotive_case_study_recommendations.csv'), index=False)
    print(f"  Saved: {os.path.join(save_dir_tab, 'automotive_case_study_recommendations.csv')}")

    print("  [Optimization] Pareto frontier...")
    df_pareto = evaluate_pareto_front(model, df_cases, save_dir_fig, save_dir_tab)

    print("  [Optimization] All optimization analyses completed.\n")
    return df_optimal, df_sens, df_cases, df_pareto
