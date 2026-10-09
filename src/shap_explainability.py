"""
shap_explainability.py
SHAP (SHapley Additive exPlanations; Lundberg & Lee, 2017) for per-prediction
explainability. Addresses Review 1 feedback item #2: "Explainability of the result
we got for a prediction."

The models are trained on min-max scaled inputs/targets, but every explanation
produced here is converted back to physical units:
  * feature values are shown in °C, mm/s, mm and %,
  * SHAP contributions and the base value are in MPa.
Because the target scaling is linear, SHAP_MPa = SHAP_scaled x (y_max - y_min) and
base_MPa = base_scaled x (y_max - y_min) + y_min, so the additivity property
(base + sum of contributions = prediction) holds exactly in MPa. This is verified
numerically for every explanation.

Outputs (both targets unless stated):
1. Beeswarm summary plot  -> global impact and direction
2. Mean |SHAP| bar plot    -> game-theoretic feature importance
3. Dependence plots        -> effect of one parameter + interaction colouring
4. Waterfall plots         -> local explanation of specific predictions
                              (optimal recipe, case-study recipes, test samples)
5. shap_local_explanations.csv / shap_global_importance.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor

from data_loader import FEATURE_COLUMNS, FEATURE_DISPLAY_NAMES, TARGET_COLUMNS
from model_utils import unwrap_model

OUTPUT_FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'outputs', 'figures'
)
OUTPUT_TAB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'outputs', 'tables'
)

TARGET_LABELS = ['Tensile', 'Compressive']


class ShapExplainerMPa:
    """
    Wraps a fitted model (see model_utils.wrap_model) and returns SHAP values in MPa
    for raw (unscaled) process parameters.
    """

    def __init__(self, model, background_X):
        import shap
        self.model = model
        self.x_scaler, self.estimator, self.y_scaler = unwrap_model(model)
        self.explainers = []
        self.kind = 'tree'
        for t in range(len(TARGET_COLUMNS)):
            base = self.estimator.estimators_[t] if isinstance(self.estimator, MultiOutputRegressor) else self.estimator
            try:
                self.explainers.append(shap.TreeExplainer(base))
            except Exception:
                # Model-agnostic fallback (e.g. ANN / SVR) working directly in MPa
                self.kind = 'kernel'
                bg = shap.kmeans(np.asarray(background_X, dtype=float), 20)
                f = (lambda X, t=t: self.model.predict(np.asarray(X, dtype=float))[:, t])
                self.explainers.append(shap.KernelExplainer(f, bg))

    def explain(self, X_raw, target_idx):
        """Return (shap_values_MPa [n, 5], base_value_MPa) for one target."""
        X_raw = np.atleast_2d(np.asarray(X_raw, dtype=float))
        explainer = self.explainers[target_idx]

        if self.kind == 'kernel':
            sv = np.asarray(explainer.shap_values(X_raw, silent=True))
            return sv, float(np.ravel(explainer.expected_value)[0])

        Xs = self.x_scaler.transform(X_raw)
        sv = explainer.shap_values(Xs, check_additivity=False)
        ev = explainer.expected_value
        sv = np.asarray(sv)
        multi = not isinstance(self.estimator, MultiOutputRegressor)
        if multi:
            # Native multi-output model (e.g. Random Forest): pick the target's slice
            if sv.ndim == 3 and sv.shape[-1] == len(TARGET_COLUMNS):
                sv = sv[:, :, target_idx]
            elif sv.ndim == 3:
                sv = sv[target_idx]
            ev = np.ravel(ev)[target_idx]
        ev = float(np.ravel(ev)[0])

        span = self.y_scaler.data_range_[target_idx]
        y_min = self.y_scaler.data_min_[target_idx]
        return sv * span, ev * span + y_min

    def check_additivity(self, X_raw, target_idx, tol=1e-3):
        """Maximum |base + sum(SHAP) - prediction| in MPa."""
        sv, base = self.explain(X_raw, target_idx)
        pred = self.model.predict(np.atleast_2d(X_raw))[:, target_idx]
        return float(np.max(np.abs(base + sv.sum(axis=1) - pred)))


def plot_shap_summary(shap_values, X_raw, target_label, save_dir=OUTPUT_FIG_DIR):
    """Beeswarm: how each parameter pushes predictions up/down across all samples."""
    import shap
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_raw, feature_names=FEATURE_DISPLAY_NAMES,
                      show=False, plot_size=(10, 6))
    plt.title(f'SHAP Summary — {target_label} Strength (contributions in MPa)\n'
              f'(each dot = one sample; red = high parameter value, blue = low)',
              fontsize=11, fontweight='bold', pad=15)
    plt.xlabel(f'SHAP value (MPa change in predicted {target_label.lower()} strength)')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'shap_summary_{target_label.lower()}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_shap_bar(shap_values, target_label, save_dir=OUTPUT_FIG_DIR):
    """Mean |SHAP| per parameter (MPa)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)

    plt.figure(figsize=(9, 5))
    colors = ['#BBDEFB', '#90CAF9', '#64B5F6', '#2196F3', '#1976D2']
    plt.barh(range(len(order)), mean_abs[order], color=colors, edgecolor='black', alpha=0.9)
    plt.yticks(range(len(order)), [FEATURE_DISPLAY_NAMES[i] for i in order], fontsize=10)
    plt.xlabel('Mean |SHAP value| (MPa)', fontsize=10, fontweight='bold')
    plt.title(f'SHAP Feature Importance — {target_label} Strength',
              fontsize=12, fontweight='bold', pad=12)
    plt.grid(axis='x', linestyle=':', alpha=0.4)
    xmax = mean_abs.max()
    for i, idx in enumerate(order):
        plt.text(mean_abs[idx] + xmax * 0.01, i, f'{mean_abs[idx]:.2f} MPa',
                 va='center', fontsize=9, fontweight='bold')
    plt.xlim(0, xmax * 1.18)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'shap_bar_{target_label.lower()}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_shap_dependence(shap_values, X_raw, feature_idx, interaction_idx, target_label,
                         save_dir=OUTPUT_FIG_DIR):
    """Dependence of one parameter's SHAP value on its value, coloured by another parameter."""
    import shap
    plt.figure(figsize=(9, 6))
    shap.dependence_plot(feature_idx, shap_values, X_raw, feature_names=FEATURE_DISPLAY_NAMES,
                         interaction_index=interaction_idx, show=False)
    plt.title(f'SHAP Dependence — {FEATURE_DISPLAY_NAMES[feature_idx]} effect on {target_label} Strength\n'
              f'(coloured by {FEATURE_DISPLAY_NAMES[interaction_idx]})',
              fontsize=11, fontweight='bold', pad=15)
    plt.ylabel(f'SHAP value for {FEATURE_DISPLAY_NAMES[feature_idx]} (MPa)')
    plt.tight_layout()
    feat_short = FEATURE_DISPLAY_NAMES[feature_idx].split('(')[0].strip().replace(' ', '_').lower()
    save_path = os.path.join(save_dir, f'shap_dependence_{feat_short}_{target_label.lower()}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_shap_waterfall(shap_row, base_value, x_row, target_label, case_name, file_tag,
                        save_dir=OUTPUT_FIG_DIR):
    """Waterfall: exact MPa contribution of each parameter to ONE prediction."""
    import shap
    explanation = shap.Explanation(values=shap_row, base_values=base_value, data=x_row,
                                   feature_names=FEATURE_DISPLAY_NAMES)
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    pred = base_value + shap_row.sum()
    input_str = ", ".join(f"{n.split(' (')[0]}={v:g}" for n, v in zip(FEATURE_DISPLAY_NAMES, x_row))
    plt.title(f'SHAP Waterfall — {case_name}: predicted {target_label} = {pred:.2f} MPa\n[{input_str}]',
              fontsize=10, fontweight='bold', pad=15)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'shap_waterfall_{target_label.lower()}_{file_tag}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def run_shap_analysis(model, X_global, local_cases, model_name='',
                      save_dir=OUTPUT_FIG_DIR, tab_dir=OUTPUT_TAB_DIR):
    """
    Master function.

    Parameters:
        model       : fitted wrapped model
        X_global    : raw parameter matrix used for the global explanation
        local_cases : list of dicts {'name': str, 'tag': str, 'x': array(5)} to explain
                      individually (e.g. optimal recipe, case-study recipes, test samples)

    Returns:
        (df_global_importance, df_local_explanations, max_additivity_error_MPa)
    """
    warnings.filterwarnings('ignore')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    explainer = ShapExplainerMPa(model, X_global)
    X_global = np.asarray(X_global, dtype=float)

    global_rows, local_rows = [], []
    max_err = 0.0

    for t, label in enumerate(TARGET_LABELS):
        print(f"\n  [SHAP] {label} strength — explaining {len(X_global)} samples with {model_name}...")
        sv, base = explainer.explain(X_global, t)
        err = explainer.check_additivity(X_global, t)
        max_err = max(max_err, err)
        print(f"  [SHAP] Additivity check (base + ΣSHAP = prediction): max error = {err:.2e} MPa")

        plot_shap_summary(sv, X_global, label, save_dir)
        plot_shap_bar(sv, label, save_dir)

        mean_abs = np.abs(sv).mean(axis=0)
        for i, f in enumerate(FEATURE_COLUMNS):
            global_rows.append({
                'Target': label, 'Parameter': FEATURE_DISPLAY_NAMES[i],
                'Mean_Abs_SHAP_MPa': round(float(mean_abs[i]), 4),
                'Share_%': round(float(mean_abs[i] / mean_abs.sum() * 100), 2)
            })

        # Dependence plots: infill x layer height, nozzle temp x print speed
        plot_shap_dependence(sv, X_global, 4, 3, label, save_dir)
        if t == 0:
            plot_shap_dependence(sv, X_global, 0, 2, label, save_dir)
            plot_shap_dependence(sv, X_global, 3, 2, label, save_dir)

        # Local explanations
        for case in local_cases:
            x = np.asarray(case['x'], dtype=float).reshape(1, -1)
            sv_loc, base_loc = explainer.explain(x, t)
            sv_loc = sv_loc[0]
            pred = float(model.predict(x)[0, t])
            row = {'Case': case['name'], 'Target': label}
            for i, f in enumerate(FEATURE_COLUMNS):
                row[f] = float(x[0, i])
            row['Base_Value_MPa'] = round(base_loc, 3)
            for i, n in enumerate(FEATURE_DISPLAY_NAMES):
                row[f'SHAP_{FEATURE_COLUMNS[i]}_MPa'] = round(float(sv_loc[i]), 3)
            row['Prediction_MPa'] = round(pred, 3)
            row['Base_plus_SHAP_MPa'] = round(base_loc + float(sv_loc.sum()), 3)
            local_rows.append(row)
            if case.get('targets') is None or label in case['targets']:
                plot_shap_waterfall(sv_loc, base_loc, x[0], label, case['name'], case['tag'], save_dir)

    df_global = pd.DataFrame(global_rows)
    df_local = pd.DataFrame(local_rows)
    df_global.to_csv(os.path.join(tab_dir, 'shap_global_importance.csv'), index=False)
    df_local.to_csv(os.path.join(tab_dir, 'shap_local_explanations.csv'), index=False)
    print(f"  Saved: {os.path.join(tab_dir, 'shap_global_importance.csv')}")
    print(f"  Saved: {os.path.join(tab_dir, 'shap_local_explanations.csv')}")
    print("  [SHAP] All explainability analyses completed.\n")
    return df_global, df_local, max_err
