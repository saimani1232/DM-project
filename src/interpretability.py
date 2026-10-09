"""
interpretability.py
Physical analysis and visualisation module:
1. Pearson Correlation Heatmap (counterpart of Figs. 11 & 12 of Munshi et al.)
2. Permutation Feature Importance
3. Parity Plots (Experimental vs Predicted)
4. 2D Process-Parameter Response Contour Surfaces

All functions take wrapped models (raw inputs -> MPa), see model_utils.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error

from data_loader import FEATURE_DISPLAY_NAMES

OUTPUT_FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'outputs',
    'figures'
)


def plot_correlation_heatmap(df, save_dir=OUTPUT_FIG_DIR):
    """
    Pearson correlation heatmap of the 5 process parameters and 2 strengths.
    """
    os.makedirs(save_dir, exist_ok=True)
    corr = df.corr()

    plt.figure(figsize=(9, 7))
    sns.set_theme(style="white")
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
        annot=True, fmt=".3f"
    )
    plt.title("Pearson Correlation Heatmap: FDM ABS Process Parameters vs Properties",
              fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "pearson_correlation_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    sns.reset_orig()
    print(f"Saved: {save_path}")
    return corr


def plot_feature_importance(model, X_test, y_test, feature_names, model_name='', save_dir=OUTPUT_FIG_DIR):
    """
    Permutation importance (drop in R² when a parameter is shuffled) on the hold-out set.
    Returns a DataFrame with mean/std importance and the relative share in %.
    """
    os.makedirs(save_dir, exist_ok=True)

    r = permutation_importance(model, X_test, y_test, n_repeats=15, random_state=42, scoring='r2')
    importances = np.clip(r.importances_mean, 0, None)
    importances_pct = importances / np.sum(importances) * 100.0

    df_imp = pd.DataFrame({
        'Parameter': feature_names,
        'R2_Drop_Mean': r.importances_mean,
        'R2_Drop_Std': r.importances_std,
        'Relative_Importance_%': importances_pct
    }).sort_values('Relative_Importance_%', ascending=False)

    plt.figure(figsize=(8, 5))
    display = dict(zip(feature_names, FEATURE_DISPLAY_NAMES))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
    bars = plt.bar([display[p] for p in df_imp['Parameter']], df_imp['Relative_Importance_%'],
                   color=colors, edgecolor='black', alpha=0.85)

    plt.title(f"Permutation Feature Importance on ABS Strengths\n({model_name}, hold-out test set)",
              fontsize=11, fontweight='bold', pad=10)
    plt.ylabel("Relative Influence (%)", fontsize=10)
    plt.xlabel("FDM Printing Parameters", fontsize=10)
    plt.xticks(rotation=20, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, "feature_importance.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
    return df_imp


def plot_parity(y_true, y_pred, model_name="Best Model", save_dir=OUTPUT_FIG_DIR):
    """
    Experimental vs predicted parity plot (counterpart of Figs. 7-10 of the paper).
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    targets = [("Tensile Strength (MPa)", 0, "#1f77b4"), ("Compressive Strength (MPa)", 1, "#d62728")]

    for title, idx, color in targets:
        ax = axes[idx]
        y_t = y_true[:, idx]
        y_p = y_pred[:, idx]

        ax.scatter(y_t, y_p, color=color, alpha=0.75, edgecolors='k', s=55, label='Test samples')

        min_val = min(np.min(y_t), np.min(y_p)) * 0.95
        max_val = max(np.max(y_t), np.max(y_p)) * 1.05
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal Fit (1:1)')
        ax.plot([min_val, max_val], [min_val * 1.1, max_val * 1.1], 'r:', alpha=0.6, label='±10% Bound')
        ax.plot([min_val, max_val], [min_val * 0.9, max_val * 0.9], 'r:', alpha=0.6)

        r2 = r2_score(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        ax.text(0.05, 0.90, f'$R^2$ = {r2:.4f}\nRMSE = {rmse:.3f} MPa',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.85))

        ax.set_title(f"{title} - {model_name}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Experimental Values (MPa)", fontsize=10)
        ax.set_ylabel("Predicted Values (MPa)", fontsize=10)
        ax.legend(loc='lower right')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "parity_plot_true_vs_pred.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_2d_response_surface(model, bounds, fixed_bed=90.0, fixed_speed=30.0,
                             layer_levels=(0.2, 0.8), save_dir=OUTPUT_FIG_DIR):
    """
    Contour maps of predicted strength over Infill Density x Nozzle Temperature,
    for each tested layer height (bed temperature and print speed held fixed at
    tested levels). Only the experimentally covered parameter window is plotted.
    """
    os.makedirs(save_dir, exist_ok=True)

    infill_vals = np.linspace(*bounds['Infill_Density_percent'], 60)
    nozzle_vals = np.linspace(*bounds['Nozzle_Temp_C'], 60)
    I_grid, N_grid = np.meshgrid(infill_vals, nozzle_vals)

    fig, axes = plt.subplots(len(layer_levels), 2, figsize=(14, 5.2 * len(layer_levels)))
    axes = np.atleast_2d(axes)

    # Common colour scale per target so the layer-height effect is visible
    preds_all = []
    for layer in layer_levels:
        grid = np.column_stack([
            N_grid.ravel(),
            np.full(N_grid.size, fixed_bed),
            np.full(N_grid.size, fixed_speed),
            np.full(N_grid.size, layer),
            I_grid.ravel()
        ])
        preds_all.append(model.predict(grid))
    stacked = np.vstack(preds_all)

    for row, (layer, preds) in enumerate(zip(layer_levels, preds_all)):
        for col, (label, cmap) in enumerate([('Tensile', 'viridis'), ('Compressive', 'plasma')]):
            Z = preds[:, col].reshape(I_grid.shape)
            levels = np.linspace(stacked[:, col].min(), stacked[:, col].max(), 21)
            c = axes[row, col].contourf(I_grid, N_grid, Z, levels=levels, cmap=cmap)
            axes[row, col].set_title(
                f"Predicted {label} Strength (MPa)\n[Layer={layer} mm, Bed={fixed_bed:.0f}°C, Speed={fixed_speed:.0f} mm/s]",
                fontsize=11, fontweight='bold')
            axes[row, col].set_xlabel("Infill Density (%)", fontsize=10)
            axes[row, col].set_ylabel("Nozzle Temperature (°C)", fontsize=10)
            plt.colorbar(c, ax=axes[row, col], label=f'{label} Strength (MPa)')

    plt.tight_layout()
    save_path = os.path.join(save_dir, "response_surface_contours.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
    return save_path
