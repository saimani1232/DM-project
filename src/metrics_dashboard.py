"""
metrics_dashboard.py
Visual dashboard of all regression metrics (Review 1 feedback item #1).

Outputs:
1. metrics_r2_comparison.png  - hold-out R² of every model vs the paper's best R²
2. metrics_cv_r2_stability.png - 5-fold CV R² mean ± std (stability across folds)
3. metrics_error_heatmap.png  - all error metrics, coloured per column (rank within metric)
4. metrics_top3_complete.png  - all 9 metrics for the top-3 models
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'outputs', 'figures'
)

PAPER_BEST_R2 = {'Tensile': 0.93, 'Compressive': 0.98}   # Adam-optimised ANN, Table 7


def _split_targets(df):
    t = df[df['Target'].str.contains('Tensile')].set_index('Model')
    c = df[df['Target'].str.contains('Compressive')].set_index('Model')
    return t, c.loc[t.index]


def plot_r2_comparison(df_holdout, save_dir=OUTPUT_FIG_DIR):
    """Grouped bar chart of hold-out R² (tensile vs compressive) for every model."""
    os.makedirs(save_dir, exist_ok=True)
    tensile, compressive = _split_targets(df_holdout)
    models = tensile.index.tolist()
    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6.5))
    b1 = ax.bar(x - width / 2, tensile['R2'], width, label='Tensile Strength R²',
                color='#2196F3', edgecolor='black', alpha=0.85)
    b2 = ax.bar(x + width / 2, compressive['R2'], width, label='Compressive Strength R²',
                color='#FF5722', edgecolor='black', alpha=0.85)
    ax.axhline(PAPER_BEST_R2['Tensile'], color='#1565C0', ls='--', lw=1.3,
               label=f"Paper Adam-ANN tensile R² ({PAPER_BEST_R2['Tensile']})")
    ax.axhline(PAPER_BEST_R2['Compressive'], color='#BF360C', ls='--', lw=1.3,
               label=f"Paper Adam-ANN compressive R² ({PAPER_BEST_R2['Compressive']})")

    lo = min(0.9, np.floor(min(tensile['R2'].min(), compressive['R2'].min()) * 50) / 50)
    ax.set_ylim(lo, 1.004)
    ax.set_ylabel('Coefficient of Determination (R²)', fontsize=11, fontweight='bold')
    ax.set_title('Hold-out R² of All Models vs Best R² Reported by Munshi et al.\n(Higher is better)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ') for m in models], rotation=25, ha='right', fontsize=9)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=7, rotation=90)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'metrics_r2_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_cv_r2_stability(df_cv, save_dir=OUTPUT_FIG_DIR):
    """5-fold CV R² mean ± std for each model and target."""
    os.makedirs(save_dir, exist_ok=True)
    tensile, compressive = _split_targets(df_cv)
    models = tensile.index.tolist()
    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, tensile['R2_Mean'], width, yerr=tensile['R2_Std'], capsize=4,
           label='Tensile (mean ± std over 5 folds)', color='#2196F3', edgecolor='black', alpha=0.85)
    ax.bar(x + width / 2, compressive['R2_Mean'], width, yerr=compressive['R2_Std'], capsize=4,
           label='Compressive (mean ± std over 5 folds)', color='#FF5722', edgecolor='black', alpha=0.85)
    lo = min(0.9, np.floor(min(tensile['R2_Mean'].min(), compressive['R2_Mean'].min()) * 50) / 50)
    ax.set_ylim(lo, 1.004)
    ax.set_ylabel('5-fold CV R²', fontsize=11, fontweight='bold')
    ax.set_title('Cross-Validation Stability: R² Mean ± Standard Deviation Across 5 Folds',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ') for m in models], rotation=25, ha='right', fontsize=9)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'metrics_cv_r2_stability.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_error_heatmap(df_holdout, save_dir=OUTPUT_FIG_DIR):
    """
    Error metrics (average of tensile & compressive) for all models.
    Colours are normalised within each column so metrics on different scales
    (MPa², MPa, %) are comparable; the annotations show the raw values.
    """
    os.makedirs(save_dir, exist_ok=True)
    error_cols = ['MSE', 'RMSE', 'MAE', 'MAPE_%', 'Max_Error', 'Median_AE']
    avg = df_holdout.groupby('Model', sort=False)[error_cols].mean()
    avg = avg.sort_values('RMSE')
    norm = (avg - avg.min()) / (avg.max() - avg.min()).replace(0, 1)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(norm, annot=avg.values, fmt='.3f', cmap='RdYlGn_r', linewidths=0.5,
                cbar_kws={'label': 'Relative error within metric (0 = best, 1 = worst)'}, ax=ax)
    ax.set_xticklabels(['MSE (MPa²)', 'RMSE (MPa)', 'MAE (MPa)', 'MAPE (%)', 'Max Error (MPa)', 'Median AE (MPa)'])
    ax.set_yticklabels([m.replace('_', ' ') for m in avg.index], rotation=0)
    ax.set_title('Hold-out Error Metrics (mean of tensile & compressive)\nModels sorted by RMSE — greener is better',
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_ylabel('Model', fontsize=11)
    ax.set_xlabel('Error Metric', fontsize=11)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'metrics_error_heatmap.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_extended_metrics_table(df_holdout, save_dir=OUTPUT_FIG_DIR):
    """Table image with all 9 metrics for the top-3 models (by mean hold-out R²)."""
    os.makedirs(save_dir, exist_ok=True)
    top3 = df_holdout.groupby('Model')['R2'].mean().sort_values(ascending=False).head(3).index.tolist()
    data = df_holdout[df_holdout['Model'].isin(top3)].copy()
    data['order'] = data['Model'].map({m: i for i, m in enumerate(top3)})
    data = data.sort_values(['order', 'Target'], ascending=[True, False])

    metric_cols = ['MSE', 'RMSE', 'MAE', 'MAPE_%', 'R2', 'Adjusted_R2',
                   'Explained_Variance', 'Max_Error', 'Median_AE']
    headers = ['Model', 'Target', 'MSE\n(MPa²)', 'RMSE\n(MPa)', 'MAE\n(MPa)', 'MAPE\n(%)', 'R²',
               'Adj. R²', 'Expl.\nVariance', 'Max Err\n(MPa)', 'Median AE\n(MPa)']
    cells = []
    for _, row in data.iterrows():
        r = [row['Model'].replace('_', ' '), row['Target'].replace('_Strength_Mpa', '')]
        for col in metric_cols:
            v = row[col]
            r.append(f'{v:.5f}' if col in ['R2', 'Adjusted_R2', 'Explained_Variance'] else f'{v:.3f}')
        cells.append(r)

    fig, ax = plt.subplots(figsize=(17, 3.8))
    ax.axis('off')
    table = ax.table(cellText=cells, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.9)
    for j in range(len(headers)):
        table[0, j].set_facecolor('#1976D2')
        table[0, j].set_text_props(color='white', fontweight='bold')
    ax.set_title('All 9 Regression Metrics — Top-3 Models (hold-out test set, n = 77)',
                 fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'metrics_top3_complete.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_metrics_dashboard(df_holdout, df_cv, save_dir=OUTPUT_FIG_DIR):
    """Master function: generates all metrics dashboard visualisations."""
    print("\n  [Metrics Dashboard] R² comparison bar chart...")
    plot_r2_comparison(df_holdout, save_dir)
    print("  [Metrics Dashboard] CV stability chart...")
    plot_cv_r2_stability(df_cv, save_dir)
    print("  [Metrics Dashboard] Error metrics heatmap...")
    plot_error_heatmap(df_holdout, save_dir)
    print("  [Metrics Dashboard] Top-3 complete metrics table...")
    plot_extended_metrics_table(df_holdout, save_dir)
    print("  [Metrics Dashboard] All dashboard visualizations generated.\n")
