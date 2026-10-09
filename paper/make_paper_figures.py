"""
make_paper_figures.py
Creates the figures used by the IEEE paper (paper/figures/):
  * fig_pipeline.pdf  - method overview diagram (vector)
  * copies of the pipeline outputs used in the paper
Run from the project root after run_project.py:  python paper/make_paper_figures.py
"""

import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG_OUT = os.path.join(HERE, 'figures')
os.makedirs(FIG_OUT, exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'font.size': 8.5, 'pdf.fonttype': 42})


def box(ax, x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02',
                                fc=fc, ec='#333333', lw=0.8))
    ax.text(x + w / 2, y + h - 0.045, title, ha='center', va='top', fontsize=8.6, fontweight='bold')
    ax.text(x + w / 2, y + h - 0.13, body, ha='center', va='top', fontsize=7.4, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', lw=0.9, color='#333333', shrinkA=0, shrinkB=0))


def pipeline_figure():
    fig, ax = plt.subplots(figsize=(7.16, 2.35))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    w, h = 0.165, 0.42
    top = 0.55
    xs = [0.005, 0.209, 0.413, 0.617, 0.821]
    boxes = [
        ('Dataset', '383 ABS records\n5 FDM parameters\n$\\sigma_t$, $\\sigma_c$ (MPa)\n(Munshi et al.)', '#eef3f8'),
        ('Pre-processing', 'min-max scaling\ninside each pipeline\n80/20 hold-out\n+ 5-fold CV', '#eef3f8'),
        ('Nine regressors', '2 ANN replications\nSVR, DT, AdaBoost\nRF, ET, HGB, XGBoost', '#eef3f8'),
        ('Evaluation', '9 metrics + normalised\nerrors; selection by\nCV $R^2$ $\\rightarrow$ HGB', '#eef3f8'),
        ('Final model', 'HGB refitted on\nall 383 records', '#fdf1e6'),
    ]
    for x, (t, b, c) in zip(xs, boxes):
        box(ax, x, top, w, h, t, b, c)
    for i in range(4):
        arrow(ax, xs[i] + w + 0.012, top + h / 2, xs[i + 1] - 0.012, top + h / 2)

    bot, hb = 0.02, 0.40
    bxs = [0.045, 0.295, 0.545, 0.795]
    bw = 0.17
    bottom = [
        ('SHAP (MPa)', 'global + local\nexplanations of\nindividual predictions', '#eaf6ee'),
        ('Robust DE', 'worst case over\n$\\pm$5-unit drift,\nexhaustive check', '#eaf6ee'),
        ('Case studies', 'brake pedal, door\nhandle, Pareto front,\nsensitivity', '#eaf6ee'),
        ('Simulator', 'exact model in the\nbrowser, live SHAP\nand optimiser', '#eaf6ee'),
    ]
    # bus: final model -> horizontal line -> each output box
    bus_y = 0.49
    fx = xs[4] + w / 2
    ax.plot([fx, fx], [top - 0.015, bus_y], color='#333333', lw=0.9)
    ax.plot([bxs[0] + bw / 2, fx], [bus_y, bus_y], color='#333333', lw=0.9)
    for x, (t, b, c) in zip(bxs, bottom):
        box(ax, x, bot, bw, hb, t, b, c)
        arrow(ax, x + bw / 2, bus_y, x + bw / 2, bot + hb + 0.012)
    fig.savefig(os.path.join(FIG_OUT, 'fig_pipeline.pdf'), bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(FIG_OUT, 'fig_pipeline.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def copy_outputs():
    src = os.path.join(ROOT, 'outputs', 'figures')
    mapping = {
        'parity_plot_true_vs_pred.png': 'fig_parity.png',
        'shap_summary_tensile.png': 'fig_shap_summary.png',
        'shap_waterfall_tensile_optimum.png': 'fig_shap_waterfall.png',
        'sensitivity_response_curves.png': 'fig_sensitivity.png',
        'pareto_frontier_automotive.png': 'fig_pareto.png',
    }
    for a, b in mapping.items():
        shutil.copy(os.path.join(src, a), os.path.join(FIG_OUT, b))
    shutil.copy(os.path.join(ROOT, 'simulator', 'simulator_screenshot.png'), os.path.join(FIG_OUT, 'fig_simulator.png'))


if __name__ == '__main__':
    pipeline_figure()
    copy_outputs()
    print('Figures written to', FIG_OUT, sorted(os.listdir(FIG_OUT)))
