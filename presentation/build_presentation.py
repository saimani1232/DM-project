"""
build_presentation.py
Builds presentation/Digital_Manufacturing_ABS_Presentation.pptx from the results in
outputs/ (run `python run_project.py` first), so every number on the slides matches
the generated tables.

Run from the project root:  python presentation/build_presentation.py
"""

import os
import json
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Emu, Pt, Inches
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, 'outputs', 'figures')
TAB = os.path.join(ROOT, 'outputs', 'tables')
OUT = os.path.join(HERE, 'Digital_Manufacturing_ABS_Presentation.pptx')

# Colour palette of the original deck
DARK = RGBColor(0x0F, 0x17, 0x2A)
SLATE = RGBColor(0x1E, 0x29, 0x3B)
MUTED = RGBColor(0x64, 0x74, 0x8B)
BLUE = RGBColor(0x02, 0x84, 0xC7)
CHIP_BG = RGBColor(0xE0, 0xF2, 0xFE)
CARD_BORDER = RGBColor(0xE2, 0xE8, 0xF0)
GREEN_BG = RGBColor(0xEC, 0xFD, 0xF5)
GREEN = RGBColor(0x06, 0x5F, 0x46)
GREEN_LINE = RGBColor(0x10, 0xB9, 0x81)
AMBER_BG = RGBColor(0xFF, 0xFB, 0xEB)
AMBER = RGBColor(0x92, 0x40, 0x0E)
AMBER_LINE = RGBColor(0xF5, 0x9E, 0x0B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_BG = RGBColor(0xF8, 0xFA, 0xFC)
INDIGO = RGBColor(0xA5, 0xB4, 0xFC)

SLIDE_W, SLIDE_H = 12191695, 6858000
MARGIN = Inches(0.8)
CONTENT_TOP = Inches(1.55)
CONTENT_W = SLIDE_W - 2 * MARGIN
CONTENT_H = SLIDE_H - CONTENT_TOP - Inches(0.45)

prs = Presentation()
prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
BLANK = prs.slide_layouts[6]
slide_no = [0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _style_run(run, size, color, bold=False, italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = 'Calibri'


def rect(slide, x, y, w, h, fill, line=None, rounded=True, line_w=1.0):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE, x, y, w, h)
    if rounded:
        shp.adjustments[0] = 0.04
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(line_w)
    shp.shadow.inherit = False
    return shp


def text(slide, x, y, w, h, paragraphs, anchor=MSO_ANCHOR.TOP, shape=None, margin=Inches(0.12)):
    """
    paragraphs: list of (text, size, color, bold[, align]) tuples or plain strings.
    Text may contain '**' pairs for bold segments.
    """
    if shape is None:
        shape = slide.shapes.add_textbox(x, y, w, h)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = margin
    tf.margin_top = tf.margin_bottom = Inches(0.06)
    first = True
    for para in paragraphs:
        if isinstance(para, str):
            para = (para, 12, DARK, False)
        t, size, color, bold = para[:4]
        align = para[4] if len(para) > 4 else PP_ALIGN.LEFT
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.space_after = Pt(4)
        parts = t.split('**')
        for k, part in enumerate(parts):
            if not part:
                continue
            r = p.add_run()
            r.text = part
            _style_run(r, size, color, bold or (k % 2 == 1))
    return shape


def card(slide, x, y, w, h, title, bullets, title_color=SLATE, size=15, fill=WHITE, border=CARD_BORDER):
    shp = rect(slide, x, y, w, h, fill, border)
    paras = [(title, size + 3, title_color, True)] if title else []
    paras += [(b, size, DARK, False) for b in bullets]
    text(slide, x, y, w, h, paras, shape=shp, margin=Inches(0.18))
    shp.text_frame.vertical_anchor = MSO_ANCHOR.TOP
    shp.text_frame.margin_top = Inches(0.14)
    return shp


def banner(slide, y, msg, kind='green', h=Inches(0.55)):
    bg, fg, ln = {'green': (GREEN_BG, GREEN, GREEN_LINE), 'amber': (AMBER_BG, AMBER, AMBER_LINE)}[kind]
    shp = rect(slide, MARGIN, y, CONTENT_W, h, bg, ln)
    text(slide, 0, 0, 0, 0, [(msg, 12.5, fg, True, PP_ALIGN.CENTER)], anchor=MSO_ANCHOR.MIDDLE, shape=shp)
    return shp


def picture(slide, path, x, y, w, h):
    """Insert an image scaled to fit inside the box (aspect ratio kept, centred)."""
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = int(iw * scale), int(ih * scale)
    return slide.shapes.add_picture(path, x + (w - pw) // 2, y + (h - ph) // 2, pw, ph)


def table(slide, x, y, w, rows, col_widths=None, font=10.5, header_fill=SLATE, row_h=Inches(0.34),
          highlight_rows=()):
    n_rows, n_cols = len(rows), len(rows[0])
    gt = slide.shapes.add_table(n_rows, n_cols, x, y, w, row_h * n_rows).table
    if col_widths:
        total = sum(col_widths)
        for j, cw in enumerate(col_widths):
            gt.columns[j].width = int(w * cw / total)
    for i, row in enumerate(rows):
        gt.rows[i].height = row_h
        for j, val in enumerate(row):
            cell = gt.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = header_fill
            elif i in highlight_rows:
                cell.fill.fore_color.rgb = GREEN_BG
            else:
                cell.fill.fore_color.rgb = LIGHT_BG if i % 2 else WHITE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(val)
            _style_run(r, font, WHITE if i == 0 else DARK, bold=(i == 0 or i in highlight_rows))
    return gt


def new_slide(chip, title, dark=False):
    s = prs.slides.add_slide(BLANK)
    slide_no[0] += 1
    rect(s, 0, 0, SLIDE_W, SLIDE_H, DARK if dark else LIGHT_BG, rounded=False)
    if not dark:
        chip_shape = rect(s, MARGIN, Inches(0.38), Inches(0.12) * (len(chip) + 6), Inches(0.32), CHIP_BG, BLUE)
        text(s, 0, 0, 0, 0, [(chip, 9.5, BLUE, True, PP_ALIGN.CENTER)], anchor=MSO_ANCHOR.MIDDLE,
             shape=chip_shape, margin=Inches(0.05))
        text(s, MARGIN, Inches(0.72), CONTENT_W, Inches(0.7), [(title, 24, SLATE, True)])
        text(s, SLIDE_W - Inches(1.3), SLIDE_H - Inches(0.42), Inches(1.0), Inches(0.3),
             [(str(slide_no[0]), 10, MUTED, False, PP_ALIGN.RIGHT)])
    return s


def fmt(v, d=2):
    return f'{v:.{d}f}'


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
summary = json.load(open(os.path.join(TAB, 'summary_key_results.json'), encoding='utf-8'))
ranking = pd.read_csv(os.path.join(TAB, 'model_ranking.csv'), index_col=0)
holdout = pd.read_csv(os.path.join(TAB, 'extended_metrics_holdout.csv'))
cv = pd.read_csv(os.path.join(TAB, '5fold_cross_validation_results.csv'))
t7 = pd.read_csv(os.path.join(TAB, 'Table7_R2_comparison_paper_vs_ours.csv'))
t4 = pd.read_csv(os.path.join(TAB, 'Table4_benchmark_comparison_paper_vs_ours.csv'))
opt = pd.read_csv(os.path.join(TAB, 'optimal_parameters.csv'))
sens = pd.read_csv(os.path.join(TAB, 'sensitivity_analysis.csv'))
cases = pd.read_csv(os.path.join(TAB, 'automotive_case_study_recommendations.csv'))
shap_g = pd.read_csv(os.path.join(TAB, 'shap_global_importance.csv'))
shap_l = pd.read_csv(os.path.join(TAB, 'shap_local_explanations.csv'))
l15 = pd.read_csv(os.path.join(TAB, 'taguchi_L15_predictions.csv'))

best = summary['best_model']
best_pretty = summary['best_model_pretty'].split(' (')[0]
b_cv = summary['best_cv']
b_ho = summary['best_holdout']
T, C = 'Tensile_Strength_Mpa', 'Compressive_Strength_Mpa'
NICE = {
    'Adam_ANN_MLP': 'Adam ANN (replication)', 'Bayesian_Regularized_ANN': 'Bayesian ANN (replication)',
    'Support_Vector_Regression_SVR': 'SVR', 'Decision_Tree': 'Decision Tree', 'AdaBoost': 'AdaBoost',
    'Random_Forest_Tuned': 'Random Forest (proposed)', 'Extra_Trees_Ensemble': 'Extra Trees (proposed)',
    'Hist_Gradient_Boosting': 'Hist. Gradient Boosting (proposed)', 'XGBoost_Ensemble': 'XGBoost (proposed)',
}


def ho(model, target, metric):
    return holdout[(holdout.Model == model) & (holdout.Target == target)][metric].values[0]


def cvm(model, target, metric):
    return cv[(cv.Model == model) & (cv.Target == target)][f'{metric}_Mean'].values[0]


comp = opt[opt.Objective == 'Composite'].iloc[0]
pedal = cases[cases.Component.str.contains('Brake')].iloc[0]
handle = cases[cases.Component.str.contains('Door')].iloc[0]
sg = shap_g[shap_g.Target == 'Tensile'].set_index('Parameter')
adam_t, adam_c = ho('Adam_ANN_MLP', T, 'R2'), ho('Adam_ANN_MLP', C, 'R2')

# ---------------------------------------------------------------------------
# Slide 1 - Title
# ---------------------------------------------------------------------------
s = new_slide('', '', dark=True)
chip = rect(s, MARGIN, Inches(0.9), Inches(3.6), Inches(0.38), DARK, INDIGO)
text(s, 0, 0, 0, 0, [('INTRODUCTION TO DIGITAL MANUFACTURING', 10, INDIGO, True, PP_ALIGN.CENTER)],
     anchor=MSO_ANCHOR.MIDDLE, shape=chip)
text(s, MARGIN, Inches(1.5), CONTENT_W, Inches(2.2), [
    ('Optimization and Machine Learning Prediction of Tensile and Compressive Strengths '
     'for Additively Manufactured ABS Automotive Components', 30, WHITE, True),
    ('Replication and extension of Munshi et al. (Next Materials, 2026), with nine ML models, '
     'complete metrics, SHAP explainability and robust process optimisation', 15, INDIGO, False)])
boxes = [
    ('REFERENCE PAPER', ['G.A. Munshi, V.M. Kulkarni, S. Yargatti', '"Computation of tensile and compressive strengths of additively manufactured ABS material for automotive applications using ANN algorithms"', 'Next Materials 10 (2026) 101420']),
    ('KEY RESULTS', [f'383-sample ABS dataset, 5 FDM parameters', f'Best model: {best_pretty}, CV R² = {fmt(b_cv[T]["R2_Mean"], 4)}',
                     f'SHAP explanations in MPa', f'Robust optimum: {comp.Predicted_Tensile_MPa:.1f} / {comp.Predicted_Compressive_MPa:.1f} MPa']),
    ('TEAM', ['016  Gedela Kiran Kumar', '021  Kapuluru Chenchu Sai Sashank', '028  Sai Mani', '035  Perumalla Krishna Murthy', 'CSE, Amrita Vishwa Vidyapeetham, Amaravati']),
]
for k, (h, lines) in enumerate(boxes):
    x = MARGIN + k * (CONTENT_W // 3)
    shp = rect(s, x + Inches(0.05), Inches(4.1), CONTENT_W // 3 - Inches(0.2), Inches(2.2), SLATE, MUTED)
    text(s, 0, 0, 0, 0, [(h, 11, INDIGO, True)] + [(l, 11, WHITE, False) for l in lines], shape=shp,
         margin=Inches(0.18))
    shp.text_frame.vertical_anchor = MSO_ANCHOR.TOP

# ---------------------------------------------------------------------------
# Slide 2 - Problem & objectives
# ---------------------------------------------------------------------------
s = new_slide('INDUSTRY CONTEXT', 'Problem Statement & Objectives')
w3 = (CONTENT_W - Inches(0.4)) // 3
card(s, MARGIN, CONTENT_TOP, w3, CONTENT_H, 'Why it matters', [
    '• FDM/MEX prints ABS parts bead by bead and layer by layer, so strength depends on how well beads fuse and how much material fills the part.',
    '• Five coupled parameters (nozzle & bed temperature, speed, layer height, infill) interact non-linearly.',
    '• Trial-and-error printing and ASTM testing is slow and wastes material.'])
card(s, MARGIN + w3 + Inches(0.2), CONTENT_TOP, w3, CONTENT_H, 'Automotive case studies (from the paper)', [
    '• **Brake pedal:** high compressive load, so we require σc ≥ 45 MPa.',
    '• **Door handle:** repeated pulling, so we require σt ≥ 30 MPa.',
    '• Goal: meet the strength requirement with the **shortest print time**.'], title_color=BLUE)
card(s, MARGIN + 2 * (w3 + Inches(0.2)), CONTENT_TOP, w3, CONTENT_H, 'Project objectives', [
    '1. Replicate the paper\'s ANN models on its dataset and compare with its Tables 4 and 7.',
    '2. Evaluate 9 models with **all regression metrics** (hold-out + 5-fold CV).',
    '3. **Explain individual predictions** (SHAP, in MPa).',
    '4. **Find optimal parameters** formally, with sensitivity analysis, case studies and a Pareto front.'])

# ---------------------------------------------------------------------------
# Slide 3 - Dataset
# ---------------------------------------------------------------------------
s = new_slide('DATASET', 'Reference Paper & Dataset (383 samples)')
left_w = Inches(5.0)
card(s, MARGIN, CONTENT_TOP, left_w, CONTENT_H, 'Provenance (Section 2.5 of the paper)', [
    '• 383 tensile/compressive strengths of virgin ABS compiled by the authors from peer-reviewed studies and test reports (ASTM D638 / D695 specimens); outliers beyond 3σ removed.',
    '• The paper\'s own PRATI_AP prints were only used for its 15-run Taguchi L15 validation.',
    '• 0 missing values, 0 duplicate rows.',
    f'• Compressive ≈ **{summary["ratio_C_over_T"]:.2f} × tensile** in every record (r = {summary["corr_T_C"]:.3f}).',
    '• Each parameter takes only a few discrete levels; layer height only 0.2 and 0.8 mm. The optimiser therefore stays inside this window.'])
stats = pd.read_csv(os.path.join(TAB, 'dataset_statistics.csv'), index_col=0)
names = ['Nozzle temp. (°C)', 'Bed temp. (°C)', 'Print speed (mm/s)', 'Layer height (mm)', 'Infill density (%)',
         'Tensile strength (MPa)', 'Compressive strength (MPa)']
rows = [['Variable', 'Min', 'Max', 'Mean', 'Std', 'Levels']]
for n, (idx, r) in zip(names, stats.iterrows()):
    rows.append([n, f'{r["min"]:g}', f'{r["max"]:g}', f'{r["mean"]:.2f}', f'{r["std"]:.2f}', f'{int(r["n_levels"])}'])
table(s, MARGIN + left_w + Inches(0.3), CONTENT_TOP, CONTENT_W - left_w - Inches(0.3), rows,
      col_widths=[3.2, 1, 1, 1.1, 1.1, 1], font=13, row_h=Inches(0.6))

# ---------------------------------------------------------------------------
# Slide 4 - Correlation
# ---------------------------------------------------------------------------
s = new_slide('EXPLORATORY ANALYSIS', 'Pearson Correlation: Which Parameters Drive Strength?')
card(s, MARGIN, CONTENT_TOP, Inches(4.9), CONTENT_H, 'Findings (paper Tables 8 & 9 agree)', [
    '1. **Infill density r = +0.927**: the dominant driver; a larger solid load-bearing area.',
    '2. **Layer height r = −0.348**: thicker layers leave larger inter-bead voids and less bonding area.',
    '3. **Print speed r = −0.162**: less time for interlayer diffusion.',
    '4. **Nozzle / bed temperature |r| < 0.04**: no linear effect inside 200–250 °C / 50–110 °C.',
    '5. Parameters are mutually uncorrelated (|r| ≤ 0.07), consistent with a designed factor space.'])
picture(s, os.path.join(FIG, 'pearson_correlation_heatmap.png'), MARGIN + Inches(5.1), CONTENT_TOP,
        CONTENT_W - Inches(5.1), CONTENT_H)

# ---------------------------------------------------------------------------
# Slide 5 - Methodology
# ---------------------------------------------------------------------------
s = new_slide('METHODOLOGY', 'Models, Validation & Pipeline')
w2 = (CONTENT_W - Inches(0.3)) // 2
card(s, MARGIN, CONTENT_TOP, w2, Inches(3.0), 'Paper replications', [
    '• **Adam ANN**: 2 tanh hidden layers (20-20), lr 1e-3, batch 32, 1000 epochs (paper §2.6.1).',
    '• **Bayesian-regularised ANN**: 2×20 tanh, linear output; MATLAB trainbr approximated by L-BFGS + L2 weight decay.',
    '• Literature comparators: SVR, Decision Tree, AdaBoost.'])
card(s, MARGIN + w2 + Inches(0.3), CONTENT_TOP, w2, Inches(3.0), 'Proposed ensembles', [
    '• **Histogram Gradient Boosting** (300 iterations, lr 0.05, depth 6)',
    '• Random Forest (300 trees), Extra Trees (300 trees)',
    '• XGBoost (350 trees, lr 0.04, depth 5)'], title_color=BLUE)
card(s, MARGIN, CONTENT_TOP + Inches(3.2), CONTENT_W, CONTENT_H - Inches(3.2), 'Leakage-free evaluation pipeline', [
    '• Min-max scaling of inputs and targets **inside** each model pipeline, re-fitted on the training part of every split.',
    '• 80/20 hold-out (306/77, seed 42) + shuffled 5-fold CV with a fresh model per fold; mean ± std of 9 metrics.',
    f'• Best model chosen by 5-fold CV R², then re-fitted on all 383 samples for SHAP and optimisation. **Selected: {best_pretty}.**'])

# ---------------------------------------------------------------------------
# Slide 6 - Review 1 feedback map
# ---------------------------------------------------------------------------
s = new_slide('REVIEW 1 FEEDBACK', 'How the Professor\'s Review 1 Feedback Was Addressed')
rows = [['#', 'Feedback', 'What we implemented', 'Evidence (slides)'],
        ['1', 'All regression metrics', 'MSE, RMSE, MAE, MAPE, R², Adjusted R², Explained Variance, Max Error, Median AE for 9 models; hold-out + 5-fold CV (mean ± std); normalised errors for comparison with the paper', '7, 8, 10, 11'],
        ['2', 'Explainability of a prediction', 'SHAP (exact TreeExplainer) in MPa for both strengths; additivity verified; waterfalls for the optimum, case-study recipes and the weakest test specimen', '12, 13'],
        ['3', 'Optimal parameters', 'Robust differential evolution (worst case within ±5 units), cross-checked by exhaustive search of 960 tested combinations; sensitivity analysis, constrained case studies, Pareto front', '14, 15, 16']]
table(s, MARGIN, CONTENT_TOP, CONTENT_W, rows, col_widths=[0.4, 2.2, 7.5, 1.6], font=12, row_h=Inches(1.0))
banner(s, CONTENT_TOP + Inches(4.2), 'Also fixed: search window now matches the data, model-reuse bug in CV, corrected paper values & citation', 'amber')

# ---------------------------------------------------------------------------
# Slide 7 - Metrics results
# ---------------------------------------------------------------------------
s = new_slide('FEEDBACK #1 · ALL METRICS', 'Model Performance: Hold-out R² of All Nine Models')
picture(s, os.path.join(FIG, 'metrics_r2_comparison.png'), MARGIN, CONTENT_TOP, Inches(7.4), CONTENT_H)
rows = [['Model', 'CV R²', 'CV RMSE']]
for m, r in ranking.iterrows():
    rows.append([NICE[m], f'{r.CV_R2:.4f}', f'{r.CV_RMSE:.2f}'])
table(s, MARGIN + Inches(7.6), CONTENT_TOP + Inches(0.1), CONTENT_W - Inches(7.6), rows,
      col_widths=[3.4, 1.2, 1.2], font=10, row_h=Inches(0.42), highlight_rows=(1,))
text(s, MARGIN + Inches(7.6), CONTENT_TOP + Inches(4.4), CONTENT_W - Inches(7.6), Inches(0.6),
     [('Mean of both targets, 5-fold CV. RMSE in MPa.', 10, MUTED, False)])

# ---------------------------------------------------------------------------
# Slide 8 - Full metrics
# ---------------------------------------------------------------------------
s = new_slide('FEEDBACK #1 · ALL METRICS', 'All 9 Metrics & Cross-Validation Stability')
picture(s, os.path.join(FIG, 'metrics_top3_complete.png'), MARGIN, CONTENT_TOP, CONTENT_W, Inches(1.9))
half = (CONTENT_W - Inches(0.3)) // 2
picture(s, os.path.join(FIG, 'metrics_error_heatmap.png'), MARGIN, CONTENT_TOP + Inches(2.0), half, CONTENT_H - Inches(2.0))
picture(s, os.path.join(FIG, 'metrics_cv_r2_stability.png'), MARGIN + half + Inches(0.3), CONTENT_TOP + Inches(2.0), half, CONTENT_H - Inches(2.0))

# ---------------------------------------------------------------------------
# Slide 9 - Parity
# ---------------------------------------------------------------------------
s = new_slide('BEST MODEL', f'{best_pretty}: Predicted vs Measured (Hold-out)')
picture(s, os.path.join(FIG, 'parity_plot_true_vs_pred.png'), MARGIN, CONTENT_TOP, Inches(7.9), CONTENT_H)
card(s, MARGIN + Inches(8.1), CONTENT_TOP, CONTENT_W - Inches(8.1), CONTENT_H, 'Best model accuracy', [
    f'• Hold-out R²: **{b_ho[T]["R2"]:.4f} / {b_ho[C]["R2"]:.4f}**',
    f'• Hold-out RMSE: {b_ho[T]["RMSE"]:.2f} / {b_ho[C]["RMSE"]:.2f} MPa',
    f'• 5-fold CV R²: {b_cv[T]["R2_Mean"]:.4f} ± {b_cv[T]["R2_Std"]:.4f}',
    f'• CV MAPE: {b_cv[T]["MAPE_%_Mean"]:.2f} % / {b_cv[C]["MAPE_%_Mean"]:.2f} %',
    f'• Max error: {b_ho[T]["Max_Error"]:.2f} / {b_ho[C]["Max_Error"]:.2f} MPa',
    '• All test samples lie inside the ±10 % band.',
    '(tensile / compressive)'], size=16)

# ---------------------------------------------------------------------------
# Slide 10 - Comparison R2 (Table 7)
# ---------------------------------------------------------------------------
s = new_slide('⭐ COMPARISON WITH PUBLISHED WORK', 'R² Comparison with Table 7 of Munshi et al.')
paper = t7[~t7.Evaluation.str.startswith('This work')]
ours = t7[t7.Evaluation.str.contains('hold-out')]
rows = [['Model', 'Source', 'Tensile R²', 'Compressive R²']]
for _, r in paper.iterrows():
    rows.append([r.Model.replace(' (literature, Table 7)', '').replace(' (Paper)', ''),
                 'Paper (L15 validation)' if 'Munshi' in r.Evaluation else 'Paper Table 7 (other studies)',
                 f'{r.Tensile_R2:g}', f'{r.Compressive_R2:g}'])
for key in ['Adam-optimised ANN (our replication)', 'Bayesian-regularised ANN (our replication)',
            'Histogram Gradient Boosting (proposed)', 'Tuned Random Forest (proposed)', 'XGBoost (proposed)']:
    r = ours[ours.Model == key].iloc[0]
    rows.append([key.split(' (')[0], 'This work (hold-out)', f'{r.Tensile_R2:.4f}', f'{r.Compressive_R2:.4f}'])
hl = tuple(i for i, r in enumerate(rows) if r[0].startswith('Histogram'))
table(s, MARGIN, CONTENT_TOP, Inches(8.2), rows, col_widths=[3.0, 3.0, 1.3, 1.5], font=10, row_h=Inches(0.33),
      highlight_rows=hl)
card(s, MARGIN + Inches(8.4), CONTENT_TOP, CONTENT_W - Inches(8.4), CONTENT_H, 'Reading the comparison', [
    f'• Our **Adam-ANN replication** ({adam_t:.3f} / {adam_c:.3f}) reaches the same level as the paper (0.93 / 0.98).',
    f'• **{best_pretty}** reaches {ho(best, T, "R2"):.4f}, above every R² in Table 7.',
    '• The paper\'s DT/SVM/RF/XGB/SVR/k-NN/AdaBoost rows come from **other studies and datasets**.',
    '• The paper\'s ANN R² is on its 15 Taguchi L15 prints; ours is on the 77 held-out records.'], size=13)

# ---------------------------------------------------------------------------
# Slide 11 - Comparison errors (Table 4)
# ---------------------------------------------------------------------------
s = new_slide('⭐ COMPARISON WITH PUBLISHED WORK', 'Error Comparison with Table 4 of Munshi et al.')
rows = [['Model', 'Validation', 'Property', 'MSE', 'RMSE', 'MAE', 'MAPE %', 'RMSE (norm.)']]
sel = t4[(t4.Algorithm.str.contains('Paper')) |
         (t4.Algorithm.str.contains('Histogram')) |
         ((t4.Algorithm.str.contains('Adam')) & (t4.Validation == 'Hold-out')) |
         ((t4.Algorithm.str.contains('Bayesian')) & (t4.Validation == '5-Fold CV'))]
for _, r in sel.iterrows():
    name = r.Algorithm.replace('-optimised', '').replace('-regularised', '').replace(' (our replication)', ' (ours)') \
        .replace('Histogram Gradient Boosting (proposed)', 'HGB (ours)')
    rows.append([name, r.Validation, r.Property.replace(' Strength', ''), f'{r.MSE:.4f}', f'{r.RMSE:.4f}',
                 f'{r.MAE:.4f}', f'{r["MAPE_%"]:.3f}', '–' if pd.isna(r.RMSE_norm) else f'{r.RMSE_norm:.4f}'])
hl = tuple(i for i, r in enumerate(rows) if r[0].startswith('HGB'))
table(s, MARGIN, CONTENT_TOP, Inches(8.6), rows, col_widths=[2.6, 1.3, 1.4, 1, 1, 1, 1, 1.2], font=9,
      row_h=Inches(0.27), highlight_rows=hl)
card(s, MARGIN + Inches(8.8), CONTENT_TOP, CONTENT_W - Inches(8.8), CONTENT_H, 'Notes', [
    '• Our values are in MPa; "norm." is on min-max scaled targets (the paper\'s values appear to be normalised).',
    f'• HGB normalised RMSE ≈ {cvm(best, T, "RMSE_norm"):.4f}, lower than all Adam-ANN values in the paper.',
    '• The paper\'s values are internally inconsistent (e.g. MSE 1.75 with RMSE 0.0125), so they are compared with caution.',
    '• Our 5-fold CV is stable for every model (R² std ≤ 0.006); the paper reports MAE 5.5 for its Bayesian 5-fold run.'], size=12)

# ---------------------------------------------------------------------------
# Slide 12 - SHAP global
# ---------------------------------------------------------------------------
s = new_slide('FEEDBACK #2 · EXPLAINABILITY', 'SHAP: How Each Parameter Moves the Prediction (MPa)')
picture(s, os.path.join(FIG, 'shap_summary_tensile.png'), MARGIN, CONTENT_TOP, Inches(7.0), CONTENT_H)
rows = [['Parameter', 'Mean |SHAP| (MPa)', 'Share']]
for p in sg.sort_values('Share_%', ascending=False).index:
    rows.append([p, f'{sg.loc[p, "Mean_Abs_SHAP_MPa"]:.2f}', f'{sg.loc[p, "Share_%"]:.1f} %'])
table(s, MARGIN + Inches(7.2), CONTENT_TOP, CONTENT_W - Inches(7.2), rows, col_widths=[2.4, 1.6, 1.0], font=10.5,
      row_h=Inches(0.36))
card(s, MARGIN + Inches(7.2), CONTENT_TOP + Inches(2.35), CONTENT_W - Inches(7.2), CONTENT_H - Inches(2.35),
     'Tensile strength (compressive is identical in share)', [
         '• High infill: up to +19 MPa; low infill: down to −17 MPa.',
         '• 0.2 mm layers: about +2.5 to +5 MPa; 0.8 mm layers: about −2 to −5 MPa.',
         '• Speed above about 40 mm/s lowers strength.',
         '• Bed temperature has no effect, as in the paper\'s conclusion.'], size=13)

# ---------------------------------------------------------------------------
# Slide 13 - SHAP local
# ---------------------------------------------------------------------------
s = new_slide('FEEDBACK #2 · EXPLAINABILITY', 'Explaining ONE Prediction: Why Does This Recipe Give This Strength?')
half = (CONTENT_W - Inches(0.3)) // 2
picture(s, os.path.join(FIG, 'shap_waterfall_tensile_optimum.png'), MARGIN, CONTENT_TOP, half, Inches(3.4))
picture(s, os.path.join(FIG, 'shap_waterfall_compressive_brake_pedal.png'), MARGIN + half + Inches(0.3), CONTENT_TOP, half, Inches(3.4))
o = shap_l[(shap_l.Case == 'Composite optimum') & (shap_l.Target == 'Tensile')].iloc[0]
eq = (f'{o.Base_Value_MPa:.2f} (dataset average) + {o.SHAP_Infill_Density_percent_MPa:+.2f} (infill) '
      f'{o.SHAP_Layer_Height_mm_MPa:+.2f} (layer) {o.SHAP_Print_Speed_mm_per_s_MPa:+.2f} (speed) '
      f'{o.SHAP_Nozzle_Temp_C_MPa:+.2f} (nozzle) {o.SHAP_Bed_Temp_C_MPa:+.2f} (bed) = **{o.Prediction_MPa:.2f} MPa**')
card(s, MARGIN, CONTENT_TOP + Inches(3.55), CONTENT_W, CONTENT_H - Inches(3.55), 'Exact decomposition of the optimal recipe\'s tensile strength', [
    eq,
    f'• Additivity verified for all 383 samples (max error {summary["shap_additivity_max_error_MPa"]:.1e} MPa). '
    'Waterfalls are also provided for the door handle and the weakest test specimen.'], size=16)

# ---------------------------------------------------------------------------
# Slide 14 - Optimum
# ---------------------------------------------------------------------------
s = new_slide('FEEDBACK #3 · OPTIMAL PARAMETERS', 'Formal Optimisation: Robust Differential Evolution')
rows = [['Objective', 'Nozzle °C', 'Bed °C', 'Speed mm/s', 'Layer mm', 'Infill %', 'σt MPa', 'σc MPa', 'Print-time τ']]
for _, r in opt.iterrows():
    rows.append([f'Max {r.Objective.lower()}', f'{r.Nozzle_Temp_C:g}', f'{r.Bed_Temp_C:g}', f'{r.Print_Speed_mm_per_s:g}',
                 f'{r.Layer_Height_mm:g}', f'{r.Infill_Density_percent:g}', f'{r.Predicted_Tensile_MPa:.2f}',
                 f'{r.Predicted_Compressive_MPa:.2f}', f'{r.Print_Time_Index:.1f}'])
g = [float(v) for v in comp.Best_Tested_Combination.split(',')]
rows.append(['Exhaustive check (960 tested combos)', f'{g[0]:g}', f'{g[1]:g}', f'{g[2]:g}', f'{g[3]:g}', f'{g[4]:g}',
             f'{comp.Best_Tested_WorstCase_Tensile_MPa:.2f}*', f'{comp.Best_Tested_WorstCase_Compressive_MPa:.2f}*',
             f'{1000 * (0.3 + 0.7 * g[4] / 100) / (g[2] * g[3]):.1f}'])
table(s, MARGIN, CONTENT_TOP, CONTENT_W, rows, col_widths=[3.0, 1, 1, 1.1, 1, 1, 1, 1, 1.1], font=11, row_h=Inches(0.4))
w3 = (CONTENT_W - Inches(0.4)) // 3
y2 = CONTENT_TOP + Inches(2.2)
h2 = CONTENT_H - Inches(2.2)
card(s, MARGIN, y2, w3, h2, 'Method', [
    '• scipy differential_evolution, 4 continuous parameters × 2 layer levels.',
    '• **Robust**: each recipe is scored by its worst case within ±5 °C, ±5 mm/s, ±5 % infill.',
    '• Ties broken by shortest print time; rounded to machine resolution.'], size=13)
card(s, MARGIN + w3 + Inches(0.2), y2, w3, h2, 'Result', [
    f'• **Layer 0.2 mm, infill ≥ 95 %, speed ≤ 35 mm/s**',
    f'• Worst case: {comp.WorstCase_Tensile_MPa:.2f} / {comp.WorstCase_Compressive_MPa:.2f} MPa',
    '• All 3 objectives give the same optimum (σc ≈ 1.25·σt).',
    '• Same strength as the best tested combination, with a shorter print time.'], title_color=BLUE, size=13)
card(s, MARGIN + 2 * (w3 + Inches(0.2)), y2, w3, h2, 'Why robust?', [
    '• Tree models are step functions; the steps lie half-way between tested levels.',
    '• A naive optimum sits on a step edge, where a 1 mm/s drift loses 4 MPa.',
    '• The robust optimum keeps its strength under process scatter.'], size=13)
text(s, MARGIN, SLIDE_H - Inches(0.45), Inches(8), Inches(0.3), [('* worst-case strengths within the tolerance band (MPa); optimiser rows show nominal predictions', 10, MUTED, False)])

# ---------------------------------------------------------------------------
# Slide 15 - Sensitivity
# ---------------------------------------------------------------------------
s = new_slide('FEEDBACK #3 · SENSITIVITY', 'Sensitivity Analysis around the Optimum')
picture(s, os.path.join(FIG, 'optimization_sensitivity.png'), MARGIN, CONTENT_TOP, Inches(7.3), Inches(2.9))
picture(s, os.path.join(FIG, 'sensitivity_response_curves.png'), MARGIN, CONTENT_TOP + Inches(3.0), Inches(7.3), CONTENT_H - Inches(3.0))
rows = [['Parameter', 'Full-range σt swing', 'Near-optimal window']]
for _, r in sens.sort_values('FullRange_Tensile_Swing_MPa', ascending=False).iterrows():
    rows.append([r.Parameter, f'{r.FullRange_Tensile_Swing_MPa:.2f} MPa', r['Near_Optimal_Window_(>=99%_composite)']])
table(s, MARGIN + Inches(7.5), CONTENT_TOP, CONTENT_W - Inches(7.5), rows, col_widths=[2.0, 1.5, 1.7], font=10,
      row_h=Inches(0.42))
card(s, MARGIN + Inches(7.5), CONTENT_TOP + Inches(2.75), CONTENT_W - Inches(7.5), CONTENT_H - Inches(2.75),
     'Engineering reading', [
         '• Very sensitive: infill, layer height.',
         '• Moderate: print speed (keep ≤ 35–40 mm/s).',
         '• Insensitive: nozzle 215–245 °C; bed has no effect, so choose it for warp control.'], size=14)

# ---------------------------------------------------------------------------
# Slide 16 - Case studies & Pareto
# ---------------------------------------------------------------------------
s = new_slide('AUTOMOTIVE CASE STUDIES', 'Fastest Recipes that Meet the Requirement + Pareto Front')
picture(s, os.path.join(FIG, 'pareto_frontier_automotive.png'), MARGIN, CONTENT_TOP, Inches(6.4), CONTENT_H)
rows = [['', 'Brake pedal', 'Door handle'],
        ['Requirement', 'σc ≥ 45 MPa', 'σt ≥ 30 MPa'],
        ['Safety margin (2×CV RMSE)', f'+{pedal.Safety_Margin_MPa:.2f} MPa', f'+{handle.Safety_Margin_MPa:.2f} MPa'],
        ['Nozzle / Bed (°C)', f'{pedal.Nozzle_Temp_C:g} / {pedal.Bed_Temp_C:g}', f'{handle.Nozzle_Temp_C:g} / {handle.Bed_Temp_C:g}'],
        ['Speed (mm/s)', f'{pedal.Print_Speed_mm_per_s:g}', f'{handle.Print_Speed_mm_per_s:g}'],
        ['Layer (mm) / Infill (%)', f'{pedal.Layer_Height_mm:g} / {pedal.Infill_Density_percent:g}', f'{handle.Layer_Height_mm:g} / {handle.Infill_Density_percent:g}'],
        ['Predicted σt / σc (MPa)', f'{pedal.Predicted_Tensile_MPa:.2f} / {pedal.Predicted_Compressive_MPa:.2f}', f'{handle.Predicted_Tensile_MPa:.2f} / {handle.Predicted_Compressive_MPa:.2f}'],
        ['Worst-case key strength', f'σc = {pedal.WorstCase_Compressive_MPa:.2f}', f'σt = {handle.WorstCase_Tensile_MPa:.2f}'],
        ['Print time vs max-strength recipe', f'−{pedal["Print_Time_Saving_vs_Max_Strength_Recipe_%"]:.1f} %', f'−{handle["Print_Time_Saving_vs_Max_Strength_Recipe_%"]:.1f} %']]
table(s, MARGIN + Inches(6.6), CONTENT_TOP, CONTENT_W - Inches(6.6), rows, col_widths=[2.6, 1.6, 1.6], font=10.5,
      row_h=Inches(0.4), highlight_rows=(8,))
text(s, MARGIN + Inches(6.6), CONTENT_TOP + Inches(3.75), CONTENT_W - Inches(6.6), Inches(1.2), [
    ('Thick 0.8 mm layers with high infill meet both requirements about 4–8× faster. '
     'Switching to 0.2 mm layers is only worth it when > 43 MPa composite strength is needed.', 13, DARK, False)])

# ---------------------------------------------------------------------------
# Slide 17 - Limitations
# ---------------------------------------------------------------------------
s = new_slide('CRITICAL ASSESSMENT', 'Taguchi L15 Check & Limitations')
r6 = l15[l15.Combination == 6].iloc[0]
r13 = l15[l15.Combination == 13].iloc[0]
card(s, MARGIN, CONTENT_TOP, w2, CONTENT_H, 'Taguchi L15 combinations (paper Table 3)', [
    '• Predicted all 15 combinations; the 5 with 0.5 mm layers are flagged (0.5 mm is not in the data).',
    f'• Combination 6: predicted {r6.Predicted_Tensile_MPa:.1f} / {r6.Predicted_Compressive_MPa:.1f} MPa vs paper about 37 / 52 MPa.',
    f'• Combination 13: predicted {r13.Predicted_Tensile_MPa:.1f} / {r13.Predicted_Compressive_MPa:.1f} MPa vs paper about 35 / 47 MPa.',
    '• Same ranking (6 and 13 are the strongest), but the paper\'s own prints are about 25–30 % weaker than the compiled dataset (C/T ratio ≈ 1.4 vs 1.25).'])
card(s, MARGIN + w2 + Inches(0.3), CONTENT_TOP, w2, CONTENT_H, 'Limitations', [
    '• The dataset is nearly deterministic (σc = 1.25·σt, no bed effect), so R² ≈ 0.9995 is not the accuracy expected on new prints.',
    '• No external validation set: the L15 measurements are only shown graphically in the paper.',
    '• Values between tested levels are interpolations; robust optimisation and safety margins reduce this risk.',
    '• The print-time index is relative (assumed 30 % shell fraction).',
    '• Next step: confirm the recipes with physical ASTM D638/D695 prints.'], title_color=AMBER)

# ---------------------------------------------------------------------------
# Slide 18 - Live demo (interactive simulator)
# ---------------------------------------------------------------------------
shot = os.path.join(ROOT, 'simulator', 'simulator_screenshot.png')
if os.path.exists(shot):
    s = new_slide('LIVE DEMO', 'ABS Print Lab: Interactive Simulator of Our Model')
    picture(s, shot, MARGIN, CONTENT_TOP, Inches(7.9), CONTENT_H)
    card(s, MARGIN + Inches(8.1), CONTENT_TOP, CONTENT_W - Inches(8.1), CONTENT_H, 'What the audience can try', [
        '• Move the 5 printing parameters; the exact trained model predicts σt and σc instantly.',
        '• The animated printer shows layer height, infill and print time.',
        '• "Why this strength?" shows live Shapley values in MPa (feedback #2).',
        '• Load-test the brake pedal and the door handle with any recipe.',
        '• Run differential evolution live and watch it converge on the Pareto map (feedback #3).',
        'Open simulator/ABS_Print_Lab.html in any browser (works offline).'], size=13)

# ---------------------------------------------------------------------------
# Slide 19 - Conclusions
# ---------------------------------------------------------------------------
s = new_slide('CONCLUSIONS', 'Conclusions & Submission Checklist')
card(s, MARGIN, CONTENT_TOP, w2, CONTENT_H, 'Key conclusions', [
    f'1. The paper\'s ANNs were replicated (Adam R² ≈ {adam_t:.2f}, matching the reported 0.93 / 0.98).',
    f'2. **{best_pretty}** is the best and most stable model: CV R² {b_cv[T]["R2_Mean"]:.4f}, RMSE {b_cv[T]["RMSE_Mean"]:.2f} / {b_cv[C]["RMSE_Mean"]:.2f} MPa.',
    '3. SHAP: infill ≈ 62 %, layer ≈ 23 %, speed ≈ 11 % of the effect; bed temperature has none.',
    f'4. Robust optimum: 0.2 mm, ≥ 95 % infill, ≤ 35 mm/s, giving {comp.Predicted_Tensile_MPa:.1f} / {comp.Predicted_Compressive_MPa:.1f} MPa.',
    f'5. Brake pedal / door handle recipes print {pedal["Print_Time_Saving_vs_Max_Strength_Recipe_%"]:.0f} % / {handle["Print_Time_Saving_vs_Max_Strength_Recipe_%"]:.0f} % faster than the max-strength recipe.'])
card(s, MARGIN + w2 + Inches(0.3), CONTENT_TOP, w2, CONTENT_H, 'Submission folder', [
    '✔ Reference paper (reference_paper/)',
    '✔ Source code: src/ + run_project.py (+ notebook, README, requirements)',
    '✔ Report in research-paper format (report/Project_Report.docx / .md)',
    '✔ PowerPoint presentation (this deck) + interactive simulator (simulator/)',
    '✔ Comparison with published work (slides 10–11, report §3.3)',
    '☐ Plagiarism report (to be generated by the team)',
    '☐ AI-generated content report (to be generated by the team)'], title_color=BLUE)

prs.save(OUT)
print(f'Saved: {OUT} ({slide_no[0]} slides)')
