# Upgrade Plan: Professor's 3 Feedback Items for Review 2

> **Status: implemented.** This is the original plan, kept for the record. For what was finally delivered (including later corrections, e.g. the robust optimiser and the SHAP outputs in MPa), see `REVIEW1_FEEDBACK_RESPONSE.md`. The file links below point to an older folder location.

Based on the professor's Review 1 feedback, three areas of the project need to be upgraded. This plan addresses all three feedback items with specific code changes.

---

## Feedback Summary

| # | Professor's Feedback | Current State | Required Upgrade |
|:--|:---|:---|:---|
| 1 | **All regression metrics** | Only MSE, RMSE, MAE, MAPE, R² | Add Adjusted R², Explained Variance, Max Error, Median AE + visual comparison charts |
| 2 | **Explainability of predictions** | Global correlation + permutation importance only | Add **SHAP** (per-prediction explainability): waterfall, beeswarm, dependence plots |
| 3 | **Optimal parameter finding** | Monte Carlo Pareto + case studies | Add formal **scipy.optimize** single-objective optimization with sensitivity analysis |

---

## Proposed Changes

### 1. Metrics Upgrade

#### [MODIFY] [evaluation.py](file:///c:/Users/asus/Downloads/download%20(1)/ABS_FDM_Digital_Manufacturing_Project/src/evaluation.py)

Add 4 new metrics to `compute_metrics()`:
- **Adjusted R²** = `1 - (1-R²)(n-1)/(n-p-1)` where `n`=samples, `p`=features
- **Explained Variance Score** — from `sklearn.metrics.explained_variance_score`
- **Max Error** — worst-case single prediction error
- **Median Absolute Error** — robust central-tendency error metric

#### [NEW] [metrics_dashboard.py](file:///c:/Users/asus/Downloads/download%20(1)/ABS_FDM_Digital_Manufacturing_Project/src/metrics_dashboard.py)

New module that generates:
1. **Grouped bar chart**: R² comparison across all 9 models (Tensile vs Compressive side-by-side)
2. **Error metrics heatmap**: Color-coded table showing MSE/RMSE/MAE/MAPE for every model
3. **Radar/spider chart**: Multi-metric comparison for top 3 models

---

### 2. SHAP Explainability (Major Addition)

#### [NEW] [shap_explainability.py](file:///c:/Users/asus/Downloads/download%20(1)/ABS_FDM_Digital_Manufacturing_Project/src/shap_explainability.py)

New module implementing SHAP (SHapley Additive exPlanations) using the `shap` library:

1. **SHAP Summary Plot (Beeswarm)** — Shows how each feature pushes predictions higher or lower across ALL samples. This answers: *"Which features matter most and in which direction?"*

2. **SHAP Waterfall Plot** — For a specific sample prediction, shows the exact contribution of each feature. This answers: *"Why did the model predict 46 MPa for THIS specific input?"*

3. **SHAP Dependence Plots** — Shows how a single feature (e.g., Infill Density) affects predictions, colored by interaction with another feature. This answers: *"How does Infill Density interact with Layer Height?"*

4. **SHAP Bar Plot** — Mean absolute SHAP values per feature (cleaner version of feature importance)

This is the **most critical upgrade** — it directly answers the professor's question about "explainability of the result we got for a prediction."

---

### 3. Formal Optimization

#### [MODIFY] [optimizer.py](file:///c:/Users/asus/Downloads/download%20(1)/ABS_FDM_Digital_Manufacturing_Project/src/optimizer.py)

Add three new functions:

1. **`optimize_single_objective()`** — Uses `scipy.optimize.differential_evolution` to find exact optimal parameters that:
   - Maximize Tensile Strength alone
   - Maximize Compressive Strength alone
   - Maximize both (weighted composite)

2. **`sensitivity_analysis()`** — Varies each parameter ±10% around the optimal point and measures the change in predicted strength. Generates a tornado/sensitivity bar chart.

3. **`generate_optimization_report()`** — Produces a clear table:
   ```
   | Objective | Nozzle°C | Bed°C | Speed | Layer | Infill% | Predicted Tensile | Predicted Compressive |
   ```

---

### 4. Pipeline Integration

#### [MODIFY] [run_project.py](file:///c:/Users/asus/Downloads/download%20(1)/ABS_FDM_Digital_Manufacturing_Project/run_project.py)

Add 3 new pipeline steps:
- Step 7: Generate metrics dashboard visualizations
- Step 8: Run SHAP explainability analysis
- Step 9: Run formal optimization and sensitivity analysis

---

## New Dependencies

```
shap          (pip install shap)
scipy         (already available)
```

## Verification Plan

### Automated Tests
- Run `python run_project.py` end-to-end and verify all new outputs are generated in `outputs/figures/` and `outputs/tables/`

### Manual Verification
- Verify SHAP waterfall plot shows per-feature contribution that sums to the prediction
- Verify optimization results fall within parameter bounds
- Verify sensitivity analysis shows Infill Density as highest-sensitivity parameter (consistent with correlation analysis)

## Expected New Output Files

| File | Type | Feedback Item |
|:---|:---|:---|
| `outputs/figures/metrics_r2_comparison.png` | Bar chart | #1 Metrics |
| `outputs/figures/metrics_error_heatmap.png` | Heatmap | #1 Metrics |
| `outputs/figures/shap_summary_tensile.png` | Beeswarm | #2 Explainability |
| `outputs/figures/shap_summary_compressive.png` | Beeswarm | #2 Explainability |
| `outputs/figures/shap_waterfall_sample.png` | Waterfall | #2 Explainability |
| `outputs/figures/shap_dependence_infill.png` | Scatter | #2 Explainability |
| `outputs/figures/optimization_sensitivity.png` | Tornado | #3 Optimization |
| `outputs/tables/extended_metrics_holdout.csv` | CSV | #1 Metrics |
| `outputs/tables/optimal_parameters.csv` | CSV | #3 Optimization |
| `outputs/tables/sensitivity_analysis.csv` | CSV | #3 Optimization |
