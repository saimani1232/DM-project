# Reference Papers for the Literature Review
## FDM ABS Mechanical-Property Prediction & Optimisation

Bibliographic details (journal, volume, pages, DOI) were checked against the publishers' records. One entry from the previous version (Giri, Sharma & Jain, NSGA-II for FDM) could not be found in any database and was removed. Reference 3 was corrected (title, pages and DOI).

---

### 1. Base paper (benchmark)
* **Title:** Computation of tensile and compressive strengths of additively manufactured ABS material for automotive applications using ANN algorithms
* **Authors:** G.A. Munshi, V.M. Kulkarni, S. Yargatti
* **Journal:** *Next Materials* 10 (2026) 101420 (available online November 2025)
* **DOI:** 10.1016/j.nxmate.2025.101420
* **Dataset:** 383 tensile/compressive strengths of ABS compiled from peer-reviewed studies and test reports (ASTM D638/D695 specimens). The authors' own PRATI_AP prints (Taguchi L15, 15 runs) were used only for validation.
* **Key findings:** Adam-optimised ANN R² = 0.93 (tensile) / 0.98 (compressive); Bayesian-regularised ANN R² = 0.90 / 0.95 on the L15 validation. Infill density is the dominant parameter.
* **Gap addressed by this project:** no tree-ensemble models on the same data, no per-prediction explainability, and no formal optimisation of the printing parameters (listed as future work in the paper).

### 2. Sood, Ohdar & Mahapatra (2010)
* *Parametric appraisal of mechanical property of fused deposition modelling processed parts*, **Materials & Design** 31(1) 287–295. DOI 10.1016/j.matdes.2009.06.016
* Response-surface study of the effect of layer thickness, orientation, raster angle, raster width and air gap on the tensile, flexural and impact strength of FDM ABS parts.

### 3. Rayegani & Onwubolu (2014)
* *Fused deposition modelling (FDM) process parameter prediction and optimization using group method for data handling (GMDH) and differential evolution (DE)*, **Int. J. Advanced Manufacturing Technology** 73, 509–519. DOI 10.1007/s00170-014-5835-2
* Predicts FDM tensile strength with GMDH and finds the optimal parameters with differential evolution, the same optimiser family used in this project.

### 4. Alafaghani, Qattawi, Alrawi & Guzman (2017)
* *Experimental optimization of fused deposition modelling processing parameters: a design-for-manufacturing approach*, **Procedia Manufacturing** 10, 791–803. DOI 10.1016/j.promfg.2017.07.079
* Taguchi study of infill percentage and pattern, layer height and extrusion temperature on the mechanical properties and dimensional accuracy of FDM parts.

### 5. Meng, McWilliams, Jarosinski, Park, Jung, Lee & Zhang (2020)
* *Machine learning in additive manufacturing: a review*, **JOM** 72(6), 2363–2377. DOI 10.1007/s11837-020-04155-y
* Review of ML for AM process–property modelling, process monitoring and design.

### 6. Goh, Sing & Yeong (2021)
* *A review on machine learning in 3D printing: applications, potential, and challenges*, **Artificial Intelligence Review** 54, 63–94. DOI 10.1007/s10462-020-09876-9

### 7. Mohamed, Masood & Bhowmik (2015)
* *Optimization of fused deposition modeling process parameters: a review of current research and future prospects*, **Advances in Manufacturing** 3(1), 42–53. DOI 10.1007/s40436-014-0097-7

### Methods references
* Lundberg & Lee (2017), *A unified approach to interpreting model predictions*, NeurIPS 30: SHAP.
* Storn & Price (1997), *Differential evolution*, **J. Global Optimization** 11, 341–359. DOI 10.1023/A:1008202821328
* Breiman (2001), *Random forests*, **Machine Learning** 45, 5–32. DOI 10.1023/A:1010933404324
* Chen & Guestrin (2016), *XGBoost: a scalable tree boosting system*, KDD '16, 785–794. DOI 10.1145/2939672.2939785
* Ke et al. (2017), *LightGBM*, NeurIPS 30: the histogram-based boosting idea behind scikit-learn's HistGradientBoosting.
* Pedregosa et al. (2011), *Scikit-learn: machine learning in Python*, **JMLR** 12, 2825–2830.
* Deb et al. (2002), *NSGA-II*, **IEEE Trans. Evolutionary Computation** 6(2), 182–197. DOI 10.1109/4235.996017

---

## Literature review matrix (for slides / report)

| Author & year | Material & process | Method | Outputs | Limitation |
|:--|:--|:--|:--|:--|
| Sood et al. (2010) | ABS, FDM | RSM | Tensile, flexural, impact strength | Low-order polynomial models |
| Rayegani & Onwubolu (2014) | ABS, FDM | GMDH + differential evolution | Tensile-strength optimisation | Small experimental set |
| Mohamed et al. (2015) | FDM polymers | Review | Parameter–property taxonomy | Qualitative |
| Alafaghani et al. (2017) | FDM | Taguchi DoE | Strength & dimensional accuracy | No predictive ML model |
| Meng et al. (2020) | AM (metals & polymers) | ML review | ML landscape for AM | Not specific to ABS |
| Goh et al. (2021) | 3D printing | ML review | ML applications & challenges | Not specific to ABS |
| Munshi et al. (2026) – base paper | ABS, MEX/FDM | Adam-ANN, Bayesian-ANN | R² 0.93/0.98 and 0.90/0.95 (L15) | No ensembles, no per-prediction explanations, no optimisation |
| **This project** | ABS, MEX/FDM (same 383 records) | 9 models incl. HGB; SHAP; robust differential evolution | CV R² 0.9995; SHAP in MPa; robust optimum and case-study recipes | Dataset nearly deterministic; physical validation pending |
