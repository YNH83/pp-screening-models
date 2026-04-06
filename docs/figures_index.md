# Figure Catalog

## Main Figures (Manuscript)

| Figure | Script | Content |
|--------|--------|---------|
| Fig1_cohort_overview.png | generate_all_figures.py | 4-panel: annual volume, age trends, Dx shift, demographics |
| Fig2_hormone_distributions.png | generate_all_figures.py | 5 hormones (LH, FSH, E2, IGF-1, LH/FSH) PP vs non-PP with AUC |
| Fig3_roc_classification.png | generate_all_figures.py | ROC curves: XGBoost (0.88) vs LH (0.53) vs logistic |
| Fig4_feature_importance.png | generate_all_figures.py | XGBoost importance + single-feature AUC, BA dominant |
| Fig5_forecast_comparison.png | generate_all_figures.py | Chronos vs ARIMA 12-month forecasts |
| Fig6_subclinical_window.png | generate_all_figures.py | LH useless (0.53) vs BA strong (0.79) vs multivariate (0.88) |

## Supplementary Figures

| Figure | Script | Content |
|--------|--------|---------|
| FigS1_sensitivity_controls.png | 06_sensitivity_analyses.py | ROC by control group definition |
| FigS2_bootstrap_calibration.png | 06_sensitivity_analyses.py | Bootstrap AUC distribution + calibration curve |
| FigS3_shap_importance.png | 06_sensitivity_analyses.py | Permutation importance with error bars |
| FigS4_lin28b_phewas.png | 07_lin28b_phewas.py | PheWAS Manhattan plot + temporal dissociation diagram |
| FigS5_control_gradient.png | 09_control_group_deep.py | AUC by control height-Z quartile |
| FigS6_psm_sensitivity.png | 09_control_group_deep.py | PSM-matched AUC comparison |
| FigS7_pp_internal.png | 09_control_group_deep.py | PP-internal validation (bypasses control group) |
| FigS8_loyo_cv.png | 08_leave_one_year_out_cv.py | LOYO-CV AUC per year + learning curve |

## Additional Analysis Figures

| Figure | Script | Content |
|--------|--------|---------|
| Fig7_external_validation.png | 04_external_validation.py | NHANES height-Z + transferable model ROC |
| Fig8_literature_validation.png | 05_literature_validation.py | Forest plot + diagnostic vs screening context |
| Fig9_male/female_manhattan.png | External (Plink2 GWAS) | UK Biobank Manhattan plots |
| model_comparison.png | 02_model_comparison.py | 4-panel multi-model benchmark |
| subclinical_*.png | 01_subclinical_window.py | Core prediction figures |
| forecast_*.png | 03_forecast_incidence.py | Time series forecast figures |
