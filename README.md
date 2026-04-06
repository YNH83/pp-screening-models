# Precocious Puberty Screening Models

**Bone age, not gonadotropins, predicts precocious puberty: convergent evidence from 234,091 individuals.**

This repository contains all prediction models and analysis code for a single-center retrospective study of pediatric precocious puberty (PP) early detection, with external validation against NHANES and UK Biobank GWAS data.

## Key Finding

Growth-axis markers (bone age advancement, height-for-age) outperform gonadotropin-axis markers (LH, FSH) for **screening** of future PP in mixed pediatric endocrine clinics, while LH remains superior for **diagnostic confirmation** after clinical referral. This context-dependent biomarker hierarchy resolves apparent contradictions in the literature.

| Marker | Screening AUC | Diagnostic AUC |
|--------|--------------|----------------|
| LH alone | 0.529 | 0.915-0.927 |
| Bone age advancement | 0.793 | N/A |
| XGBoost (7 features) | 0.880 | N/A |
| Transferable (4 features, NHANES) | 0.912 | N/A |

## Repository Structure

```
pp-screening-models/
+-- README.md                     # This file
+-- requirements.txt              # Python dependencies
+-- MODEL_CARD.md                 # Detailed model and code citations
+-- src/                          # Analysis scripts
|   +-- 01_subclinical_window.py         # Core PP prediction (LR + XGBoost)
|   +-- 02_model_comparison.py           # Multi-model benchmark (LSTM, Transformer, Chronos, XGBoost, ARIMA)
|   +-- 03_forecast_incidence.py         # Time series forecasting (Chronos + AutoARIMA/ETS)
|   +-- 04_external_validation.py        # NHANES external validation
|   +-- 05_literature_validation.py      # Literature synthesis (forest plot)
|   +-- 06_sensitivity_analyses.py       # Bootstrap CI, permutation importance, calibration
|   +-- 07_lin28b_phewas.py              # LIN28B pleiotropy evidence (GWAS Catalog)
|   +-- 08_leave_one_year_out_cv.py      # LOYO-CV temporal quasi-external validation
|   +-- 09_control_group_deep.py         # R1.1 deep sensitivity (gradient, PSM, decomposition)
+-- results/                      # Analysis output summaries
|   +-- subclinical_results.txt
|   +-- forecast_results.txt
|   +-- external_validation_results.txt
|   +-- literature_validation_results.txt
|   +-- sensitivity_results.txt
|   +-- lin28b_phewas_results.txt
|   +-- loyo_cv_results.txt
|   +-- r1_1_deep_results.txt
+-- docs/
    +-- figures_index.md          # Figure catalog
```

## Models Used

### Classification Models

| Model | Library | Version | Reference |
|-------|---------|---------|-----------|
| XGBoost (GradientBoosting) | scikit-learn | >=1.3 | Friedman (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232. Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825-2830. |
| Logistic Regression | scikit-learn | >=1.3 | Cox (1958). The regression analysis of binary sequences. *JRSS-B*, 20(2), 215-242. Pedregosa et al. (2011). |
| LSTM | PyTorch | >=2.0 | Hochreiter & Schmidhuber (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780. Paszke et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS*. |
| Transformer | PyTorch | >=2.0 | Vaswani et al. (2017). Attention is all you need. *NeurIPS*. Paszke et al. (2019). |

### Time Series Forecasting Models

| Model | Library | Version | Reference |
|-------|---------|---------|-----------|
| Chronos-T5-Small | amazon/chronos | >=1.0 | Ansari et al. (2024). Chronos: Learning the language of time series. *arXiv:2403.07815*. Hugging Face: `amazon/chronos-t5-small`. |
| AutoARIMA | statsforecast | >=1.5 | Hyndman & Khandakar (2008). Automatic time series forecasting: The forecast package for R. *J Statistical Software*, 27(3). Garza et al. (2022). StatsForecast: Lightning fast forecasting with statistical and econometric models. |
| AutoETS | statsforecast | >=1.5 | Hyndman et al. (2002). A state space framework for automatic forecasting using exponential smoothing methods. *IJF*, 18(3), 439-454. |

### Statistical Methods

| Method | Library | Reference |
|--------|---------|-----------|
| Bootstrap AUC CI | NumPy | Efron & Tibshirani (1993). *An Introduction to the Bootstrap*. Chapman & Hall. |
| Permutation Importance | scikit-learn | Breiman (2001). Random forests. *Machine Learning*, 45, 5-32. Altmann et al. (2010). Permutation importance: a corrected feature importance measure. *Bioinformatics*, 26(10), 1340-1347. |
| Mann-Whitney U | SciPy | Mann & Whitney (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Math. Stat.*, 18(1), 50-60. |
| Calibration curve / Brier | scikit-learn | Brier (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1-3. |
| Propensity Score Matching | scikit-learn (LogisticRegression) | Rosenbaum & Rubin (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55. |
| Spearman correlation | SciPy | Spearman (1904). The proof and measurement of association between two things. *Am J Psychology*, 15(1), 72-101. |

### External Data Sources

| Source | Usage | Reference |
|--------|-------|-----------|
| NHANES 2013-2014 | Population height/weight norms, Z-score computation | CDC/NCHS. National Health and Nutrition Examination Survey. [https://www.cdc.gov/nchs/nhanes/](https://www.cdc.gov/nchs/nhanes/) |
| UK Biobank | GWAS for PP-associated loci (N=228,190) | Sudlow et al. (2015). UK Biobank: An open access resource for identifying the causes of a wide range of complex diseases. *PLOS Medicine*, 12(3), e1001779. |
| GWAS Catalog | LIN28B PheWAS associations | Buniello et al. (2019). The NHGRI-EBI GWAS Catalog. *Nucleic Acids Research*, 47(D1), D1005-D1012. |
| RSNA 2017 Bone Age | Reference methodology | Halabi et al. (2019). The RSNA Pediatric Bone Age Machine Learning Challenge. *Radiology*, 290(2), 498-503. |

### Published Studies Referenced in Literature Validation

| Study | Setting | Key Finding |
|-------|---------|-------------|
| Chen 2022, *J Pediatr Endocrinol Metab* | Diagnostic (CPP vs PT, N=116) | Basal LH AUC=0.915, IGF-1 AUC=0.880 |
| Oliveira 2017, *J Pediatr (Rio)* | Screening (N=382) | BA advancement AUC=0.605 |
| Pan 2019, *JMIR Med Inform* | Diagnostic (N=1757) | XGBoost AUC=0.886 |
| Huynh 2022, *PLOS ONE* | Diagnostic (N=614) | Random Forest AUC=0.972 |
| Zhao 2025, *Front Endocrinol* | Diagnostic (CPP vs PT, N=200) | Basal LH AUC=0.927 |
| Kim 2025, *Diagnostics* | Screening (N=2464) | IGF-1 SDS AUC=0.740 |

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python >= 3.9
- pandas, numpy, openpyxl
- scikit-learn >= 1.3
- pytorch >= 2.0
- chronos (amazon/chronos-t5-small)
- statsforecast >= 1.5
- matplotlib, scipy

## Data Availability

- **Clinical data**: De-identified patient records under IRB #60073 (Taiwan). Available upon reasonable request.
- **NHANES**: Public data from CDC/NCHS ([https://www.cdc.gov/nchs/nhanes/](https://www.cdc.gov/nchs/nhanes/))
- **UK Biobank**: Restricted access (Application #1240063). See [https://www.ukbiobank.ac.uk/](https://www.ukbiobank.ac.uk/)

## Usage

Each script is self-contained and numbered for recommended execution order:

```bash
# Core prediction analysis
python src/01_subclinical_window.py

# Multi-model benchmark
python src/02_model_comparison.py

# Time series forecasting
python src/03_forecast_incidence.py

# External validation
python src/04_external_validation.py

# Literature synthesis
python src/05_literature_validation.py

# Sensitivity analyses
python src/06_sensitivity_analyses.py

# LIN28B PheWAS
python src/07_lin28b_phewas.py

# Leave-one-year-out CV
python src/08_leave_one_year_out_cv.py

# Deep control-group sensitivity
python src/09_control_group_deep.py
```

**Note**: Scripts expect decrypted data files in a `raw data/` subdirectory relative to the project root. Paths are configurable via the `DATA` constant at the top of each script.

## License

This code is provided for academic research purposes. Please cite the associated manuscript if using this code.

## Citation

> [Manuscript in preparation] Temporal dissociation at the LIN28B growth-reproduction axis defines a subclinical prediction window for precocious puberty: convergent evidence from 234,091 individuals.
