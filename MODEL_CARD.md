# Model Card: PP Screening Models

## Script-by-Script Model and Code Source Documentation

### 01_subclinical_window.py

**Purpose**: Core PP prediction analysis comparing single-feature vs multivariate models.

| Component | Source | Citation |
|-----------|--------|----------|
| `LogisticRegression(max_iter=1000, class_weight="balanced")` | scikit-learn `sklearn.linear_model` | Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830. |
| `GradientBoostingClassifier(n_estimators=100, max_depth=3)` | scikit-learn `sklearn.ensemble` | Friedman (2001). Greedy function approximation. *Annals of Statistics*, 29(5). |
| `StandardScaler` | scikit-learn `sklearn.preprocessing` | Pedregosa et al. (2011). |
| `roc_auc_score`, `roc_curve` | scikit-learn `sklearn.metrics` | Pedregosa et al. (2011). |
| `mannwhitneyu` | SciPy `scipy.stats` | Virtanen et al. (2020). SciPy 1.0. *Nature Methods*, 17, 261-272. |
| Bone age NLP extraction | Custom regex on radiology reports | Original code for this study. |

**Hyperparameters**: XGBoost: 100 trees, max_depth=3, random_state=42. LR: max_iter=1000, balanced class weights. Missing values imputed with column median.

---

### 02_model_comparison.py

**Purpose**: Benchmark 6 models on PP classification + 4 models on time series forecasting.

| Component | Source | Citation |
|-----------|--------|----------|
| `LSTMClassifier` (custom) | PyTorch `torch.nn.LSTM` | Hochreiter & Schmidhuber (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780. |
| `TransformerClassifier` (custom) | PyTorch `torch.nn.TransformerEncoder` | Vaswani et al. (2017). Attention is all you need. *NeurIPS*. |
| `ChronosPipeline` | `amazon/chronos-t5-small` | Ansari et al. (2024). Chronos: Learning the language of time series. *arXiv:2403.07815*. |
| `LSTMForecaster` (custom) | PyTorch `torch.nn.LSTM` | Hochreiter & Schmidhuber (1997). |
| `TransformerForecaster` (custom) | PyTorch `torch.nn.TransformerEncoder` | Vaswani et al. (2017). |
| `AutoARIMA(season_length=12)` | statsforecast `StatsForecast` | Hyndman & Khandakar (2008). Automatic time series forecasting. *J Stat Software*, 27(3). |
| `GradientBoostingClassifier` | scikit-learn | Friedman (2001). |
| `LogisticRegression` | scikit-learn | Pedregosa et al. (2011). |
| `BCEWithLogitsLoss` with `pos_weight` | PyTorch | Paszke et al. (2019). PyTorch. *NeurIPS*. |

**LSTM Classifier Hyperparameters**: hidden=64, layers=2, dropout=0.3, epochs=150, lr=1e-3, Adam optimizer.

**Transformer Classifier Hyperparameters**: d_model=64, nhead=4, layers=2, dropout=0.3, epochs=150, lr=1e-3.

**LSTM Forecaster Hyperparameters**: hidden=32, layers=1, lookback=12, epochs=300, lr=1e-3, MSE loss.

**Transformer Forecaster Hyperparameters**: d_model=32, nhead=4, layers=2, lookback=12, epochs=300.

**Chronos**: `amazon/chronos-t5-small` (pretrained T5-based foundation model, ~46M params), CPU inference, float32, 20 forecast samples.

---

### 03_forecast_incidence.py

**Purpose**: 12-month ahead forecasting of disease incidence and hormone trends.

| Component | Source | Citation |
|-----------|--------|----------|
| `ChronosPipeline.from_pretrained("amazon/chronos-t5-small")` | Hugging Face Hub | Ansari et al. (2024). Chronos. *arXiv:2403.07815*. |
| `AutoARIMA(season_length=12)` | statsforecast | Hyndman & Khandakar (2008). |
| `AutoETS(season_length=12)` | statsforecast | Hyndman et al. (2002). A state space framework for automatic forecasting. *IJF*, 18(3). |

**Forecast Parameters**: horizon=12 months, Chronos: 20 Monte Carlo samples, percentiles [10, 50, 90].

---

### 04_external_validation.py

**Purpose**: Validate growth-axis signal against NHANES US population norms.

| Component | Source | Citation |
|-----------|--------|----------|
| `GradientBoostingClassifier` | scikit-learn | Friedman (2001). |
| `LogisticRegression` | scikit-learn | Pedregosa et al. (2011). |
| NHANES 2013-2014 data | CDC/NCHS | CDC/NCHS. NHANES. Available: https://www.cdc.gov/nchs/nhanes/ |
| Z-score computation | Custom (age-sex-specific from NHANES norms) | WHO Multicentre Growth Reference Study Group (2006). |
| `mannwhitneyu` | SciPy | Virtanen et al. (2020). |

**Models**: Full (age+sex+height_z+weight_z+bone_age), Transferable (age+sex+height_z+weight_z), Height-Z alone. All use XGBoost with same hyperparameters as Script 01.

---

### 05_literature_validation.py

**Purpose**: Synthesize 6 published studies (2017-2025) for context-dependent hierarchy.

| Component | Source | Citation |
|-----------|--------|----------|
| Forest plot visualization | matplotlib (custom) | Hunter (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90-95. |
| Published AUC data | Manual curation from 6 studies | Chen 2022 (*JPEM*), Oliveira 2017 (*J Pediatr*), Pan 2019 (*JMIR*), Huynh 2022 (*PLOS ONE*), Zhao 2025 (*Front Endocrinol*), Kim 2025 (*Diagnostics*). |

**No ML models trained in this script** (visualization and synthesis only).

---

### 06_sensitivity_analyses.py

**Purpose**: Bootstrap CI, calibration, permutation importance, control-group sensitivity.

| Component | Source | Citation |
|-----------|--------|----------|
| Bootstrap (1000 iterations) | NumPy | Efron & Tibshirani (1993). *An Introduction to the Bootstrap*. |
| `calibration_curve` | scikit-learn `sklearn.calibration` | Pedregosa et al. (2011). |
| `brier_score_loss` | scikit-learn `sklearn.metrics` | Brier (1950). Verification of forecasts. *Monthly Weather Review*, 78(1). |
| `permutation_importance(n_repeats=30)` | scikit-learn `sklearn.inspection` | Altmann et al. (2010). Permutation importance. *Bioinformatics*, 26(10). |
| `GradientBoostingClassifier` | scikit-learn | Friedman (2001). |

---

### 07_lin28b_phewas.py

**Purpose**: Compile GWAS Catalog associations for LIN28B locus to demonstrate temporal-dissociation pleiotropy.

| Component | Source | Citation |
|-----------|--------|----------|
| GWAS Catalog associations | NHGRI-EBI GWAS Catalog REST API | Buniello et al. (2019). The NHGRI-EBI GWAS Catalog. *NAR*, 47(D1). |
| LIN28B biology | Published literature | Viswanathan et al. (2009). Lin28 promotes transformation and is associated with advanced human malignancies. *Nature Genetics*, 41, 843-848. |
| LIN28B-let-7-puberty axis | Published literature | Ong et al. (2009). Genetic variation in LIN28B is associated with the timing of puberty. *Nature Genetics*, 41, 729-733. |
| IGF pathway role | Published literature | Cousminer et al. (2013). Genome-wide association and longitudinal analyses reveal genetic loci linking pubertal height growth, pubertal timing and childhood adiposity. *Human Molecular Genetics*, 22(13), 2735-2747. |

**No ML models trained in this script** (visualization and mechanistic synthesis).

---

### 08_leave_one_year_out_cv.py

**Purpose**: LOYO-CV (10 folds, 2015-2024) for temporal quasi-external validation.

| Component | Source | Citation |
|-----------|--------|----------|
| `GradientBoostingClassifier` | scikit-learn | Friedman (2001). |
| `LogisticRegression` | scikit-learn | Pedregosa et al. (2011). |
| Spearman correlation (temporal drift) | SciPy `scipy.stats.spearmanr` | Virtanen et al. (2020). |
| LOYO-CV design | Original methodology | Inspired by: Steyerberg et al. (2010). Assessing the performance of prediction models. *Epidemiology*, 21(1), 128-138. |
| Bootstrap AUC CI per fold | NumPy | Efron & Tibshirani (1993). |

---

### 09_control_group_deep.py

**Purpose**: Systematic decomposition of control-group bias (6 sub-analyses).

| Component | Source | Citation |
|-----------|--------|----------|
| `GradientBoostingClassifier` | scikit-learn | Friedman (2001). |
| `LogisticRegression` (PSM propensity) | scikit-learn | Rosenbaum & Rubin (1983). The central role of the propensity score. *Biometrika*, 70(1). |
| `StratifiedKFold`, `RepeatedStratifiedKFold` | scikit-learn | Pedregosa et al. (2011). |
| Nearest-neighbor matching (PSM) | Custom (caliper=0.05 on propensity) | Austin (2011). An introduction to propensity score methods. *Multivariate Behavioral Research*, 46(3), 399-424. |
| Height-Z quartile gradient | Custom design | Original methodology for this study. |
| Pure endocrine model (LH+FSH+E2 only) | scikit-learn XGBoost | Original decomposition design. |
| `mannwhitneyu` | SciPy | Virtanen et al. (2020). |

---

## Pretrained Model Downloads

| Model | Source | Download |
|-------|--------|----------|
| Chronos-T5-Small | Hugging Face | `amazon/chronos-t5-small` (auto-downloaded on first run via `ChronosPipeline.from_pretrained`) |

All other models (XGBoost, LR, LSTM, Transformer) are trained from scratch on the clinical dataset. No pretrained weights are used for classification models.

## Reproducibility Notes

- All random seeds set to 42 where applicable.
- Bootstrap uses 1000 iterations; permutation importance uses 30 repeats.
- Temporal validation split: train on 2014-2021, test on 2022-2024.
- LOYO-CV: 10 folds, each calendar year (2015-2024) held out once.
- Missing lab values imputed with training-set median.
- Features standardized with `StandardScaler` (fit on train, transform test).
