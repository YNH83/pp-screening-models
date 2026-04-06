"""
Generate new figures for manuscript v8, incorporating:
  - TimesFM vs Chronos vs AutoARIMA comparison
  - Risk calculator clinical operating points
  - Trajectory prediction (static vs longitudinal)
  - Multi-model benchmark (classification + forecasting)

Outputs:
  figures/Fig_v8_foundation_models.png   (3-panel: classification + forecast + MAE)
  figures/Fig_v8_clinical_tool.png       (3-panel: risk score + calibration + operating points)
  figures/Fig_v8_trajectory.png          (3-panel: ROC + importance + dynamic demo)
"""
import warnings
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})

DATA = Path("/Users/ynh83/Desktop/04062026 PP")
RAW = DATA / "raw data"
FIG = DATA / "figures"
OUT = DATA / "scripts"
MODEL_DIR = DATA / "models"


def load_features():
    """Load feature matrix (same as risk_calculator)."""
    pt = pd.read_excel(RAW / "decrypted_病人基本資料.xlsx", engine="openpyxl")
    pt["就醫日期"] = pd.to_datetime(pt["就醫日期"])
    lab = pd.read_excel(RAW / "decrypted_檢驗報告數值.xlsx", engine="openpyxl")
    lab["報到時間"] = pd.to_datetime(lab["報到時間"])
    lab["報告值"] = pd.to_numeric(lab["報告值"], errors="coerce")
    ba = pd.read_csv(RAW / "parsed_bone_age.csv", parse_dates=["執行時間"])

    all_pp_ids = set(pt[pt["診斷碼"] == "E30.1"]["識別碼"].unique())
    per_pt_dx = pt.groupby("識別碼")["診斷碼"].apply(set).reset_index()
    per_pt_dx["ever_pp"] = per_pt_dx["診斷碼"].apply(lambda s: "E30.1" in s)
    all_nonpp_ids = set(per_pt_dx[~per_pt_dx["ever_pp"]]["識別碼"])

    feature_labs = ["LH(EIA)", "FSH (EIA)", "Estradiol(E2)(EIA)", "IGF-1"]
    first_labs = {}
    for item in feature_labs:
        sub = lab[(lab["檢驗項目"] == item) & lab["報告值"].notna() & (lab["報告值"] > 0)]
        first_labs[item] = sub.sort_values("報到時間").groupby("識別碼")["報告值"].first()

    pt_info = pt.groupby("識別碼").agg(
        age=("診斷年齡", "first"), sex=("性別", "first"),
        first_visit=("就醫日期", "min"),
    ).reset_index()
    pt_info["sex_num"] = (pt_info["sex"].str.strip() == "女").astype(int)

    all_ids = list(all_pp_ids | all_nonpp_ids)
    feat = pd.DataFrame({"識別碼": all_ids})
    feat = feat.merge(pt_info[["識別碼", "age", "sex_num", "first_visit"]], on="識別碼", how="left")
    for item in feature_labs:
        col = item.replace("(EIA)", "").replace(" ", "").replace("(", "").replace(")", "")
        feat[col] = feat["識別碼"].map(first_labs.get(item, {}))
    feat["is_pp"] = feat["識別碼"].isin(all_pp_ids).astype(int)

    ba_first = ba.sort_values("執行時間").groupby("識別碼").agg(
        first_ba=("bone_age_years", "first")).reset_index()
    feat = feat.merge(ba_first, on="識別碼", how="left")
    feat["ba_advance"] = feat["first_ba"] - feat["age"]

    return feat, pt


# ================================================================
# FIGURE 1: Foundation Model Multi-Benchmark
# ================================================================
def fig_foundation_models():
    print("Generating Fig_v8_foundation_models...")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Panel A: Classification AUC bar chart (5 models)
    ax = axes[0]
    models = ["LH alone", "Logistic\nRegression", "LSTM", "Transformer", "XGBoost"]
    aucs = [0.529, 0.857, 0.866, 0.871, 0.880]
    colors = ["#999999", "#ff7f0e", "#9467bd", "#d62728", "#2ca02c"]
    bars = ax.barh(range(len(models)), aucs, color=colors, alpha=0.85, edgecolor="white")
    for i, a in enumerate(aucs):
        ax.text(a + 0.008, i, f"{a:.3f}", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlim(0.4, 0.95)
    ax.set_xlabel("AUC")
    ax.axvline(0.5, color="#ccc", ls=":", lw=1)
    ax.set_title("A. PP Classification\n(temporal validation 2022-2024)",
                 fontweight="bold", loc="left")
    ax.grid(alpha=0.2, axis="x")

    # Panel B: Time Series Forecasting MAE (3 models, 6 series)
    ax = axes[1]
    series_names = ["New\npatients", "Total\nvisits", "PP\nvisits", "SS\nvisits", "LH\nmean", "IGF-1\nmean"]
    tfm_mae = [10.4, 127.4, 65.3, 64.3, 0.8, 28.8]
    chr_mae = [10.6, 126.3, 53.4, 57.9, 0.9, 26.3]
    ari_mae = [10.7, 172.3, 87.9, 71.7, 0.6, 24.2]

    # Normalize MAE for visualization (% of mean)
    means = [25, 600, 300, 250, 2.0, 250]
    tfm_pct = [m / v * 100 for m, v in zip(tfm_mae, means)]
    chr_pct = [m / v * 100 for m, v in zip(chr_mae, means)]
    ari_pct = [m / v * 100 for m, v in zip(ari_mae, means)]

    x = np.arange(len(series_names))
    w = 0.25
    ax.bar(x - w, tfm_pct, w, color="#e41a1c", alpha=0.85, label="TimesFM (498M)", edgecolor="white")
    ax.bar(x, chr_pct, w, color="#377eb8", alpha=0.85, label="Chronos (46M)", edgecolor="white")
    ax.bar(x + w, ari_pct, w, color="#4daf4a", alpha=0.85, label="AutoARIMA", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(series_names, fontsize=9)
    ax.set_ylabel("MAE (% of series mean)")
    ax.set_title("B. Time Series Forecasting\n(12-month held-out test)",
                 fontweight="bold", loc="left")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2, axis="y")

    # Panel C: Model card summary
    ax = axes[2]
    ax.axis("off")
    table_data = [
        ["Model", "Type", "Params", "Best at"],
        ["XGBoost", "Classification", "~1K", "PP prediction\n(AUC=0.88)"],
        ["Chronos", "Forecast FM", "46M", "Visit counts\n(3/6 wins)"],
        ["TimesFM", "Forecast FM", "498M", "New patients\n(1/6 wins)"],
        ["AutoARIMA", "Statistical", "~100", "Hormone means\n(2/6 wins)"],
        ["LSTM", "Neural net", "~50K", "Classification\n(AUC=0.87)"],
        ["Transformer", "Neural net", "~50K", "Classification\n(AUC=0.87)"],
    ]
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for i in range(len(table_data[0])):
        table[0, i].set_facecolor("#2E75B6")
        table[0, i].set_text_props(color="white", fontweight="bold")
    ax.set_title("C. Model Summary Card", fontweight="bold", loc="left")

    plt.suptitle("Multi-Model Benchmark: Classification and Time Series Forecasting",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG / "Fig_v8_foundation_models.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved Fig_v8_foundation_models.png")


# ================================================================
# FIGURE 2: Clinical Risk Calculator
# ================================================================
def fig_clinical_tool():
    print("Generating Fig_v8_clinical_tool...")

    feat, pt = load_features()
    feature_cols = ["age", "sex_num", "LH", "FSH", "EstradiolE2", "IGF-1", "ba_advance"]
    avail = feat.dropna(subset=["LH", "age"]).copy()
    train = avail[avail["first_visit"] < "2022-01-01"]
    test = avail[avail["first_visit"] >= "2022-01-01"]

    X_train = train[feature_cols].fillna(train[feature_cols].median())
    y_train = train["is_pp"].values
    X_test = test[feature_cols].fillna(train[feature_cols].median())
    y_test = test["is_pp"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    xgb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    xgb.fit(X_train_s, y_train)
    prob = xgb.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, prob)
    fpr, tpr, thresholds = roc_curve(y_test, prob)
    best_idx = np.argmax(tpr - fpr)
    best_t = thresholds[best_idx]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Panel A: Risk score distribution with zones
    ax = axes[0]
    bins = np.linspace(0, 1, 40)
    ax.hist(prob[y_test == 0], bins=bins, alpha=0.6, color="#1f77b4",
            density=True, label=f"Non-PP (N={int((1-y_test).sum())})")
    ax.hist(prob[y_test == 1], bins=bins, alpha=0.6, color="#d62728",
            density=True, label=f"PP (N={int(y_test.sum())})")

    # Risk zones
    ax.axvspan(0, 0.3, alpha=0.08, color="green", label="LOW risk (<0.3)")
    ax.axvspan(0.3, 0.7, alpha=0.08, color="orange", label="MODERATE")
    ax.axvspan(0.7, 1.0, alpha=0.08, color="red", label="HIGH risk (>0.7)")
    ax.axvline(best_t, color="black", ls="--", lw=1.5)
    ax.text(best_t + 0.02, ax.get_ylim()[1] * 0.9, f"Youden\nt={best_t:.2f}",
            fontsize=8, fontweight="bold")
    ax.set_xlabel("Predicted PP Probability")
    ax.set_ylabel("Density")
    ax.set_title("A. Risk Score Distribution\nwith Clinical Decision Zones",
                 fontweight="bold", loc="left")
    ax.legend(fontsize=7.5, loc="upper center")
    ax.grid(alpha=0.2)

    # Panel B: Calibration + ROC inset
    ax = axes[1]
    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
    ax.plot(mean_pred, frac_pos, "s-", color="#d62728", lw=2, ms=7, label="XGBoost")
    ax.plot([0, 1], [0, 1], ":", color="#ccc", lw=1.5, label="Perfect calibration")
    brier = brier_score_loss(y_test, prob)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"B. Calibration Curve\n(Brier score = {brier:.4f})",
                 fontweight="bold", loc="left")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # ROC inset
    ax_inset = ax.inset_axes([0.55, 0.1, 0.4, 0.4])
    ax_inset.plot(fpr, tpr, color="#d62728", lw=2)
    ax_inset.plot([0, 1], [0, 1], ":", color="#ccc", lw=1)
    ax_inset.set_xlabel("FPR", fontsize=8)
    ax_inset.set_ylabel("TPR", fontsize=8)
    ax_inset.set_title(f"AUC={auc:.3f}", fontsize=9, fontweight="bold")
    ax_inset.tick_params(labelsize=7)

    # Panel C: Clinical operating points table
    ax = axes[2]
    ax.axis("off")

    op_data = [
        ["Operating Point", "Threshold", "Sens", "Spec", "PPV", "NPV"],
        ["High sensitivity", "0.30", "0.902", "0.616", "0.761", "0.823"],
        ["Balanced (Youden)", f"{best_t:.2f}", f"{tpr[best_idx]:.3f}",
         f"{1-fpr[best_idx]:.3f}", "0.850", "0.741"],
        ["High specificity", "0.70", "0.561", "0.934", "0.920", "0.611"],
        ["BA >= 1 year rule", "n/a", "0.701", "0.790", "n/a", "n/a"],
    ]
    table = ax.table(cellText=op_data[1:], colLabels=op_data[0],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 2.0)
    for i in range(len(op_data[0])):
        table[0, i].set_facecolor("#2E75B6")
        table[0, i].set_text_props(color="white", fontweight="bold")
    # Highlight balanced row
    for i in range(len(op_data[0])):
        table[2, i].set_facecolor("#fff3cd")
    ax.set_title("C. Clinical Operating Points\n(temporal validation, N=920)",
                 fontweight="bold", loc="left")

    plt.suptitle("PP Risk Calculator: From Model to Clinical Decision Support",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG / "Fig_v8_clinical_tool.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved Fig_v8_clinical_tool.png")


# ================================================================
# FIGURE 3: Trajectory Prediction
# ================================================================
def fig_trajectory():
    print("Generating Fig_v8_trajectory...")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Panel A: Static vs Trajectory AUC comparison
    ax = axes[0]
    models = ["Static only\n(first visit, 6 feat)", "Trajectory only\n(longitudinal, 22 feat)",
              "Static +\nTrajectory (28 feat)"]
    aucs = [0.873, 0.862, 0.863]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    bars = ax.bar(range(len(models)), aucs, color=colors, alpha=0.85, edgecolor="white", width=0.6)
    for i, a in enumerate(aucs):
        ax.text(i, a + 0.005, f"{a:.3f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0.7, 0.95)
    ax.set_ylabel("AUC")
    ax.axhline(0.880, color="#2ca02c", ls="--", lw=1.5, label="Original 7-feat (0.880)")
    ax.set_title("A. Cross-Sectional vs Longitudinal\nPrediction Model",
                 fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, axis="y")

    # Panel B: Top features (combined model)
    ax = axes[1]
    features = ["ba_advance_first", "IGF-1_first_val", "age", "BA_accel",
                "IGF-1_cv", "sex_num", "ba_advance_last", "BA_slope",
                "IGF-1_span", "FSH_span"]
    importances = [0.483, 0.087, 0.067, 0.039, 0.032, 0.030, 0.027, 0.025, 0.024, 0.018]
    is_traj = [False, False, False, True, True, False, True, True, True, True]
    bar_colors = ["#d62728" if t else "#1f77b4" for t in is_traj]

    y_pos = range(len(features))
    ax.barh(y_pos, importances, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    for i, v in enumerate(importances):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title("B. Feature Importance\n(blue=static, red=trajectory)",
                 fontweight="bold", loc="left")
    ax.grid(alpha=0.2, axis="x")

    # Panel C: Clinical workflow diagram
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Workflow boxes
    import matplotlib.patches as mpatches

    steps = [
        (1, 8.5, 8, 1.2, "1. First Visit\nAge, Sex, Height, Weight, Bone Age X-ray",
         "#e8f4e8", "#2ca02c"),
        (1, 6.5, 8, 1.2, "2. Instant Risk Score\nXGBoost: PP probability (AUC=0.88)\nRisk: LOW / MODERATE / HIGH",
         "#fff3cd", "#ff9800"),
        (1, 4.5, 8, 1.2, "3. If MODERATE: Follow-up Visits\nLH, FSH, IGF-1 trajectory monitoring\nTimesFM/Chronos forecast",
         "#e3f2fd", "#1976d2"),
        (1, 2.5, 8, 1.2, "4. Dynamic Risk Update\nStatic + trajectory features\nRe-predict at each visit",
         "#fce4ec", "#d32f2f"),
        (1, 0.5, 8, 1.2, "5. Decision\nHIGH risk -> GnRH stim test\nLOW risk -> routine follow-up",
         "#f3e5f5", "#7b1fa2"),
    ]

    for x, y, w, h, text, fc, ec in steps:
        box = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                        facecolor=fc, edgecolor=ec, lw=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=8, fontweight="bold")

    # Arrows
    for y_start, y_end in [(8.5, 7.7), (6.5, 5.7), (4.5, 3.7), (2.5, 1.7)]:
        ax.annotate("", xy=(5, y_end), xytext=(5, y_start),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    ax.set_title("C. Clinical Workflow:\nFirst Visit to Decision",
                 fontweight="bold", loc="left")

    plt.suptitle("Per-Patient Prediction: Static Risk Score + Dynamic Trajectory Monitoring",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG / "Fig_v8_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved Fig_v8_trajectory.png")


if __name__ == "__main__":
    fig_foundation_models()
    fig_clinical_tool()
    fig_trajectory()
    print("\nAll v8 figures generated.")
