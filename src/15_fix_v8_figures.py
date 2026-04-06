"""
Fix all v8 figures: larger fonts, no overlap, proper spacing.
Also regenerate Fig_timesfm_comparison and Fig_trajectory_prediction.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

DATA = Path("/Users/ynh83/Desktop/04062026 PP")
RAW = DATA / "raw data"
FIG = DATA / "figures"


def load_features():
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
    return feat


# ================================================================
# FIG 1: Foundation Model Benchmark (FIXED)
# ================================================================
def fix_foundation_models():
    print("Fixing Fig_v8_foundation_models...")
    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1], wspace=0.35,
                          left=0.06, right=0.97, top=0.88, bottom=0.12)

    # Panel A: Classification AUC
    ax = fig.add_subplot(gs[0, 0])
    models = ["LH alone", "Logistic Reg.", "LSTM", "Transformer", "XGBoost"]
    aucs = [0.529, 0.857, 0.866, 0.871, 0.880]
    colors = ["#999999", "#ff7f0e", "#9467bd", "#d62728", "#2ca02c"]
    bars = ax.barh(range(len(models)), aucs, color=colors, alpha=0.85,
                   edgecolor="white", height=0.65)
    for i, a in enumerate(aucs):
        ax.text(a + 0.01, i, f"{a:.3f}", va="center", fontsize=13, fontweight="bold")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=12)
    ax.set_xlim(0.4, 0.96)
    ax.set_xlabel("AUC", fontsize=13)
    ax.axvline(0.5, color="#ccc", ls=":", lw=1)
    ax.set_title("A. PP Classification\n(temporal validation 2022-2024)",
                 fontweight="bold", loc="left", fontsize=14)
    ax.grid(alpha=0.2, axis="x")

    # Panel B: Forecast MAE
    ax = fig.add_subplot(gs[0, 1])
    series_names = ["New\npatients", "Total\nvisits", "PP\nvisits",
                    "SS\nvisits", "LH\nmean", "IGF-1\nmean"]
    tfm_pct = [41.6, 21.2, 21.8, 25.7, 40.0, 11.5]
    chr_pct = [42.4, 21.1, 17.8, 23.2, 45.0, 10.5]
    ari_pct = [42.8, 28.7, 29.3, 28.7, 30.0, 9.7]

    x = np.arange(len(series_names))
    w = 0.25
    ax.bar(x - w, tfm_pct, w, color="#e41a1c", alpha=0.85,
           label="TimesFM (498M)", edgecolor="white")
    ax.bar(x, chr_pct, w, color="#377eb8", alpha=0.85,
           label="Chronos (46M)", edgecolor="white")
    ax.bar(x + w, ari_pct, w, color="#4daf4a", alpha=0.85,
           label="AutoARIMA", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(series_names, fontsize=11)
    ax.set_ylabel("MAE (% of series mean)", fontsize=13)
    ax.set_title("B. Time Series Forecasting\n(12-month held-out test)",
                 fontweight="bold", loc="left", fontsize=14)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.2, axis="y")

    # Panel C: Model card (FIXED - wider, no overlap)
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    cell_text = [
        ["XGBoost", "Gradient boost.", "~1K", "AUC=0.88"],
        ["Chronos", "Forecast FM", "46M", "3/6 wins"],
        ["TimesFM", "Forecast FM", "498M", "1/6 wins"],
        ["AutoARIMA", "Statistical", "~100", "2/6 wins"],
        ["LSTM", "Neural net", "~50K", "AUC=0.87"],
        ["Transformer", "Neural net", "~50K", "AUC=0.87"],
    ]
    col_labels = ["Model", "Type", "Params", "Best Result"]
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.1, 2.2)
    for i in range(len(col_labels)):
        table[0, i].set_facecolor("#2E75B6")
        table[0, i].set_text_props(color="white", fontweight="bold", fontsize=12)
    for r in range(1, len(cell_text) + 1):
        for c in range(len(col_labels)):
            table[r, c].set_text_props(fontsize=11)
    ax.set_title("C. Model Summary", fontweight="bold", loc="left", fontsize=14)

    fig.suptitle("Multi-Model Benchmark: Classification and Time Series Forecasting",
                 fontsize=16, fontweight="bold")
    plt.savefig(FIG / "Fig_v8_foundation_models.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Done.")


# ================================================================
# FIG 2: Clinical Tool (FIXED)
# ================================================================
def fix_clinical_tool():
    print("Fixing Fig_v8_clinical_tool...")

    feat = load_features()
    feature_cols = ["age", "sex_num", "LH", "FSH", "EstradiolE2", "IGF-1", "ba_advance"]
    avail = feat.dropna(subset=["LH", "age"]).copy()
    train = avail[avail["first_visit"] < "2022-01-01"]
    test = avail[avail["first_visit"] >= "2022-01-01"]

    X_train = train[feature_cols].fillna(train[feature_cols].median())
    y_train = train["is_pp"].values
    X_test = test[feature_cols].fillna(train[feature_cols].median())
    y_test = test["is_pp"].values

    scaler = StandardScaler()
    xgb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    xgb.fit(scaler.fit_transform(X_train), y_train)
    prob = xgb.predict_proba(scaler.transform(X_test))[:, 1]
    auc = roc_auc_score(y_test, prob)
    fpr, tpr, thresholds = roc_curve(y_test, prob)
    best_idx = np.argmax(tpr - fpr)
    best_t = thresholds[best_idx]
    brier = brier_score_loss(y_test, prob)

    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.32,
                          left=0.06, right=0.97, top=0.87, bottom=0.12)

    # Panel A: Risk score distribution
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 35)
    ax.hist(prob[y_test == 0], bins=bins, alpha=0.6, color="#1f77b4",
            density=True, label=f"Non-PP (N={int((1-y_test).sum())})")
    ax.hist(prob[y_test == 1], bins=bins, alpha=0.6, color="#d62728",
            density=True, label=f"PP (N={int(y_test.sum())})")
    ax.axvspan(0, 0.3, alpha=0.06, color="green")
    ax.axvspan(0.3, 0.7, alpha=0.06, color="orange")
    ax.axvspan(0.7, 1.0, alpha=0.06, color="red")
    ax.axvline(best_t, color="black", ls="--", lw=2)
    ax.text(0.12, ax.get_ylim()[1] * 0.02, "LOW", fontsize=11,
            color="green", fontweight="bold", ha="center")
    ax.text(0.5, ax.get_ylim()[1] * 0.02, "MODERATE", fontsize=11,
            color="#cc7700", fontweight="bold", ha="center")
    ax.text(0.85, ax.get_ylim()[1] * 0.02, "HIGH", fontsize=11,
            color="red", fontweight="bold", ha="center")
    ax.set_xlabel("Predicted PP Probability", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("A. Risk Score Distribution",
                 fontweight="bold", loc="left", fontsize=14)
    ax.legend(fontsize=11, loc="upper center")
    ax.grid(alpha=0.2)

    # Panel B: Calibration
    ax = fig.add_subplot(gs[0, 1])
    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
    ax.plot(mean_pred, frac_pos, "s-", color="#d62728", lw=2.5, ms=8,
            label="XGBoost")
    ax.plot([0, 1], [0, 1], ":", color="#aaa", lw=1.5, label="Perfect")
    ax.set_xlabel("Predicted probability", fontsize=13)
    ax.set_ylabel("Observed frequency", fontsize=13)
    ax.set_title(f"B. Calibration (Brier = {brier:.4f})",
                 fontweight="bold", loc="left", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2)
    # ROC inset
    ax_in = ax.inset_axes([0.52, 0.08, 0.42, 0.42])
    ax_in.plot(fpr, tpr, color="#d62728", lw=2.5)
    ax_in.plot([0, 1], [0, 1], ":", color="#aaa", lw=1)
    ax_in.set_xlabel("FPR", fontsize=10)
    ax_in.set_ylabel("TPR", fontsize=10)
    ax_in.set_title(f"AUC = {auc:.3f}", fontsize=11, fontweight="bold")
    ax_in.tick_params(labelsize=9)

    # Panel C: Operating points table (FIXED - wider, full text)
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    cell_text = [
        ["High sensitivity", "0.30", "0.902", "0.616", "0.761", "0.823"],
        ["Balanced (Youden)", f"{best_t:.2f}", f"{tpr[best_idx]:.3f}",
         f"{1-fpr[best_idx]:.3f}", "0.850", "0.741"],
        ["High specificity", "0.70", "0.561", "0.934", "0.920", "0.611"],
        ["BA >= 1 year rule", "n/a", "0.701", "0.790", "n/a", "n/a"],
    ]
    col_labels = ["Operating Point", "Threshold", "Sens", "Spec", "PPV", "NPV"]
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.15, 2.5)
    for i in range(len(col_labels)):
        table[0, i].set_facecolor("#2E75B6")
        table[0, i].set_text_props(color="white", fontweight="bold", fontsize=12)
    # Highlight balanced row
    for i in range(len(col_labels)):
        table[2, i].set_facecolor("#fff3cd")
    for r in range(1, len(cell_text) + 1):
        for c in range(len(col_labels)):
            table[r, c].set_text_props(fontsize=12)
    ax.set_title("C. Clinical Operating Points (N = 920)",
                 fontweight="bold", loc="left", fontsize=14)

    fig.suptitle("PP Risk Calculator: From Model to Clinical Decision Support",
                 fontsize=16, fontweight="bold")
    plt.savefig(FIG / "Fig_v8_clinical_tool.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Done.")


# ================================================================
# FIG 3: Trajectory (FIXED)
# ================================================================
def fix_trajectory():
    print("Fixing Fig_v8_trajectory...")

    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1, 1.1], wspace=0.3,
                          left=0.06, right=0.97, top=0.87, bottom=0.10)

    # Panel A: AUC comparison
    ax = fig.add_subplot(gs[0, 0])
    models = ["Static only\n(first visit)", "Trajectory\nonly", "Static +\nTrajectory"]
    aucs = [0.873, 0.862, 0.863]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    bars = ax.bar(range(len(models)), aucs, color=colors, alpha=0.85,
                  edgecolor="white", width=0.55)
    for i, a in enumerate(aucs):
        ax.text(i, a + 0.004, f"{a:.3f}", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0.7, 0.95)
    ax.set_ylabel("AUC", fontsize=13)
    ax.axhline(0.880, color="#2ca02c", ls="--", lw=2,
               label="Original 7-feature (0.880)")
    ax.set_title("A. Cross-Sectional vs\nLongitudinal Prediction",
                 fontweight="bold", loc="left", fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.2, axis="y")

    # Panel B: Feature importance
    ax = fig.add_subplot(gs[0, 1])
    features = ["FSH span", "IGF-1 span", "BA slope", "BA adv. last",
                "sex", "IGF-1 cv", "BA accel.", "age",
                "IGF-1 first", "BA adv. first"]
    importances = [0.018, 0.024, 0.025, 0.027, 0.030, 0.032,
                   0.039, 0.067, 0.087, 0.483]
    is_traj = [True, True, True, True, False, True,
               True, False, False, False]
    bar_colors = ["#d62728" if t else "#1f77b4" for t in is_traj]

    ax.barh(range(len(features)), importances, color=bar_colors, alpha=0.85,
            edgecolor="white", height=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=12)
    for i, v in enumerate(importances):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=11)
    ax.set_xlabel("Feature Importance", fontsize=13)
    ax.set_title("B. Top 10 Features\n(blue = static, red = trajectory)",
                 fontweight="bold", loc="left", fontsize=14)
    ax.grid(alpha=0.2, axis="x")

    # Panel C: Clinical Workflow (FIXED - larger text, more spacing)
    ax = fig.add_subplot(gs[0, 2])
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 11)
    ax.axis("off")

    steps = [
        (0.5, 9.0, 9, 1.5,
         "1. First Visit",
         "Age, Sex, Height, Weight,\nBone Age X-ray",
         "#e8f4e8", "#2ca02c"),
        (0.5, 7.0, 9, 1.5,
         "2. Instant Risk Score",
         "XGBoost: PP probability (AUC = 0.88)\nRisk level: LOW / MODERATE / HIGH",
         "#fff3cd", "#ff9800"),
        (0.5, 5.0, 9, 1.5,
         "3. Follow-up (if MODERATE)",
         "LH, FSH, IGF-1 trajectory\nTimesFM / Chronos forecast",
         "#e3f2fd", "#1976d2"),
        (0.5, 3.0, 9, 1.5,
         "4. Dynamic Risk Update",
         "Re-predict with trajectory features\nat each visit",
         "#fce4ec", "#d32f2f"),
        (0.5, 1.0, 9, 1.5,
         "5. Decision",
         "HIGH -> GnRH stim test\nLOW -> routine follow-up",
         "#f3e5f5", "#7b1fa2"),
    ]

    for x, y, w, h, title, body, fc, ec in steps:
        box = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                                        facecolor=fc, edgecolor=ec, lw=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h * 0.65, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color=ec)
        ax.text(x + w/2, y + h * 0.25, body, ha="center", va="center",
                fontsize=10, color="#333")

    for y_top, y_bot in [(9.0, 8.5), (7.0, 6.5), (5.0, 4.5), (3.0, 2.5)]:
        ax.annotate("", xy=(5, y_bot), xytext=(5, y_top),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=2))

    ax.set_title("C. Clinical Workflow",
                 fontweight="bold", loc="left", fontsize=14)

    fig.suptitle("Per-Patient Prediction: Static Risk Score + Dynamic Trajectory Monitoring",
                 fontsize=16, fontweight="bold")
    plt.savefig(FIG / "Fig_v8_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Done.")


# ================================================================
# FIG 4: TimesFM comparison (FIXED - 2x3 instead of 6x1)
# ================================================================
def fix_timesfm():
    print("Fixing Fig_timesfm_comparison...")

    # Read results from text file
    series_data = {
        "New patients": {"tfm": 10.4, "chr": 10.6, "ari": 10.7, "winner": "TimesFM"},
        "Total visits": {"tfm": 127.4, "chr": 126.3, "ari": 172.3, "winner": "Chronos"},
        "PP visits": {"tfm": 65.3, "chr": 53.4, "ari": 87.9, "winner": "Chronos"},
        "SS visits": {"tfm": 64.3, "chr": 57.9, "ari": 71.7, "winner": "Chronos"},
        "LH mean": {"tfm": 0.8, "chr": 0.9, "ari": 0.6, "winner": "AutoARIMA"},
        "IGF-1 mean": {"tfm": 28.8, "chr": 26.3, "ari": 24.2, "winner": "AutoARIMA"},
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel A: MAE comparison grouped bar
    ax = axes[0, 0]
    names = list(series_data.keys())
    tfm_vals = [d["tfm"] for d in series_data.values()]
    chr_vals = [d["chr"] for d in series_data.values()]
    ari_vals = [d["ari"] for d in series_data.values()]
    # Normalize
    max_vals = [max(t, c, a) for t, c, a in zip(tfm_vals, chr_vals, ari_vals)]
    tfm_n = [t/m*100 for t, m in zip(tfm_vals, max_vals)]
    chr_n = [c/m*100 for c, m in zip(chr_vals, max_vals)]
    ari_n = [a/m*100 for a, m in zip(ari_vals, max_vals)]

    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, tfm_n, w, color="#e41a1c", alpha=0.85, label="TimesFM (498M)")
    ax.bar(x, chr_n, w, color="#377eb8", alpha=0.85, label="Chronos (46M)")
    ax.bar(x + w, ari_n, w, color="#4daf4a", alpha=0.85, label="AutoARIMA")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11, rotation=15, ha="right")
    ax.set_ylabel("MAE (% of worst)", fontsize=13)
    ax.set_title("A. Forecast Accuracy by Series", fontweight="bold", loc="left")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2, axis="y")

    # Panel B: Win count
    ax = axes[0, 1]
    win_names = ["TimesFM\n(498M)", "Chronos\n(46M)", "AutoARIMA"]
    win_counts = [1, 3, 2]
    win_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    ax.bar(range(3), win_counts, color=win_colors, alpha=0.85,
           edgecolor="white", width=0.5)
    for i, c in enumerate(win_counts):
        ax.text(i, c + 0.08, str(c), ha="center", fontsize=16, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(win_names, fontsize=12)
    ax.set_ylabel("Series won (out of 6)", fontsize=13)
    ax.set_ylim(0, 4.5)
    ax.set_title("B. Head-to-Head Wins", fontweight="bold", loc="left")
    ax.grid(alpha=0.2, axis="y")

    # Panel C: MAE table
    ax = axes[1, 0]
    ax.axis("off")
    cell_text = [
        ["New patients", "10.4", "10.6", "10.7", "TimesFM"],
        ["Total visits", "127.4", "126.3", "172.3", "Chronos"],
        ["PP visits", "65.3", "53.4", "87.9", "Chronos"],
        ["SS visits", "64.3", "57.9", "71.7", "Chronos"],
        ["LH mean", "0.8", "0.9", "0.6", "AutoARIMA"],
        ["IGF-1 mean", "28.8", "26.3", "24.2", "AutoARIMA"],
    ]
    col_labels = ["Series", "TimesFM", "Chronos", "ARIMA", "Winner"]
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.1, 2.0)
    for i in range(len(col_labels)):
        table[0, i].set_facecolor("#2E75B6")
        table[0, i].set_text_props(color="white", fontweight="bold", fontsize=12)
    # Bold winners
    for r in range(1, 7):
        for c in range(5):
            table[r, c].set_text_props(fontsize=11)
    ax.set_title("C. MAE Comparison (12-month held-out test)",
                 fontweight="bold", loc="left")

    # Panel D: Key insight
    ax = axes[1, 1]
    ax.axis("off")
    insights = [
        "Key Findings:",
        "",
        "1. Foundation models (TimesFM, Chronos)",
        "   dominate for visit count forecasting",
        "   (non-stationary, trend-rich data)",
        "",
        "2. Classical AutoARIMA wins for",
        "   hormone means (smooth, low-noise)",
        "",
        "3. Chronos (46M) slightly outperforms",
        "   TimesFM (498M): model size does",
        "   not determine clinical utility",
        "",
        "4. All models agree on trend direction:",
        "   PP visits plateau, SS visits rising",
    ]
    for i, line in enumerate(insights):
        fw = "bold" if i == 0 or line.startswith("   ") is False and line.strip().startswith(tuple("1234")) else "normal"
        ax.text(0.05, 0.95 - i * 0.068, line, transform=ax.transAxes,
                fontsize=12, fontweight=fw, fontfamily="monospace",
                va="top")
    ax.set_title("D. Interpretation", fontweight="bold", loc="left")

    fig.suptitle("Foundation Model Comparison: TimesFM (Google, 498M) vs "
                 "Chronos (Amazon, 46M) vs AutoARIMA",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(FIG / "Fig_timesfm_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Done.")


if __name__ == "__main__":
    fix_foundation_models()
    fix_clinical_tool()
    fix_trajectory()
    fix_timesfm()
    print("\nAll figures fixed.")
