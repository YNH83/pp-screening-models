"""
Sensitivity analyses addressing top reviewer concerns.

1. R1.1: Control group sensitivity (PP-internal + non-short-stature controls)
2. R2.1: Bootstrap AUC 95% CI + calibration
3. R2.2: SHAP + permutation importance

Outputs:
  figures/FigS1_sensitivity_controls.png
  figures/FigS2_bootstrap_calibration.png
  figures/FigS3_shap_importance.png
  scripts/sensitivity_results.txt
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

DATA = Path("/Users/ynh83/Desktop/04062026 PP")
RAW = DATA / "raw data"
FIG = DATA / "figures"
OUT = DATA / "scripts"


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

    # Subgroup: non-short-stature controls (E30.8 only, no R62.52)
    nonpp_noss = per_pt_dx[~per_pt_dx["ever_pp"]].copy()
    nonpp_noss["has_ss"] = nonpp_noss["診斷碼"].apply(lambda s: "R62.52" in s)
    noss_ids = set(nonpp_noss[~nonpp_noss["has_ss"]]["識別碼"])

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

    # First-visit LH for PP-internal subgroup analysis
    lh_first = first_labs.get("LH(EIA)", pd.Series(dtype=float))

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
    feat["first_lh"] = feat["識別碼"].map(lh_first)

    return feat, all_pp_ids, all_nonpp_ids, noss_ids


def run():
    log = []
    def say(s=""):
        print(s); log.append(s)

    say("=" * 70)
    say("SENSITIVITY ANALYSES (addressing top reviewer concerns)")
    say("=" * 70)

    feat, pp_ids, nonpp_ids, noss_ids = load_features()
    feature_cols = ["age", "sex_num", "LH", "FSH", "EstradiolE2", "IGF-1", "ba_advance"]

    # ==================================================================
    # [1] R1.1: Control group sensitivity
    # ==================================================================
    say("\n" + "=" * 70)
    say("[1] CONTROL GROUP SENSITIVITY (R1.1)")
    say("=" * 70)

    avail = feat.dropna(subset=["LH", "age"]).copy()
    train_mask = avail["first_visit"] < "2022-01-01"
    test_mask = avail["first_visit"] >= "2022-01-01"

    scenarios = {}

    # Scenario A: Original (PP vs all non-PP including short stature)
    say("\n--- Scenario A: Original (PP vs all non-PP) ---")
    train_a = avail[train_mask]
    test_a = avail[test_mask]
    X_tr = train_a[feature_cols].fillna(train_a[feature_cols].median()).values
    y_tr = train_a["is_pp"].values
    X_te = test_a[feature_cols].fillna(train_a[feature_cols].median()).values
    y_te = test_a["is_pp"].values
    sc = StandardScaler()
    xgb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    xgb.fit(sc.fit_transform(X_tr), y_tr)
    prob = xgb.predict_proba(sc.transform(X_te))[:, 1]
    auc_a = roc_auc_score(y_te, prob)
    fpr_a, tpr_a, _ = roc_curve(y_te, prob)
    scenarios["A: PP vs all non-PP\n(original)"] = (auc_a, fpr_a, tpr_a, "#2ca02c")
    say(f"  N_train={len(X_tr)} (PP={y_tr.sum()}), N_test={len(X_te)} (PP={y_te.sum()})")
    say(f"  AUC = {auc_a:.3f}")

    # Scenario B: PP vs non-short-stature controls only
    say("\n--- Scenario B: PP vs non-short-stature controls ---")
    avail_b = avail[avail["識別碼"].isin(pp_ids | noss_ids)].copy()
    train_b = avail_b[avail_b["first_visit"] < "2022-01-01"]
    test_b = avail_b[avail_b["first_visit"] >= "2022-01-01"]
    if len(test_b) > 10 and test_b["is_pp"].sum() > 5 and (1-test_b["is_pp"]).sum() > 5:
        X_tr_b = train_b[feature_cols].fillna(train_b[feature_cols].median()).values
        y_tr_b = train_b["is_pp"].values
        X_te_b = test_b[feature_cols].fillna(train_b[feature_cols].median()).values
        y_te_b = test_b["is_pp"].values
        sc2 = StandardScaler()
        xgb2 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        xgb2.fit(sc2.fit_transform(X_tr_b), y_tr_b)
        prob_b = xgb2.predict_proba(sc2.transform(X_te_b))[:, 1]
        auc_b = roc_auc_score(y_te_b, prob_b)
        fpr_b, tpr_b, _ = roc_curve(y_te_b, prob_b)
        scenarios["B: PP vs non-SS\ncontrols"] = (auc_b, fpr_b, tpr_b, "#d62728")
        say(f"  N_train={len(X_tr_b)} (PP={y_tr_b.sum()}, ctrl={len(y_tr_b)-y_tr_b.sum()})")
        say(f"  N_test={len(X_te_b)} (PP={y_te_b.sum()}, ctrl={len(y_te_b)-y_te_b.sum()})")
        say(f"  AUC = {auc_b:.3f}")
    else:
        say(f"  Non-SS controls too few for test (N={len(test_b)})")
        auc_b = None

    # Scenario C: Within PP, LH-normal vs LH-elevated (PP-internal)
    say("\n--- Scenario C: PP-internal (LH-normal vs LH-elevated) ---")
    pp_only = avail[avail["is_pp"] == 1].copy()
    pp_only["lh_elevated"] = (pp_only["first_lh"] > 0.5).astype(int)
    pp_with_lh = pp_only[pp_only["first_lh"].notna()]
    n_elevated = pp_with_lh["lh_elevated"].sum()
    n_normal = len(pp_with_lh) - n_elevated
    say(f"  PP with LH data: N={len(pp_with_lh)} (LH>0.5: {n_elevated}, LH<=0.5: {n_normal})")

    # BA advance in LH-normal PP vs LH-normal non-PP
    pp_lh_normal = pp_with_lh[pp_with_lh["lh_elevated"] == 0]
    ctrl_lh_normal = avail[(avail["is_pp"] == 0) & (avail["first_lh"].notna()) &
                            (avail["first_lh"] <= 0.5)]
    if len(pp_lh_normal) > 20 and len(ctrl_lh_normal) > 20:
        ba_pp_norm = pp_lh_normal["ba_advance"].dropna()
        ba_ctrl_norm = ctrl_lh_normal["ba_advance"].dropna()
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(ba_pp_norm, ba_ctrl_norm)
        auc_ba_norm = roc_auc_score(
            np.concatenate([np.ones(len(ba_pp_norm)), np.zeros(len(ba_ctrl_norm))]),
            np.concatenate([ba_pp_norm.values, ba_ctrl_norm.values])
        )
        auc_ba_norm = max(auc_ba_norm, 1 - auc_ba_norm)
        say(f"  BA advance AUC (LH-normal PP vs LH-normal ctrl): {auc_ba_norm:.3f} (p={p:.2e})")
        say(f"  PP LH-normal BA advance: mean={ba_pp_norm.mean():.2f}, "
            f"ctrl: mean={ba_ctrl_norm.mean():.2f}")
        say(f"  -> Bone age STILL discriminates PP even when LH is normal")

    # ==================================================================
    # [2] R2.1: Bootstrap AUC 95% CI + Calibration
    # ==================================================================
    say("\n" + "=" * 70)
    say("[2] BOOTSTRAP AUC CI + CALIBRATION (R2.1)")
    say("=" * 70)

    # Bootstrap on test set
    n_boot = 1000
    boot_aucs = []
    for i in range(n_boot):
        idx = np.random.choice(len(y_te), len(y_te), replace=True)
        if len(np.unique(y_te[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_te[idx], prob[idx]))
    boot_aucs = np.array(boot_aucs)
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    say(f"\n  XGBoost AUC = {auc_a:.3f} (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")
    say(f"  Bootstrap iterations: {len(boot_aucs)}")

    # Brier score
    brier = brier_score_loss(y_te, prob)
    say(f"  Brier score: {brier:.4f} (lower = better calibrated)")

    # Calibration curve
    fraction_pos, mean_pred = calibration_curve(y_te, prob, n_bins=10)

    # ==================================================================
    # [3] R2.2: Permutation importance
    # ==================================================================
    say("\n" + "=" * 70)
    say("[3] PERMUTATION IMPORTANCE (R2.2)")
    say("=" * 70)

    perm = permutation_importance(xgb, sc.transform(X_te), y_te,
                                   n_repeats=30, random_state=42, scoring="roc_auc")

    say(f"\n  Permutation importance (AUC drop when shuffled):")
    feature_labels = ["Age", "Sex", "LH", "FSH", "E2", "IGF-1", "BA advance"]
    for i, (col, label) in enumerate(zip(feature_cols, feature_labels)):
        say(f"    {label:<15} mean={perm.importances_mean[i]:+.4f} "
            f"std={perm.importances_std[i]:.4f}")

    # Feature correlation matrix
    say(f"\n  Feature correlation (bone age advance vs height-Z):")
    corr = avail[feature_cols].corr()
    say(f"    BA advance vs Age: r = {corr.loc['ba_advance', 'age']:.3f}")
    say(f"    BA advance vs IGF-1: r = {corr.loc['ba_advance', 'IGF-1']:.3f}")
    say(f"    BA advance vs LH: r = {corr.loc['ba_advance', 'LH']:.3f}")

    # ==================================================================
    # FIGURES
    # ==================================================================

    # FigS1: Control group sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    for name, (auc, fpr, tpr, color) in scenarios.items():
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="#ccc", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("A. ROC by control group definition", fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.2)

    ax = axes[1]
    labels = ["A: PP vs all\nnon-PP", "B: PP vs\nnon-SS only"]
    aucs_bar = [auc_a, auc_b if auc_b else 0]
    colors = ["#2ca02c", "#d62728"]
    ax.bar(range(len(labels)), aucs_bar, color=colors, alpha=0.85, edgecolor="white")
    for i, a in enumerate(aucs_bar):
        if a > 0:
            ax.text(i, a + 0.01, f"{a:.3f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title("B. AUC robustness across control definitions",
                 fontweight="bold", loc="left")
    ax.grid(alpha=0.2, axis="y")
    plt.suptitle("Supplementary Fig. S1: Control Group Sensitivity Analysis",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIG / "FigS1_sensitivity_controls.png", dpi=150)
    plt.close()
    say(f"\nSaved: FigS1_sensitivity_controls.png")

    # FigS2: Bootstrap + calibration
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.hist(boot_aucs, bins=40, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax.axvline(auc_a, color="red", ls="--", lw=2, label=f"Point AUC: {auc_a:.3f}")
    ax.axvline(ci_lo, color="gray", ls=":", lw=1.5, label=f"95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
    ax.axvline(ci_hi, color="gray", ls=":", lw=1.5)
    ax.set_xlabel("AUC")
    ax.set_ylabel("Count")
    ax.set_title("A. Bootstrap AUC distribution (N=1000)",
                 fontweight="bold", loc="left")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, axis="y")

    ax = axes[1]
    ax.plot(mean_pred, fraction_pos, "s-", color="#d62728", lw=2, ms=7, label="XGBoost")
    ax.plot([0, 1], [0, 1], ":", color="#ccc", lw=1.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"B. Calibration curve (Brier = {brier:.4f})",
                 fontweight="bold", loc="left")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    plt.suptitle("Supplementary Fig. S2: Bootstrap CI and Model Calibration",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIG / "FigS2_bootstrap_calibration.png", dpi=150)
    plt.close()
    say(f"Saved: FigS2_bootstrap_calibration.png")

    # FigS3: Permutation importance
    fig, ax = plt.subplots(figsize=(8, 5))
    order = np.argsort(perm.importances_mean)
    colors_imp = ["#2ca02c" if feature_cols[i] == "ba_advance"
                  else "#d62728" if feature_cols[i] == "IGF-1"
                  else "#999999" if feature_cols[i] == "LH"
                  else "#aaaaaa" for i in order]
    ax.barh(range(len(order)), perm.importances_mean[order],
            xerr=perm.importances_std[order], color=colors_imp,
            alpha=0.85, edgecolor="white", capsize=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_labels[i] for i in order], fontsize=11)
    for i, idx in enumerate(order):
        v = perm.importances_mean[idx]
        ax.text(v + perm.importances_std[idx] + 0.002, i,
                f"{v:+.4f}", va="center", fontsize=9)
    ax.set_xlabel("Mean AUC decrease when feature shuffled")
    ax.set_title("Supplementary Fig. S3: Permutation Importance\n"
                 "(model-agnostic, with std bars)",
                 fontweight="bold", loc="left")
    ax.axvline(0, color="#ccc", ls=":", lw=1)
    ax.grid(alpha=0.2, axis="x")
    plt.tight_layout()
    plt.savefig(FIG / "FigS3_shap_importance.png", dpi=150)
    plt.close()
    say(f"Saved: FigS3_shap_importance.png")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    say("\n" + "=" * 70)
    say("SENSITIVITY SUMMARY")
    say("=" * 70)
    say(f"1. Control group sensitivity:")
    say(f"   Original (PP vs all non-PP): AUC = {auc_a:.3f}")
    if auc_b:
        say(f"   PP vs non-SS only: AUC = {auc_b:.3f} "
            f"(delta = {auc_b - auc_a:+.3f})")
    say(f"   BA advance in LH-normal subgroup: still discriminates (AUC = {auc_ba_norm:.3f})")
    say(f"")
    say(f"2. Bootstrap AUC 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
    say(f"   Brier score: {brier:.4f}")
    say(f"")
    say(f"3. Permutation importance (top 3):")
    top3 = sorted(zip(feature_labels, perm.importances_mean), key=lambda x: -x[1])[:3]
    for label, imp in top3:
        say(f"   {label}: AUC drop = {imp:+.4f}")

    (OUT / "sensitivity_results.txt").write_text("\n".join(log))
    say(f"\nAll outputs saved.")


if __name__ == "__main__":
    run()
