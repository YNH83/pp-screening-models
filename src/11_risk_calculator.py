"""
PP Risk Calculator: Instant prediction of precocious puberty probability.

Given a patient's first-visit clinical features, outputs the probability
of PP (E30.1) using the trained XGBoost model (AUC=0.880 on temporal
validation). Also provides a simple bone-age-only screening rule.

Usage:
  # Train and save model
  python pp_risk_calculator.py --train

  # Predict for a single patient
  python pp_risk_calculator.py --predict --age 7 --sex F --lh 0.3 --fsh 2.1 --e2 15 --igf1 280 --ba 9.5

  # Batch predict from CSV
  python pp_risk_calculator.py --batch input.csv --output predictions.csv

Models:
  - XGBoost (GradientBoostingClassifier): scikit-learn
    Friedman (2001). Annals of Statistics, 29(5), 1189-1232.
  - Logistic Regression: scikit-learn
    Pedregosa et al. (2011). JMLR, 12, 2825-2830.

Output:
  models/pp_xgboost.pkl          -- trained XGBoost model
  models/pp_logistic.pkl         -- trained Logistic Regression model
  models/pp_scaler.pkl           -- StandardScaler
  models/model_metadata.json     -- training metadata + performance
  scripts/risk_calculator_results.txt
"""
import warnings
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATA = Path("/Users/ynh83/Desktop/04062026 PP")
RAW = DATA / "raw data"
FIG = DATA / "figures"
OUT = DATA / "scripts"
MODEL_DIR = DATA / "models"

FEATURE_COLS = ["age", "sex_num", "LH", "FSH", "EstradiolE2", "IGF-1", "ba_advance"]
FEATURE_LABELS = ["Age (years)", "Sex (Female=1)", "LH (mIU/mL)", "FSH (mIU/mL)",
                  "Estradiol (pg/mL)", "IGF-1 (ng/mL)", "Bone Age Advance (years)"]


def load_and_build_features():
    """Load raw data and build feature matrix."""
    print("Loading clinical data...")
    pt = pd.read_excel(RAW / "decrypted_病人基本資料.xlsx", engine="openpyxl")
    pt["就醫日期"] = pd.to_datetime(pt["就醫日期"])
    lab = pd.read_excel(RAW / "decrypted_檢驗報告數值.xlsx", engine="openpyxl")
    lab["報到時間"] = pd.to_datetime(lab["報到時間"])
    lab["報告值"] = pd.to_numeric(lab["報告值"], errors="coerce")
    ba = pd.read_csv(RAW / "parsed_bone_age.csv", parse_dates=["執行時間"])

    # PP vs non-PP
    all_pp_ids = set(pt[pt["診斷碼"] == "E30.1"]["識別碼"].unique())
    per_pt_dx = pt.groupby("識別碼")["診斷碼"].apply(set).reset_index()
    per_pt_dx["ever_pp"] = per_pt_dx["診斷碼"].apply(lambda s: "E30.1" in s)
    all_nonpp_ids = set(per_pt_dx[~per_pt_dx["ever_pp"]]["識別碼"])

    # First-visit labs
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

    return feat, all_pp_ids, all_nonpp_ids


def train_and_save():
    """Train models, evaluate, and save to disk."""
    log = []
    def say(s=""):
        print(s); log.append(s)

    say("=" * 72)
    say("PP RISK CALCULATOR: Training and Evaluation")
    say("=" * 72)

    feat, pp_ids, nonpp_ids = load_and_build_features()
    avail = feat.dropna(subset=["LH", "age"]).copy()

    # Temporal split
    train = avail[avail["first_visit"] < "2022-01-01"]
    test = avail[avail["first_visit"] >= "2022-01-01"]

    X_train = train[FEATURE_COLS].fillna(train[FEATURE_COLS].median())
    y_train = train["is_pp"].values
    X_test = test[FEATURE_COLS].fillna(train[FEATURE_COLS].median())
    y_test = test["is_pp"].values

    # Store training medians for imputation at inference time
    train_medians = train[FEATURE_COLS].median().to_dict()

    say(f"Train: {len(X_train)} (PP={y_train.sum()}, Non-PP={len(y_train)-y_train.sum()})")
    say(f"Test:  {len(X_test)} (PP={y_test.sum()}, Non-PP={len(y_test)-y_test.sum()})")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---- XGBoost ----
    xgb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42,
        learning_rate=0.1, min_samples_leaf=20,
    )
    xgb.fit(X_train_s, y_train)
    prob_xgb = xgb.predict_proba(X_test_s)[:, 1]
    auc_xgb = roc_auc_score(y_test, prob_xgb)

    # ---- Logistic Regression ----
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    lr.fit(X_train_s, y_train)
    prob_lr = lr.predict_proba(X_test_s)[:, 1]
    auc_lr = roc_auc_score(y_test, prob_lr)

    say(f"\n--- Test Set Performance ---")
    say(f"XGBoost AUC:           {auc_xgb:.3f}")
    say(f"Logistic Reg AUC:      {auc_lr:.3f}")

    # Bootstrap CI
    boot_aucs = []
    for _ in range(1000):
        idx = np.random.choice(len(y_test), len(y_test), replace=True)
        if len(np.unique(y_test[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_test[idx], prob_xgb[idx]))
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    say(f"XGBoost 95% CI:        [{ci_lo:.3f}, {ci_hi:.3f}]")

    brier = brier_score_loss(y_test, prob_xgb)
    say(f"Brier score:           {brier:.4f}")

    # Feature importance
    say(f"\n--- Feature Importance ---")
    for col, label, imp in sorted(
        zip(FEATURE_COLS, FEATURE_LABELS, xgb.feature_importances_),
        key=lambda x: -x[2]
    ):
        say(f"  {label:<25} {imp:.3f}")

    # Optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_test, prob_xgb)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    say(f"\n--- Optimal Threshold (Youden's J) ---")
    say(f"  Threshold: {best_threshold:.3f}")
    say(f"  Sensitivity: {tpr[best_idx]:.3f}")
    say(f"  Specificity: {1 - fpr[best_idx]:.3f}")

    # Clinical operating points
    say(f"\n--- Clinical Operating Points ---")
    for thresh_name, thresh in [("High sensitivity (0.3)", 0.3),
                                  ("Balanced (Youden)", best_threshold),
                                  ("High specificity (0.7)", 0.7)]:
        pred = (prob_xgb >= thresh).astype(int)
        tp = ((pred == 1) & (y_test == 1)).sum()
        tn = ((pred == 0) & (y_test == 0)).sum()
        fp = ((pred == 1) & (y_test == 0)).sum()
        fn = ((pred == 0) & (y_test == 1)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        say(f"  {thresh_name}:")
        say(f"    Sens={sens:.3f} Spec={spec:.3f} PPV={ppv:.3f} NPV={npv:.3f}")

    # Simple screening rule: bone age advance >= 1 year
    say(f"\n--- Simple Screening Rule: BA advance >= 1 year ---")
    ba_test = X_test["ba_advance"].values
    ba_pred = (ba_test >= 1.0).astype(int)
    ba_sens = ((ba_pred == 1) & (y_test == 1)).sum() / max(y_test.sum(), 1)
    ba_spec = ((ba_pred == 0) & (y_test == 0)).sum() / max((1 - y_test).sum(), 1)
    say(f"  Sensitivity: {ba_sens:.3f}")
    say(f"  Specificity: {ba_spec:.3f}")

    # ---- Save models ----
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "pp_xgboost.pkl", "wb") as f:
        pickle.dump(xgb, f)
    with open(MODEL_DIR / "pp_logistic.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open(MODEL_DIR / "pp_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    metadata = {
        "feature_cols": FEATURE_COLS,
        "feature_labels": FEATURE_LABELS,
        "train_medians": train_medians,
        "train_n": len(X_train),
        "test_n": len(X_test),
        "auc_xgboost": round(auc_xgb, 4),
        "auc_logistic": round(auc_lr, 4),
        "auc_95ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "brier_score": round(brier, 4),
        "optimal_threshold": round(float(best_threshold), 4),
        "feature_importance": {
            col: round(float(imp), 4)
            for col, imp in zip(FEATURE_COLS, xgb.feature_importances_)
        },
    }
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    say(f"\nModels saved to {MODEL_DIR}/")

    # ---- Risk stratification figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: ROC
    ax = axes[0]
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, prob_xgb)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, prob_lr)
    ax.plot(fpr_xgb, tpr_xgb, color="#d62728", lw=2.5, label=f"XGBoost (AUC={auc_xgb:.3f})")
    ax.plot(fpr_lr, tpr_lr, color="#1f77b4", lw=1.8, label=f"Logistic (AUC={auc_lr:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="#ccc", lw=1)
    ax.plot(fpr[best_idx], tpr[best_idx], "r*", ms=15, label=f"Optimal (t={best_threshold:.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("A. ROC Curve (temporal validation)", fontweight="bold", loc="left")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # Panel B: Calibration
    ax = axes[1]
    frac_pos, mean_pred = calibration_curve(y_test, prob_xgb, n_bins=10)
    ax.plot(mean_pred, frac_pos, "s-", color="#d62728", lw=2, ms=7, label="XGBoost")
    ax.plot([0, 1], [0, 1], ":", color="#ccc", lw=1.5, label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"B. Calibration (Brier={brier:.4f})", fontweight="bold", loc="left")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # Panel C: Risk distribution
    ax = axes[2]
    ax.hist(prob_xgb[y_test == 0], bins=30, alpha=0.6, color="#1f77b4",
            density=True, label=f"Non-PP (N={int((1-y_test).sum())})")
    ax.hist(prob_xgb[y_test == 1], bins=30, alpha=0.6, color="#d62728",
            density=True, label=f"PP (N={int(y_test.sum())})")
    ax.axvline(best_threshold, color="black", ls="--", lw=1.5,
               label=f"Threshold={best_threshold:.2f}")
    ax.set_xlabel("Predicted PP probability")
    ax.set_ylabel("Density")
    ax.set_title("C. Risk score distribution", fontweight="bold", loc="left")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    plt.suptitle("PP Risk Calculator: Model Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG / "Fig_risk_calculator.png", dpi=150)
    plt.close()
    say(f"\nSaved: Fig_risk_calculator.png")

    (OUT / "risk_calculator_results.txt").write_text("\n".join(log))
    say("Training complete.")


def predict_single(age, sex, lh, fsh, e2, igf1, ba_years):
    """Predict PP probability for a single patient."""
    # Load model
    with open(MODEL_DIR / "pp_xgboost.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(MODEL_DIR / "pp_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / "model_metadata.json") as f:
        meta = json.load(f)

    medians = meta["train_medians"]
    sex_num = 1 if sex.upper() in ("F", "女") else 0
    ba_advance = ba_years - age if ba_years is not None else medians["ba_advance"]

    features = {
        "age": age if age is not None else medians["age"],
        "sex_num": sex_num,
        "LH": lh if lh is not None else medians["LH"],
        "FSH": fsh if fsh is not None else medians["FSH"],
        "EstradiolE2": e2 if e2 is not None else medians["EstradiolE2"],
        "IGF-1": igf1 if igf1 is not None else medians["IGF-1"],
        "ba_advance": ba_advance,
    }

    X = np.array([[features[c] for c in FEATURE_COLS]])
    X_s = scaler.transform(X)
    prob = xgb.predict_proba(X_s)[0, 1]

    threshold = meta["optimal_threshold"]
    risk_level = "HIGH" if prob >= 0.7 else ("MODERATE" if prob >= threshold else "LOW")

    print("=" * 60)
    print("PP RISK CALCULATOR RESULT")
    print("=" * 60)
    print(f"\nPatient Profile:")
    print(f"  Age:               {age} years")
    print(f"  Sex:               {'Female' if sex_num else 'Male'}")
    print(f"  LH:                {lh} mIU/mL")
    print(f"  FSH:               {fsh} mIU/mL")
    print(f"  Estradiol:         {e2} pg/mL")
    print(f"  IGF-1:             {igf1} ng/mL")
    print(f"  Bone Age:          {ba_years} years (advance: {ba_advance:+.1f})")
    print(f"\nPrediction:")
    print(f"  PP Probability:    {prob:.1%}")
    print(f"  Risk Level:        {risk_level}")
    print(f"  Threshold:         {threshold:.3f}")
    print(f"\nModel: XGBoost (AUC={meta['auc_xgboost']}, 95% CI [{meta['auc_95ci'][0]}, {meta['auc_95ci'][1]}])")

    # Clinical interpretation
    print(f"\nClinical Notes:")
    if ba_advance >= 1.0:
        print(f"  [!] Bone age advanced {ba_advance:.1f} years (>= 1 year threshold)")
    if lh is not None and lh > 5:
        print(f"  [!] LH elevated ({lh} > 5 mIU/mL, suggestive of central PP)")
    if prob >= threshold:
        print(f"  [!] Recommend close monitoring and consider GnRH stimulation test")
    else:
        print(f"  [ ] Routine follow-up")

    return prob, risk_level


def batch_predict(input_csv, output_csv):
    """Predict PP probability for a batch of patients from CSV."""
    with open(MODEL_DIR / "pp_xgboost.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(MODEL_DIR / "pp_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / "model_metadata.json") as f:
        meta = json.load(f)

    df = pd.read_csv(input_csv)
    medians = meta["train_medians"]

    # Map column names
    col_map = {
        "age": "age", "sex": "sex_num", "lh": "LH", "fsh": "FSH",
        "e2": "EstradiolE2", "estradiol": "EstradiolE2",
        "igf1": "IGF-1", "igf_1": "IGF-1",
        "bone_age": "ba_years", "ba": "ba_years",
    }
    df.columns = [col_map.get(c.lower(), c) for c in df.columns]

    if "sex_num" in df.columns:
        df["sex_num"] = df["sex_num"].apply(
            lambda x: 1 if str(x).upper() in ("F", "女", "1", "FEMALE") else 0
        )
    if "ba_years" in df.columns:
        df["ba_advance"] = df["ba_years"] - df["age"]

    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = medians.get(c, 0)
        df[c] = df[c].fillna(medians.get(c, 0))

    X = df[FEATURE_COLS].values
    X_s = scaler.transform(X)
    probs = xgb.predict_proba(X_s)[:, 1]

    threshold = meta["optimal_threshold"]
    df["pp_probability"] = probs
    df["risk_level"] = pd.cut(probs,
                               bins=[0, threshold, 0.7, 1.0],
                               labels=["LOW", "MODERATE", "HIGH"])

    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    print(f"  Total: {len(df)}, HIGH: {(probs >= 0.7).sum()}, "
          f"MODERATE: {((probs >= threshold) & (probs < 0.7)).sum()}, "
          f"LOW: {(probs < threshold).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PP Risk Calculator")
    parser.add_argument("--train", action="store_true", help="Train and save models")
    parser.add_argument("--predict", action="store_true", help="Predict for single patient")
    parser.add_argument("--batch", type=str, help="Batch predict from CSV file")
    parser.add_argument("--output", type=str, default="predictions.csv")

    # Single patient features
    parser.add_argument("--age", type=float)
    parser.add_argument("--sex", type=str, default="F")
    parser.add_argument("--lh", type=float)
    parser.add_argument("--fsh", type=float)
    parser.add_argument("--e2", type=float)
    parser.add_argument("--igf1", type=float)
    parser.add_argument("--ba", type=float, help="Bone age in years")

    args = parser.parse_args()

    if args.train:
        train_and_save()
    elif args.predict:
        predict_single(args.age, args.sex, args.lh, args.fsh, args.e2, args.igf1, args.ba)
    elif args.batch:
        batch_predict(args.batch, args.output)
    else:
        parser.print_help()
