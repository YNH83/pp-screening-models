"""
Multi-model comparison for precocious puberty prediction.

Models compared:
  1. Chronos (Amazon) - pretrained time series foundation model
  2. LSTM - PyTorch recurrent neural network
  3. Transformer - PyTorch self-attention model
  4. XGBoost - gradient boosting (our existing best: AUC 0.880)
  5. Logistic Regression - interpretable baseline
  6. AutoARIMA - classical statistical baseline

Two tasks on one figure:
  Panel A: Classification (predict PP vs non-PP from first-visit features)
  Panel B: Time series forecasting (monthly disease incidence)

Output: figures/model_comparison.png
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA = Path("/Users/ynh83/Desktop/04062026 PP")
RAW = DATA / "raw data"
FIG = DATA / "figures"
RNG = np.random.default_rng(42)


# ================================================================
#  LSTM classifier
# ================================================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (batch, features) -> treat as (batch, 1, features)
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


# ================================================================
#  Transformer classifier
# ================================================================
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)  # (batch, 1, d_model)
        x = self.encoder(x)
        return self.fc(x[:, 0, :]).squeeze(-1)


def train_pytorch_model(model, X_train, y_train, epochs=100, lr=1e-3):
    """Train a PyTorch binary classifier."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)])
    )
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def predict_pytorch(model, X):
    with torch.no_grad():
        logits = model(torch.FloatTensor(X))
        return torch.sigmoid(logits).numpy()


# ================================================================
#  Chronos forecasting wrapper
# ================================================================
def chronos_forecast(series_values, horizon=12):
    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small", device_map="cpu", torch_dtype=torch.float32
    )
    context = torch.tensor(series_values.astype(np.float32)).unsqueeze(0)
    fc = pipeline.predict(context, prediction_length=horizon, num_samples=20)
    arr = fc[0].numpy()
    return np.median(arr, axis=0), np.percentile(arr, 10, axis=0), np.percentile(arr, 90, axis=0)


# ================================================================
#  LSTM forecaster
# ================================================================
class LSTMForecaster(nn.Module):
    def __init__(self, hidden=32, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_lstm_forecaster(series, lookback=12, epochs=200, lr=1e-3):
    """Train LSTM on univariate series and return trained model."""
    vals = series.astype(np.float32)
    mean_, std_ = vals.mean(), max(vals.std(), 1e-6)
    normed = (vals - mean_) / std_

    X, y = [], []
    for i in range(lookback, len(normed)):
        X.append(normed[i - lookback:i])
        y.append(normed[i])
    X = torch.FloatTensor(np.array(X)).unsqueeze(-1)
    y = torch.FloatTensor(np.array(y))

    model = LSTMForecaster(hidden=32, layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    model.eval()

    # Forecast
    last_seq = torch.FloatTensor(normed[-lookback:]).unsqueeze(0).unsqueeze(-1)
    preds = []
    with torch.no_grad():
        for _ in range(12):
            p = model(last_seq).item()
            preds.append(p * std_ + mean_)
            new = torch.FloatTensor([[[p]]])
            last_seq = torch.cat([last_seq[:, 1:, :], new], dim=1)
    return np.array(preds)


# ================================================================
#  Transformer forecaster
# ================================================================
class TransformerForecaster(nn.Module):
    def __init__(self, d_model=32, nhead=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.proj(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :]).squeeze(-1)


def train_transformer_forecaster(series, lookback=12, epochs=200, lr=1e-3):
    vals = series.astype(np.float32)
    mean_, std_ = vals.mean(), max(vals.std(), 1e-6)
    normed = (vals - mean_) / std_

    X, y = [], []
    for i in range(lookback, len(normed)):
        X.append(normed[i - lookback:i])
        y.append(normed[i])
    X = torch.FloatTensor(np.array(X)).unsqueeze(-1)
    y = torch.FloatTensor(np.array(y))

    model = TransformerForecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    model.eval()

    last_seq = torch.FloatTensor(normed[-lookback:]).unsqueeze(0).unsqueeze(-1)
    preds = []
    with torch.no_grad():
        for _ in range(12):
            p = model(last_seq).item()
            preds.append(p * std_ + mean_)
            new = torch.FloatTensor([[[p]]])
            last_seq = torch.cat([last_seq[:, 1:, :], new], dim=1)
    return np.array(preds)


# ================================================================
#  Main
# ================================================================
def run():
    print("=" * 70)
    print("MULTI-MODEL COMPARISON: PP Prediction + Time Series Forecasting")
    print("=" * 70)

    # ---- Load data for classification ----
    pt = pd.read_excel(RAW / "decrypted_病人基本資料.xlsx", engine="openpyxl")
    pt["就醫日期"] = pd.to_datetime(pt["就醫日期"])
    lab = pd.read_excel(RAW / "decrypted_檢驗報告數值.xlsx", engine="openpyxl")
    lab["報到時間"] = pd.to_datetime(lab["報到時間"])
    lab["報告值"] = pd.to_numeric(lab["報告值"], errors="coerce")
    ba = pd.read_csv(RAW / "parsed_bone_age.csv", parse_dates=["執行時間"])

    # Build feature matrix (same as subclinical_window.py)
    all_pp_ids = set(pt[pt["診斷碼"] == "E30.1"]["識別碼"].unique())
    per_pt_dx = pt.groupby("識別碼")["診斷碼"].apply(set).reset_index()
    per_pt_dx["ever_pp"] = per_pt_dx["診斷碼"].apply(lambda s: "E30.1" in s)
    all_nonpp_ids = set(per_pt_dx[~per_pt_dx["ever_pp"]]["識別碼"])

    feature_labs = ["LH(EIA)", "FSH (EIA)", "Estradiol(E2)(EIA)", "IGF-1"]
    first_labs = {}
    for item in feature_labs:
        sub = lab[(lab["檢驗項目"] == item) & lab["報告值"].notna() & (lab["報告值"] > 0)]
        first_labs[item] = sub.sort_values("報到時間").groupby("識別碼")["報告值"].first()

    pt_info = pt.groupby("識別碼").agg(age=("診斷年齡", "first"), sex=("性別", "first")).reset_index()
    pt_info["sex_num"] = (pt_info["sex"].str.strip() == "女").astype(int)

    all_ids = list(all_pp_ids | all_nonpp_ids)
    feat_df = pd.DataFrame({"識別碼": all_ids})
    feat_df = feat_df.merge(pt_info[["識別碼", "age", "sex_num"]], on="識別碼", how="left")
    for item in feature_labs:
        col = item.replace("(EIA)", "").replace(" ", "").replace("(", "").replace(")", "")
        feat_df[col] = feat_df["識別碼"].map(first_labs.get(item, {}))
    feat_df["is_pp"] = feat_df["識別碼"].isin(all_pp_ids).astype(int)

    ba_first = ba.sort_values("執行時間").groupby("識別碼").agg(first_ba=("bone_age_years", "first")).reset_index()
    feat_df = feat_df.merge(ba_first, on="識別碼", how="left")
    feat_df["ba_advance"] = feat_df["first_ba"] - feat_df["age"]

    feature_cols = ["age", "sex_num", "LH", "FSH", "EstradiolE2", "IGF-1", "ba_advance"]
    available = feat_df.dropna(subset=["LH", "age"]).copy()

    first_visit_date = pt.groupby("識別碼")["就醫日期"].min()
    available["fvd"] = available["識別碼"].map(first_visit_date)
    train_df = available[available["fvd"] < "2022-01-01"]
    test_df = available[available["fvd"] >= "2022-01-01"]

    X_train = train_df[feature_cols].fillna(train_df[feature_cols].median()).values
    y_train = train_df["is_pp"].values
    X_test = test_df[feature_cols].fillna(train_df[feature_cols].median()).values
    y_test = test_df["is_pp"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"\nClassification: Train N={len(X_train)} (PP={y_train.sum()}), "
          f"Test N={len(X_test)} (PP={y_test.sum()})")

    # ---- Classification models ----
    clf_results = {}

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train_s, y_train)
    y_prob = lr.predict_proba(X_test_s)[:, 1]
    clf_results["Logistic Reg"] = {"auc": roc_auc_score(y_test, y_prob),
                                    "fpr": None, "tpr": None, "prob": y_prob, "color": "#ff7f0e"}
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    clf_results["Logistic Reg"]["fpr"] = fpr
    clf_results["Logistic Reg"]["tpr"] = tpr

    # 2. XGBoost
    xgb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    xgb.fit(X_train_s, y_train)
    y_prob = xgb.predict_proba(X_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    clf_results["XGBoost"] = {"auc": roc_auc_score(y_test, y_prob),
                               "fpr": fpr, "tpr": tpr, "prob": y_prob, "color": "#2ca02c"}

    # 3. LSTM
    print("Training LSTM classifier...")
    lstm = LSTMClassifier(input_dim=X_train_s.shape[1])
    lstm = train_pytorch_model(lstm, X_train_s, y_train, epochs=150)
    y_prob = predict_pytorch(lstm, X_test_s)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    clf_results["LSTM"] = {"auc": roc_auc_score(y_test, y_prob),
                            "fpr": fpr, "tpr": tpr, "prob": y_prob, "color": "#9467bd"}

    # 4. Transformer
    print("Training Transformer classifier...")
    tfm = TransformerClassifier(input_dim=X_train_s.shape[1])
    tfm = train_pytorch_model(tfm, X_train_s, y_train, epochs=150)
    y_prob = predict_pytorch(tfm, X_test_s)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    clf_results["Transformer"] = {"auc": roc_auc_score(y_test, y_prob),
                                    "fpr": fpr, "tpr": tpr, "prob": y_prob, "color": "#d62728"}

    # 5. LH alone
    lh_test = X_test[:, feature_cols.index("LH")]
    auc_lh = roc_auc_score(y_test, lh_test)
    auc_lh = max(auc_lh, 1 - auc_lh)
    fpr_lh, tpr_lh, _ = roc_curve(y_test, lh_test)
    if auc_lh < 0.5:
        fpr_lh, tpr_lh, _ = roc_curve(y_test, -lh_test)
    clf_results["LH alone"] = {"auc": auc_lh, "fpr": fpr_lh, "tpr": tpr_lh,
                                 "prob": lh_test, "color": "#999999"}

    print("\n--- Classification AUCs (temporal validation) ---")
    for name, r in sorted(clf_results.items(), key=lambda x: -x[1]["auc"]):
        print(f"  {name:<18} AUC = {r['auc']:.3f}")

    # ---- Time series forecasting ----
    print("\nBuilding monthly time series for forecasting...")
    pt_ts = pt.copy()
    pt_ts["yearmonth"] = pt_ts["就醫日期"].dt.to_period("M")
    total_visits = pt_ts.groupby("yearmonth").size()
    full_idx = pd.period_range(total_visits.index.min(), total_visits.index.max(), freq="M")
    total_visits = total_visits.reindex(full_idx, fill_value=0)
    series_vals = total_visits.values.astype(np.float32)

    # Split: use last 12 months as test
    train_ts = series_vals[:-12]
    test_ts = series_vals[-12:]
    test_dates = total_visits.index[-12:].to_timestamp()

    ts_results = {}

    # 1. Chronos
    print("Running Chronos forecast...")
    chr_med, chr_lo, chr_hi = chronos_forecast(train_ts, horizon=12)
    ts_results["Chronos (FM)"] = {"pred": chr_med, "lo": chr_lo, "hi": chr_hi,
                                    "color": "#1f77b4", "style": "s-"}

    # 2. LSTM forecaster
    print("Training LSTM forecaster...")
    lstm_pred = train_lstm_forecaster(train_ts, lookback=12, epochs=300)
    ts_results["LSTM"] = {"pred": lstm_pred, "lo": None, "hi": None,
                           "color": "#9467bd", "style": "^--"}

    # 3. Transformer forecaster
    print("Training Transformer forecaster...")
    tfm_pred = train_transformer_forecaster(train_ts, lookback=12, epochs=300)
    ts_results["Transformer"] = {"pred": tfm_pred, "lo": None, "hi": None,
                                   "color": "#d62728", "style": "v--"}

    # 4. AutoARIMA
    print("Running AutoARIMA...")
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
    sf_df = pd.DataFrame({
        "unique_id": "ts",
        "ds": total_visits.index[:-12].to_timestamp(),
        "y": train_ts.astype(float),
    })
    sf = StatsForecast(models=[AutoARIMA(season_length=12)], freq="MS")
    sf.fit(sf_df)
    arima_pred = sf.predict(h=12)["AutoARIMA"].values
    ts_results["AutoARIMA"] = {"pred": arima_pred, "lo": None, "hi": None,
                                "color": "#ff7f0e", "style": "D--"}

    # Forecast accuracy (MAE on last 12 months)
    print("\n--- Forecasting MAE (last 12 months as test) ---")
    for name, r in ts_results.items():
        mae = np.mean(np.abs(r["pred"] - test_ts))
        mape = np.mean(np.abs(r["pred"] - test_ts) / np.maximum(test_ts, 1)) * 100
        print(f"  {name:<18} MAE = {mae:.1f}  MAPE = {mape:.1f}%")

    # ================================================================
    #  FIGURE: 2x2 layout, clean, no overlap
    # ================================================================
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28,
                          left=0.07, right=0.97, top=0.93, bottom=0.06)

    # --- Panel A: ROC curves ---
    ax_roc = fig.add_subplot(gs[0, 0])
    for name in ["LH alone", "Logistic Reg", "LSTM", "Transformer", "XGBoost"]:
        r = clf_results[name]
        lw = 2.5 if name == "XGBoost" else 1.8
        ax_roc.plot(r["fpr"], r["tpr"], color=r["color"], lw=lw,
                    label=f"{name} ({r['auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], ":", color="#cccccc", lw=1)
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    ax_roc.set_title("A. PP Classification (temporal validation 2022-2024)",
                     fontsize=12, fontweight="bold", loc="left")
    ax_roc.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.grid(alpha=0.2)

    # --- Panel B: AUC bar chart ---
    ax_bar = fig.add_subplot(gs[0, 1])
    sorted_clf = sorted(clf_results.items(), key=lambda x: x[1]["auc"])
    bar_names = [x[0] for x in sorted_clf]
    bar_aucs = [x[1]["auc"] for x in sorted_clf]
    bar_colors = [x[1]["color"] for x in sorted_clf]
    bars = ax_bar.barh(range(len(bar_names)), bar_aucs, color=bar_colors, alpha=0.85,
                       edgecolor="white", linewidth=0.8)
    ax_bar.set_yticks(range(len(bar_names)))
    ax_bar.set_yticklabels(bar_names, fontsize=10)
    for i, (n, a) in enumerate(zip(bar_names, bar_aucs)):
        ax_bar.text(a + 0.01, i, f"{a:.3f}", va="center", fontsize=10, fontweight="bold")
    ax_bar.set_xlim(0.4, 1.0)
    ax_bar.set_xlabel("AUC", fontsize=11)
    ax_bar.set_title("B. Model Comparison (AUC on test set)",
                     fontsize=12, fontweight="bold", loc="left")
    ax_bar.axvline(0.5, color="#cccccc", ls=":", lw=1)
    ax_bar.grid(alpha=0.2, axis="x")

    # --- Panel C: Time series forecast ---
    ax_ts = fig.add_subplot(gs[1, 0])
    # History
    hist_dates = total_visits.index[:-12].to_timestamp()
    ax_ts.plot(hist_dates, train_ts, "k-", lw=1.2, label="Observed", alpha=0.7)
    ax_ts.plot(test_dates, test_ts, "ko", ms=5, label="Actual (held out)")
    # Forecasts
    for name, r in ts_results.items():
        ax_ts.plot(test_dates, r["pred"], r["style"], color=r["color"],
                   lw=1.8, ms=5, label=name)
        if r["lo"] is not None:
            ax_ts.fill_between(test_dates, r["lo"], r["hi"],
                               alpha=0.12, color=r["color"])
    ax_ts.set_xlabel("Date", fontsize=11)
    ax_ts.set_ylabel("Monthly visits", fontsize=11)
    ax_ts.set_title("C. Visit Forecasting (12-month ahead, held-out test)",
                     fontsize=12, fontweight="bold", loc="left")
    ax_ts.legend(fontsize=8.5, loc="upper left", framealpha=0.9, ncol=2)
    ax_ts.grid(alpha=0.2)

    # --- Panel D: Forecast error bar chart ---
    ax_err = fig.add_subplot(gs[1, 1])
    err_data = []
    for name, r in ts_results.items():
        mae = np.mean(np.abs(r["pred"] - test_ts))
        mape = np.mean(np.abs(r["pred"] - test_ts) / np.maximum(test_ts, 1)) * 100
        err_data.append((name, mae, mape, r["color"]))
    err_data.sort(key=lambda x: x[1])
    err_names = [x[0] for x in err_data]
    err_maes = [x[1] for x in err_data]
    err_colors = [x[3] for x in err_data]
    bars2 = ax_err.barh(range(len(err_names)), err_maes, color=err_colors, alpha=0.85,
                        edgecolor="white", linewidth=0.8)
    ax_err.set_yticks(range(len(err_names)))
    ax_err.set_yticklabels(err_names, fontsize=10)
    for i, (n, m) in enumerate(zip(err_names, err_maes)):
        ax_err.text(m + 3, i, f"MAE={m:.0f}", va="center", fontsize=10)
    ax_err.set_xlabel("Mean Absolute Error (visits/month)", fontsize=11)
    ax_err.set_title("D. Forecast Accuracy (lower is better)",
                     fontsize=12, fontweight="bold", loc="left")
    ax_err.grid(alpha=0.2, axis="x")

    fig.suptitle("Precocious Puberty: Multi-Model Comparison\n"
                 "Foundation Models vs Classical Methods",
                 fontsize=15, fontweight="bold", y=0.98)

    plt.savefig(FIG / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: figures/model_comparison.png")

    # ---- XGBoost feature importance (consistent across all models) ----
    print("\n--- XGBoost feature importance (confirmation) ---")
    for col, imp in sorted(zip(feature_cols, xgb.feature_importances_), key=lambda x: -x[1]):
        print(f"  {col:<15} {imp:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    run()
