"""
Precocious Puberty disease incidence forecasting.

Uses Amazon Chronos (pretrained time series foundation model, structurally
equivalent to Google TimesFM but PyTorch-compatible on macOS) plus
StatsForecast AutoARIMA as baseline.

Data: 小兒科性早熟 clinical records (2014-11 to 2024-09, 116 months).
Targets:
  1. Monthly new patient count (新增患者數)
  2. Monthly visits by diagnosis: E30.1 (青春期過早), R62.52 (身材短小)
  3. Monthly mean LH, FSH, E2, IGF-1 levels (population trend)

Outputs:
  forecast_new_patients.png
  forecast_diagnosis_visits.png
  forecast_lab_trends.png
  forecast_results.txt
"""
import warnings
import io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings("ignore")

DATA = Path("/Users/ynh83/Desktop/04062026 PP")
FIG = DATA


def load_data():
    """Load decrypted Excel files."""
    patients = pd.read_excel(DATA / "decrypted_病人基本資料.xlsx", engine="openpyxl")
    patients["就醫日期"] = pd.to_datetime(patients["就醫日期"])
    patients["yearmonth"] = patients["就醫日期"].dt.to_period("M")

    labs = pd.read_excel(DATA / "decrypted_檢驗報告數值.xlsx", engine="openpyxl")
    labs["報到時間"] = pd.to_datetime(labs["報到時間"])
    labs["yearmonth"] = labs["報到時間"].dt.to_period("M")
    labs["報告值"] = pd.to_numeric(labs["報告值"], errors="coerce")

    return patients, labs


def build_time_series(patients, labs):
    """Build monthly time series for forecasting."""
    ts = {}

    # 1. Monthly new patients (first visit per patient)
    first_visit = patients.groupby("識別碼")["就醫日期"].min().reset_index()
    first_visit["yearmonth"] = first_visit["就醫日期"].dt.to_period("M")
    new_patients = first_visit.groupby("yearmonth").size()
    full_idx = pd.period_range(new_patients.index.min(), new_patients.index.max(), freq="M")
    new_patients = new_patients.reindex(full_idx, fill_value=0)
    ts["新增患者數"] = new_patients

    # 2. Monthly visits by top diagnoses
    for dx, name in [("E30.1", "青春期過早_就診數"),
                      ("R62.52", "身材短小_就診數"),
                      ("E30.8", "其他青春期_就診數")]:
        sub = patients[patients["診斷碼"] == dx]
        monthly = sub.groupby("yearmonth").size()
        monthly = monthly.reindex(full_idx, fill_value=0)
        ts[name] = monthly

    # 3. Total monthly visits
    total = patients.groupby("yearmonth").size()
    total = total.reindex(full_idx, fill_value=0)
    ts["總就診數"] = total

    # 4. Monthly mean lab values for key hormones
    for lab_item in ["LH(EIA)", "FSH (EIA)", "Estradiol(E2)(EIA)", "IGF-1"]:
        sub = labs[labs["檢驗項目"] == lab_item].copy()
        sub = sub[sub["報告值"].notna() & (sub["報告值"] > 0)]
        monthly_mean = sub.groupby("yearmonth")["報告值"].mean()
        monthly_mean = monthly_mean.reindex(full_idx)
        monthly_mean = monthly_mean.interpolate(method="linear").bfill().ffill()
        short = lab_item.replace("(EIA)", "").replace(" ", "").strip()
        ts[f"{short}_月均值"] = monthly_mean

    return ts


def forecast_chronos(series, horizon=12, model_name="amazon/chronos-t5-small"):
    """Forecast using Amazon Chronos pretrained model."""
    from chronos import ChronosPipeline

    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    values = series.values.astype(np.float32)
    context = torch.tensor(values).unsqueeze(0)

    forecast = pipeline.predict(context, prediction_length=horizon, num_samples=20)
    # forecast shape: (1, num_samples, horizon)
    median = np.median(forecast[0].numpy(), axis=0)
    low = np.percentile(forecast[0].numpy(), 10, axis=0)
    high = np.percentile(forecast[0].numpy(), 90, axis=0)

    return median, low, high


def forecast_arima(series, horizon=12):
    """Forecast using AutoARIMA as baseline."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS

    sf_df = pd.DataFrame({
        "unique_id": "ts",
        "ds": series.index.to_timestamp(),
        "y": series.values.astype(float),
    })

    sf = StatsForecast(
        models=[AutoARIMA(season_length=12), AutoETS(season_length=12)],
        freq="MS",
    )
    sf.fit(sf_df)
    pred = sf.predict(h=horizon)

    return {
        "AutoARIMA": pred["AutoARIMA"].values,
        "AutoETS": pred["AutoETS"].values,
    }


def run():
    log = []
    def say(s=""):
        print(s); log.append(s)

    say("=" * 72)
    say("Precocious Puberty Disease Incidence Forecasting")
    say("Data: 2014-11 to 2024-09 (116 months)")
    say("Models: Amazon Chronos (foundation model) + AutoARIMA/AutoETS")
    say("=" * 72)

    patients, labs = load_data()
    ts = build_time_series(patients, labs)

    say(f"\nBuilt {len(ts)} time series:")
    for name, s in ts.items():
        say(f"  {name}: {len(s)} months, mean={s.mean():.1f}, last={s.iloc[-1]:.1f}")

    horizon = 12  # forecast 12 months ahead
    say(f"\nForecast horizon: {horizon} months (2024-10 to 2025-09)")

    # ---- Load Chronos model once ----
    say("\nLoading Chronos model (amazon/chronos-t5-small)...")
    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    say("  Model loaded.")

    def chronos_predict(values, h=12):
        context = torch.tensor(values.astype(np.float32)).unsqueeze(0)
        fc = pipeline.predict(context, prediction_length=h, num_samples=20)
        arr = fc[0].numpy()
        return np.median(arr, axis=0), np.percentile(arr, 10, axis=0), np.percentile(arr, 90, axis=0)

    results = {}
    future_idx = pd.period_range(
        ts["新增患者數"].index[-1] + 1, periods=horizon, freq="M"
    )

    # ---- Forecast each series ----
    for name, series in ts.items():
        say(f"\n--- Forecasting: {name} ---")
        vals = series.values.astype(float)

        # Chronos
        med, lo, hi = chronos_predict(vals, horizon)
        say(f"  Chronos: next 3m = [{med[0]:.1f}, {med[1]:.1f}, {med[2]:.1f}]")

        # ARIMA
        arima_res = forecast_arima(series, horizon)
        say(f"  AutoARIMA: next 3m = [{arima_res['AutoARIMA'][0]:.1f}, "
            f"{arima_res['AutoARIMA'][1]:.1f}, {arima_res['AutoARIMA'][2]:.1f}]")

        results[name] = {
            "history": series,
            "chronos_median": med, "chronos_lo": lo, "chronos_hi": hi,
            "arima": arima_res["AutoARIMA"],
            "ets": arima_res["AutoETS"],
            "future_idx": future_idx,
        }

    # ---- PLOTS ----
    # Plot 1: New patients + total visits
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for ax, name in zip(axes, ["新增患者數", "總就診數"]):
        r = results[name]
        hist_x = r["history"].index.to_timestamp()
        fut_x = r["future_idx"].to_timestamp()

        ax.plot(hist_x, r["history"].values, "k-", lw=1.5, label="Observed")
        ax.plot(fut_x, r["chronos_median"], "s-", color="#d62728", lw=2,
                ms=5, label="Chronos forecast")
        ax.fill_between(fut_x, r["chronos_lo"], r["chronos_hi"],
                         alpha=0.2, color="#d62728")
        ax.plot(fut_x, r["arima"], "^--", color="#1f77b4", lw=1.5,
                ms=5, label="AutoARIMA")
        ax.plot(fut_x, r["ets"], "v--", color="#2ca02c", lw=1.5,
                ms=5, label="AutoETS")
        ax.set_ylabel(name, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_title("Precocious Puberty: Monthly Patient Counts Forecast\n"
                       "(12-month ahead, 2024-10 to 2025-09)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG / "forecast_new_patients.png", dpi=150)
    plt.close()
    say(f"\nSaved: forecast_new_patients.png")

    # Plot 2: Diagnosis-specific visits
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    for ax, name in zip(axes, ["青春期過早_就診數", "身材短小_就診數", "其他青春期_就診數"]):
        r = results[name]
        hist_x = r["history"].index.to_timestamp()
        fut_x = r["future_idx"].to_timestamp()
        ax.plot(hist_x, r["history"].values, "k-", lw=1.5, label="Observed")
        ax.plot(fut_x, r["chronos_median"], "s-", color="#d62728", lw=2,
                ms=5, label="Chronos")
        ax.fill_between(fut_x, r["chronos_lo"], r["chronos_hi"],
                         alpha=0.2, color="#d62728")
        ax.plot(fut_x, r["arima"], "^--", color="#1f77b4", lw=1.5, ms=5, label="ARIMA")
        ax.set_ylabel(name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_title("Diagnosis-Specific Visit Forecasts (12 months)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG / "forecast_diagnosis_visits.png", dpi=150)
    plt.close()
    say(f"Saved: forecast_diagnosis_visits.png")

    # Plot 3: Lab value trends
    lab_names = [n for n in results if "月均值" in n]
    fig, axes = plt.subplots(len(lab_names), 1, figsize=(12, 3*len(lab_names)), sharex=False)
    if len(lab_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, lab_names):
        r = results[name]
        hist_x = r["history"].index.to_timestamp()
        fut_x = r["future_idx"].to_timestamp()
        ax.plot(hist_x, r["history"].values, "k-", lw=1.5, label="Observed")
        ax.plot(fut_x, r["chronos_median"], "s-", color="#d62728", lw=2,
                ms=5, label="Chronos")
        ax.fill_between(fut_x, r["chronos_lo"], r["chronos_hi"],
                         alpha=0.2, color="#d62728")
        ax.plot(fut_x, r["arima"], "^--", color="#1f77b4", lw=1.5, ms=5, label="ARIMA")
        ax.set_ylabel(name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_title("Key Hormone Trend Forecasts (monthly mean, 12 months)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG / "forecast_lab_trends.png", dpi=150)
    plt.close()
    say(f"Saved: forecast_lab_trends.png")

    # ---- Summary table ----
    say("\n" + "=" * 72)
    say("FORECAST SUMMARY (next 12 months)")
    say("=" * 72)
    say(f"{'Series':<30} {'Last obs':>10} {'Chronos mean':>13} {'ARIMA mean':>11} {'Trend':>8}")
    for name, r in results.items():
        last = r["history"].iloc[-1]
        chr_mean = np.mean(r["chronos_median"])
        ari_mean = np.mean(r["arima"])
        trend = "UP" if chr_mean > last * 1.05 else ("DOWN" if chr_mean < last * 0.95 else "STABLE")
        say(f"{name:<30} {last:>10.1f} {chr_mean:>13.1f} {ari_mean:>11.1f} {trend:>8}")

    (DATA / "forecast_results.txt").write_text("\n".join(log))
    say(f"\nAll outputs saved to {DATA}")


if __name__ == "__main__":
    run()
