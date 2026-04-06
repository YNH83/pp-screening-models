"""
TimesFM (Google Research) forecasting for precocious puberty time series.

TimesFM is a decoder-only foundation model for time-series forecasting,
pretrained on 100B time points from Google Trends, Wiki Pageviews, and
synthetic data. This script runs TimesFM alongside Chronos and AutoARIMA
for direct comparison on our clinical time series.

Model: google/timesfm-2.0-500m-pytorch (498M params)
Reference: Das et al. "A decoder-only foundation model for time-series
           forecasting." ICML 2024.
GitHub: https://github.com/google-research/timesfm
HuggingFace: google/timesfm-2.0-500m-pytorch

Outputs:
  figures/Fig_timesfm_comparison.png
  scripts/timesfm_results.txt
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings("ignore")

DATA = Path("/Users/ynh83/Desktop/04062026 PP")
RAW = DATA / "raw data"
FIG = DATA / "figures"
OUT = DATA / "scripts"


def load_time_series():
    """Build monthly time series from clinical data."""
    pt = pd.read_excel(RAW / "decrypted_病人基本資料.xlsx", engine="openpyxl")
    pt["就醫日期"] = pd.to_datetime(pt["就醫日期"])
    pt["yearmonth"] = pt["就醫日期"].dt.to_period("M")

    lab = pd.read_excel(RAW / "decrypted_檢驗報告數值.xlsx", engine="openpyxl")
    lab["報到時間"] = pd.to_datetime(lab["報到時間"])
    lab["yearmonth"] = lab["報到時間"].dt.to_period("M")
    lab["報告值"] = pd.to_numeric(lab["報告值"], errors="coerce")

    ts = {}

    # Monthly new patients
    first_visit = pt.groupby("識別碼")["就醫日期"].min().reset_index()
    first_visit["yearmonth"] = first_visit["就醫日期"].dt.to_period("M")
    new_patients = first_visit.groupby("yearmonth").size()
    full_idx = pd.period_range(new_patients.index.min(), new_patients.index.max(), freq="M")
    ts["New patients"] = new_patients.reindex(full_idx, fill_value=0)

    # Total visits
    total = pt.groupby("yearmonth").size().reindex(full_idx, fill_value=0)
    ts["Total visits"] = total

    # PP visits
    pp = pt[pt["診斷碼"] == "E30.1"].groupby("yearmonth").size().reindex(full_idx, fill_value=0)
    ts["PP visits (E30.1)"] = pp

    # Short stature visits
    ss = pt[pt["診斷碼"] == "R62.52"].groupby("yearmonth").size().reindex(full_idx, fill_value=0)
    ts["SS visits (R62.52)"] = ss

    # Monthly mean LH
    lh = lab[lab["檢驗項目"] == "LH(EIA)"]
    lh = lh[lh["報告值"].notna() & (lh["報告值"] > 0)]
    lh_monthly = lh.groupby("yearmonth")["報告值"].mean().reindex(full_idx)
    lh_monthly = lh_monthly.interpolate(method="linear").bfill().ffill()
    ts["LH monthly mean"] = lh_monthly

    # Monthly mean IGF-1
    igf = lab[lab["檢驗項目"] == "IGF-1"]
    igf = igf[igf["報告值"].notna() & (igf["報告值"] > 0)]
    igf_monthly = igf.groupby("yearmonth")["報告值"].mean().reindex(full_idx)
    igf_monthly = igf_monthly.interpolate(method="linear").bfill().ffill()
    ts["IGF-1 monthly mean"] = igf_monthly

    return ts


def forecast_timesfm(series_values, horizon=12):
    """Forecast using TimesFM 2.0 500M (PyTorch)."""
    from transformers import TimesFmModelForPrediction

    model = TimesFmModelForPrediction.from_pretrained(
        "google/timesfm-2.0-500m-pytorch",
        dtype=torch.float32,
    )
    model.eval()

    context = torch.tensor(series_values.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        outputs = model(context, freq=[0])  # 0 = monthly

    mean_pred = outputs.mean_predictions[0].numpy()[:horizon]
    full_pred = outputs.full_predictions[0].numpy()[:horizon, :]  # quantiles

    # Extract prediction intervals (quantile indices: 0=10%, 4=50%, 8=90%)
    if full_pred.ndim == 2 and full_pred.shape[1] >= 9:
        lo = full_pred[:, 0]   # 10th percentile
        hi = full_pred[:, 8]   # 90th percentile
    else:
        lo = mean_pred * 0.9
        hi = mean_pred * 1.1

    return mean_pred, lo, hi


def forecast_chronos(series_values, horizon=12):
    """Forecast using Amazon Chronos T5-small."""
    from chronos import ChronosPipeline

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    context = torch.tensor(series_values.astype(np.float32)).unsqueeze(0)
    fc = pipeline.predict(context, prediction_length=horizon, num_samples=20)
    arr = fc[0].numpy()
    return np.median(arr, axis=0), np.percentile(arr, 10, axis=0), np.percentile(arr, 90, axis=0)


def forecast_arima(series, horizon=12):
    """Forecast using AutoARIMA."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    sf_df = pd.DataFrame({
        "unique_id": "ts",
        "ds": series.index.to_timestamp(),
        "y": series.values.astype(float),
    })
    sf = StatsForecast(models=[AutoARIMA(season_length=12)], freq="MS")
    sf.fit(sf_df)
    pred = sf.predict(h=horizon)
    return pred["AutoARIMA"].values


def run():
    log = []
    def say(s=""):
        print(s); log.append(s)

    say("=" * 72)
    say("TimesFM vs Chronos vs AutoARIMA: Foundation Model Comparison")
    say("=" * 72)
    say("TimesFM: Das et al. (2024). ICML. google/timesfm-2.0-500m-pytorch")
    say("Chronos: Ansari et al. (2024). amazon/chronos-t5-small")
    say("AutoARIMA: Hyndman & Khandakar (2008). statsforecast")
    say("=" * 72)

    ts = load_time_series()
    say(f"\nLoaded {len(ts)} time series")

    horizon = 12
    # Use last 12 months as held-out test for accuracy comparison
    say(f"\nStrategy: train on all-but-last-12, test on last 12 months")

    results = {}

    for name, series in ts.items():
        vals = series.values.astype(np.float64)
        train_vals = vals[:-12]
        test_vals = vals[-12:]

        say(f"\n--- {name} ---")
        say(f"  Total months: {len(vals)}, Train: {len(train_vals)}, Test: 12")

        # TimesFM
        say("  Running TimesFM...")
        tfm_med, tfm_lo, tfm_hi = forecast_timesfm(train_vals, horizon)

        # Chronos
        say("  Running Chronos...")
        chr_med, chr_lo, chr_hi = forecast_chronos(train_vals, horizon)

        # AutoARIMA
        say("  Running AutoARIMA...")
        train_series = series.iloc[:-12]
        arima_pred = forecast_arima(train_series, horizon)

        # Compute errors
        mae_tfm = np.mean(np.abs(tfm_med - test_vals))
        mae_chr = np.mean(np.abs(chr_med - test_vals))
        mae_ari = np.mean(np.abs(arima_pred - test_vals))

        mape_tfm = np.mean(np.abs(tfm_med - test_vals) / np.maximum(np.abs(test_vals), 1)) * 100
        mape_chr = np.mean(np.abs(chr_med - test_vals) / np.maximum(np.abs(test_vals), 1)) * 100
        mape_ari = np.mean(np.abs(arima_pred - test_vals) / np.maximum(np.abs(test_vals), 1)) * 100

        results[name] = {
            "test": test_vals,
            "timesfm": {"med": tfm_med, "lo": tfm_lo, "hi": tfm_hi, "mae": mae_tfm, "mape": mape_tfm},
            "chronos": {"med": chr_med, "lo": chr_lo, "hi": chr_hi, "mae": mae_chr, "mape": mape_chr},
            "arima": {"pred": arima_pred, "mae": mae_ari, "mape": mape_ari},
            "history": series,
        }

        say(f"  TimesFM  MAE={mae_tfm:7.1f}  MAPE={mape_tfm:5.1f}%")
        say(f"  Chronos  MAE={mae_chr:7.1f}  MAPE={mape_chr:5.1f}%")
        say(f"  AutoARIMA MAE={mae_ari:7.1f}  MAPE={mape_ari:5.1f}%")

        best = min([("TimesFM", mae_tfm), ("Chronos", mae_chr), ("AutoARIMA", mae_ari)],
                    key=lambda x: x[1])
        say(f"  Winner: {best[0]} (MAE={best[1]:.1f})")

    # ================================================================
    # Summary table
    # ================================================================
    say("\n" + "=" * 72)
    say("SUMMARY: Head-to-Head Foundation Model Comparison")
    say("=" * 72)
    say(f"\n{'Series':<25} {'TimesFM MAE':>12} {'Chronos MAE':>12} {'ARIMA MAE':>10} {'Winner':>10}")
    say("-" * 72)

    wins = {"TimesFM": 0, "Chronos": 0, "AutoARIMA": 0}
    for name, r in results.items():
        maes = [("TimesFM", r["timesfm"]["mae"]),
                ("Chronos", r["chronos"]["mae"]),
                ("AutoARIMA", r["arima"]["mae"])]
        best = min(maes, key=lambda x: x[1])
        wins[best[0]] += 1
        say(f"{name:<25} {r['timesfm']['mae']:>12.1f} {r['chronos']['mae']:>12.1f} "
            f"{r['arima']['mae']:>10.1f} {best[0]:>10}")

    say(f"\nWin count: TimesFM={wins['TimesFM']}, Chronos={wins['Chronos']}, AutoARIMA={wins['AutoARIMA']}")

    # ================================================================
    # Figure
    # ================================================================
    n_series = len(results)
    fig, axes = plt.subplots(n_series, 1, figsize=(14, 3.5 * n_series))
    if n_series == 1:
        axes = [axes]

    test_dates = list(results.values())[0]["history"].index[-12:].to_timestamp()

    for ax, (name, r) in zip(axes, results.items()):
        hist = r["history"]
        hist_dates = hist.index[:-12].to_timestamp()

        # History
        ax.plot(hist_dates, hist.values[:-12], "k-", lw=1, alpha=0.5, label="History")
        ax.plot(test_dates, r["test"], "ko", ms=5, label="Actual (held out)")

        # TimesFM
        ax.plot(test_dates, r["timesfm"]["med"], "s-", color="#e41a1c", lw=2, ms=5,
                label=f'TimesFM (MAE={r["timesfm"]["mae"]:.0f})')
        ax.fill_between(test_dates, r["timesfm"]["lo"], r["timesfm"]["hi"],
                         alpha=0.12, color="#e41a1c")

        # Chronos
        ax.plot(test_dates, r["chronos"]["med"], "^--", color="#377eb8", lw=1.5, ms=5,
                label=f'Chronos (MAE={r["chronos"]["mae"]:.0f})')
        ax.fill_between(test_dates, r["chronos"]["lo"], r["chronos"]["hi"],
                         alpha=0.12, color="#377eb8")

        # AutoARIMA
        ax.plot(test_dates, r["arima"]["pred"], "D--", color="#4daf4a", lw=1.5, ms=4,
                label=f'AutoARIMA (MAE={r["arima"]["mae"]:.0f})')

        ax.set_ylabel(name, fontsize=10)
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        ax.grid(alpha=0.2)

    axes[0].set_title(
        "Foundation Model Comparison: TimesFM (Google, 498M) vs Chronos (Amazon, 46M) vs AutoARIMA\n"
        "12-month held-out test on PP clinical time series",
        fontsize=13, fontweight="bold"
    )
    axes[-1].set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(FIG / "Fig_timesfm_comparison.png", dpi=150)
    plt.close()
    say(f"\nSaved: Fig_timesfm_comparison.png")

    # ================================================================
    # Model citations
    # ================================================================
    say("\n" + "=" * 72)
    say("MODEL CITATIONS")
    say("=" * 72)
    say("TimesFM 2.0:")
    say("  Das A, Kong W, Leber A, Mathews R, Rajagopalan K, et al.")
    say("  'A decoder-only foundation model for time-series forecasting.'")
    say("  ICML 2024. arXiv:2310.10688.")
    say("  HuggingFace: google/timesfm-2.0-500m-pytorch (498M params)")
    say("  GitHub: https://github.com/google-research/timesfm")
    say("")
    say("Chronos:")
    say("  Ansari AF, Stella L, Turkmen C, Zhang X, et al.")
    say("  'Chronos: Learning the language of time series.'")
    say("  arXiv:2403.07815, 2024.")
    say("  HuggingFace: amazon/chronos-t5-small (46M params)")
    say("")
    say("AutoARIMA:")
    say("  Hyndman RJ, Khandakar Y.")
    say("  'Automatic time series forecasting: the forecast package for R.'")
    say("  J Statistical Software, 27(3), 2008.")

    (OUT / "timesfm_results.txt").write_text("\n".join(log))
    say(f"\nAll outputs saved.")


if __name__ == "__main__":
    run()
