"""
Part 1 - Data retrieval, hourly resampling, EDA and stationarity testing
for the Appliance Energy Prediction dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import json
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.dpi"] = 110
FIGDIR = "figs"
RESDIR = "results"


def load_raw(path="data/energydata_complete.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def resample_hourly(df):
    """Bin the 10-minute readings up to hourly values.
    Appliances/lights energy (Wh per 10-min interval) are summed to give
    total Wh consumed in the hour; sensor/weather readings are averaged."""
    energy_cols = ["Appliances", "lights"]
    other_cols = [c for c in df.columns if c not in energy_cols]

    hourly_energy = df[energy_cols].resample("h").sum()
    hourly_other = df[other_cols].resample("h").mean()
    hourly = hourly_energy.join(hourly_other)
    hourly = hourly.dropna()
    return hourly


def missing_value_report(df):
    miss = df.isna().sum()
    return miss[miss > 0]


def plot_series(hourly):
    fig, ax = plt.subplots(figsize=(12, 4))
    hourly["Appliances"].plot(ax=ax, lw=0.7, color="#1f6f78")
    ax.set_title("Hourly Appliance Energy Use (Wh)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliances (Wh/hour)")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/01_hourly_series.png")
    plt.close(fig)

    # zoom on 2 weeks
    fig, ax = plt.subplots(figsize=(12, 4))
    hourly["Appliances"].iloc[:24 * 14].plot(ax=ax, color="#c0562d")
    ax.set_title("Hourly Appliance Energy Use - First 14 Days")
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliances (Wh/hour)")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/02_hourly_series_2weeks.png")
    plt.close(fig)

    # distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    hourly["Appliances"].plot(kind="hist", bins=40, ax=ax, color="#7a8fbf")
    ax.set_title("Distribution of Hourly Appliance Energy Use")
    ax.set_xlabel("Wh/hour")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/03_hist.png")
    plt.close(fig)

    # average daily profile (hour of day)
    hod = hourly.copy()
    hod["hour"] = hod.index.hour
    prof = hod.groupby("hour")["Appliances"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    prof.plot(ax=ax, marker="o", color="#1f6f78")
    ax.set_title("Average Appliance Energy Use by Hour of Day")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean Wh/hour")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/04_hourly_profile.png")
    plt.close(fig)

    # average weekly profile (day of week)
    dow = hourly.copy()
    dow["dow"] = dow.index.dayofweek
    profw = dow.groupby("dow")["Appliances"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    profw.plot(ax=ax, kind="bar", color="#c0562d")
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)
    ax.set_title("Average Appliance Energy Use by Day of Week")
    ax.set_ylabel("Mean Wh/hour")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/05_weekly_profile.png")
    plt.close(fig)


def decomposition(hourly):
    # additive decomposition with a daily period (24) on a representative window
    result = seasonal_decompose(hourly["Appliances"], model="additive", period=24)
    fig = result.plot()
    fig.set_size_inches(11, 7)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/06_decomposition_daily.png")
    plt.close(fig)

    # weekly period decomposition (168 hours)
    result_w = seasonal_decompose(hourly["Appliances"], model="additive", period=168)
    fig = result_w.plot()
    fig.set_size_inches(11, 7)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/07_decomposition_weekly.png")
    plt.close(fig)
    return result, result_w


def acf_pacf(series, lags, tag):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title(f"ACF - {tag}")
    axes[1].set_title(f"PACF - {tag}")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/08_acf_pacf_{tag.replace(' ', '_')}.png")
    plt.close(fig)


def stationarity_tests(series):
    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(series.dropna(), autolag="AIC")
    kpss_stat, kpss_p, _, kpss_crit = kpss(series.dropna(), regression="c", nlags="auto")
    out = {
        "ADF_statistic": float(adf_stat),
        "ADF_pvalue": float(adf_p),
        "ADF_critical_values": {k: float(v) for k, v in adf_crit.items()},
        "ADF_stationary_at_5pct": bool(adf_p < 0.05),
        "KPSS_statistic": float(kpss_stat),
        "KPSS_pvalue": float(kpss_p),
        "KPSS_critical_values": {k: float(v) for k, v in kpss_crit.items()},
        "KPSS_stationary_at_5pct": bool(kpss_p > 0.05),
    }
    return out


def run():
    df = load_raw()
    miss = missing_value_report(df)
    hourly = resample_hourly(df)

    plot_series(hourly)
    decomposition(hourly)
    acf_pacf(hourly["Appliances"], lags=72, tag="level series")

    stats_level = stationarity_tests(hourly["Appliances"])
    diff1 = hourly["Appliances"].diff().dropna()
    stats_diff1 = stationarity_tests(diff1)
    diff_seasonal = hourly["Appliances"].diff(24).dropna()
    stats_diff_seasonal = stationarity_tests(diff_seasonal)

    acf_pacf(diff1, lags=72, tag="first differenced")
    acf_pacf(diff_seasonal, lags=72, tag="24h seasonally differenced")

    summary = {
        "n_raw_rows": int(len(df)),
        "n_hourly_rows": int(len(hourly)),
        "date_range": [str(hourly.index.min()), str(hourly.index.max())],
        "missing_values_raw": {k: int(v) for k, v in miss.items()},
        "target_mean": float(hourly["Appliances"].mean()),
        "target_std": float(hourly["Appliances"].std()),
        "target_min": float(hourly["Appliances"].min()),
        "target_max": float(hourly["Appliances"].max()),
        "stationarity_level": stats_level,
        "stationarity_first_diff": stats_diff1,
        "stationarity_seasonal_diff_24h": stats_diff_seasonal,
    }
    with open(f"{RESDIR}/part1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    hourly.to_csv(f"{RESDIR}/hourly_data.csv")
    print(json.dumps(summary, indent=2))
    return hourly, summary


if __name__ == "__main__":
    run()
