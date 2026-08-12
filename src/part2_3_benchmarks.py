"""
Part 2 - Forecasting problem definition
Part 3 - Benchmark models: mean, naive, daily seasonal naive,
         weekly seasonal naive, drift. 24-hour forecast horizon.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

FIGDIR = "figs"
RESDIR = "results"
HORIZON = 24


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100)


def all_metrics(y_true, y_pred):
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred),
            "MAPE": mape(y_true, y_pred), "sMAPE": smape(y_true, y_pred)}


def train_test_split(hourly, test_days=14):
    """Hold out the last `test_days` (24 * test_days hours) as the test set.
    Everything before that is training data. This mirrors the brief's
    instruction to use the last 14 days as the test/evaluation period and
    keeps a single consistent split across every model in the study."""
    test_hours = 24 * test_days
    train = hourly.iloc[:-test_hours]
    test = hourly.iloc[-test_hours:]
    return train, test


def mean_forecast(train, horizon):
    m = train["Appliances"].mean()
    return np.full(horizon, m)


def naive_forecast(train, horizon):
    last = train["Appliances"].iloc[-1]
    return np.full(horizon, last)


def seasonal_naive_forecast(train, horizon, season):
    """y_hat(t+h) = y(t+h-season)"""
    last_season = train["Appliances"].iloc[-season:].values
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season, reps)[:horizon]


def drift_forecast(train, horizon):
    y = train["Appliances"].values
    n = len(y)
    slope = (y[-1] - y[0]) / (n - 1)
    return y[-1] + slope * np.arange(1, horizon + 1)


def rolling_origin_evaluation(hourly, test_days=14, horizon=24):
    """Rolling-origin (walk-forward) evaluation: for every day in the test
    block, forecast the next 24 hours from a growing training window, then
    slide forward. This gives a robust, non-single-window estimate of each
    benchmark's typical 24h-ahead accuracy, rather than relying on one lucky
    (or unlucky) forecast origin."""
    test_hours = 24 * test_days
    origins = list(range(len(hourly) - test_hours - horizon, len(hourly) - horizon, 24))

    results = {name: {"RMSE": [], "MAE": [], "MAPE": [], "sMAPE": []}
               for name in ["Mean", "Naive", "Seasonal_Naive_24h", "Seasonal_Naive_168h", "Drift"]}

    example_store = None
    for origin in origins:
        train = hourly.iloc[:origin]
        test = hourly.iloc[origin:origin + horizon]
        y_true = test["Appliances"].values

        preds = {
            "Mean": mean_forecast(train, horizon),
            "Naive": naive_forecast(train, horizon),
            "Seasonal_Naive_24h": seasonal_naive_forecast(train, horizon, 24),
            "Seasonal_Naive_168h": seasonal_naive_forecast(train, horizon, 168),
            "Drift": drift_forecast(train, horizon),
        }
        for name, y_pred in preds.items():
            m = all_metrics(y_true, y_pred)
            for k in results[name]:
                results[name][k].append(m[k])

        if origin == origins[-1]:
            example_store = (test.index, y_true, preds)

    summary = {name: {k: float(np.mean(v)) for k, v in metrics.items()}
               for name, metrics in results.items()}
    return summary, example_store


def plot_example(example_store, best_name):
    idx, y_true, preds = example_store
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(idx, y_true, label="Actual", color="black", lw=2)
    colors = {"Mean": "#999999", "Naive": "#c0562d", "Seasonal_Naive_24h": "#1f6f78",
              "Seasonal_Naive_168h": "#7a8fbf", "Drift": "#e0a800"}
    for name, y_pred in preds.items():
        ax.plot(idx, y_pred, label=name.replace("_", " "), color=colors.get(name), lw=1.3, ls="--")
    ax.set_title("Benchmark 24-hour Forecasts vs Actual (final test-window origin)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Appliances (Wh/hour)")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/09_benchmark_forecasts.png")
    plt.close(fig)


def run(hourly):
    train, test = train_test_split(hourly, test_days=14)
    summary, example_store = rolling_origin_evaluation(hourly, test_days=14, horizon=HORIZON)
    best_name = min(summary, key=lambda k: summary[k]["RMSE"])
    plot_example(example_store, best_name)

    problem_def = {
        "target_variable": "Appliances (Wh consumed per hour, hourly-resampled)",
        "forecast_horizon_hours": HORIZON,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_period_days": 14,
        "evaluation_metrics": ["RMSE", "MAE", "MAPE", "sMAPE"],
        "evaluation_protocol": "Rolling-origin evaluation: one 24h forecast issued at "
                                "the start of each of the 14 test days, training window "
                                "grows to include all prior data each time; metrics are "
                                "averaged over the 14 origins.",
        "benchmark_results_mean_over_origins": summary,
        "best_benchmark": best_name,
    }
    with open(f"{RESDIR}/part2_3_benchmarks.json", "w") as f:
        json.dump(problem_def, f, indent=2)
    print(json.dumps(problem_def, indent=2))
    return train, test, problem_def


if __name__ == "__main__":
    hourly = pd.read_csv(f"{RESDIR}/hourly_data.csv", index_col=0, parse_dates=True)
    run(hourly)
