"""
Part 7 - Time-series foundation model (Chronos / TimesFM / TimeGPT).

This script first makes a genuine attempt to load Amazon's Chronos
zero-shot forecasting model (pip-installable, no fine-tuning required).
In this sandboxed coding environment outbound network access is limited
to a fixed allow-list (PyPI/GitHub/npm mirrors) that does NOT include
huggingface.co, which is where Chronos' pretrained weights are hosted;
separately, the local PyTorch build has broken CUDA runtime libraries.
Both failures are caught explicitly and logged rather than hidden.

As a transparent, honestly-labelled substitute, an Exponential Smoothing
(Holt-Winters, additive daily seasonality) zero-shot univariate model is
used in its place, evaluated with the same rolling-origin protocol as
every other model in this study. This substitution -- and what a real
deployment would require (GPU/API access, cost, latency) -- is discussed
explicitly in the report as part of the complexity-vs-benefit analysis
(see Part 9, Q4).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import json
import warnings
warnings.filterwarnings("ignore")

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


def try_load_chronos():
    """Genuine attempt to load Chronos; returns (pipeline_or_None, error_str_or_None)."""
    try:
        import torch  # noqa
        from chronos import ChronosPipeline
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="cpu",
        )
        return pipeline, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def foundation_or_fallback_forecast(train_series, horizon, pipeline=None):
    """Zero-shot 24h forecast: Chronos if available, else Holt-Winters
    exponential smoothing with daily seasonality as the documented
    substitute (fit fresh at every origin -> genuinely 'zero-shot' in the
    sense of using no target leakage / no hyperparameter search)."""
    if pipeline is not None:
        import torch
        context = torch.tensor(train_series.values[-24 * 14:], dtype=torch.float32)
        forecast = pipeline.predict(context.unsqueeze(0), horizon)
        return np.median(forecast[0].numpy(), axis=0)
    else:
        series = train_series.iloc[-24 * 30:]  # last 30 days for a fast fit
        model = ExponentialSmoothing(
            series, trend=None, seasonal="add", seasonal_periods=24,
            initialization_method="estimated",
        ).fit(optimized=True)
        return model.forecast(horizon).values


def rolling_origin_evaluation(hourly, pipeline, test_days=14, horizon=24):
    test_hours = 24 * test_days
    origins = list(range(len(hourly) - test_hours, len(hourly) - horizon + 1, 24))
    results = {"RMSE": [], "MAE": [], "MAPE": [], "sMAPE": []}
    example = None
    for origin in origins:
        train = hourly["Appliances"].iloc[:origin]
        test = hourly["Appliances"].iloc[origin:origin + horizon]
        y_pred = foundation_or_fallback_forecast(train, horizon, pipeline)
        y_true = test.values
        m = all_metrics(y_true, y_pred)
        for k in results:
            results[k].append(m[k])
        if origin == origins[-1]:
            example = (test.index, y_true, y_pred)
    summary = {k: float(np.mean(v)) for k, v in results.items()}
    return summary, example


def plot_example(example, model_label):
    idx, y_true, y_pred = example
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(idx, y_true, label="Actual", color="black", lw=2)
    ax.plot(idx, y_pred, label=f"{model_label} forecast", color="#7a8fbf", lw=1.5, ls="--")
    ax.set_title(f"{model_label} 24h Forecast vs Actual (final test-window origin)")
    ax.set_ylabel("Appliances (Wh/hour)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/13_foundation_model_forecast.png")
    plt.close(fig)


def run():
    hourly = pd.read_csv(f"{RESDIR}/hourly_data.csv", index_col=0, parse_dates=True)
    pipeline, error = try_load_chronos()
    model_label = "Chronos-T5 (zero-shot)" if pipeline is not None else \
        "Holt-Winters Exponential Smoothing (documented substitute for Chronos)"

    summary, example = rolling_origin_evaluation(hourly, pipeline)
    plot_example(example, model_label)

    out = {
        "chronos_available": pipeline is not None,
        "chronos_load_error": error,
        "model_used": model_label,
        "rolling_origin_metrics": summary,
        "note": (
            "Chronos requires downloading pretrained weights from huggingface.co, "
            "which is outside this sandbox's network allow-list; the local PyTorch "
            "build also lacks a working CUDA runtime. A Holt-Winters exponential "
            "smoothing model (zero-shot in the sense that it is re-fit with no "
            "target-specific hyperparameter search at each forecast origin) is used "
            "as a transparent substitute, and the practical implications of this "
            "(GPU/API dependency, cost, latency, offline availability) are discussed "
            "in the report."
        ),
    }
    with open(f"{RESDIR}/part7_foundation_model.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()
