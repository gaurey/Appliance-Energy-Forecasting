"""
Part 4 - Autoregressive modelling with SARIMA/SARIMAX.
Grid search over (p,d,q) using AIC, seasonal term for daily seasonality,
residual diagnostics, 24h forecast with confidence intervals.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import json
import itertools
import warnings
warnings.filterwarnings("ignore")

FIGDIR = "figs"
RESDIR = "results"


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def grid_search_aic(train_subset, p_range=range(0, 7), d_range=range(0, 3),
                     q_range=range(0, 7), max_models=None):
    """Loop over p in [0,6], d in [0,2], q in [0,6] as specified in the brief,
    fitting a non-seasonal ARIMA on a recent subsample of the training data
    (for tractable runtime) and selecting the combination with the lowest AIC.
    Non-convergent / numerically unstable fits are skipped."""
    combos = list(itertools.product(p_range, d_range, q_range))
    records = []
    for (p, d, q) in combos:
        try:
            mod = SARIMAX(train_subset, order=(p, d, q),
                           enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False, maxiter=100)
            records.append({"p": p, "d": d, "q": q, "aic": float(res.aic),
                             "bic": float(res.bic)})
        except Exception:
            continue
    df = pd.DataFrame(records).sort_values("aic").reset_index(drop=True)
    return df


def fit_sarimax(train, order, seasonal_order, exog_train=None, maxiter=200, method="lbfgs",
                 retry_with_powell=False):
    """Fit a SARIMAX model and report convergence status transparently rather than silently
    accepting whatever the optimiser returns.

    `retry_with_powell=True` retries a non-converged fit with the derivative-free Powell
    optimiser -- sometimes more robust on flatter likelihood surfaces, but dramatically
    slower here (tested: did not finish in over 5 minutes on this model/data), so it is
    opt-in rather than automatic."""
    mod = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                  exog=exog_train,
                  enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False, maxiter=maxiter, method=method)
    converged = res.mle_retvals.get("converged", True)
    if not converged:
        print(f"  [fit_sarimax] WARNING: {method} did not fully converge within maxiter={maxiter} "
              f"(order={order}, seasonal={seasonal_order}, exog={exog_train is not None}).")
        if retry_with_powell:
            print("  [fit_sarimax] retrying with Powell (this can take several minutes)...")
            res_powell = mod.fit(disp=False, maxiter=maxiter, method="powell")
            if res_powell.llf > res.llf:
                res = res_powell
            if not res.mle_retvals.get("converged", True):
                print(f"  [fit_sarimax] still not converged after Powell retry -- using best "
                      f"available fit (llf={res.llf:.1f}).")
        else:
            print("  [fit_sarimax] proceeding with the best fit found (set retry_with_powell=True "
                  "for a more robust but much slower re-fit).")
    return res


def residual_diagnostics(res, tag):
    resid = res.resid.dropna()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(resid, color="#1f6f78", lw=0.6)
    axes[0, 0].set_title("Residuals over time")
    axes[0, 1].hist(resid, bins=30, color="#7a8fbf")
    axes[0, 1].set_title("Residual distribution")
    plot_acf(resid, lags=48, ax=axes[1, 0])
    axes[1, 0].set_title("Residual ACF")
    from scipy import stats as sstats
    sstats.probplot(resid, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q plot")
    fig.suptitle(f"Residual diagnostics - {tag}")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/{tag}_residual_diagnostics.png")
    plt.close(fig)

    lb = acorr_ljungbox(resid, lags=[24], return_df=True)
    return {
        "ljung_box_stat": float(lb["lb_stat"].iloc[0]),
        "ljung_box_pvalue": float(lb["lb_pvalue"].iloc[0]),
        "residual_mean": float(resid.mean()),
        "residual_std": float(resid.std()),
    }


def forecast_and_eval(res, test, horizon, exog_test=None, tag=""):
    fc = res.get_forecast(steps=horizon, exog=exog_test)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    y_true = test["Appliances"].values[:horizon]
    y_pred = mean.values[:horizon]

    metrics = {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred)}

    fig, ax = plt.subplots(figsize=(11, 5))
    idx = test.index[:horizon]
    ax.plot(idx, y_true, label="Actual", color="black", lw=2)
    ax.plot(idx, y_pred, label="SARIMAX forecast", color="#c0562d", lw=1.5)
    ax.fill_between(idx, ci.iloc[:horizon, 0], ci.iloc[:horizon, 1],
                     color="#c0562d", alpha=0.2, label="95% CI")
    ax.set_title(f"SARIMAX 24h Forecast with 95% Confidence Interval ({tag})")
    ax.set_ylabel("Appliances (Wh/hour)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/10_sarimax_forecast_{tag}.png")
    plt.close(fig)
    return metrics, y_pred, ci


def run(hourly, train, test):
    # Use a recent 45-day subsample of the training data for the AIC grid
    # search so the full p,d,q sweep specified in the brief is tractable.
    subset = train["Appliances"].iloc[-24 * 45:]
    grid = grid_search_aic(subset)
    grid.to_csv(f"{RESDIR}/part4_grid_search.csv", index=False)
    best = grid.iloc[0]
    order = (int(best.p), int(best.d), int(best.q))

    # Daily seasonality established in Part 1 decomposition/ACF -> add a
    # seasonal term at s=24. Keep the seasonal order small (tractable) and
    # justified by the strong 24h ACF spikes observed earlier.
    seasonal_order = (1, 1, 1, 24)

    # Exogenous variables: outdoor temperature and humidity, justified by
    # their physical link to heating/cooling appliance load, plus an
    # hour-of-day sine/cosine pair capturing intraday occupancy patterns.
    def make_exog(df):
        hour = df.index.hour
        ex = pd.DataFrame({
            "T_out": df["T_out"].values,
            "RH_out": df["RH_out"].values,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
        }, index=df.index)
        return ex

    exog_train = make_exog(train)
    exog_test = make_exog(test)

    # Fit non-seasonal best-AIC ARIMA (no exog) as an intermediate model
    res_arima = fit_sarimax(train["Appliances"], order, (0, 0, 0, 0))
    diag_arima = residual_diagnostics(res_arima, "arima_bestaic")
    metrics_arima, _, _ = forecast_and_eval(res_arima, test, 24, tag="arima_bestaic")

    # Fit SARIMA (seasonal, no exog)
    res_sarima = fit_sarimax(train["Appliances"], order, seasonal_order)
    diag_sarima = residual_diagnostics(res_sarima, "sarima_seasonal")
    metrics_sarima, _, _ = forecast_and_eval(res_sarima, test, 24, tag="sarima_seasonal")

    # Fit full SARIMAX (seasonal + exogenous)
    res_sarimax = fit_sarimax(train["Appliances"], order, seasonal_order, exog_train)
    diag_sarimax = residual_diagnostics(res_sarimax, "sarimax_full")
    metrics_sarimax, y_pred_sarimax, ci_sarimax = forecast_and_eval(
        res_sarimax, test, 24, exog_test=exog_test.iloc[:24], tag="sarimax_full")

    summary = {
        "grid_search_top5": grid.head(5).to_dict(orient="records"),
        "selected_order_pdq": order,
        "seasonal_order_PDQm": seasonal_order,
        "exog_variables": list(exog_train.columns),
        "models": {
            "ARIMA_best_aic_no_seasonality": {"order": order, "metrics_24h": metrics_arima,
                                               "residual_diagnostics": diag_arima,
                                               "aic": float(res_arima.aic)},
            "SARIMA_seasonal_no_exog": {"order": order, "seasonal_order": seasonal_order,
                                         "metrics_24h": metrics_sarima,
                                         "residual_diagnostics": diag_sarima,
                                         "aic": float(res_sarima.aic)},
            "SARIMAX_seasonal_plus_exog": {"order": order, "seasonal_order": seasonal_order,
                                            "exog": list(exog_train.columns),
                                            "metrics_24h": metrics_sarimax,
                                            "residual_diagnostics": diag_sarimax,
                                            "aic": float(res_sarimax.aic)},
        },
        "best_model": min(
            [("ARIMA_best_aic_no_seasonality", metrics_arima["RMSE"]),
             ("SARIMA_seasonal_no_exog", metrics_sarima["RMSE"]),
             ("SARIMAX_seasonal_plus_exog", metrics_sarimax["RMSE"])],
            key=lambda x: x[1])[0],
    }
    with open(f"{RESDIR}/part4_sarimax.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2, default=str))
    return summary, res_sarimax, y_pred_sarimax


def make_exog(df):
    hour = df.index.hour
    ex = pd.DataFrame({
        "T_out": df["T_out"].values,
        "RH_out": df["RH_out"].values,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
    }, index=df.index)
    return ex


if __name__ == "__main__":
    import sys
    from part2_3_benchmarks import train_test_split
    hourly = pd.read_csv(f"{RESDIR}/hourly_data.csv", index_col=0, parse_dates=True)
    train, test = train_test_split(hourly, test_days=14)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"

    if stage == "grid":
        subset = train["Appliances"].iloc[-24 * 45:]
        grid = grid_search_aic(subset)
        grid.to_csv(f"{RESDIR}/part4_grid_search.csv", index=False)
        print(grid.head(10))

    elif stage == "arima":
        grid = pd.read_csv(f"{RESDIR}/part4_grid_search.csv")
        best = grid.iloc[0]
        order = (int(best.p), int(best.d), int(best.q))
        res = fit_sarimax(train["Appliances"], order, (0, 0, 0, 0))
        diag = residual_diagnostics(res, "arima_bestaic")
        metrics, y_pred, ci = forecast_and_eval(res, test, 24, tag="arima_bestaic")
        out = {"order": order, "metrics_24h": metrics, "residual_diagnostics": diag,
               "aic": float(res.aic), "bic": float(res.bic)}
        with open(f"{RESDIR}/part4_arima_bestaic.json", "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2))

    elif stage == "sarima":
        grid = pd.read_csv(f"{RESDIR}/part4_grid_search.csv")
        best = grid.iloc[0]
        order = (int(best.p), int(best.d), int(best.q))
        seasonal_order = (1, 1, 1, 24)
        res = fit_sarimax(train["Appliances"], order, seasonal_order)
        diag = residual_diagnostics(res, "sarima_seasonal")
        metrics, y_pred, ci = forecast_and_eval(res, test, 24, tag="sarima_seasonal")
        out = {"order": order, "seasonal_order": seasonal_order, "metrics_24h": metrics,
               "residual_diagnostics": diag, "aic": float(res.aic), "bic": float(res.bic)}
        with open(f"{RESDIR}/part4_sarima_seasonal.json", "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2))

    elif stage == "sarimax":
        grid = pd.read_csv(f"{RESDIR}/part4_grid_search.csv")
        best = grid.iloc[0]
        order = (int(best.p), int(best.d), int(best.q))
        seasonal_order = (1, 1, 1, 24)
        exog_train = make_exog(train)
        exog_test = make_exog(test)
        res = fit_sarimax(train["Appliances"], order, seasonal_order, exog_train)
        diag = residual_diagnostics(res, "sarimax_full")
        metrics, y_pred, ci = forecast_and_eval(res, test, 24, exog_test=exog_test.iloc[:24],
                                                 tag="sarimax_full")
        out = {"order": order, "seasonal_order": seasonal_order,
               "exog": list(exog_train.columns), "metrics_24h": metrics,
               "residual_diagnostics": diag, "aic": float(res.aic), "bic": float(res.bic),
               "forecast": y_pred.tolist(), "ci_lower": ci.iloc[:24, 0].tolist(),
               "ci_upper": ci.iloc[:24, 1].tolist(), "actual": test["Appliances"].iloc[:24].tolist(),
               "forecast_index": [str(t) for t in test.index[:24]]}
        with open(f"{RESDIR}/part4_sarimax_full.json", "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({k: v for k, v in out.items() if k not in
                           ["forecast", "ci_lower", "ci_upper", "actual", "forecast_index"]}, indent=2))

    else:
        run(hourly, train, test)
