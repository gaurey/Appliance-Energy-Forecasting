"""
Part 5 - Feature engineering: lag, rolling-window, time-of-day, day-of-week,
         indoor/outdoor sensor and weather covariates.
Part 6 - Feature-based ML model (XGBoost). Last 14 days held out as test.
         24h-ahead forecasts produced via rolling-origin walk-forward
         evaluation, matching the protocol used for the benchmarks/SARIMAX.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
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


def build_features(hourly):
    """Construct a feature set safe for genuine multi-step forecasting:
    - calendar features (known at any future time -> safe)
    - lag features of the target at lags >= 24h (so that forecasting the
      next 24 hours never needs unseen target values)
    - rolling statistics computed on lagged windows (also safe)
    - weather/indoor sensor features are included via their own lag-24
      versions (see note in Part 9 discussion re: true vs conditional
      forecasts) so the model never conditions on future weather.
    """
    df = hourly.copy()

    # calendar / time features
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # lag features of the target, all >=24h so a 24h-ahead forecast never
    # needs a target value that would not yet be known at the forecast origin
    for lag in [24, 25, 48, 72, 168]:
        df[f"appl_lag{lag}"] = df["Appliances"].shift(lag)

    # rolling statistics on lagged windows (computed on data ending 24h back)
    lag24 = df["Appliances"].shift(24)
    df["appl_roll24_mean"] = lag24.rolling(24).mean()
    df["appl_roll24_std"] = lag24.rolling(24).std()
    df["appl_roll168_mean"] = lag24.rolling(168).mean()

    # sensor / weather covariates, lagged 24h (available at forecast origin
    # for a genuine forecast; using same-hour values would be a conditional
    # forecast - see Part 9 discussion)
    weather_cols = ["T_out", "RH_out", "Windspeed", "Visibility", "Tdewpoint",
                     "Press_mm_hg", "T1", "RH_1", "T3", "RH_3"]
    for col in weather_cols:
        df[f"{col}_lag24"] = df[col].shift(24)

    df = df.dropna()
    feature_cols = [c for c in df.columns if c not in
                     ["Appliances", "lights", "hour", "dow"] and
                     c not in [c2 for c2 in hourly.columns if c2 not in
                               ["hour", "dow"]]]
    # keep only engineered features (exclude raw un-lagged sensor columns)
    feature_cols = [c for c in df.columns if c.endswith(("_sin", "_cos")) or
                     c.startswith("appl_lag") or c.startswith("appl_roll") or
                     c.endswith("_lag24") or c == "is_weekend"]
    return df, feature_cols


def rolling_origin_xgb(df, feature_cols, test_days=14, horizon=24):
    test_hours = 24 * test_days
    origins = list(range(len(df) - test_hours, len(df) - horizon + 1, 24))

    all_true, all_pred = [], []
    per_origin_metrics = []
    importances = []
    last_model = None

    for origin in origins:
        train_df = df.iloc[:origin]
        test_df = df.iloc[origin:origin + horizon]
        if len(test_df) < horizon:
            continue

        X_train, y_train = train_df[feature_cols], train_df["Appliances"]
        X_test, y_test = test_df[feature_cols], test_df["Appliances"]

        model = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            reg_lambda=1.0, n_jobs=4,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_true.extend(y_test.values.tolist())
        all_pred.extend(y_pred.tolist())
        per_origin_metrics.append(all_metrics(y_test.values, y_pred))
        importances.append(model.feature_importances_)
        last_model = model
        last_test_index = test_df.index
        last_y_test = y_test.values
        last_y_pred = y_pred

    overall = all_metrics(all_true, all_pred)
    mean_importance = np.mean(importances, axis=0)
    imp_series = pd.Series(mean_importance, index=feature_cols).sort_values(ascending=False)

    return overall, per_origin_metrics, imp_series, (last_test_index, last_y_test, last_y_pred), last_model


def ablation_study(df, feature_cols, test_days=14, horizon=24):
    """Compare feature groups to see which add most value."""
    groups = {
        "lags_only": [c for c in feature_cols if c.startswith("appl_lag")],
        "lags_plus_calendar": [c for c in feature_cols if c.startswith("appl_lag") or
                                c.endswith(("_sin", "_cos")) or c == "is_weekend"],
        "lags_plus_rolling": [c for c in feature_cols if c.startswith("appl_lag") or
                               c.startswith("appl_roll")],
        "full_feature_set": feature_cols,
    }
    results = {}
    for name, cols in groups.items():
        overall, _, _, _, _ = rolling_origin_xgb(df, cols, test_days, horizon)
        results[name] = overall
    return results


def plot_importance(imp_series):
    fig, ax = plt.subplots(figsize=(8, 6))
    imp_series.head(15).sort_values().plot(kind="barh", ax=ax, color="#1f6f78")
    ax.set_title("XGBoost Feature Importance (top 15, mean over rolling origins)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/11_xgb_feature_importance.png")
    plt.close(fig)


def plot_forecast(example, tag="xgb"):
    idx, y_true, y_pred = example
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(idx, y_true, label="Actual", color="black", lw=2)
    ax.plot(idx, y_pred, label="XGBoost forecast", color="#1f6f78", lw=1.5, ls="--")
    ax.set_title("XGBoost 24h Forecast vs Actual (final test-window origin)")
    ax.set_ylabel("Appliances (Wh/hour)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/12_xgb_forecast.png")
    plt.close(fig)


def run():
    hourly = pd.read_csv(f"{RESDIR}/hourly_data.csv", index_col=0, parse_dates=True)
    df, feature_cols = build_features(hourly)

    overall, per_origin, imp_series, example, model = rolling_origin_xgb(df, feature_cols)
    plot_importance(imp_series)
    plot_forecast(example)

    ablation = ablation_study(df, feature_cols)

    summary = {
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "rolling_origin_overall_metrics": overall,
        "feature_group_ablation_rmse": {k: v["RMSE"] for k, v in ablation.items()},
        "feature_group_ablation_full": ablation,
        "top_10_features_by_importance": imp_series.head(10).to_dict(),
    }
    with open(f"{RESDIR}/part5_6_xgboost.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    run()
