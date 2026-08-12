"""
Part 8 - Consolidated comparison of every model in the study, using the
same rolling-origin 24h protocol and metric set throughout.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt

RESDIR = "results"
FIGDIR = "figs"


def run():
    with open(f"{RESDIR}/part2_3_benchmarks.json") as f:
        p23 = json.load(f)
    with open(f"{RESDIR}/part4_arima_bestaic.json") as f:
        p4a = json.load(f)
    with open(f"{RESDIR}/part4_sarima_seasonal.json") as f:
        p4s = json.load(f)
    with open(f"{RESDIR}/part4_sarimax_full.json") as f:
        p4x = json.load(f)
    with open(f"{RESDIR}/part5_6_xgboost.json") as f:
        p6 = json.load(f)
    with open(f"{RESDIR}/part7_foundation_model.json") as f:
        p7 = json.load(f)

    rows = []
    for name, m in p23["benchmark_results_mean_over_origins"].items():
        rows.append({"Model": name.replace("_", " "), "RMSE": m["RMSE"], "MAE": m["MAE"],
                      "MAPE": m["MAPE"], "sMAPE": m["sMAPE"], "Family": "Benchmark"})

    rows.append({"Model": "ARIMA(1,1,6) best-AIC, no seasonality", "RMSE": p4a["metrics_24h"]["RMSE"],
                 "MAE": p4a["metrics_24h"]["MAE"], "MAPE": None, "sMAPE": None, "Family": "SARIMA/X (single origin)"})
    rows.append({"Model": "SARIMA(1,1,6)(1,1,1)[24]", "RMSE": p4s["metrics_24h"]["RMSE"],
                 "MAE": p4s["metrics_24h"]["MAE"], "MAPE": None, "sMAPE": None, "Family": "SARIMA/X (single origin)"})
    rows.append({"Model": "SARIMAX + weather/hour exog", "RMSE": p4x["metrics_24h"]["RMSE"],
                 "MAE": p4x["metrics_24h"]["MAE"], "MAPE": None, "sMAPE": None, "Family": "SARIMA/X (single origin)"})

    m = p6["rolling_origin_overall_metrics"]
    rows.append({"Model": "XGBoost (full feature set)", "RMSE": m["RMSE"], "MAE": m["MAE"],
                 "MAPE": m["MAPE"], "sMAPE": m["sMAPE"], "Family": "ML"})

    m7 = p7["rolling_origin_metrics"]
    rows.append({"Model": p7["model_used"], "RMSE": m7["RMSE"], "MAE": m7["MAE"],
                 "MAPE": m7["MAPE"], "sMAPE": m7["sMAPE"], "Family": "Foundation model"})

    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    table.to_csv(f"{RESDIR}/part8_comparison_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"Benchmark": "#999999", "SARIMA/X (single origin)": "#c0562d",
              "ML": "#1f6f78", "Foundation model": "#7a8fbf"}
    bar_colors = [colors[f] for f in table["Family"]]
    ax.barh(table["Model"], table["RMSE"], color=bar_colors)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (Wh/hour)")
    ax.set_title("Model Comparison - 24h-ahead Forecast RMSE")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/14_model_comparison.png")
    plt.close(fig)

    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    run()
