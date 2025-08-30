import argparse
import warnings
import os
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

def load_data(path, freq="D"):
    df = pd.read_csv(path)
    if "date" not in df.columns or "sales" not in df.columns:
        raise ValueError("CSV must have columns: date,sales")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df

def prepare_prophet_df(df, freq="D"):
    df_resampled = df.set_index("date").resample(freq).sum().reset_index()
    df_resampled = df_resampled.rename(columns={"date": "ds", "sales": "y"})
    return df_resampled

def train_prophet(df, freq="D"):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=(freq in ["D", "W"]),
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.9
    )
    if freq == "M":
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.fit(df)
    return model

def evaluate(model, df, horizon, freq="D"):
    if horizon >= len(df):
        return None, None
    train = df.iloc[:-horizon]
    test = df.iloc[-horizon:]

    m = train_prophet(train, freq)
    future = m.make_future_dataframe(periods=horizon, freq=freq)
    forecast = m.predict(future)
    forecast = forecast.set_index("ds").loc[test["ds"]]

    mae = mean_absolute_error(test["y"], forecast["yhat"])
    mse = mean_squared_error(test["y"], forecast["yhat"])
    rmse = mse ** 0.5
    return mae, rmse

def plot_forecast(df, forecast, outdir, label=""):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(12,6))
    plt.plot(df["ds"], df["y"], label="Actual")
    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast")
    plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.2, label="Confidence Interval")
    plt.title(f"Sales Forecast {label}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"forecast_{label}.png"), dpi=150)
    plt.close()

def monthly_yearly_comparisons(df, outdir, label=""):
    df_copy = df.copy()
    df_copy["month"] = df_copy["ds"].dt.to_period("M").astype(str)
    df_copy["year"] = df_copy["ds"].dt.year

    monthly = df_copy.groupby("month")["y"].sum()
    yearly = df_copy.groupby("year")["y"].sum()

    # Plot monthly
    plt.figure(figsize=(12,5))
    monthly.plot(kind="bar", color="skyblue")
    plt.title(f"Monthly Sales Comparison {label}")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"monthly_sales_{label}.png"), dpi=150)
    plt.close()

    # Plot yearly
    plt.figure(figsize=(8,5))
    yearly.plot(kind="bar", color="lightgreen")
    plt.title(f"Yearly Sales Comparison {label}")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"yearly_sales_{label}.png"), dpi=150)
    plt.close()

    return monthly, yearly

def generate_insights(df, forecast, outdir, label=""):
    insights = []
    last_actual = df["y"].iloc[-1]
    next_forecast = forecast["yhat"].iloc[-1]

    if next_forecast > last_actual:
        insights.append("📈 Sales are expected to increase in the coming period.")
    else:
        insights.append("📉 Sales are expected to decrease in the coming period.")

    df_copy = df.copy()
    df_copy["month"] = df_copy["ds"].dt.month
    monthly_avg = df_copy.groupby("month")["y"].mean()
    best_month = monthly_avg.idxmax()
    worst_month = monthly_avg.idxmin()
    insights.append(f"🏆 Highest average sales occur in month: {best_month}.")
    insights.append(f"📉 Lowest average sales occur in month: {worst_month}.")

    with open(os.path.join(outdir, f"insights_{label}.txt"), "w", encoding="utf-8") as f:
        for line in insights:
            f.write(line + "\n")

    return insights

def run_forecast(df, group_cols, args):
    if group_cols:
        grouped = df.groupby(group_cols)
        for keys, group in grouped:
            label = "_".join(str(k) for k in (keys if isinstance(keys, tuple) else (keys,)))
            outdir = os.path.join(args.outdir, label)
            run_single(group, args, label, outdir)
    else:
        run_single(df, args, "overall", args.outdir)

def run_single(df, args, label, outdir):
    df_resampled = prepare_prophet_df(df, args.freq)
    model = train_prophet(df_resampled, args.freq)

    future = model.make_future_dataframe(periods=args.horizon, freq=args.freq)
    forecast = model.predict(future)

    os.makedirs(outdir, exist_ok=True)
    forecast.to_csv(os.path.join(outdir, f"forecast_{label}.csv"), index=False)

    mae, rmse = evaluate(model, df_resampled, min(args.horizon, max(1, len(df_resampled)//10)), args.freq)
    metrics = pd.DataFrame([{"mae": mae, "rmse": rmse}])
    metrics.to_csv(os.path.join(outdir, f"metrics_{label}.csv"), index=False)

    plot_forecast(df_resampled, forecast, outdir, label)
    monthly_yearly_comparisons(df_resampled, outdir, label)
    insights = generate_insights(df_resampled, forecast, outdir, label)

    print(f"✅ Forecast saved for {label} in {outdir}")
    print(metrics)
    for line in insights:
        print(" -", line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="CSV file with columns: date,sales,[category,store,region...]")
    parser.add_argument("--freq", type=str, default="D", choices=["D", "W", "M"], help="Aggregation frequency")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon (steps)")
    parser.add_argument("--outdir", type=str, default="artifacts", help="Output folder")
    args = parser.parse_args()

    df = load_data(args.data, args.freq)
    group_cols = [c for c in df.columns if c not in ["date", "sales"]]
    run_forecast(df, group_cols, args)

if __name__ == "__main__":
    main()
