import pandas as pd
from prophet import Prophet
from utils.logging_config import get_logger
from utils.metrics import calculate_metrics
from data_preparation import generate_rolling_windows

logger = get_logger(__name__)

def train_prophet(df, barcode):
    df = df.dropna()
    df = df.sort_values("Date")
    results = []
    windows = generate_rolling_windows(df[df["Date"] < "2024-01-01"])

    for i, w in enumerate(windows):
        train, test = w["train"], w["test"]
        df_train = train[["Date", "Quantity"]].rename(columns={"Date": "ds", "Quantity": "y"})
        df_test = test[["Date", "Quantity"]].rename(columns={"Date": "ds", "Quantity": "y"})

        model = Prophet()
        model.fit(df_train)
        forecast = model.predict(df_test)
        y_pred = forecast["yhat"].values
        y_test = df_test["y"].values
        metrics = calculate_metrics(y_test, y_pred)

        logger.info(f"{barcode} | Prophet - Janela {i+1} | Treino: {train['Date'].min()} a {train['Date'].max()} | Teste: {test['Date'].min()} a {test['Date'].max()} | MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%")

        test_out = test[["Date", "Quantity"]].copy()
        test_out["prediction_prophet"] = y_pred
        results.append(test_out)

    # Predição mensal sequencial para 2024
    forecast_2024 = []
    full_train = df[df["Date"] < "2024-01-01"][["Date", "Quantity"]].rename(columns={"Date": "ds", "Quantity": "y"})
    model = Prophet()
    model.fit(full_train)

    for month in range(1, 13):
        future_dates = pd.date_range(start=f"2024-{month:02d}-01", end=f"2024-{month:02d}-28")
        df_future = pd.DataFrame({"ds": future_dates})
        forecast = model.predict(df_future)
        y_pred = forecast["yhat"].values
        y_real = df[df["Date"].isin(future_dates)]["Quantity"].values
        metrics = calculate_metrics(y_real, y_pred)
        logger.info(f"{barcode} | Prophet - Predição 2024-{month:02d} | MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%")
        forecast_2024.append(pd.DataFrame({"Date": future_dates, "prediction_prophet": y_pred}))

    return pd.concat(results), metrics, pd.concat(forecast_2024)
