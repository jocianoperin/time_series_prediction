import pandas as pd
import numpy as np
from prophet import Prophet
import gc
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def train_prophet(df, barcode):
    """
    Recebe DataFrame com [Date, Quantity, ...].
    """
    # Separar treino/teste
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    # Monta DF para Prophet
    train_prophet = train_df.rename(columns={"Date": "ds", "Quantity": "y"})
    test_prophet = test_df.rename(columns={"Date": "ds", "Quantity": "y"})

    # Instancia Prophet
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(train_prophet[["ds", "y"]])

    # Previsão no conjunto de teste
    future = test_prophet[["ds"]].copy()
    forecast = model.predict(future)

    # Prophet retorna coluna 'yhat' como previsão
    test_df["prediction_prophet"] = forecast["yhat"].values

    mae = np.mean(np.abs(test_df["Quantity"] - test_df["prediction_prophet"]))
    mape = np.mean(np.abs((test_df["Quantity"] - test_df["prediction_prophet"]) / test_df["Quantity"])) * 100
    logger.info(f"{barcode} - Prophet: MAE={mae:.2f}, MAPE={mape:.2f}%")

    del model
    gc.collect()

    return test_df[["Date", "Quantity", "prediction_prophet"]], (mae, mape)
