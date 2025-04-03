import pmdarima as pm
import numpy as np
import pandas as pd
import gc
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def train_arima(df, barcode):
    """
    df: DataFrame com colunas [Date, Quantity, ... features ...]
    barcode: para salvar modelo e logs.
    Retorna previsões e métricas.
    """
    # Separar treino e teste (ex.: últimos 30 dias para teste)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    # Para ARIMA, geralmente usamos apenas a série principal:
    # se quiser exógenas, passe as features a 'exogenous'
    y_train = train_df["Quantity"]
    y_test = test_df["Quantity"]

    # Se quiser exógenas, exog_train = train_df[["Holiday", "OnPromotion", ...]]
    # exog_test = test_df[["Holiday", "OnPromotion", ...]]
    # Aqui, exemplificando sem exógenas:
    model = pm.auto_arima(
        y_train,
        seasonal=False,  # Se quiser SARIMA, setar True e definir m (sazonalidade)
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    # Previsão no conjunto de teste
    forecast = model.predict(n_periods=len(y_test))
    test_df["prediction_arima"] = forecast

    # Métricas simples
    mae = np.mean(np.abs(y_test - forecast))
    mape = np.mean(np.abs((y_test - forecast) / y_test)) * 100
    logger.info(f"{barcode} - ARIMA: MAE={mae:.2f}, MAPE={mape:.2f}%")

    # Limpa memória se precisar
    del model
    gc.collect()

    return test_df[["Date", "Quantity", "prediction_arima"]], (mae, mape)
