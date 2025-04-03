import pandas as pd
import numpy as np
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def create_features(df, n_lags=7):
    """
    Gera colunas de lag e rolling para a coluna 'Quantity'.
    Exemplo: lag1, lag2, ..., lag7.
    Também cria colunas de calendário (dia_da_semana, mês, etc.).
    Retorna um DataFrame com as novas features.
    """
    df = df.copy()

    # Exemplo de lags
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["Quantity"].shift(lag)

    # Exemplo de média móvel de 7 dias
    df["rolling_7"] = df["Quantity"].rolling(window=7).mean()

    # Variáveis de calendário
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    # Remove linhas iniciais que ficarão com NaN por conta dos lags
    df = df.dropna().reset_index(drop=True)

    return df
