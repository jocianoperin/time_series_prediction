# Atualizado feature_engineering.py com novas features
# ============================================================
#  FEATURE ENGINEERING PARA MODELOS DE PREVISÃO
#  Geração de lags, médias móveis, variáveis temporais e taxas
# ============================================================
 
import pandas as pd
import numpy as np
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------
# GERA AS FEATURES PARA TREINAMENTO DO MODELO
# ------------------------------------------------------------
def create_features(df: pd.DataFrame, n_lags: int = 7) -> pd.DataFrame:
    """
    Gera colunas adicionais com base na série temporal recebida,
    como lags, médias móveis, feriados, promoções e variáveis
    temporais cíclicas.

    Parâmetros:
    - df: DataFrame original contendo vendas e atributos por data
    - n_lags: número de lags personalizados (default = 7)

    Retorna:
    - df enriquecido com novas features, sem valores nulos
    """

    logger.info("Iniciando criação de features...")

    df = df.copy()
    df = df.sort_values("Date")

    # Remove 'TotalValue' para evitar vazamento de informação
    if "TotalValue" in df.columns:
        df.drop(columns=["TotalValue"], inplace=True)
        logger.debug("Coluna 'TotalValue' removida para evitar vazamento")

    # --------------------------------------------------------
    # FEATURES TEMPORAIS BÁSICAS
    # --------------------------------------------------------
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # --------------------------------------------------------
    # HOLIDAY (BINÁRIO + LAG)
    # --------------------------------------------------------
    if "Holiday" in df.columns:
        df["Holiday"]      = df["Holiday"].astype(int)
        df["holiday_lag_1"]= df["Holiday"].shift(1).fillna(0).astype(int)
        logger.debug("Feature de feriado e seu lag adicionados")

    # --------------------------------------------------------
    # LAGS E MÉDIAS MÓVEIS
    # --------------------------------------------------------
    for lag in [1, 7, 14, 21, 30]:
        df[f"lag_{lag}"] = df["Quantity"].shift(lag)

    df["rolling_mean_7"] = df["Quantity"].shift(1).rolling(window=7).mean()
    df["rolling_std_7"] = df["Quantity"].shift(1).rolling(window=7).std()

    # --------------------------------------------------------
    # INFLUÊNCIAS DO DIA ANTERIOR
    # --------------------------------------------------------
    df["promo_lag_1"] = df["OnPromotion"].shift(1)

    # --------------------------------------------------------
    # TAXAS DERIVADAS
    # --------------------------------------------------------
    df["discount_rate"] = df["Discount"] / (df["UnitValue"] + df["Discount"])
    df["increase_rate"] = df["Increase"] / (df["UnitValue"] + 1e-5)
    df["returned_rate"]  = df["ReturnedQuantity"].shift(1) / (df["lag_1"] + 1e-5)
    df["profit_margin"] = (df["UnitValue"] - df["CostValue"]) / (df["UnitValue"] + 1e-5)

    # --------------------------------------------------------
    # MARCADORES TEMPORAIS E PADRÕES RECORRENTES
    # --------------------------------------------------------
    df["day_of_year"] = df["Date"].dt.dayofyear
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["start_month"] = (df["Date"].dt.day <= 3).astype(int)  # primeiros 3 dias do mês
    df["end_month"] = (df["Date"].dt.day >= df["Date"].dt.days_in_month - 2).astype(int)  # últimos 3 dias do mês

    # --------------------------------------------------------
    # ENCODING CÍCLICO – DIA DA SEMANA / MÊS
    # --------------------------------------------------------
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["m_sin"]   = np.sin(2 * np.pi * (df["month"]-1) / 12)
    df["m_cos"]   = np.cos(2 * np.pi * (df["month"]-1) / 12)

    # --------------------------------------------------------
    # REMOVE LINHAS INICIAIS COM NaNs DEVIDO A SHIFTS E MÉDIAS
    # --------------------------------------------------------
    df = df.dropna()

    logger.info(f"Features criadas com sucesso — shape final: {df.shape}")
    return df