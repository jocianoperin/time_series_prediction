# Atualizado feature_engineering.py com novas features
import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame, n_lags: int = 7) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("Date")

    # ——— Evita vazamento: remove TotalValue (que inclui Quantity do mesmo dia)
    if "TotalValue" in df.columns:
        df.drop(columns=["TotalValue"], inplace=True)

    # Data/time features
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # — Holiday binário e lag de feriado —
    # garante que Holiday venha como 0/1
    if "Holiday" in df.columns:
        df["Holiday"]      = df["Holiday"].astype(int)
        df["holiday_lag_1"]= df["Holiday"].shift(1).fillna(0).astype(int)

    # Lags e médias móveis
    for lag in [1, 7, 14, 21, 30]:
        df[f"lag_{lag}"] = df["Quantity"].shift(lag)

    df["rolling_mean_7"] = df["Quantity"].shift(1).rolling(window=7).mean()
    df["rolling_std_7"] = df["Quantity"].shift(1).rolling(window=7).std()

    # Promoção e feriado anteriores
    df["promo_lag_1"] = df["OnPromotion"].shift(1)

    # Taxas derivadas
    df["discount_rate"] = df["Discount"] / (df["UnitValue"] + df["Discount"])
    df["increase_rate"] = df["Increase"] / (df["UnitValue"] + 1e-5)
    df["returned_rate"]  = df["ReturnedQuantity"].shift(1) / (df["lag_1"] + 1e-5)
    df["profit_margin"] = (df["UnitValue"] - df["CostValue"]) / (df["UnitValue"] + 1e-5)

    # variáveis adicionais para reforçar padrões temporais
    df["day_of_year"] = df["Date"].dt.dayofyear
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["start_month"] = (df["Date"].dt.day <= 3).astype(int)  # primeiros 3 dias do mês
    df["end_month"] = (df["Date"].dt.day >= df["Date"].dt.days_in_month - 2).astype(int)  # últimos 3 dias do mês

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["m_sin"]   = np.sin(2 * np.pi * (df["month"]-1) / 12)
    df["m_cos"]   = np.cos(2 * np.pi * (df["month"]-1) / 12)

    # Remove linhas iniciais com NaNs por causa dos shifts
    df = df.dropna()

    return df