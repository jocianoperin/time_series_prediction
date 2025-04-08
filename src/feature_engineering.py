# Atualizado feature_engineering.py com novas features
import pandas as pd

def create_features(df: pd.DataFrame, n_lags: int = 7) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("Date")

    # Data/time features
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Lags e médias móveis
    for lag in [1, 7, 14]:
        df[f"lag_{lag}"] = df["Quantity"].shift(lag)

    df["rolling_mean_7"] = df["Quantity"].shift(1).rolling(window=7).mean()
    df["rolling_std_7"] = df["Quantity"].shift(1).rolling(window=7).std()

    # Promoção e feriado anteriores
    df["promo_lag_1"] = df["OnPromotion"].shift(1)

    # Taxas derivadas
    df["discount_rate"] = df["Discount"] / (df["UnitValue"] + df["Discount"])
    df["increase_rate"] = df["Increase"] / (df["UnitValue"] + 1e-5)
    df["returned_rate"] = df["ReturnedQuantity"] / (df["Quantity"] + 1e-5)
    df["profit_margin"] = (df["UnitValue"] - df["CostValue"]) / (df["UnitValue"] + 1e-5)

    # Remove linhas iniciais com NaNs por causa dos shifts
    df = df.dropna()

    return df