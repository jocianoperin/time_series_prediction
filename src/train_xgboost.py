# Refatorado src/train_nn.py com arquitetura aprimorada + Hiperparâmetros XGBoost otimizados em train_xgboost.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from utils.logging_config import get_logger
from utils.metrics import calculate_metrics
from data_preparation import generate_rolling_windows

logger = get_logger(__name__)

def train_xgboost(df, barcode):
    df = df.dropna()
    df = df.sort_values("Date")
    results = []
    windows = generate_rolling_windows(df[df["Date"] < "2024-01-01"])

    for i, w in enumerate(windows):
        train, test = w["train"], w["test"]
        features = [col for col in df.columns if col not in ["Date", "Quantity"]]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[features])
        y_train = train["Quantity"].values
        X_test = scaler.transform(test[features])
        y_test = test["Quantity"].values

        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="gpu_hist"
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)

        logger.info(f"{barcode} | XGBoost - Janela {i+1} | Treino: {train['Date'].min()} a {train['Date'].max()} | Teste: {test['Date'].min()} a {test['Date'].max()} | MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%")

        test_out = test[["Date", "Quantity"]].copy()
        test_out["prediction_xgboost"] = y_pred
        results.append(test_out)

    # Predição mês a mês 2024 com fine-tuning
    forecast_2024 = []
    features = [col for col in df.columns if col not in ["Date", "Quantity"]]
    scaler = StandardScaler()
    train_df = df[df["Date"] < "2024-01-01"].copy().dropna()
    X_train_full = scaler.fit_transform(train_df[features])
    y_train_full = train_df["Quantity"].values

    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="gpu_hist"
    )
    model.fit(X_train_full, y_train_full)

    for month in range(1, 13):
        future_df = df[(df["Date"] >= f"2024-{month:02d}-01") & (df["Date"] <= f"2024-{month:02d}-28")].copy()
        if future_df.empty:
            continue

        X_future = scaler.transform(future_df[features])
        y_real = future_df["Quantity"].values
        y_pred = model.predict(X_future)

        metrics = calculate_metrics(y_real, y_pred)
        logger.info(f"{barcode} | XGBoost - Predição 2024-{month:02d} | MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%")
        forecast_2024.append(pd.DataFrame({"Date": future_df["Date"].values, "prediction_xgboost": y_pred}))

        # Fine-tune incremental
        X_month = scaler.transform(future_df[features])
        y_month = y_real
        model.fit(X_month, y_month, xgb_model=model.get_booster())

    return pd.concat(results), metrics, pd.concat(forecast_2024)
