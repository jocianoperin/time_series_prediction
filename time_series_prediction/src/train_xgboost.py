import xgboost as xgb
import numpy as np
import pandas as pd
import gc
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def train_xgboost(df, barcode):
    """
    df já deve conter colunas de features (lags, rolling, etc.) e a target 'Quantity'.
    """
    # Exemplo de separação
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    # X e y
    feature_cols = [col for col in df.columns if col not in ["Date", "Quantity"]]
    X_train = train_df[feature_cols]
    y_train = train_df["Quantity"]
    X_test = test_df[feature_cols]
    y_test = test_df["Quantity"]

    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

    preds = model.predict(X_test)
    test_df["prediction_xgb"] = preds

    mae = np.mean(np.abs(y_test - preds))
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    logger.info(f"{barcode} - XGBoost: MAE={mae:.2f}, MAPE={mape:.2f}%")

    # Salvar modelo (opcional)
    # model.save_model(f"models/xgboost/{barcode}_xgb.json")

    del model
    gc.collect()

    return test_df[["Date", "Quantity", "prediction_xgb"]], (mae, mape)
