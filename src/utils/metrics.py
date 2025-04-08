import numpy as np

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    safe_y_true = np.where(y_true == 0, 1e-5, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / safe_y_true)) * 100
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-5))
    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape}