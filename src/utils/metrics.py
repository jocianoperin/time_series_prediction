import numpy as np

def _to_np(a):          # garante ndarray unidimensional
    return np.asarray(a).reshape(-1)

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = _to_np(y_true), _to_np(y_pred)

    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # evita divisão por zero
    mask        = y_true != 0
    mape        = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    smape_denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_mask  = smape_denom != 0
    smape       = np.mean(np.abs(y_true[smape_mask] - y_pred[smape_mask]) / smape_denom[smape_mask]) * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
    }
