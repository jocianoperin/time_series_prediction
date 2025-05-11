# utils/pred_loader.py  (NOVO ARQUIVO)
# -----------------------------------------------------
import glob, os
import pandas as pd

def load_saved_predictions(barcode: str, model_name: str) -> pd.DataFrame:
    """
    Lê todos os CSVs de predição diária já exportados para um modelo
    específico (XGBoost, NN, etc.) e devolve um único DataFrame
    concatenado, padronizado com colunas:
       Date | real | forecast
    """
    base_dir = f"data/predictions/{model_name}/{barcode}"
    pattern  = f"{model_name}_daily_{barcode}_*.csv"
    paths = glob.glob(os.path.join(base_dir, pattern))
    if not paths:
        return pd.DataFrame()          # nada a reutilizar

    dfs = []
    for p in paths:
        df = pd.read_csv(p, parse_dates=["Date"])
        # cobertura para nomes de colunas ligeiramente diferentes
        if "Quantity" in df.columns:
            df.rename(columns={"Quantity": "real"}, inplace=True)
        dfs.append(df[["Date", "real", "forecast"]])

    return pd.concat(dfs).sort_values("Date").reset_index(drop=True)
