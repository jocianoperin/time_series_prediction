import pandas as pd
import os
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def compare_and_save_results(barcode, results, out_path="predictions"):
    """
    results: dicionário com { 'arima': (mae, mape), 'prophet': (mae, mape), ...}
    Salva o melhor modelo, etc.
    """
    # Converter dict em DataFrame para facilitar
    df_results = pd.DataFrame.from_dict(results, orient='index', columns=["MAE", "MAPE"])
    df_results.reset_index(inplace=True)
    df_results.rename(columns={"index": "Model"}, inplace=True)

    # Salva CSV de métricas do produto
    os.makedirs(os.path.join(out_path, "metrics"), exist_ok=True)
    metrics_file = os.path.join(out_path, "metrics", f"{barcode}_metrics.csv")
    df_results.to_csv(metrics_file, index=False)

    # Log
    best_model = df_results.loc[df_results["MAE"].idxmin(), "Model"]
    logger.info(f"Para o produto {barcode}, o melhor modelo foi: {best_model}")
