import os
import pandas as pd
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def load_data(raw_data_path="data/raw"):
    """
    Lê todos os CSVs de 'produto_*.csv' no diretório fornecido e retorna um dicionário {barcode: DataFrame}.
    """
    data_dict = {}
    for file_name in os.listdir(raw_data_path):
        if file_name.startswith("produto_") and file_name.endswith(".csv"):
            barcode = file_name.replace("produto_", "").replace(".csv", "")
            file_path = os.path.join(raw_data_path, file_name)
            try:
                df = pd.read_csv(file_path)
                # Valida colunas mínimas necessárias
                required_cols = ["Date", "Barcode", "Quantity"]
                if all(col in df.columns for col in required_cols):
                    # Converte Date para datetime
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.sort_values("Date").reset_index(drop=True)
                    data_dict[barcode] = df
                else:
                    logger.info(f"Arquivo {file_name} não possui todas as colunas necessárias. Ignorando.")
            except Exception as e:
                logger.info(f"Falha ao ler {file_name}: {e}")

    logger.info(f"Foram carregados {len(data_dict)} DataFrames de produtos.")
    return data_dict
