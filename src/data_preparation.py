# ============================================================
#  PREPARAÇÃO INICIAL DOS DADOS
#  Carregamento e estruturação em janelas deslizantes (rolling)
# ============================================================

import os
import pandas as pd
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------
# CARREGA DADOS CSV DA PASTA 'raw' → DICIONÁRIO POR BARCODE
# ------------------------------------------------------------
def carregar_dados(pasta):
    """
    Lê todos os arquivos .csv da pasta especificada e retorna um dicionário
    com os DataFrames, indexados pelo código de barras do produto.

    Espera arquivos nomeados como: produto_<barcode>.csv
    """
    dados = {}

    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".csv") and arquivo.startswith("produto_"):
            caminho = os.path.join(pasta, arquivo)
            try:
                df = pd.read_csv(caminho, parse_dates=["Date"])
                cod_barras = arquivo.replace("produto_", "").replace(".csv", "")
                dados[cod_barras] = df
                logger.info(f"Arquivo carregado com sucesso: {arquivo}")

            except Exception as e:
                logger.error(f"Erro ao carregar {arquivo}: {e}")

    return dados

# ------------------------------------------------------------
# GERA JANELAS ROLLING DE TREINO E TESTE (365+31)
# ------------------------------------------------------------
def generate_rolling_windows(df: pd.DataFrame, train_days=365, test_days=31, step_days=30):
    """
    Divide um DataFrame em janelas temporais deslizantes para treino e teste.

    Parâmetros:
    - train_days: número de dias para treino (default: 365)
    - test_days: número de dias para teste (default: 31)
    - step_days: avanço em dias entre uma janela e outra (default: 30)

    Retorna uma lista de dicionários no formato:
    [
        {"train": df_treino_1, "test": df_teste_1},
        {"train": df_treino_2, "test": df_teste_2},
        ...
    ]
    """

    df = df.sort_values("Date").reset_index(drop=True)
    windows = []

    start = 0
    total_days = train_days + test_days

    while start + total_days <= len(df):
        df_train = df.iloc[start:start + train_days].copy()
        df_test = df.iloc[start + train_days:start + total_days].copy()

        windows.append({"train": df_train, "test": df_test})
        
        start += step_days  # Move para a próxima janela

    logger.info(f"{len(windows)} janelas geradas (train: {train_days} dias, test: {test_days} dias)")
    return windows