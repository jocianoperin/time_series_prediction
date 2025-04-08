import os
import pandas as pd


def carregar_dados(pasta):
    """
    Carrega todos os arquivos CSV de uma pasta e retorna um dicionário com o código de barras como chave.
    """
    dados = {}
    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".csv") and arquivo.startswith("produto_"):
            caminho = os.path.join(pasta, arquivo)
            try:
                df = pd.read_csv(caminho, parse_dates=["Date"])
                cod_barras = arquivo.replace("produto_", "").replace(".csv", "")
                dados[cod_barras] = df
            except Exception as e:
                print(f"Erro ao carregar {arquivo}: {e}")
    return dados


def generate_rolling_windows(df, window_size=395, train_size=365, step=30):
    """
    Gera janelas deslizantes para séries temporais.

    Cada janela tem `window_size` dias no total:
    - `train_size` dias para treino
    - o restante para teste (ex: 30 dias)

    Parâmetros:
    - df: DataFrame com as colunas 'Date' e 'Quantity'
    - window_size: tamanho total da janela
    - train_size: tamanho da parte de treino
    - step: passo de deslizamento (ex: 30 dias)

    Retorna:
    - Lista de dicionários com 'train' e 'test'
    """
    windows = []
    df = df.sort_values("Date").reset_index(drop=True)
    for start in range(0, len(df) - window_size + 1, step):
        end = start + window_size
        window_df = df.iloc[start:end].copy()
        if len(window_df) < window_size:
            continue
        train = window_df.iloc[:train_size]
        test = window_df.iloc[train_size:]
        windows.append({"train": train, "test": test})
    return windows
