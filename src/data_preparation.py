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

def generate_rolling_windows(df: pd.DataFrame, train_days=365, test_days=31, step_days=30):
    df = df.sort_values("Date").reset_index(drop=True)
    
    windows = []
    start = 0
    total_days = train_days + test_days

    while start + total_days <= len(df):
        df_train = df.iloc[start:start + train_days].copy()
        df_test = df.iloc[start + train_days:start + total_days].copy()

        windows.append({"train": df_train, "test": df_test})
        
        start += step_days  # Avança exatamente 30 dias por vez

    return windows