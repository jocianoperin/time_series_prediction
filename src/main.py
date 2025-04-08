# main.py (exemplo simplificado)
import os
import pandas as pd
from data_preparation import carregar_dados
from train_arima import train_arima_daily_2024
from utils.logging_config import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Iniciando pipeline ARIMA diário 2024...")

    dados = carregar_dados("data/raw")
    logger.info(f"Dados carregados: {len(dados)} produtos encontrados.")

    for barcode, df in dados.items():
        try:
            logger.info(f"Processando produto {barcode}...")
            df_2024 = train_arima_daily_2024(df, barcode)
            # df_2024 conterá todas as previsões diárias consolidadas para 2024 (se quiser usar)
        except Exception as e:
            logger.error(f"Erro no produto {barcode}: {e}")

    logger.info("Finalizado processo diário de ARIMA.")

if __name__ == "__main__":
    main()
