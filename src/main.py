# main.py (exemplo simplificado)
import os
import traceback
import pandas as pd
from data_preparation import carregar_dados
from train_arima import train_arima_daily_2024
from utils.logging_config import get_logger
from train_xgboost import train_xgboost
from train_nn import train_neural_network
from train_prophet import train_prophet
from compare_models import compare_and_save_results
from feature_engineering import create_features
from utils.metrics import calculate_metrics

logger = get_logger(__name__)

def main():
    logger.info("Iniciando pipeline diário 2024...")

    dados = carregar_dados("data/raw")
    logger.info(f"Dados carregados: {len(dados)} produtos encontrados.")

    for barcode, df in dados.items():
        try:
            logger.info(f"Processando produto {barcode}...")

            # Criação de features
            df = create_features(df)

            # Rodar todos os modelos
            results = {}

            # ARIMA
            logger.info("Iniciando pipeline ARIMA diário 2024...")
            df_arima = train_arima_daily_2024(df, barcode)
            df_arima.rename(columns={"forecast": "prediction_arima", "real": "Quantity"}, inplace=True)
            arima_metrics = calculate_metrics(df_arima["Quantity"], df_arima["prediction_arima"])
            results["arima"] = {"metrics": arima_metrics, "predictions": df_arima}
            logger.info("Finalizado processo diário de ARIMA.")

            """# XGBoost
            logger.info("Iniciando pipeline XGBoost diário 2024...")
            df_xgb_all, xgb_metrics, df_xgb_2024 = train_xgboost(df, barcode)
            results["xgboost"] = {"metrics": xgb_metrics, "predictions": df_xgb_2024}
            logger.info("Finalizado processo diário de XGBoost.")

            # LSTM
            logger.info("Iniciando pipeline LSTM diário 2024...")
            df_nn_all, nn_metrics, df_nn_2024 = train_neural_network(df, barcode)
            results["nn"] = {"metrics": nn_metrics, "predictions": df_nn_2024}
            logger.info("Finalizado processo diário de LSTM.")"""

            """# Prophet
            logger.info("Iniciando pipeline Prophet diário 2024...")
            df_prophet_all, prophet_metrics, df_prophet_2024 = train_prophet(df, barcode)
            results["prophet"] = {"metrics": prophet_metrics, "predictions": df_prophet_2024}
            logger.info("Finalizado processo diário de Prophet.")"""

            # Salvar e comparar
            compare_and_save_results(barcode, results)

        except Exception as e:
            logger.error(f"Erro no produto {barcode}: {e}")
            logger.error(traceback.format_exc())

    logger.info("Finalizado processo diário de todos os modelos.")

if __name__ == "__main__":
    main()
