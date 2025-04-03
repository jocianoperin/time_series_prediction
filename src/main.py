import os
import gc
from src.utils.logging_config import get_logger
from src.data_preparation import load_data
from src.feature_engineering import create_features
from src.train_arima import train_arima
from src.train_prophet import train_prophet
from src.train_xgboost import train_xgboost
from src.train_nn import train_neural_network
from src.compare_models import compare_and_save_results

logger = get_logger(__name__)

def main():
    logger.info("Iniciando pipeline de comparacao de modelos de series temporais...")

    # 1) Carregar dados
    data_dict = load_data(raw_data_path="data/raw")

    # 2) Loop em cada produto
    for barcode, df in data_dict.items():
        # Exemplo de critério mínimo
        if len(df) < 50:
            logger.info(f"Dados insuficientes para {barcode}. Ignorando...")
            continue

        # 3) Criar features (se formos usar em XGBoost e NN)
        df_features = create_features(df, n_lags=7)

        # Se for ARIMA e Prophet, normalmente usamos outra abordagem,
        # mas podemos usar df_features também se quisermos (exógenas).

        # 4) Treinar/capturar previsões e métricas
        arima_preds, arima_metrics = train_arima(df, barcode)
        prophet_preds, prophet_metrics = train_prophet(df, barcode)
        xgb_preds, xgb_metrics = train_xgboost(df_features, barcode)  # usa df_features
        nn_preds, nn_metrics = train_neural_network(df_features, barcode)

        # 5) Salvar previsões em CSV
        # Poderia criar subpastas "predictions/arima/barcode.csv" etc.
        arima_preds.to_csv(f"predictions/arima/{barcode}_preds.csv", index=False)
        prophet_preds.to_csv(f"predictions/prophet/{barcode}_preds.csv", index=False)
        xgb_preds.to_csv(f"predictions/xgboost/{barcode}_preds.csv", index=False)
        nn_preds.to_csv(f"predictions/nn/{barcode}_preds.csv", index=False)

        # 6) Comparar métricas e salvar
        results = {
            "arima": arima_metrics,
            "prophet": prophet_metrics,
            "xgboost": xgb_metrics,
            "nn": nn_metrics
        }
        compare_and_save_results(barcode, results, out_path="predictions")

        # Limpar memória se forem muitos produtos
        del df_features, arima_preds, prophet_preds, xgb_preds, nn_preds
        gc.collect()

    logger.info("Pipeline finalizado com sucesso!")


if __name__ == "__main__":
    main()
