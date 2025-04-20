# ============================================================
#  MAIN – PIPELINE DE PREVISÃO DIÁRIA 2024
#  (execução **sequencial** ‑ mas deixando o trecho paralelo
#   comentado para uso futuro)
# ============================================================

# ---------- Ajustes globais de GPU ---------------------------------
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"      # devolve VRAM assim que possível
# -------------------------------------------------------------------

import traceback
import multiprocessing as mp            # ainda importado p/ uso futuro
mp.set_start_method("spawn", force=True)   # garante contexto CUDA limpo
gpu_lock = mp.Semaphore(10)               # semáforo segue, mesmo sequencial

from utils.logging_config import get_logger

from data_preparation   import carregar_dados
from feature_engineering import create_features

from train_xgboost import train_xgboost
from train_nn      import train_neural_network
# from train_arima import train_arima_daily_2024
# from train_prophet import train_prophet
from compare_models import compare_and_save_results
from utils.metrics  import calculate_metrics
from utils.gpu_utils import free_gpu_memory

logger = get_logger(__name__)

# ============================================================
#  FUNÇÃO ISOLADA PARA PROCESSAR UM ÚNICO PRODUTO
# ============================================================
def processar_produto(barcode: str, df_raw, gpu_lock):
    """Executa features → modelos → métricas para um produto."""
    # logger exclusivo por arquivo
    logger = get_logger(f"PROD_{barcode}", log_file=f"logs/{barcode}.log")

    try:
        logger.info(f"Processando produto {barcode}…")

        # 1) Feature engineering
        df = create_features(df_raw)
        results = {}

        # 2) ARIMA ----------------------------------------------------
        # (desabilitado por padrão)
        # try:
        #     from train_arima import train_arima_daily_2024
        #     logger.info("Iniciando ARIMA…")
        #     df_arima = train_arima_daily_2024(df, barcode)
        #     arima_metrics = calculate_metrics(df_arima["Quantity"],
        #                                       df_arima["prediction_arima"])
        #     results["arima"] = {"metrics": arima_metrics,
        #                         "predictions": df_arima}
        #     logger.info("ARIMA concluído.")
        # except Exception as e:
        #     logger.warning(f"ARIMA não executado para {barcode}: {e}")

        # 3) XGBoost --------------------------------------------------
        with gpu_lock:                          # exclusão de GPU
            logger.info("Iniciando XGBoost…")
            df_xgb_all, xgb_metrics, df_xgb_2024 = train_xgboost(df, barcode)
            results["xgboost"] = {"metrics": xgb_metrics,
                                  "predictions": df_xgb_2024}
            logger.info("XGBoost concluído.")

        # 4) LSTM / Attention ----------------------------------------
        logger.info("Iniciando LSTM…")
        df_nn_all, nn_metrics, df_nn_2024 = train_neural_network(df, barcode)
        results["nn"] = {"metrics": nn_metrics, "predictions": df_nn_2024}
        logger.info("LSTM concluído.")

        # 5) Prophet --------------------------------------------------
        # try:
        #     from train_prophet import train_prophet
        #     logger.info("Iniciando Prophet…")
        #     df_prophet_all, prophet_metrics, df_prophet_2024 = \
        #         train_prophet(df, barcode)
        #     results["prophet"] = {"metrics": prophet_metrics,
        #                           "predictions": df_prophet_2024}
        #     logger.info("Prophet concluído.")
        # except Exception as e:
        #     logger.warning(f"Prophet não executado para {barcode}: {e}")

        # 6) Comparativo + salvamento ---------------------------------
        compare_and_save_results(barcode, results)

    except Exception as exc:
        logger.error(f"Erro no produto {barcode}: {exc}")
        logger.error(traceback.format_exc())
    finally:
        free_gpu_memory()        # libera VRAM

# ============================================================
#  EXECUÇÃO SEQUENCIAL
# ============================================================
def main():
    logger.info("Iniciando pipeline diário 2024…")

    dados = carregar_dados("data/raw")
    logger.info(f"Dados carregados: {len(dados)} produtos encontrados.")

    # --------- EXECUÇÃO SEQUENCIAL -----------------------------
    for barcode, df in dados.items():
        processar_produto(barcode, df, gpu_lock)

    # --------- BLOCO PARA EXECUÇÃO EM SUBPROCESSOS -------------
    # (mantenha comentado; habilite se precisar de paralelismo)
    #
    # processes = []
    # for barcode, df in dados.items():
    #     p = mp.Process(target=processar_produto,
    #                    args=(barcode, df, gpu_lock))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
    # -----------------------------------------------------------

    logger.info("Finalizado processo diário de todos os modelos.")

if __name__ == "__main__":
    main()
