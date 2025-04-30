# ============================================================
#  MAIN – PIPELINE DE PREVISÃO DIÁRIA 2024 (PARALELO CONTROLADO)
#  ------------------------------------------------------------
#  Objetivo:
#    • Rodar XGBoost de forma concorrente (GPU leve)
#    • Rodar Rede Neural (LSTM) com exclusividade da GPU
#    • Consolidar previsões e métricas por produto
# ============================================================

# ----- AJUSTES DE GPU (OTIMIZAÇÃO DE USO DE VRAM) ----------
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# ----- MULTIPROCESSAMENTO COM CONTEXTO LIMPO (CUDA) --------
import multiprocessing as mp
import traceback
mp.set_start_method("spawn", force=True)

# ----- LIMITES DE CONCORRÊNCIA -----------------------------
MAX_PARALLEL_PROCS = 4     # Máximo de processos simultâneos
MAX_XGB_CONCURRENT = 2     # XGBoosts simultâneos na GPU
proc_lock    = mp.Semaphore(MAX_PARALLEL_PROCS)   # controle global
xgb_gpu_lock = mp.Semaphore(MAX_XGB_CONCURRENT)   # GPU leve
nn_gpu_lock  = mp.Semaphore(1)                    # GPU exclusiva para NN

# ----- IMPORTS DO PIPELINE ----------------------------------
from utils.logging_config import get_logger
from data_preparation     import carregar_dados
from feature_engineering  import create_features
from train_xgboost        import train_xgboost
from train_nn             import train_neural_network
from compare_models       import compare_and_save_results
from utils.metrics        import calculate_metrics
from utils.gpu_utils      import free_gpu_memory

logger = get_logger(__name__)

# ============================================================
#  PROCESSA UM ÚNICO PRODUTO – ETAPAS COMPLETAS
# ============================================================
def processar_produto(barcode: str, df_raw, xgb_gpu_lock, nn_gpu_lock, proc_lock):
    """
    Executa a cadeia completa de predição para um produto:
    feature engineering → XGBoost → LSTM → consolidação de métricas.
    """
    
    slot_released = False
    
    try:
        logger.info(f"[{barcode}] Iniciando processamento…")

        # ----- FEATURE ENGINEERING --------------------------
        df = create_features(df_raw)
        results = {}

        # ----- XGBOOST (GPU leve, concorrente) --------------
        with xgb_gpu_lock:
            logger.info("Iniciando XGBoost…")
            df_xgb_all, xgb_metrics, df_xgb_2024 = train_xgboost(df, barcode)
            results["xgboost"] = {"metrics": xgb_metrics,
                                  "predictions": df_xgb_2024}
            
            logger.info(f"[{barcode}] XGBoost concluído")
            free_gpu_memory()
            
            slot_released = True
            proc_lock.release()
            logger.debug(f"[{barcode}] Slot global liberado após XGBoost")

        # ----- LSTM / NN (GPU pesada, uso exclusivo) ---------
        with nn_gpu_lock:
            logger.info("Iniciando LSTM (NN)…")
            df_nn_all, nn_metrics, df_nn_2024 = train_neural_network(df, barcode)
            results["nn"] = {"metrics": nn_metrics,
                             "predictions": df_nn_2024}
            
            logger.info(f"[{barcode}] LSTM concluído")
        
        # ----- CONSOLIDAÇÃO FINAL ----------------------------
        compare_and_save_results(barcode, results)

    except Exception as exc:
        logger.error(f"[{barcode}] Erro no pipeline: {exc}")
        logger.error(traceback.format_exc())
    finally:
        free_gpu_memory() 
        if not slot_released:
            proc_lock.release()

# ============================================================
#  EXECUÇÃO PRINCIPAL – PARALELO CONTROLADO
# ============================================================
def main():
    logger.info("🚀 Iniciando pipeline de predição diária 2024")

    dados = carregar_dados("data/raw")
    logger.info(f"Dados carregados: {len(dados)} produtos encontrados.")

    # ----- EXECUÇÃO EM MULTIPROCESSAMENTO -------------------
    processes = []

    for barcode, df in dados.items():
        proc_lock.acquire()   # respeita MAX_PARALLEL_PROCS
        p = mp.Process(target=processar_produto,
                       args=(barcode, df,
                             xgb_gpu_lock, nn_gpu_lock, proc_lock))
        p.start()
        processes.append(p)
        logger.info(f"{len(mp.active_children())} processos ativos…")

    for p in processes:
        p.join()
    
    # ----- VERSÃO SEQUENCIAL (desativada) -------------------
    # for barcode, df in dados.items():
    #     processar_produto(barcode, df, xgb_gpu_lock, nn_gpu_lock, proc_lock)

    logger.info("✅ Pipeline concluído para todos os produtos")

if __name__ == "__main__":
    main()
