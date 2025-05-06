# ============================================================
#  MAIN – PIPELINE DE PREVISÃO DIÁRIA 2024 (PARALELO CONTROLADO)
#  ------------------------------------------------------------
#  Objetivo:
#    • Rodar XGBoost de forma concorrente (GPU leve)
#    • Rodar Rede Neural (LSTM) com exclusividade da GPU
#    • Consolidar previsões e métricas por produto
# ============================================================

# ----- AJUSTES DE GPU (OTIMIZAÇÃO DE USO DE VRAM) ----------
import signal
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# ----- MULTIPROCESSAMENTO COM CONTEXTO LIMPO (CUDA) --------
import multiprocessing as mp
import traceback
import pandas as pd
#mp.set_start_method("spawn", force=True)

# ----- LIMITES DE CONCORRÊNCIA -----------------------------
#MAX_PARALLEL_PROCS = 4     # Máximo de processos simultâneos
#MAX_XGB_CONCURRENT = 2     # XGBoosts simultâneos na GPU
#proc_lock    = mp.Semaphore(MAX_PARALLEL_PROCS)   # controle global
#xgb_gpu_lock = mp.Semaphore(MAX_XGB_CONCURRENT)   # GPU leve
#nn_gpu_lock  = mp.Semaphore(1)                    # GPU exclusiva para NN

# ----- IMPORTS DO PIPELINE ----------------------------------
from utils.logging_config import get_logger
from data_preparation     import carregar_dados
from feature_engineering  import create_features
from train_xgboost        import train_xgboost
from train_nn             import train_neural_network
from compare_models       import compare_and_save_results
from utils.metrics        import calculate_metrics
from utils.gpu_utils      import free_gpu_memory
import tensorflow as tf
import gc

logger = get_logger(__name__)

# ============================================================
#  PROCESSA UM ÚNICO PRODUTO – ETAPAS COMPLETAS
# ============================================================
def processar_produto(barcode: str, df_raw, xgb_gpu_lock, nn_gpu_lock, proc_lock):
    """
    Executa a cadeia completa de predição para um produto:
    feature engineering → XGBoost → LSTM → consolidação de métricas.
    """
    
    # ————— Normaliza `barcode` removendo prefixo "produto_" e extensão ".csv"
    base = os.path.splitext(barcode)[0]
    if base.startswith("produto_"):
        barcode = base[len("produto_"):]
    else:
        barcode = base

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

            # Comentando para liberar apenas ao final de tudo, evitando sobrecarga da RAM
            # slot_released = True
            # proc_lock.release()
            # logger.debug(f"[{barcode}] Slot global liberado após XGBoost")

        # ----- LSTM / NN (GPU pesada, uso exclusivo) ---------
        """with nn_gpu_lock:
            logger.info("Iniciando LSTM (NN)…")
            df_nn_all, nn_metrics, df_nn_2024 = train_neural_network(df, barcode)
            results["nn"] = {"metrics": nn_metrics,
                             "predictions": df_nn_2024}
            
            logger.info(f"[{barcode}] LSTM concluído")
        
        # ----- CONSOLIDAÇÃO FINAL ----------------------------
        compare_and_save_results(barcode, results)"""

    except Exception as exc:
        logger.error(f"[{barcode}] Erro no pipeline: {exc}")
        logger.error(traceback.format_exc())
    finally:
        free_gpu_memory() 
        # limpa sessão TF/Keras para realmente liberar a VRAM
        tf.keras.backend.clear_session()
        gc.collect()
        free_gpu_memory()
        proc_lock.release()

# ============================================================
#  EXECUÇÃO PRINCIPAL – PARALELO CONTROLADO
# ============================================================
def main(proc_lock, xgb_gpu_lock, nn_gpu_lock):
    logger.info("🚀 Iniciando pipeline de predição diária 2024")

    # ----- EXECUÇÃO EM MULTIPROCESSAMENTO -------------------
    # Processa em batches de 20 CSVs por vez
    batch_size = 20

    # ------------------------------------------------------------
    #  CARREGA .CSV E MOVE PARA processed APÓS PROCESSAMENTO OK
    # ------------------------------------------------------------
    raw_dir = "data/raw"
    processed_dir = os.path.join(raw_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Lista todos os CSVs ainda não processados
    arquivos = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    total = len(arquivos)
    logger.info(f"Arquivos encontrados: {total} CSV(s) em '{raw_dir}'.")

    for batch_idx in range(0, total, batch_size):
        batch_files = arquivos[batch_idx:batch_idx + batch_size]
        n_batch = len(batch_files)
        logger.info(f"🔄 Iniciando batch {batch_idx//batch_size + 1}: "
                    f"{n_batch} arquivos CSV (índices {batch_idx}–{batch_idx + n_batch - 1})")
        processes = []

        for filename in batch_files:
            # ——— Extrai o barcode: remove 'produto_' e '.csv'
            base = os.path.splitext(filename)[0]
            if base.startswith("produto_"):
                barcode = base[len("produto_"):]
            else:
                barcode = base

            filepath = os.path.join(raw_dir, filename)
            # Lê o CSV já convertendo a coluna Date para datetime
            df = pd.read_csv(filepath, parse_dates=["Date"])

            proc_lock.acquire()   # respeita MAX_PARALLEL_PROCS
            p = mp.Process(
                target=processar_produto,
                args=(filename, df, xgb_gpu_lock, nn_gpu_lock, proc_lock)
            )
            p.start()
            processes.append((p, filepath))
            logger.info(f"{len(mp.active_children())} processos ativos…")

        ec = p.exitcode

        if ec == 0:
            # sucesso
            dest = os.path.join(processed_dir, filename)
            os.rename(filepath, dest)
            logger.info(f"✅ '{filename}' movido para '{processed_dir}'")
        elif ec == -signal.SIGKILL:
            # morto por SIGKILL (OOM-killer)
            logger.error(f"❌ '{filename}' foi morto por SIGKILL (possível OOM)")
        elif ec < 0:
            # outro sinal
            sig = -ec
            logger.error(f"❌ '{filename}' morto pelo sinal {sig}")
        else:
            # exitcode > 0 (exceção Python)
            logger.error(f"❌ Falha ao processar '{filename}' (exitcode={ec})")

        logger.info(f"✅ Batch {batch_idx//batch_size + 1} concluído")
    
    # ----- VERSÃO SEQUENCIAL (desativada) -------------------
    # for barcode, df in dados.items():
    #     processar_produto(barcode, df, xgb_gpu_lock, nn_gpu_lock, proc_lock)

    logger.info("✅ Pipeline concluído para todos os produtos")

if __name__ == "__main__":
    # ----- MULTIPROCESSAMENTO COM CONTEXTO LIMPO (CUDA) --------
    mp.set_start_method("spawn", force=True)

    # ----- LIMITES DE CONCORRÊNCIA -----------------------------
    MAX_PARALLEL_PROCS = 8     # Máximo de processos simultâneos
    MAX_XGB_CONCURRENT = 8     # XGBoosts simultâneos na GPU
    proc_lock    = mp.Semaphore(MAX_PARALLEL_PROCS)   # controle global
    xgb_gpu_lock = mp.Semaphore(MAX_XGB_CONCURRENT)   # GPU leve
    nn_gpu_lock  = mp.Semaphore(1)                    # GPU exclusiva para NN

    # Inicia o pipeline passando os semáforos criados no escopo do pai
    main(proc_lock, xgb_gpu_lock, nn_gpu_lock)