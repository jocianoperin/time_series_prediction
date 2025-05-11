# Arquivo: main.py  (trecho inicial – novas flags e argparse)
# ============================================================
#  MAIN – PIPELINE DE PREVISÃO DIÁRIA 2024 (PARALELO CONTROLADO)
#  ------------------------------------------------------------
#  Agora suporta:
#    • --run-xgb / --run-nn ................... treinar/predizer modelos
#    • --reuse-xgb / --reuse-nn ............... somente ler previsões já salvas
#    • --reset-xgb / --reset-nn ............... APAGA artefatos e treina do ZERO
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
# mp.set_start_method("spawn", force=True)  # configurado no bloco __main__

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
# -------------------------------------------------------------------
RUN_XGB   = True   # True ⇒ treina/prevê com XGBoost
RUN_NN    = False  # True ⇒ treina/prevê com NN
REUSE_XGB = False  # True ⇒ carrega CSVs antigos do XGB se RUN_XGB=False

from utils.pred_loader import load_saved_predictions  # NOVO

logger = get_logger(__name__)

# ============================================================
#  PROCESSA UM ÚNICO PRODUTO – ETAPAS COMPLETAS
# ============================================================
def processar_produto(barcode: str, df_raw: pd.DataFrame, xgb_gpu_lock, nn_gpu_lock, proc_lock, run_xgb=RUN_XGB, run_nn=RUN_NN, reuse_xgb=REUSE_XGB):
    """
    Executa a cadeia completa de predição para um produto:
    feature engineering → XGBoost → LSTM → consolidação de métricas.
    """
    results = {}
    df = create_features(df_raw)

    # ————— Normaliza `barcode` removendo prefixo "produto_" e extensão ".csv"
    base = os.path.splitext(barcode)[0]
    if base.startswith("produto_"):
        barcode = base[len("produto_"):]
    else:
        barcode = base

    # ————— Normaliza nome
    barcode = os.path.splitext(barcode)[0].removeprefix("produto_")

    try:
        logger.info(f"[{barcode}] Iniciando processamento…")

        # ----- XGBOOST (GPU leve, concorrente) --------------
        if run_xgb:
            with xgb_gpu_lock:
                logger.info(f"[{barcode}] Iniciando XGBoost…")
                df_xgb_all, xgb_metrics, df_xgb_2024 = train_xgboost(df, barcode)
                results["xgboost"] = {"metrics": xgb_metrics,
                                    "predictions": df_xgb_2024}
                logger.info(f"[{barcode}] XGBoost concluído")
                free_gpu_memory()

        elif reuse_xgb:
            # Reutiliza previsões anteriores do XGBoost
            logger.info(f"[{barcode}] Reutilizando previsões do XGBoost…")
            df_xgb_2024 = load_saved_predictions(barcode, "XGBoost")
            if df_xgb_2024.empty:
                raise ValueError(f"⚠️  Não há previsões anteriores do XGBoost para '{barcode}'")
            else:
                mets = calculate_metrics(df_xgb_2024["real"], df_xgb_2024["forecast"])
                results["xgboost"] = {"metrics": mets,
                                "predictions": df_xgb_2024}

        # ----- LSTM / NN (GPU pesada, uso exclusivo) ---------
        if run_nn:
            with nn_gpu_lock:
                logger.info(f"[{barcode}] Iniciando LSTM (NN)…")
                df_nn_all, nn_metrics, df_nn_2024 = train_neural_network(df, barcode)
                results["nn"] = {"metrics": nn_metrics,
                                "predictions": df_nn_2024}
                
                logger.info(f"[{barcode}] LSTM concluído")
        
        # ----- CONSOLIDAÇÃO FINAL ----------------------------
        if results:
            compare_and_save_results(barcode, results)

    except Exception as exc:
        logger.error(f"[{barcode}] Erro no pipeline: {exc}")
        logger.error(traceback.format_exc())
    finally:
        # Limpa completamente a sessão para liberar VRAM
        free_gpu_memory() 
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
    batch_size = 50                      # processa em batches de 20 CSVs por vez

    # ------------------------------------------------------------
    #  CARREGA .CSV E MOVE PARA processed APÓS PROCESSAMENTO OK
    # ------------------------------------------------------------
    raw_dir = "data/raw"
    processed_dir = os.path.join(raw_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    arquivos = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    total = len(arquivos)
    logger.info(f"Arquivos encontrados: {total} CSV(s) em '{raw_dir}'.")

    # ------------------------------------------------------------
    #  LOOP DE BATCHES (20 EM 20)
    # ------------------------------------------------------------
    for batch_idx in range(0, total, batch_size):
        batch_files = arquivos[batch_idx : batch_idx + batch_size]
        n_batch = len(batch_files)

        logger.info(
            f"🔄 Iniciando batch {batch_idx//batch_size + 1}: "
            f"{n_batch} arquivos CSV "
            f"(índices {batch_idx}–{batch_idx + n_batch - 1})"
        )

        processes = []  # [(Process, filepath, filename)]

        # ----- LANÇA TODOS OS PROCESSOS DO BATCH -------------
        for filename in batch_files:
            base = os.path.splitext(filename)[0]
            barcode = base[len("produto_"):] if base.startswith("produto_") else base

            filepath = os.path.join(raw_dir, filename)
            df = pd.read_csv(filepath, parse_dates=["Date"])

            proc_lock.acquire()  # respeita MAX_PARALLEL_PROCS
            p = mp.Process(
                target=processar_produto,
                args=(filename, df, xgb_gpu_lock, nn_gpu_lock, proc_lock),
                name=f"proc_{barcode}",
            )
            p.start()

            processes.append((p, filepath, filename))
            logger.info(f"{len(mp.active_children())} processos ativos…")

        # ----- ESPERA TODOS OS PROCESSOS TERMINAREM -----------
        for proc, path, fname in processes:
            proc.join()                        # bloqueia até finalizar
            ec = proc.exitcode                 # exitcode confiável

            # ----- AVALIA EXITCODE & MOVE ARQUIVO --------------
            if ec == 0:
                dest = os.path.join(processed_dir, fname)
                try:
                    os.rename(path, dest)
                    logger.info(f"✅ '{fname}' movido para '{processed_dir}'")
                except OSError as e:
                    logger.error(f"⚠️  Não foi possível mover '{fname}': {e}")
            elif ec == -signal.SIGKILL:
                logger.error(f"❌ '{fname}' foi morto por SIGKILL (possível OOM)")
            elif ec is not None and ec < 0:
                logger.error(f"❌ '{fname}' morto pelo sinal {-ec}")
            else:
                logger.error(f"❌ Falha ao processar '{fname}' (exitcode={ec})")

        logger.info(f"✅ Batch {batch_idx//batch_size + 1} concluído")

    logger.info("✅ Pipeline concluído para todos os produtos")

# ============================================================
#  PONTO DE ENTRADA
# ============================================================
if __name__ == "__main__":
    # ----- MULTIPROCESSAMENTO COM CONTEXTO LIMPO (CUDA) -----
    mp.set_start_method("spawn", force=True)

    # ----- LIMITES DE CONCORRÊNCIA ---------------------------
    MAX_PARALLEL_PROCS   = 6  # Máx. de processos simultâneos
    MAX_XGB_CONCURRENT   = 6  # Máx. de XGBoosts simultâneos na GPU
    MAX_NN_CONCURRENT    = 1  # Máx. de NNs simultâneas na GPU
    proc_lock    = mp.Semaphore(MAX_PARALLEL_PROCS)   # controle global
    xgb_gpu_lock = mp.Semaphore(MAX_XGB_CONCURRENT)   # GPU leve
    nn_gpu_lock  = mp.Semaphore(MAX_NN_CONCURRENT)    # GPU exclusiva para NN

    # ----- INICIA O PIPELINE --------------------------------
    main(proc_lock, xgb_gpu_lock, nn_gpu_lock)
