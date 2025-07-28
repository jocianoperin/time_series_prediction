#!/usr/bin/env python3
"""
MAIN – PIPELINE DE PREVISÃO DIÁRIA 2024 (PARALELO CONTROLADO)
----------------------------------------------------------------
• Mantém comportamentos do script legado quando NENHUMA flag é dada.
• Permite override via CLI: --run-xgb, --run-nn, --reuse-xgb, --reuse-nn,
  --reset-xgb, --reset-nn, --batch-size.
• Controle de concorrência igual ao legado (semaphores em __main__).
"""

# ============================================================
#  IMPORTS E AJUSTES DE AMBIENTE
# ============================================================
import os, gc, shutil, signal, argparse, traceback, multiprocessing as mp
import pandas as pd
import tensorflow as tf

# Minimiza consumo de threads/VRAM em operações BLAS
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # descomente se preciso

# ============================================================
#  IMPORTS DO PROJETO
# ============================================================
from utils.logging_config import get_logger
from feature_engineering  import create_features
from train_xgboost        import train_xgboost
from train_nn             import train_neural_network
from compare_models       import compare_and_save_results
from utils.metrics        import calculate_metrics
from utils.gpu_utils      import free_gpu_memory
from utils.pred_loader    import load_saved_predictions

logger = get_logger(__name__)

# ============================================================
#  PARÂMETROS PADRÃO (SEM FLAGS)
# ============================================================
RUN_XGB   = True      # treina/prevê XGB
RUN_NN    = False     # treina/prevê NN
REUSE_XGB = False     # lê previsões XGB
REUSE_NN  = False     # lê previsões NN
RESET_XGB = True      # apaga artefatos XGB
RESET_NN  = False     # apaga artefatos NN
RUN_COMPARE  = False   # compara resultados
BATCH_DEF = 50        # tamanho de batch

MAX_PARALLEL_PROCS = 1  # processos simultâneos
MAX_XGB_CONCURRENT = 1  # XGBoost simultâneos
MAX_NN_CONCURRENT  = 1  # NN simultâneas

# ============================================================
#  ARGUMENTOS DE LINHA DE COMANDO (OVERRIDE OPCIONAL)
# ============================================================
parser = argparse.ArgumentParser(description="Pipeline de predição diária 2024")
parser.add_argument("--run-xgb",    action="store_true", help="Treinar e prever com XGBoost")
parser.add_argument("--run-nn",     action="store_true", help="Treinar e prever com Rede Neural (LSTM)")
parser.add_argument("--reuse-xgb",  action="store_true", help="Usar previsões XGBoost existentes no disco")
parser.add_argument("--reuse-nn",   action="store_true", help="Usar previsões NN existentes no disco")
parser.add_argument("--reset-xgb",  action="store_true", help="Apagar artefatos XGBoost antes de treinar")
parser.add_argument("--reset-nn",   action="store_true", help="Apagar artefatos NN antes de treinar")
parser.add_argument("--batch-size", type=int, default=BATCH_DEF, help="Qtd. de CSVs por batch (default 50)")
parser.add_argument("--no-compare", action="store_false", dest="run_compare", help="Não consolidar comparação entre modelos")
parser.add_argument("--max-procs", type=int, default=1, help="Número máximo de processos simultâneos")
parser.add_argument("--max-xgb", type=int, default=1, help="Número máximo de XGBoost simultâneos")
parser.add_argument("--max-nn", type=int, default=1, help="Número máximo de NN simultâneas")
ARGS, _ = parser.parse_known_args()

# Override apenas se flag presente
RUN_XGB   = ARGS.run_xgb   or RUN_XGB
RUN_NN    = ARGS.run_nn    or RUN_NN
REUSE_XGB = ARGS.reuse_xgb or REUSE_XGB
REUSE_NN  = ARGS.reuse_nn  or REUSE_NN
RESET_XGB = ARGS.reset_xgb or RESET_XGB
RESET_NN  = ARGS.reset_nn  or RESET_NN
RUN_COMPARE = ARGS.run_compare and RUN_COMPARE
BATCH_SZ  = ARGS.batch_size
MAX_PARALLEL_PROCS = ARGS.max_procs
MAX_XGB_CONCURRENT = ARGS.max_xgb
MAX_NN_CONCURRENT = ARGS.max_nn

# ============================================================
#  FUNÇÃO: processar_produto
# ============================================================

def processar_produto(csv_name: str,
                      df_raw: pd.DataFrame,
                      xgb_gpu_lock,
                      nn_gpu_lock,
                      proc_lock,
                      raw_dir=None):
    """Pipeline completo (feature eng → model(s) → consolidação) p/ um produto.
    
    Args:
        csv_name: Nome do arquivo CSV
        df_raw: DataFrame com os dados brutos
        xgb_gpu_lock: Lock para controle de concorrência do XGBoost
        nn_gpu_lock: Lock para controle de concorrência da rede neural
        proc_lock: Lock para controle de processos
        raw_dir: Diretório de origem dos arquivos brutos (opcional, usado para mover o arquivo após processamento)
    """

    base = os.path.splitext(csv_name)[0]
    barcode = base[len("produto_"):] if base.startswith("produto_") else base
    processed_successfully = False

    logger.info(f"[{barcode}] ▶️  início")

    try:
        df = create_features(df_raw)
        results = {}

        # Helper para limpar pastas
        def _reset_paths(tag: str):
            for p in (
                os.path.join("data/predictions", tag, barcode),
                os.path.join("data/plots",       tag, barcode),
                os.path.join("models",           tag, barcode),
            ):
                if os.path.exists(p):
                    shutil.rmtree(p, ignore_errors=True)
                    logger.debug(f"[{barcode}] reset → {p}")

        # ---------- XGBoost ----------
        if RESET_XGB:
            _reset_paths("XGBoost")
        if RUN_XGB:
            with xgb_gpu_lock:
                logger.info(f"[{barcode}] XGB: treino/pred")
                _, mets, preds_2024 = train_xgboost(df, barcode)
                results["xgboost"] = {"metrics": mets, "predictions": preds_2024}
                free_gpu_memory()
        elif REUSE_XGB:
            logger.info(f"[{barcode}] XGB: reuse")
            preds_2024 = load_saved_predictions(barcode, "XGBoost")
            if preds_2024.empty:
                raise RuntimeError("Sem preds XGB em disco")
            mets = calculate_metrics(preds_2024["real"], preds_2024["forecast"])
            results["xgboost"] = {"metrics": mets, "predictions": preds_2024}

        # ---------- NN/LSTM ----------
        if RESET_NN:
            _reset_paths("NN")
        if RUN_NN:
            with nn_gpu_lock:
                logger.info(f"[{barcode}] NN: treino/pred")
                _, mets, preds_2024 = train_neural_network(df, barcode)
                results["nn"] = {"metrics": mets, "predictions": preds_2024}
        elif REUSE_NN:
            logger.info(f"[{barcode}] NN: reuse")
            preds_2024 = load_saved_predictions(barcode, "NN")
            if preds_2024.empty:
                raise RuntimeError("Sem preds NN em disco")
            mets = calculate_metrics(preds_2024["real"], preds_2024["forecast"])
            results["nn"] = {"metrics": mets, "predictions": preds_2024}

        # ---------- Consolidation ----------
        if RUN_COMPARE and {"xgboost", "nn"}.issubset(results):
            compare_and_save_results(barcode, results)
            logger.info(f"[{barcode}] comparação consolidada")

    except Exception as exc:
        logger.error(f"[{barcode}] erro: {exc}")
        logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"[{barcode}] Erro durante o processamento: {str(e)}")
        processed_successfully = False
    finally:
        # Limpeza de recursos
        free_gpu_memory()
        tf.keras.backend.clear_session()
        gc.collect()
        free_gpu_memory()
        
        # Se o processamento foi bem-sucedido e raw_dir foi fornecido, move o arquivo
        if processed_successfully and raw_dir and os.path.exists(os.path.join(raw_dir, csv_name)):
            processed_dir = os.path.join(raw_dir, "processed")
            os.makedirs(processed_dir, exist_ok=True)
            try:
                shutil.move(
                    os.path.join(raw_dir, csv_name),
                    os.path.join(processed_dir, csv_name)
                )
                logger.info(f"[{barcode}] ✅ Arquivo movido para {processed_dir}")
            except Exception as e:
                logger.error(f"[{barcode}] ❌ Falha ao mover o arquivo: {str(e)}")
        
        proc_lock.release()
        logger.info(f"[{barcode}] ⏹️  fim")

# ============================================================
#  FUNÇÃO: main (multiprocess & batches)
# ============================================================

def main(proc_lock, xgb_gpu_lock, nn_gpu_lock):
    logger.info("🚀 Pipeline global iniciado – batch_size=%s", BATCH_SZ)

    raw_dir = "data/raw"
    processed_dir = os.path.join(raw_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Lista todos os arquivos CSV que ainda não foram processados
    csv_files = [f for f in os.listdir(raw_dir) 
                if f.endswith(".csv") and not f.startswith(".")]
    total = len(csv_files)
    logger.info("📂 %s CSV(s) detectados para processamento", total)

    # Processa em lotes, mas move os arquivos individualmente
    for offset in range(0, total, BATCH_SZ):
        batch = csv_files[offset:offset + BATCH_SZ]
        logger.info("🔄 Lote %s: processando %s arquivos", 
                   offset // BATCH_SZ + 1, len(batch))
        
        processes = []
        
        # Inicia os processos para o lote atual
        for fname in batch:
            path = os.path.join(raw_dir, fname)
            try:
                df = pd.read_csv(path, parse_dates=["Date"])
                
                # Adquire o lock antes de iniciar o processo
                proc_lock.acquire()
                
                # Cria e inicia o processo de processamento
                p = mp.Process(
                    target=processar_produto,
                    args=(fname, df, xgb_gpu_lock, nn_gpu_lock, proc_lock, raw_dir),
                    name=f"proc_{fname}",
                    daemon=True
                )
                p.start()
                processes.append(p)
                
            except Exception as e:
                logger.error(f"❌ Erro ao processar {fname}: {str(e)}")
                proc_lock.release()  # Libera o lock em caso de erro
        
        # Aguarda todos os processos do lote atual terminarem
        for p in processes:
            p.join()
            if p.exitcode != 0:
                logger.warning(f"Processo {p.name} terminou com código de saída {p.exitcode}")
        
        logger.info(f"✅ Lote {offset // BATCH_SZ + 1} concluído")

    logger.info("🎉 Pipeline completo para %s produtos", total)

# ============================================================
#  PONTO DE ENTRADA
# ============================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    proc_lock    = mp.Semaphore(MAX_PARALLEL_PROCS)
    xgb_gpu_lock = mp.Semaphore(MAX_XGB_CONCURRENT)
    nn_gpu_lock  = mp.Semaphore(MAX_NN_CONCURRENT)

    main(proc_lock, xgb_gpu_lock, nn_gpu_lock)
