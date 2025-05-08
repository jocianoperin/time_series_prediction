# ============================================================
#  TREINAMENTO DO MODELO XGBoost
#  Aplica rolling windows (2019–2023) + predições mensais 2024
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import gc

from sklearn.preprocessing import StandardScaler
from utils.logging_config import get_logger
from utils.metrics import calculate_metrics
from data_preparation import generate_rolling_windows

logger = get_logger(__name__)

# ------------------------------------------------------------
# PADRÕES DE ERRO DE GPU QUE CAUSAM FALLBACK PARA CPU
# ------------------------------------------------------------
GPU_ERROR_PATTERNS = (
    "cudaErrorInitializationError",
    "ctx_->Ordinal()",
    "Must have at least one device",
)

# ------------------------------------------------------------
# FALLBACK – FORÇA CPU SE GPU ESTIVER INDISPONÍVEL
# ------------------------------------------------------------
def _fallback_to_cpu(e, params, barcode, where):
    if any(pat in str(e) for pat in GPU_ERROR_PATTERNS):
        logger.warning(f"{barcode} | GPU falhou em {where} → usando CPU")
        params["tree_method"] = "hist"
        params["device"]   = "cpu"
        return True
    return False

# ------------------------------------------------------------
# FUNÇÃO PRINCIPAL – TREINAMENTO COM ROLLING E PREVISÃO 2024
# ------------------------------------------------------------
def train_xgboost(df, barcode):
    """
    Executa o treinamento do XGBoost com janelas deslizantes para validação
    e gera predições mensais para 2024 com fine-tune incremental.
    """
    logger.info(f"{barcode} | Iniciando treinamento XGBoost.")

    # ----- CAMINHOS PARA SALVAMENTO DO MODELO -------------
    model_dir   = f"models/XGBoost/{barcode}"
    model_path  = os.path.join(model_dir, f"xgb_{barcode}.json")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    booster_loaded = None

    # ----- TENTA CARREGAR MODELO E SCALER SALVOS ----------
    if os.path.isfile(model_path) and os.path.isfile(scaler_path):
        try:
            booster_loaded = xgb.Booster()
            booster_loaded.load_model(model_path)
            scaler = joblib.load(scaler_path)
            logger.info(f"{barcode} | Modelo XGBoost carregado de disco. "
                        f"Pulando re-treino global.")
        except Exception as e:
            logger.warning(f"{barcode} | Falha ao carregar modelo salvo: {e}")
            booster_loaded = None

    # ----- PREPARAÇÃO DOS DADOS ---------------------------
    df = df.dropna().sort_values("Date").reset_index(drop=True)

    # Separação de treino e predição
    df_treino = df[df["Date"] < "2024-01-01"].copy()
    df_2024 = df[df["Date"].dt.year == 2024].copy()
    features = [col for col in df.columns if col not in ["Date", "Quantity"]]

    # ----- PARÂMETROS BASE DO XGBoost ---------------------
    params = {
        "objective": "reg:squarederror",        # Função de perda para regressão (erro quadrático)
        "eval_metric": "mae",                   # Métrica de avaliação: erro absoluto médio
        "learning_rate": 0.01,                  # Taxa de aprendizado (eta) — menor = mais estável
        "max_depth": 5,                         # Profundidade máxima de cada árvore
        "subsample": 0.8,                       # Proporção de amostras para cada árvore (evita overfitting)
        "colsample_bytree": 0.8,                # Proporção de features usadas por árvore (aleatorização)
        "min_child_weight": 3,                  # Mínimo de instâncias por folha (controle de complexidade)
        "gamma": 0.1,                           # Ganho mínimo para realizar split (regularização)
        "lambda": 1.0,                          # Termo L2 de regularização dos pesos (Ridge)
        "tree_method": "gpu_hist",              # Algoritmo baseado em histogramas (eficiente p/ CPU e GPU)
        "max_bin": 64,                          # ↓ bins 256→64 ⇒ –VRAM ~3×
        "single_precision_histogram": True,     # usa FP32 em vez de FP64
        "device": "cuda",                       # Executa o treinamento na GPU via CUDA
        "verbosity": 0,                         # Silencia logs internos do XGBoost (0 = silencioso)
        "seed": 42                              # Semente para reprodutibilidade dos resultados
    }

    results = []

    # ----- ESCALONAMENTO (SCALER GLOBAL) ------------------
    if booster_loaded:
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(df_treino[features])

    # ----- CRIA JANELAS DE ROLLING (365+31) ---------------
    windows = generate_rolling_windows(df_treino, train_days=365, test_days=7, step_days=7)

    logger.info(f"{barcode} | Número de janelas geradas para rolling window: {len(windows)}.")

    # ------------------------------------------------------------
    # TREINAMENTO POR JANELAS (2019–2023) – ROLLING WINDOW
    # ------------------------------------------------------------
    for i, w in enumerate(windows):
        train, test = w["train"], w["test"]

        logger.info(
            f"{barcode} | Iniciando treinamento na Janela {i+1}: "
            f"Treino de {train['Date'].min()} a {train['Date'].max()}, "
            f"Teste de {test['Date'].min()} a {test['Date'].max()}"
        )
        
        # ----- PREPARA ENTRADAS ESCALADAS ----------------------
        X_train = scaler.transform(train[features]).astype(np.float32)
        y_train = train["Quantity"].values
        X_test  = scaler.transform(test[features]).astype(np.float32)
        y_test  = test["Quantity"].values

        # ----- CHECK DE CORRELAÇÃO (PREVENÇÃO DE VAZAMENTO) ----
        for idx, col in enumerate(features):
            xi = X_train[:, idx]
            if np.std(xi) == 0 or np.std(y_train) == 0:
                continue  # ignora colunas ou targets constantes
            corr = np.corrcoef(xi, y_train)[0, 1]
            if abs(corr) > 0.99:
                logger.warning(f"{barcode} | Correlação extrema em '{col}': {corr:.3f}")

        # ----- SPLIT PARA EARLY STOPPING -----------------------
        split_idx = int(len(X_train) * 0.8)
        X_tr, y_tr = X_train[:split_idx], y_train[:split_idx]
        X_val, y_val = X_train[split_idx:], y_train[split_idx:]

        dtrain = xgb.DMatrix(X_tr,    label=y_tr)
        dval   = xgb.DMatrix(X_val,   label=y_val)
        dtest  = xgb.DMatrix(X_test,  label=y_test)

        evals = [(dtrain, "train"), (dval, "valid"), (dtest, "eval")]

        # ----- TREINAMENTO DO MODELO ---------------------------
        try:
            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=10000,
                evals=evals,
                early_stopping_rounds=50,
                verbose_eval=False
            )
        except xgb.core.XGBoostError as e:
            if _fallback_to_cpu(e, params, barcode, f"janela {i+1}"):
                logger.warning(f"{barcode} | Janela {i+1} | GPU falhou → retreinando em CPU")
                booster = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    evals=evals,
                    num_boost_round=10000,
                    early_stopping_rounds=50,
                    verbose_eval=False
                )
            else:
                raise
        
        # ----- PREVISÃO E MÉTRICAS DA JANELA -------------------
        y_pred = booster.predict(dtest)
        y_pred = np.clip(y_pred, 0, None)        # força previsão ≥ 0

        metrics = calculate_metrics(y_test, y_pred)

        logger.info(
            f"{barcode} | XGBoost - Janela {i+1} | Treino: {train['Date'].min()} a {train['Date'].max()} | "
            f"Teste: {test['Date'].min()} a {test['Date'].max()} | "
            f"MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%, BestIter={booster.best_iteration}"
        )

        test_out = test[["Date", "Quantity"]].copy()
        test_out["prediction_xgboost"] = y_pred
        results.append(test_out)

    # ------------------------------------------------------------
    # BASELINE COMPARATIVO (NAÏVE = DADO ANTERIOR)
    # ------------------------------------------------------------
    naive = df_treino["Quantity"].shift(1).dropna()
    y_true_naive = df_treino["Quantity"].iloc[1:].values
    baseline_mae = np.mean(np.abs(y_true_naive - naive.values))

    logger.info(f"{barcode} | Baseline naïve MAE (treino): {baseline_mae:.2f}")
    logger.info(f"{barcode} | Finalizadas as janelas rolling. Iniciando predição mensal para 2024.")

    # ------------------------------------------------------------
    # RE-TREINO GLOBAL (2019–2023) PARA PREVISÃO DE 2024
    # ------------------------------------------------------------
    if booster_loaded is None:
        logger.info(f"{barcode} | Re-treinando modelo com todos os dados de 2019–2023")

        X_train_full = scaler.transform(df_treino[features]).astype(np.float32)
        y_train_full = df_treino["Quantity"].values

        # Split para early stopping (80/20)
        split_idx_full = int(len(X_train_full) * 0.8)
        X_tr_f, y_tr_f = X_train_full[:split_idx_full], y_train_full[:split_idx_full]
        X_val_f, y_val_f = X_train_full[split_idx_full:], y_train_full[split_idx_full:]

        dtrain = xgb.DMatrix(X_tr_f, label=y_tr_f)
        dval   = xgb.DMatrix(X_val_f, label=y_val_f)
        evals  = [(dtrain, "train"), (dval, "valid")]

        try:
            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=10000,
                evals=evals,
                early_stopping_rounds=50,
                verbose_eval=False,
            )
        except xgb.core.XGBoostError as e:
            if _fallback_to_cpu(e, params, barcode, "re-treino global"):
                logger.warning(f"{barcode} | Re-treino global na CPU")
                booster = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    evals=evals,
                    num_boost_round=10000,
                    early_stopping_rounds=50,
                    verbose_eval=False,
                )
            else:
                raise
    else:
        booster = booster_loaded
        X_train_full = scaler.transform(df_treino[features])

    forecast_2024 = []

    # ------------------------------------------------------------
    # PREDIÇÃO MENSAL PARA 2024 + FINE-TUNE INCREMENTAL
    # ------------------------------------------------------------
    for month in range(1, 13):
        df_month = df_2024[df_2024["Date"].dt.month == month].copy()
        if df_month.empty:
            logger.warning(f"{barcode} | XGBoost - Mês {month:02d} ignorado: nenhum dado disponível.")
            continue

        logger.info(f"{barcode} | Iniciando predição para o mês {month:02d}/2024.")

        X_future = scaler.transform(df_month[features])
        y_real = df_month["Quantity"].values
        dfuture = xgb.DMatrix(X_future)

        y_pred = booster.predict(dfuture)
        y_pred = np.clip(y_pred, 0, None)        # ← garante somente valores ≥ 0

        metrics = calculate_metrics(y_real, y_pred)

        logger.info(f"{barcode} | XGBoost - Predição 2024-{month:02d} | MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%")

        # ----- ORGANIZAÇÃO DOS RESULTADOS ---------------------
        df_result = df_month[["Date", "Quantity"]].copy()
        df_result["forecast"] = y_pred

        df_result["mae"]   = np.abs(df_result["Quantity"] - df_result["forecast"])
        df_result["rmse"]  = np.sqrt((df_result["Quantity"] - df_result["forecast"])**2)
        df_result["ape"]   = df_result["mae"] / df_result["Quantity"].replace(0, np.nan)
        df_result["mape"]  = df_result["ape"] * 100
        df_result["smape"] = (df_result["mae"] / ( (df_result["Quantity"].abs() + df_result["forecast"].abs())/2 ).replace(0, np.nan)) * 100

        forecast_2024.append(df_result.rename(columns={"Quantity": "real"})[["Date", "real", "forecast"]])
        
        # ----- SALVA RESULTADO EM CSV -------------------------
        output_csv = df_result[["Date", "Quantity", "forecast", "mae", "rmse", "mape", "smape"]].copy()
        output_csv.rename(columns={"Quantity": "real"}, inplace=True)

        out_dir = f"data/predictions/XGBoost/{barcode}"
        os.makedirs(out_dir, exist_ok=True)

        csv_filename = f"XGBoost_daily_{barcode}_2024_{month:02d}.csv"
        output_csv.to_csv(os.path.join(out_dir, csv_filename), index=False)
        logger.info(f"{barcode} | CSV salvo: {os.path.join(out_dir, csv_filename)}")

        # ----- GERA E SALVA O GRÁFICO --------------------------
        plot_xgboost_monthly(
            df_result,
            barcode,
            month
        )

        logger.info(f"{barcode} | Gráfico salvo para {month:02d}/2024.")

        # ----- FINE-TUNE COM DADOS DO MÊS ----------------------
        try:
            booster = xgb.train(
                params=params,
                dtrain=xgb.DMatrix(X_future, label=y_real),
                num_boost_round=10000,
                xgb_model=booster,
                verbose_eval=False
            )
        except xgb.core.XGBoostError as e:
            if _fallback_to_cpu(e, params, barcode, f"janela {i+1}"):
                logger.warning(f"{barcode} | GPU indisponível → voltando ao CPU")
                booster = xgb.train(
                    params=params,
                    dtrain=xgb.DMatrix(X_future, label=y_real),
                    num_boost_round=10000,
                    xgb_model=booster,
                    verbose_eval=False
                )
            else:
                raise

        logger.info(f"{barcode} | Fine-tune incremental realizado com dados de {month:02d}/2024.")

    # ------------------------------------------------------------
    # SALVA MODELO E SCALER FINAL
    # ------------------------------------------------------------
    os.makedirs(model_dir, exist_ok=True)
    booster.save_model(model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"{barcode} | Modelo XGBoost + scaler salvos em {model_dir}")

    # ------------------------------------------------------------
    # FINALIZAÇÃO
    # ------------------------------------------------------------
    logger.info(f"{barcode} | Pipeline XGBoost finalizado com sucesso")

    # --- 1) Constrói as saídas antes de deletar anything ---
    # --- Concatena resultados / previsões ou devolve DataFrame vazio -----------
    df_results  = pd.concat(results,       ignore_index=True) if results       else pd.DataFrame()
    df_forecast = pd.concat(forecast_2024, ignore_index=True) if forecast_2024 else pd.DataFrame()

    if df_results.empty:
        logger.warning(f"{barcode} | 'results' vazio — possivelmente não houve janelas rolling suficientes.")
    if df_forecast.empty:
        logger.warning(f"{barcode} | 'forecast_2024' vazio — possivelmente não existem dados de 2024 para este produto.")

    # --- 2) Limpeza de memória (agora que já temos as saídas) ---
    del booster, scaler, windows, results, forecast_2024
    gc.collect()

    # --- 3) Retorno ---
    # Se nenhuma janela foi processada, 'metrics' não existe → devolve dicionário vazio
    return df_results, metrics if "metrics" in locals() else {}, df_forecast

# ------------------------------------------------------------
# GERA GRÁFICO MENSAL – REAL × PREVISTO (XGBoost)
# ------------------------------------------------------------
def plot_xgboost_monthly(df_plot, barcode, month):
    """
    Gera e salva o gráfico de comparação entre valores reais e preditos
    pelo modelo XGBoost para o mês especificado.

    Parâmetros:
    - df_plot: DataFrame contendo 'Date', 'Quantity' e 'forecast'
    - barcode: identificador do produto
    - month: mês (1 a 12) a ser plotado
    """

    df_plot = df_plot.sort_values("Date")
    plt.figure(figsize=(10, 5))
    
    # ----- CURVAS REAL E PREVISTA -------------------------
    plt.plot(df_plot["Date"], df_plot["Quantity"], label="Real", marker="o")
    plt.plot(df_plot["Date"], df_plot["forecast"], label="Previsto", marker="x")

    # ----- ANOTAÇÕES COM VALORES NOS PONTOS ---------------
    for x, y in zip(df_plot["Date"], df_plot["Quantity"]):
        plt.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=8)
    for x, y in zip(df_plot["Date"], df_plot["forecast"]):
        plt.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=8, color="blue")

    # ----- CONFIGURAÇÕES VISUAIS DO GRÁFICO ---------------
    plt.title(f"Comparação Real vs. XGBoost - {barcode} - {month:02d}/2024")
    plt.xlabel("Dia")
    plt.ylabel("Quantidade")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # ----- SALVA GRÁFICO EM PNG ---------------------------
    out_dir = f"data/plots/XGBoost/{barcode}"
    os.makedirs(out_dir, exist_ok=True)
    
    plt.savefig(os.path.join(out_dir, f"XGBoost_{barcode}_2024_{month:02d}.png"))
    plt.close()
