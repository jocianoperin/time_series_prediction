import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from utils.logging_config import get_logger
from utils.metrics import calculate_metrics
from data_preparation import generate_rolling_windows

logger = get_logger(__name__)

GPU_ERROR_PATTERNS = (
    "cudaErrorInitializationError",
    "ctx_->Ordinal()",
    "Must have at least one device",
)

def _fallback_to_cpu(e, params, barcode, where):
    if any(pat in str(e) for pat in GPU_ERROR_PATTERNS):
        logger.warning(f"{barcode} | GPU falhou em {where} → usando CPU")
        params["tree_method"] = "hist"
        params["device"]   = "cpu"
        return True
    return False

def train_xgboost(df, barcode):
    logger.info(f"{barcode} | Iniciando treinamento XGBoost.")

    df = df.dropna().sort_values("Date").reset_index(drop=True)

    # Separação de treino e predição
    df_treino = df[df["Date"] < "2024-01-01"].copy()
    df_2024 = df[df["Date"].dt.year == 2024].copy()

    features = [col for col in df.columns if col not in ["Date", "Quantity"]]

    # Parâmetros XGBoost (GPU + aprendizado intenso)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "lambda": 1.0,
        # — nova sintaxe GPU —
        "tree_method": "hist",     # algoritmo
        "device": "cuda",          # executa em GPU
        "verbosity": 0,            # silencia prints do backend
        "seed": 42,                # semente para reprodutibilidade
    }

    results = []
    windows = generate_rolling_windows(df_treino)
    logger.info(f"{barcode} | Número de janelas geradas para rolling window: {len(windows)}.")

    # Rolling windows de 2019 a 2023
    for i, w in enumerate(windows):
        train, test = w["train"], w["test"]

        logger.info(
            f"{barcode} | Iniciando treinamento na Janela {i+1}: "
            f"Treino de {train['Date'].min()} a {train['Date'].max()}, "
            f"Teste de {test['Date'].min()} a {test['Date'].max()}"
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[features])
        y_train = train["Quantity"].values
        X_test = scaler.transform(test[features])
        y_test = test["Quantity"].values

        # Diagnóstico: verifica correlação para detectar possíveis vazamentos
        for idx, col in enumerate(features):
            corr = np.corrcoef(X_train[:, idx], y_train)[0, 1]
            if abs(corr) > 0.99:
                logger.warning(f"{barcode} | Alta correlação no recurso {col}: {corr:.3f}")

        # split 80/20 do train para early stopping
        split_idx = int(len(X_train) * 0.8)
        X_tr, y_tr = X_train[:split_idx], y_train[:split_idx]
        X_val, y_val = X_train[split_idx:], y_train[split_idx:]

        dtrain = xgb.DMatrix(X_tr,    label=y_tr)
        dval   = xgb.DMatrix(X_val,   label=y_val)
        dtest  = xgb.DMatrix(X_test,  label=y_test)

        evals = [(dtrain, "train"), (dval, "valid"), (dtest, "eval")]
        try:
            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=10000,
                evals=evals,
                early_stopping_rounds=200,
                verbose_eval=False
            )
        except xgb.core.XGBoostError as e:
            if _fallback_to_cpu(e, params, barcode, f"janela {i+1}"):
                logger.warning(f"{barcode} | GPU indisponível → voltando ao CPU")
                booster = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    evals=evals,
                    num_boost_round=10000,
                    early_stopping_rounds=200,
                    verbose_eval=False
                )
            else:
                raise

        y_pred = booster.predict(dtest)
        metrics = calculate_metrics(y_test, y_pred)

        logger.info(
            f"{barcode} | XGBoost - Janela {i+1} | Treino: {train['Date'].min()} a {train['Date'].max()} | "
            f"Teste: {test['Date'].min()} a {test['Date'].max()} | "
            f"MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%, BestIter={booster.best_iteration}"
        )

        test_out = test[["Date", "Quantity"]].copy()
        test_out["prediction_xgboost"] = y_pred
        results.append(test_out)

    # Baseline naïve: previsão = valor do dia anterior
    naive = df_treino["Quantity"].shift(1).dropna()
    y_true_naive = df_treino["Quantity"].iloc[1:].values
    baseline_mae = np.mean(np.abs(y_true_naive - naive.values))
    logger.info(f"{barcode} | Baseline naïve MAE (treino): {baseline_mae:.2f}")

    logger.info(f"{barcode} | Finalizadas as janelas rolling. Iniciando predição mensal para 2024.")

    # Previsão mês a mês de 2024
    forecast_2024 = []
    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(df_treino[features])
    y_train_full = df_treino["Quantity"].values
    dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full)

    # split 80/20 do train_full para early stopping
    split_idx_full = int(len(X_train_full) * 0.8)
    X_tr_f, y_tr_f = X_train_full[:split_idx_full], y_train_full[:split_idx_full]
    X_val_f, y_val_f = X_train_full[split_idx_full:], y_train_full[split_idx_full:]

    dtrain = xgb.DMatrix(X_tr_f,  label=y_tr_f)
    dval   = xgb.DMatrix(X_val_f,  label=y_val_f)

    evals_full = [(dtrain, "train"), (dval, "valid")]
    try:
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=10000,
            evals=evals_full,
            early_stopping_rounds=200,
            verbose_eval=False
        )
    except xgb.core.XGBoostError as e:
        if _fallback_to_cpu(e, params, barcode, "re-treino global"):
            logger.warning(f"{barcode} | GPU indisponível no re-treino global → voltando ao CPU")
            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                evals=evals_full,
                num_boost_round=10000,
                early_stopping_rounds=200,
                verbose_eval=False
            )
        else:
            raise

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
        metrics = calculate_metrics(y_real, y_pred)

        logger.info(
            f"{barcode} | XGBoost - Predição 2024-{month:02d} | MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%"
        )

        df_result = df_month[["Date", "Quantity"]].copy()
        df_result["forecast"] = y_pred

        # métricas linha a linha
        df_result["mae"]   = np.abs(df_result["Quantity"] - df_result["forecast"])
        df_result["rmse"]  = np.sqrt((df_result["Quantity"] - df_result["forecast"])**2)
        df_result["ape"]   = df_result["mae"] / df_result["Quantity"].replace(0, np.nan)
        df_result["mape"]  = df_result["ape"] * 100
        df_result["smape"] = (df_result["mae"] / ( (df_result["Quantity"].abs() + df_result["forecast"].abs())/2 ).replace(0, np.nan)) * 100

        forecast_2024.append(
            df_result.rename(columns={"Quantity": "real"})[["Date", "real", "forecast"]]
        )
        
        output_csv = df_result[["Date", "Quantity", "forecast", "mae", "rmse", "mape", "smape"]].copy()

        output_csv.rename(columns={"Quantity": "real"}, inplace=True)

        # >>> Ajuste para que cada barcode tenha sua subpasta em data/predictions <<<
        out_dir = f"data/predictions/XGBoost/{barcode}"
        os.makedirs(out_dir, exist_ok=True)
        csv_filename = f"XGBoost_daily_{barcode}_2024_{month:02d}.csv"
        output_csv.to_csv(os.path.join(out_dir, csv_filename), index=False)
        logger.info(f"{barcode} | CSV salvo: {os.path.join(out_dir, csv_filename)}")

        plot_xgboost_monthly(
            df_result,
            barcode,
            month
        )


        logger.info(f"{barcode} | Gráfico salvo para {month:02d}/2024.")

        # Fine-tune incremental com dados reais
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

    logger.info(f"{barcode} | Treinamento XGBoost concluído.")

    return pd.concat(results), metrics, pd.concat(forecast_2024)


def plot_xgboost_monthly(df_plot, barcode, month):
    """
    Gera e salva gráfico comparando previsão XGBoost vs. valores reais para cada mês de 2024.
    """
    df_plot = df_plot.sort_values("Date")
    plt.figure(figsize=(10, 5))
    plt.plot(df_plot["Date"], df_plot["Quantity"], label="Real", marker="o")
    plt.plot(df_plot["Date"], df_plot["forecast"], label="Previsto", marker="x")

    # valores explícitos no gráfico
    for x, y in zip(df_plot["Date"], df_plot["Quantity"]):
        plt.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=8)
    for x, y in zip(df_plot["Date"], df_plot["forecast"]):
        plt.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=8, color="blue")

    plt.title(f"Comparação Real vs. XGBoost - {barcode} - {month:02d}/2024")
    plt.xlabel("Dia")
    plt.ylabel("Quantidade")
    plt.legend()
    plt.xticks(rotation=45)

    out_dir = f"data/plots/XGBoost/{barcode}"
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"XGBoost_{barcode}_2024_{month:02d}.png"))
    plt.close()
