# ============================================================
#  COMPARAÇÃO DE RESULTADOS ENTRE MODELOS – GRÁFICOS E CSVs
#  Desenvolvido para consolidação, métricas e visualização
#  por produto (via código de barras) no pipeline de predição
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------
# FUNÇÃO AUXILIAR – NORMALIZA COLUNA DE PREVISÃO
# ------------------------------------------------------------
def _normalizar_preds(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Renomeia a coluna de predição do DataFrame para o nome do modelo,
    permitindo padronização durante o merge.

    Exemplo: 'forecast' → 'xgboost'
    """
    
    possiveis = [
        "forecast",
        f"prediction_{model_name}",
        "prediction",
        "predicted",
        "yhat",
    ]
    for col in possiveis:
        if col in df.columns:
            logger.debug(f"Coluna '{col}' renomeada para '{model_name}'")
            return df.rename(columns={col: model_name})

    logger.warning(f"Coluna de previsão não encontrada para o modelo: {model_name}")

    raise ValueError(
        f"Coluna de predição não encontrada no DF de {model_name} "
        f"({', '.join(possiveis)})"
    )

# ------------------------------------------------------------
# FUNÇÃO PRINCIPAL – CONSOLIDA RESULTADOS E GERA GRÁFICOS
# ------------------------------------------------------------
def compare_and_save_results(
        barcode: str,
        results: dict,
        base_pred_dir: str = "data/predictions",
        base_plot_dir: str = "data/plots",
    ) -> None:
    """
    Consolida métricas e predições de diferentes modelos para um produto
    específico (`barcode`) e produz:

    • CSV …/predictions/comparativo/<barcode>_metrics.csv
    • CSV …/predictions/comparativo/predicoes_2024_<barcode>.csv
    • PNGs mensais …/plots/comparativo/<barcode>/comparativo_<barcode>_2024_MM.png
    """
    
    logger.info(f"Iniciando consolidação para o produto {barcode}")

    metrics_list: list[dict] = []
    merged_preds: pd.DataFrame | None = None
    real_col_name: str | None = None

    # --------------------------------------------------------
    # ITERA MODELOS PARA UNIFICAR PREVISÕES E MÉTRICAS
    # --------------------------------------------------------
    for model_name, data in results.items():
        if not data or "predictions" not in data:
            logger.warning(f"{model_name} sem predições — será ignorado")
            continue

        df = data["predictions"].copy()
        if df.empty:
            continue
        
        # Define coluna real na primeira iteração válida
        if merged_preds is None:
            if "Quantity" in df.columns:
                real_col_name = "Quantity"
            elif "real" in df.columns:
                real_col_name = "real"
            else:
                logger.warning(f"[{barcode}] {model_name} ignorado (sem coluna real)")
                continue
            merged_preds = df[["Date", real_col_name]].copy()

        # Normaliza e adiciona coluna de previsão do modelo atual
        df_norm = _normalizar_preds(df, model_name)[["Date", model_name]]
        merged_preds = pd.merge(merged_preds, df_norm, on="Date", how="outer")

        # Armazena métricas (se existirem)
        metrics = data.get("metrics", {})
        metrics["model"] = model_name
        metrics_list.append(metrics)

    if merged_preds is None:
        logger.warning(f"[{barcode}] Nenhuma predição consolidada – processo abortado")
        return
    # ------------------- TRATAMENTO DE DADOS -----------------
    # Renomeia coluna real para "real" (se necessário)
    if real_col_name and real_col_name != "real":
        merged_preds = merged_preds.rename(columns={real_col_name: "real"})

    # --------------------------------------------------------
    # GERA ESTRUTURA DE SAÍDAS (CSV E PLOT)
    # --------------------------------------------------------
    out_cmp_dir = os.path.join(base_pred_dir, "comparativo", barcode)
    os.makedirs(out_cmp_dir, exist_ok=True)

    # Salva CSV com métricas agregadas
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(os.path.join(out_cmp_dir, f"{barcode}_metrics.csv"), index=False)
        logger.info(f"[{barcode}] Métricas salvas")
    else:
        metrics_df = pd.DataFrame()

    # Salva CSV com todas as predições do ano
    merged_preds.to_csv(os.path.join(out_cmp_dir, f"predicoes_2024_{barcode}.csv"),index=False,)

    # Salva resumo anual (soma das quantidades por ano)
    resumo_anual = (
        merged_preds
        .assign(year=merged_preds["Date"].dt.year)
        .groupby("year", as_index=False)[["real", "xgboost", "nn"]]
        .sum()
        .sort_values("year")
    )
    resumo_anual.to_csv(
        os.path.join(out_cmp_dir, f"resumo_anual_{barcode}.csv"),
        index=False
    )

    # --------------------------------------------------------
    # GERA CSVs DIÁRIOS COM MÉTRICAS POR MODELO E POR MÊS
    # --------------------------------------------------------
    merged_preds["year"]  = merged_preds["Date"].dt.year
    merged_preds["month"] = merged_preds["Date"].dt.month

    for (yr, mn), df_m in merged_preds.groupby(["year", "month"]):
        for mdl in [c for c in df_m.columns if c not in {"Date", "real", "year", "month"}]:
            # Cálculo de métricas por dia
            df_m[f"mae_{mdl}"]  = (df_m[mdl] - df_m["real"]).abs()
            df_m[f"rmse_{mdl}"] = np.sqrt((df_m[mdl] - df_m["real"])**2)
            df_m[f"mape_{mdl}"] = df_m[f"mae_{mdl}"] / df_m["real"].replace(0, np.nan) * 100
            df_m[f"smape_{mdl}"]= df_m[f"mae_{mdl}"] / ((df_m["real"].abs() + df_m[mdl].abs())/2).replace(0, np.nan) * 100

        csv_path = os.path.join(out_cmp_dir, f"{barcode}_{yr}_{mn:02d}_diario.csv")
        df_m.drop(columns=["year", "month"]).to_csv(csv_path, index=False)
        logger.debug(f"[{barcode}] CSV diário salvo: {csv_path}")

    # --------------------------------------------------------
    # GERA GRÁFICOS MÊS A MÊS (SOMENTE 2024)
    # --------------------------------------------------------
    plot_dir = os.path.join(base_plot_dir, "comparativo", barcode)
    os.makedirs(plot_dir, exist_ok=True)

    merged_preds["Date"] = pd.to_datetime(merged_preds["Date"])
    merged_preds_2024 = merged_preds[merged_preds["Date"].dt.year == 2024]

    for month, df_month in merged_preds_2024.groupby(merged_preds_2024["Date"].dt.month):
        plt.figure(figsize=(10, 5))

        # Linha real
        plt.plot(df_month["Date"], df_month["real"], label="REAL", marker="o", linestyle="-")

        # Linha XGBOOST
        if "xgboost" in df_month.columns:
            plt.plot(df_month["Date"], df_month["xgboost"], label="XGBOOST", marker="x", linestyle="-")

        # Linha NN
        if "nn" in df_month.columns:
            plt.plot(df_month["Date"], df_month["nn"], label="NN", marker="x", linestyle="-")

        # Finalização do gráfico
        plt.title(f"COMPARATIVO – {barcode} – {month:02d}/2024")
        plt.xlabel("Dia do mês")
        plt.ylabel("Quantidade")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(os.path.join(plot_dir, f"comparativo_{barcode}_2024_{month:02d}.png"))
        plt.close()
        
        logger.info(f"{barcode} | Comparativo salvo para {month:02d}/2024")