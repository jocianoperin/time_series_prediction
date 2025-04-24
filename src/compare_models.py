# ============================================================
#  COMPARAÇÃO, CONSOLIDAÇÃO E PLOTAGEM DE RESULTADOS
# ------------------------------------------------------------
#  ‣ Consolida CSVs de métricas e predições de cada modelo
#  ‣ Gera gráficos mensais REAL × MODELOS
#  ‣ Mantém compatibilidade com o pipeline existente
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
#  FUNÇÃO AUXILIAR – NORMALIZA NOME DA COLUNA DE PREVISÃO
# ------------------------------------------------------------
def _normalizar_preds(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Renomeia, se necessário, a coluna de previsão para `<model_name>`.

    Aceita as variações:
        • forecast
        • prediction_<model_name>
        • predicted
        • yhat
    """
    possiveis = [
        "forecast",
        f"prediction_{model_name}",
        "predicted",
        "yhat",
    ]
    for col in possiveis:
        if col in df.columns:
            return df.rename(columns={col: model_name})

    raise ValueError(
        f"Coluna de predição não encontrada no DF de {model_name} "
        f"({', '.join(possiveis)})"
    )


# ------------------------------------------------------------
#  FUNÇÃO PRINCIPAL – COMPARAÇÃO / CONSOLIDAÇÃO / PLOTAGEM
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
    metrics_list: list[dict] = []
    merged_preds: pd.DataFrame | None = None
    real_col_name: str | None = None

    # ------------------- ITERA MODELOS ---------------------
    for model_name, data in results.items():
        # `data` deve conter "predictions" (obrigatório) e "metrics" (opcional)
        if not data or "predictions" not in data:
            continue

        # ------- PREVISÕES ---------------------------------
        df = data["predictions"].copy()
        if df.empty:
            continue

        # define coluna REAL na 1ª iteração que a contiver
        if merged_preds is None:
            if "Quantity" in df.columns:
                real_col_name = "Quantity"
            elif "real" in df.columns:
                real_col_name = "real"
            else:
                # sem coluna real → pula este modelo e tenta o próximo
                continue
            merged_preds = df[["Date", real_col_name]].copy()

        # adiciona coluna do modelo
        df_norm = _normalizar_preds(df, model_name)[["Date", model_name]]
        merged_preds = pd.merge(
            merged_preds, df_norm, on="Date", how="outer"
        )

        # ------- MÉTRICAS ----------------------------------
        if "metrics" in data and data["metrics"]:
            m = data["metrics"].copy()
            m["model"] = model_name
            metrics_list.append(m)

    # nada para consolidar → aborta silenciosamente
    if merged_preds is None:
        return

    # renomeia coluna real → real
    if real_col_name and real_col_name != "real":
        merged_preds = merged_preds.rename(columns={real_col_name: "real"})

    # ------------------- SAÍDAS ---------------------------
    out_cmp_dir = os.path.join(base_pred_dir, "comparativo")
    os.makedirs(out_cmp_dir, exist_ok=True)

    # — métricas agregadas —
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(
            os.path.join(out_cmp_dir, f"{barcode}_metrics.csv"), index=False
        )
    else:
        metrics_df = pd.DataFrame()

    # — predições consolidadas —
    merged_preds.to_csv(
        os.path.join(out_cmp_dir, f"predicoes_2024_{barcode}.csv"),
        index=False,
    )

    # ------------------- GRÁFICOS -------------------------
    plot_dir = os.path.join(base_plot_dir, "comparativo", barcode)
    os.makedirs(plot_dir, exist_ok=True)

    merged_preds["Date"] = pd.to_datetime(merged_preds["Date"])
    merged_preds_2024 = merged_preds[merged_preds["Date"].dt.year == 2024]

    for month, df_month in merged_preds_2024.groupby(
        merged_preds_2024["Date"].dt.month
    ):
        plt.figure(figsize=(10, 5))

        # — série real —
        plt.plot(
            df_month["Date"],
            df_month["real"],
            label="REAL",
            marker="o",
        )

        # — séries dos modelos —
        for model in metrics_df["model"].tolist() if not metrics_df.empty else []:
            if model in df_month.columns:
                plt.plot(
                    df_month["Date"],
                    df_month[model],
                    label=model.upper(),
                    marker="x",
                )

        plt.title(f"COMPARATIVO – {barcode} – {month:02d}/2024")
        plt.xlabel("Dia")
        plt.ylabel("Quantidade")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                plot_dir, f"comparativo_{barcode}_2024_{month:02d}.png"
            )
        )
        plt.close()