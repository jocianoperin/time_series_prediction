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
import numpy as np
from utils.logging_config import get_logger

logger = get_logger(__name__)

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
        "prediction",
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
        """if "metrics" in data and data["metrics"]:
            m = data["metrics"].copy()
            m["model"] = model_name
            metrics_list.append(m)"""
        
        m = data.get("metrics", {})
        m["model"] = model_name
        metrics_list.append(m)

    # nada para consolidar → aborta silenciosamente
    if merged_preds is None:
        return

    # renomeia coluna real → real
    if real_col_name and real_col_name != "real":
        merged_preds = merged_preds.rename(columns={real_col_name: "real"})

    # ------------------- SAÍDAS ---------------------------
    # --- agora cada barcode tem uma subpasta própria -------------
    out_cmp_dir = os.path.join(base_pred_dir, "comparativo", barcode)
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

    # -------- CSVs diários por mês -----------------------------
    merged_preds["year"]  = merged_preds["Date"].dt.year
    merged_preds["month"] = merged_preds["Date"].dt.month

    for (yr, mn), df_m in merged_preds.groupby(["year", "month"]):
        # métricas diárias por modelo
        for mdl in [c for c in df_m.columns if c not in {"Date", "real", "year", "month"}]:
            df_m[f"mae_{mdl}"]  = (df_m[mdl] - df_m["real"]).abs()
            df_m[f"rmse_{mdl}"] = np.sqrt((df_m[mdl] - df_m["real"])**2)
            df_m[f"mape_{mdl}"] = df_m[f"mae_{mdl}"] / df_m["real"].replace(0, np.nan) * 100
            df_m[f"smape_{mdl}"]= df_m[f"mae_{mdl}"] / ((df_m["real"].abs() + df_m[mdl].abs())/2).replace(0, np.nan) * 100

        csv_path = os.path.join(
            out_cmp_dir,
            f"{barcode}_{yr}_{mn:02d}_diario.csv"
        )
        df_m.drop(columns=["year", "month"]).to_csv(csv_path, index=False)


    # ------------------- GRÁFICOS -------------------------
    plot_dir = os.path.join(base_plot_dir, "comparativo", barcode)
    os.makedirs(plot_dir, exist_ok=True)

    merged_preds["Date"] = pd.to_datetime(merged_preds["Date"])
    merged_preds_2024 = merged_preds[merged_preds["Date"].dt.year == 2024]

    for month, df_month in merged_preds_2024.groupby(
        merged_preds_2024["Date"].dt.month
    ):
        plt.figure(figsize=(10, 5))

        # --- REAL ---
        plt.plot(
            df_month["Date"],
            df_month["real"],
            label="REAL",
            marker="o",
            linestyle="-",
        )

        # --- XGBOOST ---
        if "xgboost" in df_month.columns:
            plt.plot(
                df_month["Date"],
                df_month["xgboost"],
                label="XGBOOST",
                marker="x",
                linestyle="-",
            )

        # --- NN ---
        if "nn" in df_month.columns:
            plt.plot(
                df_month["Date"],
                df_month["nn"],
                label="NN",
                marker="x",
                linestyle="-",
            )

        # --- formatação final ---
        plt.title(f"COMPARATIVO – {barcode} – {month:02d}/2024")
        plt.xlabel("Dia do mês")
        plt.ylabel("Quantidade")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # salva
        plt.savefig(
            os.path.join(
                plot_dir, f"comparativo_{barcode}_2024_{month:02d}.png"
            )
        )
        plt.close()
        logger.info(f"{barcode} | Comparativo salvo para {month:02d}/2024")