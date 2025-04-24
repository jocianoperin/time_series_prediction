# ============================================================
#  COMPARAÇÃO, CONSOLIDAÇÃO E PLOTAGEM DE RESULTADOS
# ============================================================
import os
import pandas as pd
import matplotlib.pyplot as plt

def _normalizar_preds(df, model_name):
    """
    Renomeia a coluna‑de‑previsão para o nome do modelo.
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
    raise ValueError(f"Coluna de predição não encontrada para {model_name}")

def compare_and_save_results(barcode, results,
                             base_pred_dir="data/predictions",
                             base_plot_dir="data/plots"):
    """
    Consolida métricas + predições, salva em
        data/predictions/comparativo/
    e gera gráficos mensais em
        data/plots/comparativo/<barcode>/.
    """
    metrics_list  = []
    merged_preds  = None
    real_col_name = None

    # ------------------- CONSOLIDAÇÃO -----------------------
    for model_name, data in results.items():
        if not data:
            continue

        # ---- métricas --------------------------------------
        m = data["metrics"].copy()
        m["model"] = model_name
        metrics_list.append(m)

        # ---- predições -------------------------------------
        # ---------- define a série REAL (primeiro DF que a contiver) ----------
        if merged_preds is None:
            if "Quantity" in df.columns:
                real_col_name = "Quantity"
            elif "real" in df.columns:
                real_col_name = "real"
            else:
                # este modelo não tem coluna real → pula e tenta no próximo
                continue

            merged_preds = df[["Date", real_col_name]].copy()


        df = _normalizar_preds(df, model_name)[["Date", model_name]]
        merged_preds = pd.merge(merged_preds, df, on="Date", how="outer")

    # Renomeia coluna real → real
    if real_col_name and real_col_name != "real":
        merged_preds = merged_preds.rename(columns={real_col_name: "real"})

    # ------------------- SAÍDAS -----------------------------
    out_cmp_dir = os.path.join(base_pred_dir, "comparativo")
    os.makedirs(out_cmp_dir, exist_ok=True)

    # métricas
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(
        os.path.join(out_cmp_dir, f"{barcode}_metrics.csv"), index=False
    )

    # predições consolidadas
    merged_preds.to_csv(
        os.path.join(out_cmp_dir, f"predicoes_2024_{barcode}.csv"), index=False
    )

    # melhor modelo
    best_model = metrics_df.sort_values("mae").iloc[0]["model"]
    print(f"Para o produto {barcode}, o melhor modelo foi: {best_model}")

    # ------------------- GRÁFICOS ---------------------------
    plot_dir = os.path.join(base_plot_dir, "comparativo", barcode)
    os.makedirs(plot_dir, exist_ok=True)

    merged_preds["Date"] = pd.to_datetime(merged_preds["Date"])
    merged_preds_2024 = merged_preds[merged_preds["Date"].dt.year == 2024]

    for month, df_month in merged_preds_2024.groupby(
        merged_preds_2024["Date"].dt.month
    ):
        plt.figure(figsize=(10, 5))
        # linha real
        plt.plot(
            df_month["Date"],
            df_month["real"],
            label="REAL",
            marker="o",
        )
        # linhas dos modelos rodados
        for model in metrics_df["model"]:
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
