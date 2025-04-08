# Atualizado src/compare_models.py para consolidar resultados refinados
import os
import pandas as pd

def compare_and_save_results(barcode, results, out_path="data/predictions"):
    """
    Recebe um dicionário com previsões e métricas dos modelos e salva os dados.
    Exemplo do dicionário:
    {
        'arima': {'metrics': ..., 'predictions': DataFrame},
        'prophet': {...},
        ...
    }
    """
    metrics_list = []
    merged_preds = None

    for model_name, data in results.items():
        # Salvar métricas
        m = data["metrics"].copy()
        m["model"] = model_name
        metrics_list.append(m)

        # Salvar predições (2024)
        df = data["predictions"]
        if merged_preds is None:
            merged_preds = df.copy()
        else:
            merged_preds = pd.merge(merged_preds, df, on="Date", how="outer")

    # Salvar métricas
    metrics_df = pd.DataFrame(metrics_list)
    metrics_out = os.path.join("data/metrics", f"{barcode}_metrics.csv")
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    metrics_df.to_csv(metrics_out, index=False)

    # Salvar predições combinadas
    pred_out = os.path.join(out_path, f"predicoes_2024_{barcode}.csv")
    os.makedirs(out_path, exist_ok=True)
    merged_preds.to_csv(pred_out, index=False)

    # Determinar melhor modelo por menor MAE
    best_model = metrics_df.sort_values("mae").iloc[0]["model"]
    print(f"Para o produto {barcode}, o melhor modelo foi: {best_model}")
