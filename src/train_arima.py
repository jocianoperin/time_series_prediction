import os
import pandas as pd
import numpy as np
import pmdarima as pm
import matplotlib.pyplot as plt
from utils.metrics import calculate_metrics
from utils.logging_config import get_logger

logger = get_logger(__name__)


def fill_missing_dates_with_bfill_first_day(df, date_col="Date"):
    """
    Reindexa o DataFrame de 2019-01-01 até 2024-12-31, criando linhas se faltarem.
    
    Passos:
    1) Ordena o DataFrame e define 'Date' como índice.
    2) Cria o range diário entre 2019-01-01 e 2024-12-31.
    3) Reindexa, gerando linhas vazias (NaN) se estiverem faltando datas.
    4) Se a primeira linha era NaN, faz "bfill" para Barcode, UnitValue, CostValue
       a partir do segundo dia, e define zero para Quantity etc. e Holiday=1 (ou outro valor).
    5) Para os demais dias criados (do segundo em diante):
       - faz forward fill (ffill) nas colunas que se repetem (Barcode, UnitValue, CostValue);
       - e fillna(0) nas colunas que devem ser zero se não havia registro (Quantity, OnPromotion etc.).
    """
    df = df.sort_values(date_col).copy()
    df.set_index(date_col, inplace=True)

    # Cria índice diário de 2019-01-01 até 2024-12-31
    all_dates = pd.date_range(start="2019-01-01", end="2024-12-31", freq="D")
    df = df.reindex(all_dates)

    # Transformar de volta em coluna normal
    df.reset_index(inplace=True)
    df.rename(columns={"index": date_col}, inplace=True)

    # Se a primeira linha (2019-01-01) não existia e ficou NaN, 
    # iremos "bfill" de 2019-01-02 APENAS para Barcode, UnitValue, CostValue
    # e forçar zero nas demais colunas e Holiday=1, por exemplo.
    if pd.isna(df.loc[0, "Barcode"]):
        # Copiamos do dia seguinte (linha 1) se não for NaN
        if df.shape[0] > 1:  # existe pelo menos mais uma linha
            df.loc[0, "Barcode"] = df.loc[1, "Barcode"]
            df.loc[0, "UnitValue"] = df.loc[1, "UnitValue"]
            df.loc[0, "CostValue"] = df.loc[1, "CostValue"]

        # Zera colunas de venda/promo
        df.loc[0, "Quantity"] = 0
        df.loc[0, "OnPromotion"] = 0
        df.loc[0, "ReturnedQuantity"] = 0
        df.loc[0, "Discount"] = 0
        df.loc[0, "Increase"] = 0
        df.loc[0, "TotalValue"] = 0
        # Se quiser forçar Holiday=1 no primeiro dia
        df.loc[0, "Holiday"] = 1

    # Agora, fforward fill das colunas que devem se repetir nos demais dias:
    for col in ["Barcode", "UnitValue", "CostValue"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    # fillna(0) para colunas de vendas/promo etc. que devem ser zero se faltava registro
    fill_zero_cols = [
        "Quantity",
        "OnPromotion",
        "ReturnedQuantity",
        "Discount",
        "Increase",
        "TotalValue",
        "Holiday",
    ]
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def train_arima_daily_2024(df, barcode):
    """
    Treina e faz previsões diárias para cada dia de 2024, mês a mês, usando ARIMA.
    - Para prever o dia D, utiliza dados de 01/01/2019 até o dia D-1.
    - Se não houver dado real, a função 'fill_missing_dates_with_bfill_first_day'
      insere essa data com Quantity=0 ou o que definirmos no 1º dia etc.
    - Salva previsões diárias em data/predictions/
    - Calcula métricas diárias e gera sumário mensal
    - Gera gráficos mensais comparando Previsto vs. Real.
    """
    # Remove NaNs críticos em Date
    df = df.dropna(subset=["Date"]).copy()

    # Converte para datetime se necessário
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    # Mantém somente a partir de 2019-01-01
    df = df[df["Date"] >= "2019-01-01"].copy()

    # Preenche datas ausentes até 31/12/2024, 
    # inclusive tratando 01/01/2019 se faltava
    df = fill_missing_dates_with_bfill_first_day(df, date_col="Date")

    # Ordena novamente
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Lista para armazenar predições diárias
    all_daily_predictions = []

    # Loop pelos meses de 2024
    for month in range(1, 13):
        # Filtra datas do mês/ano
        mask_month = (df["Date"].dt.year == 2024) & (df["Date"].dt.month == month)
        df_month = df[mask_month].copy()
        if df_month.empty:
            continue

        daily_metrics = []
        # Pega cada dia do mês
        days_in_month = sorted(df_month["Date"].unique())

        for current_day in days_in_month:
            train_cutoff = current_day - pd.Timedelta(days=1)
            df_train = df[df["Date"] <= train_cutoff].copy()
            if df_train.empty:
                # se não existe treino, pula
                continue

            # Ajusta ARIMA
            model = pm.auto_arima(
                df_train["Quantity"],
                start_p=1, start_q=1,
                seasonal=True, m=30,
                stepwise=True, suppress_warnings=True,
                error_action="ignore"
            )

            # Previsão de 1 passo
            forecast = model.predict(n_periods=1).iloc[0]

            # Valor real do dia (pode ser 0 se foi "forçado")
            real_value = df.loc[df["Date"] == current_day, "Quantity"].values[0]

            # Calcula métricas (dia único)
            day_metrics = calculate_metrics(
                np.array([real_value]),
                np.array([forecast])
            )
            day_metrics["date"] = current_day
            day_metrics["barcode"] = barcode
            day_metrics["forecast"] = forecast
            day_metrics["real"] = real_value
            daily_metrics.append(day_metrics)

        df_daily_metrics = pd.DataFrame(daily_metrics)
        if not df_daily_metrics.empty:
            # Calcula resumo do mês
            mae_mean = df_daily_metrics["mae"].mean()
            mape_mean = df_daily_metrics["mape"].mean()
            rmse_mean = df_daily_metrics["rmse"].mean()

            logger.info(
                f"[ARIMA - {barcode}] Mês {month:02d}/2024 -> "
                f"MAE médio={mae_mean:.2f}, MAPE médio={mape_mean:.2f}%, RMSE médio={rmse_mean:.2f}"
            )

            # Salva CSV de métricas diárias
            os.makedirs("data/predictions", exist_ok=True)
            df_daily_metrics.to_csv(
                f"data/predictions/ARIMA_daily_{barcode}_{2024}_{month:02d}.csv",
                index=False
            )

            # Monta df para plot
            df_plot = df_daily_metrics[["date", "real", "forecast"]].copy()
            df_plot.rename(columns={"date": "Date"}, inplace=True)
            all_daily_predictions.append(df_plot)

            # Gera gráfico
            plot_arima_monthly(df_plot, barcode, month)

    # Retorna concatenação de todos os meses
    if all_daily_predictions:
        return pd.concat(all_daily_predictions, ignore_index=True)
    else:
        return pd.DataFrame(columns=["Date", "real", "forecast"])


def plot_arima_monthly(df_plot, barcode, month):
    """
    Gera e salva um gráfico de linhas comparando (Previsto vs. Real)
    dia a dia para o mês em questão (2024).
    """
    df_plot = df_plot.sort_values("Date")
    plt.figure(figsize=(10, 5))

    plt.plot(df_plot["Date"], df_plot["real"], label="Real", marker="o")
    plt.plot(df_plot["Date"], df_plot["forecast"], label="Previsto", marker="x")

    # Exibe valor numérico em cada ponto
    for x, y in zip(df_plot["Date"], df_plot["real"]):
        plt.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=8)
    for x, y in zip(df_plot["Date"], df_plot["forecast"]):
        plt.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=8, color="blue")

    plt.title(f"Comparação Real vs. ARIMA - {barcode} - {month:02d}/2024")
    plt.xlabel("Dia")
    plt.ylabel("Quantidade")
    plt.legend()
    plt.xticks(rotation=45)

    out_dir = f"data/plots/ARIMA/{barcode}"
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"ARIMA_{barcode}_{2024}_{month:02d}.png"))
    plt.close()