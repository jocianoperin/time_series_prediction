import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from utils.logging_config import get_logger
from utils.metrics import calculate_metrics
from data_preparation import generate_rolling_windows

logger = get_logger(__name__)
TIME_STEPS = 30

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Construção flexível do modelo LSTM
def build_lstm_model(input_shape, layers_config, learning_rate=0.001):
    model = Sequential()
    
    for i, layer in enumerate(layers_config):
        layer_type = layer.get("type", "LSTM")
        units = layer.get("units", 64)
        activation = layer.get("activation", "relu")
        dropout = layer.get("dropout", 0.0)
        return_sequences = layer.get("return_sequences", False)
        bidirectional = layer.get("bidirectional", False)

        LayerClass = LSTM if layer_type == "LSTM" else GRU if layer_type == "GRU" else Dense

        if i == 0:
            if bidirectional and layer_type in ["LSTM", "GRU"]:
                model.add(Bidirectional(LayerClass(units, activation=activation, return_sequences=return_sequences),
                                        input_shape=input_shape))
            else:
                model.add(LayerClass(units, activation=activation, return_sequences=return_sequences, input_shape=input_shape))
        else:
            if bidirectional and layer_type in ["LSTM", "GRU"]:
                model.add(Bidirectional(LayerClass(units, activation=activation, return_sequences=return_sequences)))
            else:
                model.add(LayerClass(units, activation=activation, return_sequences=return_sequences))

        if dropout > 0:
            model.add(Dropout(dropout))

    model.add(Dense(1))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mae')

    return model

def train_neural_network(df, barcode):
    df = df.dropna().sort_values("Date").reset_index(drop=True)
    df_treino = df[df["Date"] < "2024-01-01"].copy()
    df_2024 = df[df["Date"].dt.year == 2024].copy()
    features = [col for col in df.columns if col not in ["Date", "Quantity"]]

    scaler = StandardScaler()
    df_treino[features] = scaler.fit_transform(df_treino[features])
    df_2024[features] = scaler.transform(df_2024[features])

    results = []
    forecast_2024 = []
    windows = generate_rolling_windows(df_treino, train_days=365, test_days=31)

    # ===== CONFIGURAÇÕES DO MODELO (EXEMPLOS) ===== #
    """
    Exemplo 1: Mais camadas LSTM:
    layers_config = [
        {"type": "LSTM", "units": 256, "activation": "relu", "dropout": 0.3, "return_sequences": True, "bidirectional": True},
        {"type": "LSTM", "units": 128, "activation": "relu", "dropout": 0.2, "return_sequences": True, "bidirectional": False},
        {"type": "LSTM", "units": 64, "activation": "relu", "dropout": 0.2, "return_sequences": False},
    ]
    Exemplo 2: Usar camadas GRU e Dense junto com LSTM:
    layers_config = [
        {"type": "GRU", "units": 128, "activation": "relu", "dropout": 0.2, "return_sequences": True, "bidirectional": True},
        {"type": "LSTM", "units": 64, "activation": "relu", "dropout": 0.2, "return_sequences": False},
        {"type": "Dense", "units": 32, "activation": "relu", "dropout": 0.1}, # Dense não precisa de 'return_sequences'
        ]"""

    layers_config = [
        {"type": "LSTM", "units": 128, "activation": "relu", "dropout": 0.2, "return_sequences": True, "bidirectional": True},
        {"type": "LSTM", "units": 64, "activation": "relu", "dropout": 0.1, "return_sequences": False},
    ]

    """Exemplo 3: Alterar learning rate, epochs e batch size:
    learning_rate = 0.0005  # Menor para aprendizado mais suave
    batch_size = 64         # Batch maior para treinamento mais rápido, caso tenha recursos
    epochs = 150            # Aumentar para mais oportunidades de aprendizado
    patience = 20           # Mais paciência para interromper o treinamento"""

    learning_rate = 0.001   # Menor para aprendizado mais suave
    batch_size = 32         # Batch maior para treinamento mais rápido, caso tenha recursos
    epochs = 1000            # Aumentar para mais oportunidades de aprendizado
    patience = 50           # Mais paciência para interromper o treinamento

    """Exemplo 4: Camada simples para testes rápidos:
    layers_config = [
        {"type": "LSTM", "units": 32, "activation": "relu", "dropout": 0.1, "return_sequences": False},
    ]

    learning_rate = 0.01   # Aprendizado mais rápido para testes
    batch_size = 16        # Batches menores para testes rápidos
    epochs = 30            # Menos épocas para testes rápidos
    patience = 5           # Interrupção rápida caso não haja melhora"""

    # ============================================== #

    for i, w in enumerate(windows):
        train, test = w["train"], w["test"]

        logger.info(f"Janela {i+1}: Treino={len(train)} dias ({train['Date'].min()} a {train['Date'].max()}), "
                    f"Teste={len(test)} dias ({test['Date'].min()} a {test['Date'].max()})")

        if len(train) <= TIME_STEPS or len(test) <= TIME_STEPS:
            logger.warning(f"{barcode} | LSTM - Janela {i+1} ignorada: dados insuficientes.")
            continue

        X_train, y_train = create_sequences(train[features].values, train["Quantity"].values)
        X_test, y_test = create_sequences(test[features].values, test["Quantity"].values)

        if len(X_test) == 0 or len(X_train) == 0:
            logger.warning(f"{barcode} | LSTM - Janela {i+1} ignorada: nenhuma sequência válida.")
            continue

        model = build_lstm_model((X_train.shape[1], X_train.shape[2]), layers_config, learning_rate)
        es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

        model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=1)

        y_pred = model.predict(X_test).flatten()
        metrics = calculate_metrics(y_test, y_pred)

        logger.info(f"{barcode} | LSTM - Janela {i+1} | Treino: {train['Date'].min()} a {train['Date'].max()} | "
                    f"Teste: {test['Date'].min()} a {test['Date'].max()} | "
                    f"MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%")

        test_out = test.iloc[TIME_STEPS:].copy()
        test_out["prediction_nn"] = y_pred
        results.append(test_out)

    # Previsão mês a mês de 2024
    if len(df_treino) > TIME_STEPS:
        X_full, y_full = create_sequences(df_treino[features].values, df_treino["Quantity"].values)
        model = build_lstm_model((X_full.shape[1], X_full.shape[2]), layers_config, learning_rate)
        model.fit(X_full, y_full, epochs=epochs, batch_size=batch_size, verbose=1)

        for month in range(1, 13):
            future_df = df_2024[df_2024["Date"].dt.month == month].copy()
    
            if future_df.empty:
                continue

            # Verificação ajustada (não mais <= TIME_STEPS)
            if len(future_df) < 2:  # menos de 2 dias não faz sentido prever
                logger.warning(f"{barcode} | Mês {month:02d} ignorado por ter menos de 2 dias válidos.")
                continue

            # A ideia é criar sequências menores se não houver dados suficientes
            steps = min(TIME_STEPS, len(future_df) - 1)

            # Criação de sequências mais robusta
            X_future, y_real = create_sequences(future_df[features].values, future_df["Quantity"].values, time_steps=steps)

            if len(X_future) == 0:
                logger.warning(f"{barcode} | Nenhuma sequência válida no mês {month:02d}.")
                continue

            # Predição
            y_pred = model.predict(X_future).flatten()

            # Ajustar alinhamento do DataFrame para exportar resultados corretamente
            future_out = future_df.iloc[steps:].copy()
            future_out["prediction_nn"] = y_pred
            forecast_2024.append(future_out[["Date", "prediction_nn"]])

            # Exportação CSV mensal padrão ARIMA
            output_csv = future_out[["Date", "Quantity", "prediction_nn"]].copy()
            output_csv.rename(columns={"Quantity": "real", "prediction_nn": "forecast"}, inplace=True)

            os.makedirs("data/predictions", exist_ok=True)
            output_csv.to_csv(f"data/predictions/NN_daily_{barcode}_2024_{month:02d}.csv", index=False)

            plot_nn_monthly(future_out, barcode, month)

            # Fine-tune incremental com dados reais
            model.fit(X_future, y_real, epochs=10, batch_size=batch_size, verbose=1)

            # Calcula métricas
            metrics = calculate_metrics(y_real, y_pred)
            logger.info(f"{barcode} | LSTM - Predição 2024-{month:02d} | MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%")

    return (
        pd.concat(results) if results else pd.DataFrame(columns=["Date", "Quantity", "prediction_nn"]),
        metrics if 'metrics' in locals() else {"mae": None, "mape": None, "rmse": None},
        pd.concat(forecast_2024) if forecast_2024 else pd.DataFrame(columns=["Date", "prediction_nn"])
    )

def plot_nn_monthly(df_plot, barcode, month):
    df_plot = df_plot.sort_values("Date")
    plt.figure(figsize=(10, 5))
    plt.plot(df_plot["Date"], df_plot["Quantity"], label="Real", marker="o")
    plt.plot(df_plot["Date"], df_plot["prediction_nn"], label="Previsto", marker="x")

    # valores explícitos no gráfico
    for x, y in zip(df_plot["Date"], df_plot["Quantity"]):
        plt.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=8)
    for x, y in zip(df_plot["Date"], df_plot["prediction_nn"]):
        plt.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=8, color="blue")

    plt.title(f"Comparação Real vs. NN (LSTM) - {barcode} - {month:02d}/2024")
    plt.xlabel("Dia")
    plt.ylabel("Quantidade")
    plt.legend()
    plt.xticks(rotation=45)

    out_dir = f"data/plots/NN/{barcode}"
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"NN_{barcode}_2024_{month:02d}.png"))
    plt.close()

