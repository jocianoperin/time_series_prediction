import pandas as pd
import gc
import numpy as np
import os
import tensorflow as tf

# ======== AJUSTE DE VRAM (apenas para evitar OOM) ====================
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"          # Habilita memory growth
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"        # Desabilita o XLA (auto JIT)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=3584)],  # MB; ajuste se quiser mais/menos
        )
    except RuntimeError as e:
        # Será lançado se o TF já inicializou a GPU antes deste ponto
        print("[TensorFlow] logical_device_configuration já definida:", e)
# ====================================================================

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
from utils.logging_config import get_logger
from utils.metrics import calculate_metrics
from data_preparation import generate_rolling_windows

logger = get_logger(__name__)

# ========================================================= #
# --------------  CONFIGURAÇÕES GERAIS -------------------- #
# ========================================================= #
TIME_STEPS = 30  # número de passos usados nas sequências

def create_sequences(X, y, time_steps: int = TIME_STEPS):
    """
    Constrói pares (X_seq, y_target) a partir de séries multivariadas.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i : i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


# ========================================================= #
# --------------  CONSTRUÇÃO DINÂMICA DO MODELO ----------- #
# ========================================================= #
def build_lstm_model(input_shape, layers_config, learning_rate: float = 1e-3):
    """
    Constrói, de forma dinâmica, uma pilha que pode misturar:
      • LSTM / GRU (unidirecionais ou bidirecionais)
      • Dense intermediárias
      • Camadas de Atenção Multi‑Head (“ATTN”)
    
    A lista `layers_config` deve conter dicionários, por exemplo:
        {"type": "LSTM",  "units": 256, "return_sequences": True,  ...}
        {"type": "ATTN",  "heads": 4,   "key_dim": 32,  "dropout": 0.1}
        {"type": "Dense", "units": 64,  "activation": "relu"}
    
    Observação:
      – Para que a atenção funcione, a entrada dela precisa ser 3‑D
        (batch, timesteps, features). Portanto, garanta que a camada
        imediatamente anterior tenha `return_sequences=True`.
    """
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Input, LSTM, GRU, Dense, Dropout, Bidirectional,
        MultiHeadAttention, Add, LayerNormalization,
        GlobalAveragePooling1D
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    inputs = Input(shape=input_shape)
    x = inputs  # tensor que vamos encadear

    for i, layer in enumerate(layers_config):
        layer_type       = layer.get("type", "LSTM").upper()
        units            = layer.get("units", 64)
        activation       = layer.get("activation", "relu")
        dropout          = layer.get("dropout", 0.0)
        return_sequences = layer.get("return_sequences", False)
        bidirectional    = layer.get("bidirectional", False)

        # ---------- LSTM / GRU / Dense -------------------------------- #
        if layer_type in ["LSTM", "GRU", "DENSE"]:
            if layer_type == "LSTM":
                core_layer = LSTM(
                    units,
                    activation=activation,
                    return_sequences=return_sequences,
                )
            elif layer_type == "GRU":
                core_layer = GRU(
                    units,
                    activation=activation,
                    return_sequences=return_sequences,
                )
            else:  # Dense
                core_layer = Dense(units, activation=activation)

            x = Bidirectional(core_layer)(x) if bidirectional and layer_type in ["LSTM", "GRU"] else core_layer(x)

            if dropout > 0:
                x = Dropout(dropout)(x)

        # ---------- Camada de Atenção --------------------------------- #
        elif layer_type == "ATTN":
            heads     = layer.get("heads", 4)
            key_dim   = layer.get("key_dim", 32)
            attn_drop = layer.get("dropout", 0.0)

            attn_out = MultiHeadAttention(
                num_heads=heads,
                key_dim=key_dim,
                dropout=attn_drop,
            )(x, x)
            attn_out = Dropout(attn_drop)(attn_out)
            x = Add()([x, attn_out])
            x = LayerNormalization()(x)
        else:
            raise ValueError(f"Tipo de camada desconhecido: {layer_type}")

    if len(x.shape) == 3:
        x = GlobalAveragePooling1D()(x)

    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="mae",
    )
    return model

# ========================================================= #
# --------------  TREINAMENTO / PREDIÇÃO ------------------ #
# ========================================================= #
def train_neural_network(df: pd.DataFrame, barcode: str):
    """
    - Treina janelas rolling de 365 × 31 dias para avaliar o modelo.
    - Depois re‑treina com todo o histórico até 31/12/2023.
    - Gera predições **diárias** para 2024 em esquema walk‑forward.
    - Salva CSVs e gráficos no mesmo padrão usado pelo ARIMA.
    """
    # ---------------------------------------------------------------- #
    # ---------- 1) LIMPEZA / NORMALIZAÇÃO DE FEATURES ---------------- #
    # ---------------------------------------------------------------- #
    df = df.dropna().sort_values("Date").reset_index(drop=True)

    df_treino = df[df["Date"] < "2024-01-01"].copy()
    df_2024 = df[df["Date"].dt.year == 2024].copy()

    features = [c for c in df.columns if c not in ["Date", "Quantity"]]

    df = df.dropna().replace([np.inf, -np.inf], 0).sort_values("Date").reset_index(drop=True)

    scaler = StandardScaler()
    df_treino[features] = scaler.fit_transform(df_treino[features])
    df_2024[features] = scaler.transform(df_2024[features])

    # ---------------------------------------------------------------- #
    # ---------- 2) AVALIAÇÃO ROLLING 365 × 31 ------------------------ #
    # ---------------------------------------------------------------- #
    results = []
    windows = generate_rolling_windows(df_treino, train_days=365, test_days=31)

    layers_config = [
        # Camada 1 removida (a de 512 unidades)
        #{"type": "LSTM", "units": 512, "activation": "relu", "dropout": 0.4, "return_sequences": True, "bidirectional": True},
        # Nova 1: LSTM com 256 unidades, bidirecional, permanece a mesma
        {"type": "LSTM", "units": 256, "activation": "relu", "dropout": 0.3, "return_sequences": True, "bidirectional": True},
        # A camada GRU permanece
        {"type": "GRU",  "units": 128, "activation": "relu", "dropout": 0.25, "return_sequences": True},
        # Camada de atenção permanece
        {"type": "ATTN", "heads": 4,   "key_dim": 32, "dropout": 0.1},
        # Camada 5 removida (a de 64 unidades)
        #{"type": "LSTM", "units": 64,  "activation": "relu", "dropout": 0.2, "return_sequences": False},
        # Camada final densa permanece
        {"type": "Dense","units": 32,  "activation": "relu", "dropout": 0.1},
    ]

    learning_rate = 1e-4
    batch_size = 8
    epochs = 800
    patience = 80

    for i, w in enumerate(windows):
        train, test = w["train"], w["test"]

        logger.info(
            f"Janela {i+1}: Treino={len(train)} dias "
            f"({train['Date'].min()} a {train['Date'].max()}), "
            f"Teste={len(test)} dias "
            f"({test['Date'].min()} a {test['Date'].max()})"
        )

        if len(train) <= TIME_STEPS or len(test) <= TIME_STEPS:
            logger.warning(f"{barcode} | LSTM - Janela {i+1} ignorada: dados insuficientes.")
            continue

        X_train, y_train = create_sequences(
            train[features].values, train["Quantity"].values
        )
        X_test, y_test = create_sequences(
            test[features].values, test["Quantity"].values
        )

        if len(X_train) == 0 or len(X_test) == 0:
            logger.warning(f"{barcode} | LSTM - Janela {i+1} ignorada: nenhuma sequência válida.")
            continue

        model = build_lstm_model(
            (X_train.shape[1], X_train.shape[2]), layers_config, learning_rate
        )
        es = EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1,
        )

        y_pred = model.predict(X_test).flatten()
        metrics = calculate_metrics(y_test, y_pred)

        logger.info(
            f"{barcode} | LSTM - Janela {i+1} | "
            f"MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%"
        )

        test_out = test.iloc[TIME_STEPS:].copy()
        test_out["prediction_nn"] = y_pred
        results.append(test_out)

        # ========== LIBERA MEMÓRIA DA JANELA ========================= #
        del X_train, y_train, X_test, y_test, y_pred, model
        tf.keras.backend.clear_session()   # limpa grafo da GPU/CPU
        gc.collect()                       # força coleta de lixo em Python
        # ============================================================= #

    # ---------------------------------------------------------------- #
    # ---------- 3) PREDIÇÃO DIÁRIA PARA TODO 2024 -------------------- #
    # ---------------------------------------------------------------- #

    if len(df_treino) <= TIME_STEPS:
        logger.warning(f"{barcode} | Histórico insuficiente para predição 2024.")
        return pd.DataFrame(), {}, pd.DataFrame()

    # Re‑treina no histórico completo até 31/12/2023
    df_hist = df_treino.copy()
    X_hist, y_hist = create_sequences(df_hist[features].values, df_hist["Quantity"].values)

    model = build_lstm_model(
        (X_hist.shape[1], X_hist.shape[2]), layers_config, learning_rate
    )
    model.fit(
        X_hist,
        y_hist,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[EarlyStopping(monitor="loss", patience=patience)],
        verbose=1,
    )

    # Concatena histórico + 2024 já normalizados
    df_full = pd.concat([df_hist, df_2024]).reset_index(drop=True)

    daily_rows = []
    for idx in range(TIME_STEPS, len(df_full)):
        current_date = df_full.loc[idx, "Date"]
        if current_date.year != 2024:
            continue

        seq = df_full.loc[idx - TIME_STEPS : idx - 1, features].values.reshape(
            (1, TIME_STEPS, len(features))
        )
        forecast = model.predict(seq, verbose=0).flatten()[0]
        real = df_full.loc[idx, "Quantity"]

        met = calculate_metrics(np.array([real]), np.array([forecast]))
        daily_rows.append(
            {
                "mae": met["mae"],
                "rmse": met["rmse"],
                "mape": met["mape"],
                "smape": met["smape"],
                "date": current_date,
                "barcode": barcode,
                "forecast": forecast,
                "real": real,
            }
        )

    if not daily_rows:
        logger.warning(f"{barcode} | Nenhuma predição gerada para 2024.")
        return pd.DataFrame(), {}, pd.DataFrame()

    df_daily = pd.DataFrame(daily_rows)

    # ---------------------------------------------------------------- #
    # ---------- 4) EXPORTAÇÃO DE CSVs + GRÁFICOS --------------------- #
    # ---------------------------------------------------------------- #
    # Ajuste para criar subpasta para cada barcode em data/predictions/NN
    out_dir = f"data/predictions/NN/{barcode}"
    os.makedirs(out_dir, exist_ok=True)
    for month, df_month in df_daily.groupby(df_daily["date"].dt.month):
        csv_path = os.path.join(out_dir, f"NN_daily_{barcode}_2024_{month:02d}.csv")
        df_month.to_csv(csv_path, index=False)
        plot_nn_monthly(df_month, barcode, month)

    # Métricas agregadas (média dos erros diários)
    nn_metrics_2024 = {
        "mae": df_daily["mae"].mean(),
        "rmse": df_daily["rmse"].mean(),
        "mape": df_daily["mape"].mean(),
        "smape": df_daily["smape"].mean(),
    }

    # ---------------------------------------------------------------- #
    # ---------- 5) RETORNO PARA O PIPELINE PRINCIPAL ---------------- #
    # ---------------------------------------------------------------- #
    df_all_results = (
        pd.concat(results).reset_index(drop=True) if results else pd.DataFrame()
    )
    return (
        df_all_results,
        nn_metrics_2024,
        df_daily[["date", "forecast"]].rename(columns={"date": "Date"}),
    )

# ========================================================= #
# --------------  FUNÇÃO DE PLOTAGEM MENSAL --------------- #
# ========================================================= #
def plot_nn_monthly(df_plot: pd.DataFrame, barcode: str, month: int):
    """
    Gera gráfico Real vs. Previsto para um mês específico de 2024.
    Espera DataFrame no formato (date, real, forecast).
    """
    df_plot = df_plot.sort_values("date")

    plt.figure(figsize=(10, 5))
    plt.plot(df_plot["date"], df_plot["real"], label="Real", marker="o")
    plt.plot(df_plot["date"], df_plot["forecast"], label="Previsto", marker="x")

    for x, y in zip(df_plot["date"], df_plot["real"]):
        plt.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=8)
    for x, y in zip(df_plot["date"], df_plot["forecast"]):
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