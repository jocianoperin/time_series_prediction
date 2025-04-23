"""
TRAIN_NN.PY – BLOCO 1/3  (IMPORTS, AJUSTES DE GPU, CONSTANTES E FUNÇÕES AUXILIARES)
-----------------------------------------------------------------
Nesta primeira parte permanecem **apenas**:
  • imports e configuração de alocador/VRAM
  • definição de constantes
  • função helper `create_sequences`
Os próximos blocos (modelo, treino, exportações) serão adicionados
na sequência, preservando TODOS os comentários descritivos.
"""

# ================================================================
# IMPORTS PRINCIPAIS
# ================================================================
import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    LSTM,
    GRU,
    Dense,
    Dropout,
    Bidirectional,
    Input,
    MultiHeadAttention,
    Add,
    LayerNormalization,
    GlobalAveragePooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils.logging_config import get_logger
from utils.metrics import calculate_metrics
from data_preparation import generate_rolling_windows
from utils.gpu_utils import free_gpu_memory  # liberação total de VRAM

# ================================================================
# AJUSTES GLOBAIS DE GPU / VRAM
# ================================================================
# 1) Usa alocador "cuda_malloc_async" para devolver blocos de VRAM
#    imediatamente após o tensor ser desalocado.
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# 2) Permite que o TensorFlow cresça a memória conforme a demanda
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# 3) Desativa o XLA JIT – costuma reter buffers extras na VRAM
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

# → Limita a placa (ex.: GTX 1650 4 GB) a ~3.5 GB para evitar OOM.
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
        )
    except RuntimeError:
        # A configuração só funciona se a GPU ainda não foi inicializada
        pass

# ================================================================
# CONSTANTES E LOGGER
# ================================================================
logger = get_logger(__name__)
TIME_STEPS = 30  # número de passos de tempo usados nas sequências

# ================================================================
# FUNÇÃO AUXILIAR – CRIAÇÃO DE JANELAS SEQUENCIAIS
# ================================================================

def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int = TIME_STEPS):
    """Gera pares `(X_seq, y_target)` para entrada em redes recorrentes.

    Parâmetros
    ----------
    X : ndarray, shape (n_samples, n_features)
        Matriz com as features já escalonadas.
    y : ndarray, shape (n_samples,)
        Vetor alvo (quantidade vendida).
    time_steps : int, opcional (default=30)
        Quantidade de passos consecutivos que formam uma sequência.

    Retorno
    -------
    Xs : ndarray, shape (n_samples - time_steps, time_steps, n_features)
    ys : ndarray, shape (n_samples - time_steps,)
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i : i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# ========================================================= #
# --------------  CONSTRUÇÃO DINÂMICA DO MODELO ----------- #
# ========================================================= #
def build_lstm_model(
    input_shape: tuple,
    layers_config: list,
    learning_rate: float = 1e-3,
):
    """Constrói, de forma *dinâmica*, uma pilha de camadas composta por
    LSTM, GRU, camadas densas e/ou blocos de Atenção Multi‑Head.

    A função recebe uma lista `layers_config`, onde cada elemento é um
    dicionário com chaves:
      • type   : "LSTM", "GRU", "DENSE" ou "ATTN"
      • units  : tamanho da camada (para LSTM / GRU / Dense)
      • dropout: taxa de dropout (padrão 0)
      • activation : função de ativação (padrão "relu")
      • return_sequences : bool (apenas para LSTM / GRU)
      • bidirectional    : bool (apenas para LSTM / GRU)
      • heads / key_dim  : nº de cabeças e dimensão‑chave (para ATTN)

    Observações importantes
    -----------------------
    • Para que um bloco de Atenção receba tensores 3‑D, a camada anterior
      **deve** ter `return_sequences=True`.
    • Caso o tensor de saída ainda seja 3‑D após o loop principal, ele é
      reduzido via **GlobalAveragePooling1D** antes da densa final.
    """

    # ---------- Entrada do modelo ---------------------------------
    inputs = Input(shape=input_shape)
    x = inputs  # tensor corrente

    # ---------- Loop pelas camadas definidas em layers_config ------
    for idx, layer in enumerate(layers_config):
        layer_type = layer.get("type", "LSTM").upper()
        units      = layer.get("units", 64)
        activation = layer.get("activation", "relu")
        dropout    = layer.get("dropout", 0.0)
        return_seq = layer.get("return_sequences", False)
        bidir      = layer.get("bidirectional", False)

        if layer_type in {"LSTM", "GRU", "DENSE"}:
            # --- cria a camada core --------------------------------
            if layer_type == "LSTM":
                core = LSTM(units, activation=activation, return_sequences=return_seq)
            elif layer_type == "GRU":
                core = GRU(units, activation=activation, return_sequences=return_seq)
            else:  # Dense
                core = Dense(units, activation=activation)

            # --- aplica bidirecional se solicitado -----------------
            if bidir and layer_type in {"LSTM", "GRU"}:
                x = Bidirectional(core)(x)
            else:
                x = core(x)

            # --- dropout opcional ----------------------------------
            if dropout > 0:
                x = Dropout(dropout)(x)

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
            x        = Add()([x, attn_out])
            x        = LayerNormalization()(x)
        else:
            raise ValueError(f"Tipo de camada desconhecido: {layer_type}")

    # ---------- Pooling se tensor ainda for 3‑D ---------------------
    if len(x.shape) == 3:
        x = GlobalAveragePooling1D()(x)

    # ---------- Saída densa de regressão ----------------------------
    outputs = Dense(1)(x)

    # ---------- Compilação do modelo --------------------------------
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="mae",
    )
    return model

# ================================================================
# BLOCO 3 – FUNÇÃO train_neural_network (ROLLING + WALK‑FORWARD)
# ================================================================

def train_neural_network(df: pd.DataFrame, barcode: str):
    """PIPELINE COMPLETO DE REDE NEURAL (LSTM + ATTN)

    Etapas
    ------
    1.  Limpeza, escalonamento e split temporal (treino < 2024; teste = 2024)
    2.  Avaliação rolling 365×31 sobre 2019‑2023
    3.  Re‑treino global até 31‑12‑2023
    4.  Walk‑forward 2024 **com fine‑tune mensal** (paridade c/ XGBoost online)
    5.  Exporta CSVs + gráficos e devolve métricas agregadas
    """

    # ---------- 1) PREPARAÇÃO & ESCALONAMENTO ---------------------
    df = df.dropna().sort_values("Date").reset_index(drop=True)

    df_treino = df[df["Date"] < "2024-01-01"].copy()
    df_2024   = df[df["Date"].dt.year == 2024].copy()

    features = [c for c in df.columns if c not in {"Date", "Quantity"}]

    scaler = StandardScaler()
    df_treino[features] = scaler.fit_transform(df_treino[features])
    df_2024[features]   = scaler.transform(df_2024[features])

    # ---------- 2) AVALIAÇÃO ROLLING 365×31 ----------------------
    windows = generate_rolling_windows(df_treino, train_days=365, test_days=31)

    layers_cfg = [
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

    lr          = 1e-4
    batch_size  = 8
    epochs      = 800
    patience    = 80
    fine_epochs = 10    # fine‑tune mensal
    fine_batch  = 8

    rolling_frames = []
    for idx, win in enumerate(windows):
        train, test = win["train"], win["test"]

        logger.info(
            f"{barcode} | NN – janela {idx+1}: "
            f"treino {train['Date'].min()}→{train['Date'].max()} | "
            f"teste {test['Date'].min()}→{test['Date'].max()}"
        )

        if len(train) <= TIME_STEPS or len(test) <= TIME_STEPS:
            logger.warning(f"{barcode} | NN – janela {idx+1} ignorada (dados insuficientes)")
            continue

        X_train, y_train = create_sequences(train[features].values, train["Quantity"].values)
        X_test,  y_test  = create_sequences(test[features].values,  test["Quantity"].values)
        if len(X_train) == 0 or len(X_test) == 0:
            logger.warning(f"{barcode} | NN – janela {idx+1} ignorada (seqs vazias)")
            continue

        model_tmp = build_lstm_model((X_train.shape[1], X_train.shape[2]), layers_cfg, lr)
        es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        model_tmp.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=0,
        )

        y_pred = model_tmp.predict(X_test, verbose=0).flatten()
        mets   = calculate_metrics(y_test, y_pred)
        logger.info(
            f"{barcode} | NN – janela {idx+1}: MAE={mets['mae']:.2f}  "
            f"MAPE={mets['mape']:.2f}%"
        )

        out = test.iloc[TIME_STEPS:].copy()
        out["prediction_nn"] = y_pred
        rolling_frames.append(out)

        # limpeza da janela
        del X_train, y_train, X_test, y_test, y_pred, model_tmp
        tf.keras.backend.clear_session()
        gc.collect()

    # ---------- 3) RE‑TREINO GLOBAL (até 31‑12‑2023) -------------
    if len(df_treino) <= TIME_STEPS:
        logger.warning(f"{barcode} | NN – histórico insuficiente para 2024")
        return pd.DataFrame(), {}, pd.DataFrame()

    X_hist, y_hist = create_sequences(
        df_treino[features].values, df_treino["Quantity"].values
    )
    model = build_lstm_model((X_hist.shape[1], X_hist.shape[2]), layers_cfg, lr)
    model.fit(
        X_hist, y_hist,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[EarlyStopping(monitor="loss", patience=patience)],
        verbose=0,
    )
    logger.info(f"{barcode} | NN – re‑treino global concluído")

    # ---------- 4) WALK‑FORWARD 2024 + FINE‑TUNE MENSAL ----------
    df_full = pd.concat([df_treino, df_2024]).reset_index(drop=True)
    daily_rows = []

    for month in range(1, 13):
        df_month = df_2024[df_2024["Date"].dt.month == month]
        if df_month.empty:
            logger.warning(f"{barcode} | NN – {month:02d}/2024 ignorado (sem dados)")
            continue

        # ---- predição diária -----------------------------------
        for idx in df_month.index:
            seq_start = idx - TIME_STEPS
            if seq_start < 0:
                continue
            seq = df_full.loc[seq_start: idx - 1, features].values.reshape(
                (1, TIME_STEPS, len(features))
            )
            forecast = float(model.predict(seq, verbose=0).flatten()[0])
            real     = float(df_full.loc[idx, "Quantity"])

            met = calculate_metrics(np.array([real]), np.array([forecast]))
            daily_rows.append({
                "date": df_full.loc[idx, "Date"],
                "barcode": barcode,
                "forecast": forecast,
                "real": real,
                "mae": met["mae"],
                "rmse": met["rmse"],
                "mape": met["mape"],
                "smape": met["smape"],
            })

        # ---- fine‑tune com dados reais do mês ------------------
        X_ft, y_ft = create_sequences(
            df_month[features].values, df_month["Quantity"].values
        )
        if len(X_ft):
            model.fit(X_ft, y_ft, epochs=fine_epochs, batch_size=fine_batch, verbose=0)
            logger.info(f"{barcode} | NN – fine‑tune aplicado em {month:02d}/2024")

    if not daily_rows:
        logger.warning(f"{barcode} | NN – nenhuma predição gerada para 2024")
        return pd.DataFrame(), {}, pd.DataFrame()

    df_daily = pd.DataFrame(daily_rows)

    # ---------- 5) EXPORTAÇÃO DE CSVs + GRÁFICOS -----------------
    out_dir = f"data/predictions/NN/{barcode}"
    os.makedirs(out_dir, exist_ok=True)

    for month, df_m in df_daily.groupby(df_daily["date"].dt.month):
        csv_path = os.path.join(out_dir, f"NN_daily_{barcode}_2024_{month:02d}.csv")
        df_m[["date", "real", "forecast", "mae", "rmse", "mape", "smape"]].to_csv(csv_path, index=False)

        plot_nn_monthly(df_m, barcode, month)
        logger.info(f"{barcode} | NN – CSV+plot salvos para {month:02d}/2024")

    nn_metrics_2024 = {
        "mae":   df_daily["mae"].mean(),
        "rmse":  df_daily["rmse"].mean(),
        "mape":  df_daily["mape"].mean(),
        "smape": df_daily["smape"].mean(),
    }

    # ---------- 6) RETORNO + LIMPEZA FINAL -----------------------
    df_roll = (
        pd.concat(rolling_frames).reset_index(drop=True)
        if rolling_frames else pd.DataFrame()
    )

    del model
    free_gpu_memory()

    return df_roll, nn_metrics_2024, (
        df_daily.rename(columns={"date": "Date"})[["Date", "forecast"]]
    )


# ========================================================= #
# --------------  FUNÇÃO DE PLOTAGEM MENSAL --------------- #
# ========================================================= #
def plot_nn_monthly(df_plot: pd.DataFrame, barcode: str, month: int) -> None:
    """Plota e salva “Real vs Previsto” para um mês específico de 2024.

    Parâmetros
    ----------
    df_plot : DataFrame
        Deve conter colunas:  date, real, forecast  (já filtrado para um mês).
    barcode : str
        Código do produto (usado no título e no caminho de saída).
    month : int
        Mês de 1 a 12 correspondente ao `df_plot`.

    Arquivos gerados
    ----------------
    • PNG em data/plots/NN/<barcode>/NN_<barcode>_2024_<MM>.png
    """

    # --- Garantia de ordenação cronológica ------------------------
    df_plot = df_plot.sort_values("date")

    # --- Figura ---------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(df_plot["date"], df_plot["real"],
             label="Real",     marker="o")
    plt.plot(df_plot["date"], df_plot["forecast"],
             label="Previsto", marker="x")

    # --- Texto com valores sobre os pontos ------------------------
    for x, y in zip(df_plot["date"], df_plot["real"]):
        plt.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=8)
    for x, y in zip(df_plot["date"], df_plot["forecast"]):
        plt.text(x, y, f"{y:.0f}", ha="center", va="bottom",
                 fontsize=8, color="blue")

    # --- Estética -------------------------------------------------
    plt.title(f"REAL vs NN (LSTM) – {barcode} – {month:02d}/2024")
    plt.xlabel("Dia"); plt.ylabel("Quantidade")
    plt.legend(); plt.xticks(rotation=45)
    plt.tight_layout()

    # --- Diretório de saída + salvamento --------------------------
    out_dir = f"data/plots/NN/{barcode}"
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir,
                            f"NN_{barcode}_2024_{month:02d}.png")
    plt.savefig(filepath)
    plt.close()

    logger.info(f"{barcode} | NN – plot salvo: {filepath}")