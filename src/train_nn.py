# ================================================================
# IMPORTS PRINCIPAIS
# ================================================================
import os
import random
import matplotlib.pyplot as plt
import gc
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# FIX REPRODUCIBILIDADE
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Evita alocar toda a VRAM de uma vez e melhora liberação de contexto
physical_gpus = tf.config.list_physical_devices('GPU')
for gpu in physical_gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.layers import ( # type: ignore
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
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import load_model as keras_load_model # type: ignore
from utils.logging_config import get_logger
from utils.metrics import calculate_metrics
from data_preparation import generate_rolling_windows
from utils.gpu_utils import free_gpu_memory  # liberação total de VRAM
from utils.build_nn_model import layers_cfg

# ================================================================
# AJUSTES GLOBAIS DE GPU / VRAM
# ================================================================
# 1) Usa alocador "cuda_malloc_async" para devolver blocos de VRAM
#    imediatamente após o tensor ser desalocado.
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# 2) Permite que o TensorFlow cresça a memória conforme a demanda
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# 3) Desativa o XLA JIT – costuma reter buffers extras na VRAM
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

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

TIME_STEPS = 7  # número de passos de tempo usados nas sequências

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
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

# ========================================================= #
# --------------  CONSTRUÇÃO DINÂMICA DO MODELO ----------- #
# ========================================================= #
def build_lstm_model(
    input_shape: tuple,
    layers_config: list,
    learning_rate: float = 5e-4,
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
    # sanity check
    for layer in layers_config:
        if not isinstance(layer, dict):
            raise TypeError(f"Cada layer deve ser um dict, mas recebeu {type(layer)}")

    # ---------- Entrada do modelo ---------------------------------
    inputs = Input(shape=input_shape)
    x = inputs  # tensor corrente

    # ---------- Loop pelas camadas definidas em layers_config ------
    for layer in layers_config:
        layer_type = layer.get("type", "LSTM").upper()
        units      = layer.get("units", 64)
        activation = layer.get("activation", "tanh" if layer_type in {"LSTM", "GRU"} else "relu")
        rec_act = layer.get("recurrent_activation", "sigmoid")
        dropout    = layer.get("dropout", 0.0)
        return_seq = layer.get("return_sequences", False)
        bidir      = layer.get("bidirectional", False)

         # ——— Camadas de sequência ————————————————————————
        if layer_type == "LSTM":
            core = LSTM(
                units,
                activation=activation,
                recurrent_activation=rec_act,
                return_sequences=return_seq
            )
            out = Bidirectional(core)(x) if bidir else core(x)

        elif layer_type == "GRU":
            core = GRU(
                units,
                activation=activation,
                return_sequences=return_seq
            )
            out = Bidirectional(core)(x) if bidir else core(x)

        # ——— Camada densa —————————————————————————————
        elif layer_type == "DENSE":
            out = Dense(units, activation=activation)(x)

        # ——— Bloco de Atenção Multi-Head ——————————————————
        elif layer_type == "ATTN":
            heads = layer.get("heads", 4)
            key_dim = layer.get("key_dim", 32)
            a_drop = layer.get("dropout", 0.0)

            attn = MultiHeadAttention(
                num_heads=heads,
                key_dim=key_dim,
                dropout=a_drop
            )(x, x)
            attn = Dropout(a_drop)(attn)
            out = Add()([x, attn])
            out = LayerNormalization()(out)

        else:
            raise ValueError(f"Tipo de camada desconhecido: {layer_type}")

        # ——— Dropout opcional (fora do core) ——————————
        x = Dropout(dropout)(out) if (dropout and layer_type != "ATTN") else out

    # Se ainda for 3-D, aplica pooling para achatar
    if len(x.shape) == 3:
        x = GlobalAveragePooling1D()(x)

    # Camada de saída linear para regressão
    outputs = Dense(1, activation="linear")(x)

    # Compilação com MAE para favorecer picos
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="mean_absolute_error"
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

    # --- escala linear (mantém zeros e picos sem atenuação) ---
    df_treino["Quantity_tr"] = df_treino["Quantity"].astype(np.float32)
    df_2024["Quantity_tr"]   = df_2024["Quantity"].astype(np.float32)

    # ---------------- ARQUIVOS DE MODELO -----------------
    model_dir   = f"models/NN/{barcode}"
    model_path  = os.path.join(model_dir, f"nn_{barcode}.h5")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    model_loaded  = None
    scaler_loaded = None
    if os.path.isfile(model_path) and os.path.isfile(scaler_path):
        try:
            model_loaded  = keras_load_model(model_path, compile=False)
            scaler_loaded = joblib.load(scaler_path)

            # -------------------------------------------------------------
            # O MODELO FOI CARREGADO SEM COMPILE ⇒ (compile=False):
            # recompilamos aqui para permitir .fit() e .predict()
            # -------------------------------------------------------------
            try:
                model_loaded.compile(
                    optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
                    loss="huber",
                )
            except Exception as e:
                logger.warning(f"{barcode} | Falha ao compilar rede carregada: {e}")
                model_loaded = None      # força reconstrução se der erro


            # ----------- checa compatibilidade ------------------
            num_feat_model = int(model_loaded.input_shape[-1])
            features = [c for c in df.columns if c not in {"Date", "Quantity"}]

            if num_feat_model != len(features):
                logger.warning(
                    f"{barcode} | Nº de features mudou "
                    f"({num_feat_model} ➜ {len(features)}) – descartando modelo salvo."
                )
                model_loaded  = None
                scaler_loaded = None
                # (opcional) remove arquivos obsoletos
                os.remove(model_path)
                os.remove(scaler_path)
            else:
                logger.info(f"{barcode} | Rede NN carregada de disco.")
        except Exception as e:
            logger.warning(f"{barcode} | Falha ao carregar rede salva: {e}")
            model_loaded  = None
            scaler_loaded = None

    features = [c for c in df.columns if c not in {"Date", "Quantity"}]

    if scaler_loaded is None:
        scaler = StandardScaler()
        df_treino[features] = scaler.fit_transform(df_treino[features]).astype(np.float32)
    else:
        scaler = scaler_loaded
        df_treino[features] = scaler.transform(df_treino[features]).astype(np.float32)

    df_2024[features] = scaler.transform(df_2024[features])


    # ---------- 2) AVALIAÇÃO ROLLING 365×31 ----------------------
    windows = generate_rolling_windows(df_treino, train_days=365, test_days=TIME_STEPS + 1, step_days=7)

    lr          = 1e-4
    batch_size  = 128
    epochs      = 50
    patience    = 20
    fine_epochs = 5    # fine‑tune mensal
    fine_batch  = 64

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

        X_train, y_train = create_sequences(train[features].values, train["Quantity_tr"].values)
        X_test,  y_test  = create_sequences(test[features].values,  test["Quantity_tr"].values)

        if len(X_train) == 0 or len(X_test) == 0:
            logger.warning(f"{barcode} | NN – janela {idx+1} ignorada (seqs vazias)")
            continue

        model_tmp = build_lstm_model((X_train.shape[1], X_train.shape[2]), layers_cfg, lr)
        es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        lr_sched = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=30, verbose=0
        )

        model_tmp.fit(
            X_train, y_train,
            validation_split=0.2,
            shuffle=False,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es, lr_sched],
            verbose=0,
        )

        y_pred = model_tmp.predict(X_test, verbose=0).flatten()
        # use raw → sem inversão
        mets = calculate_metrics(y_test, y_pred)

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
        df_treino[features].values, df_treino["Quantity_tr"].values
    )

    if model_loaded is None:
        model = build_lstm_model((X_hist.shape[1], X_hist.shape[2]),
                                 layers_cfg, lr)
    else:
        model = model_loaded

    # early stopping com validação 20% e sem shuffle
    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    lr_sched = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=30, verbose=0
    )

    model.fit(
        X_hist, y_hist,
        validation_split=0.2,
        shuffle=False,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, lr_sched],
        verbose=0,
    )

    logger.info(f"{barcode} | NN – re-treino global concluído")


    # ---------- 4) WALK‑FORWARD 2024 + FINE‑TUNE MENSAL ----------
    df_full    = pd.concat([df_treino, df_2024]).reset_index(drop=True)
    daily_rows = []

    for month in range(1, 13):
        # índices desse mês já dentro de df_full (garante contexto anterior)
        idxs_month = df_full.index[
            (df_full["Date"].dt.year == 2024) & (df_full["Date"].dt.month == month)
        ]

        if len(idxs_month) == 0:
            logger.warning(f"{barcode} | NN – {month:02d}/2024 ignorado (sem dados)")
            continue

        # ------------- PREVISÃO DIÁRIA ----------------------------
        for idx in idxs_month:
            """seq_start = idx - TIME_STEPS
            if seq_start < 0:
                # histórico ainda insuficiente; pula o primeiro(s) dia(s) de janeiro
                continue"""
            if idx < TIME_STEPS:
                # histórico ainda insuficiente; pula o(s) dia(s) de janeiro
                continue
            seq_start = idx - TIME_STEPS

            X_seq = df_full.loc[seq_start:idx - 1, features] \
                            .values.reshape(1, TIME_STEPS, len(features))
            y_real = float(df_full.loc[idx, "Quantity"])
            y_pred = float(model.predict(X_seq, verbose=0).flatten()[0])

            met = calculate_metrics(np.array([y_real]), np.array([y_pred]))

            daily_rows.append({
                "date":     df_full.loc[idx, "Date"],
                "barcode":  barcode,
                "forecast": y_pred,
                "real":     y_real,
                "mae":      met["mae"],
                "rmse":     met["rmse"],
                "mape":     met["mape"],
                "smape":    met["smape"],
            })

        # ------------- FINE-TUNE DO MÊS ---------------------------
        # cria as sequências usando o CONTEXTO COMPLETO (TIME_STEPS anteriores
        # + todo o mês), evitando descartar meses curtos (fev, abr, jun, set, nov)
        start_ft = max(int(idxs_month[0]) - TIME_STEPS, 0)
        end_ft   = int(idxs_month[-1])

        X_ft, y_ft = create_sequences(
            df_full.loc[start_ft:end_ft, features].values,
            df_full.loc[start_ft:end_ft, "Quantity"].values,
            time_steps=TIME_STEPS,
        )

        if len(X_ft):
            model.fit(X_ft, y_ft,
                    epochs=fine_epochs,
                    batch_size=fine_batch,
                    verbose=0)
            logger.info(f"{barcode} | NN – fine-tune aplicado em {month:02d}/2024")
        else:
            logger.warning(f"{barcode} | NN – fine-tune {month:02d}/2024 ignorado (seqs insuficientes)")

    if not daily_rows:
        logger.warning(f"{barcode} | NN – nenhuma predição gerada para 2024")
        return pd.DataFrame(), {}, pd.DataFrame()

    #df_daily = pd.DataFrame(daily_rows)
    df_daily = pd.DataFrame(daily_rows).dropna(subset=["forecast"])

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
    
    # ---------------- SALVAMENTO -------------------------
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_path, include_optimizer=False)
    joblib.dump(scaler, scaler_path)
    logger.info(f"{barcode} | Rede NN + scaler salvos em {model_dir}")

    del model
    free_gpu_memory()
    tf.keras.backend.clear_session()
    gc.collect()

    return df_roll, nn_metrics_2024, (
        df_daily.rename(columns={"date": "Date"})[["Date", "real", "forecast"]]
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
                fontsize=8, color="orange")


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