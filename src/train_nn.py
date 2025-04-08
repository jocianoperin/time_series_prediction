# Novo pipeline de Rede Neural com LSTM
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from utils.logging_config import get_logger
from utils.metrics import calculate_metrics
from data_preparation import generate_rolling_windows

logger = get_logger(__name__)

TIME_STEPS = 30  # janela deslizante de 30 dias

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')
    return model

def train_neural_network(df, barcode):
    df = df.dropna().sort_values("Date")
    features = [col for col in df.columns if col not in ["Date", "Quantity"]]

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    results = []
    windows = generate_rolling_windows(df[df["Date"] < "2024-01-01"])

    for i, w in enumerate(windows):
        train, test = w["train"], w["test"]
        X_train, y_train = create_sequences(train[features].values, train["Quantity"].values)
        X_test, y_test = create_sequences(test[features].values, test["Quantity"].values)

        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[es], verbose=0)

        y_pred = model.predict(X_test).flatten()
        metrics = calculate_metrics(y_test, y_pred)

        logger.info(f"{barcode} | LSTM - Janela {i+1} | Treino: {train['Date'].min()} a {train['Date'].max()} | Teste: {test['Date'].min()} a {test['Date'].max()} | MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%")

        test_out = test.iloc[TIME_STEPS:].copy()
        test_out["prediction_nn"] = y_pred
        results.append(test_out)

    # Predição mês a mês de 2024
    full_train = df[df["Date"] < "2024-01-01"].copy()
    X_full, y_full = create_sequences(full_train[features].values, full_train["Quantity"].values)
    model = build_lstm_model((X_full.shape[1], X_full.shape[2]))
    model.fit(X_full, y_full, epochs=50, batch_size=32, verbose=0)

    forecast_2024 = []
    for month in range(1, 13):
        future_df = df[(df["Date"] >= f"2024-{month:02d}-01") & (df["Date"] <= f"2024-{month:02d}-28")].copy()
        if len(future_df) <= TIME_STEPS:
            continue

        X_future, y_real = create_sequences(future_df[features].values, future_df["Quantity"].values)
        y_pred = model.predict(X_future).flatten()
        m = calculate_metrics(y_real, y_pred)

        logger.info(f"{barcode} | LSTM - Predição 2024-{month:02d} | MAE={m['mae']:.2f}, MAPE={m['mape']:.2f}%")

        future_out = future_df.iloc[TIME_STEPS:].copy()
        future_out["prediction_nn"] = y_pred
        forecast_2024.append(future_out[["Date", "prediction_nn"]])

        # Fine-tune
        model.fit(X_future, y_real, epochs=10, batch_size=32, verbose=0)

    return pd.concat(results), m, pd.concat(forecast_2024)
