import numpy as np
import pandas as pd
import gc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def train_neural_network(df, barcode, epochs=50, batch_size=32):
    # Separação de treino/teste
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    # Definir features e target
    feature_cols = [col for col in df.columns if col not in ["Date", "Quantity"]]
    X_train = train_df[feature_cols].values
    y_train = train_df["Quantity"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["Quantity"].values

    # Monta modelo MLP simples
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse')

    # EarlyStopping para evitar overfitting
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0  # ou 1 para ver logs
    )

    preds = model.predict(X_test).flatten()
    test_df["prediction_nn"] = preds

    mae = np.mean(np.abs(y_test - preds))
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    logger.info(f"{barcode} - NN: MAE={mae:.2f}, MAPE={mape:.2f}%")

    # Opcional: model.save(f"models/nn/{barcode}_nn.h5")

    del model
    gc.collect()

    return test_df[["Date", "Quantity", "prediction_nn"]], (mae, mape)
