import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import optuna
import os

# -----------------------------
# Utility Metrics
# -----------------------------
def rmse(obs, pred):
    return np.sqrt(mean_squared_error(obs, pred))

def nse(obs, pred):
    return 1 - np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2)

# -----------------------------
# Load Data
# -----------------------------
case_study = "feeagh" #sau
base_dir = os.getcwd()
data_dir = os.path.join(base_dir,"../",  case_study)

df = pd.read_csv(os.path.join(data_dir, "data", "data.csv"))
df["date"] = pd.to_datetime(df["date"])
df["cyday"] = np.cos(df["date"].dt.dayofyear * np.pi / 180)

if case_study == "feeagh":
    df = df[["swt", "sr", "st100", "st255", "sm100", "sm255",
             "doc_gwlf", "cyday", "fdom", "date"]]

if case_study == "sau":
    df = df[["v", "st255", "sm100", "sm255",
              "cyday", "fdom", "date"]]
              
# Train/test split (last 15%)
n = len(df)
n_holdout = int(round(n * 0.15))
train = df.iloc[:n - n_holdout]
test = df.iloc[n - n_holdout:]

X_train = train.drop(columns=["fdom", "date"]).values
y_train = train["fdom"].values
X_test = test.drop(columns=["fdom", "date"]).values
y_test = test["fdom"].values

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_mean = y_train.mean()
y_sd = y_train.std()
y_train_scaled = (y_train - y_mean)/y_sd
y_test_scaled = (y_test - y_mean)/y_sd

n_features = X_train_scaled.shape[1]

# -----------------------------
# LSTM helper
# -----------------------------
def create_sequences(X, y, lag):
    Xs, ys = [], []
    for i in range(lag, len(X)):
        Xs.append(X[i-lag:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# ===================================================
# Optuna Objective for MLP
# ===================================================
def objective_mlp(trial):
    # Hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 3)
    units = trial.suggest_int("units", 16, 128, step=16)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Build model
    model = Sequential()
    model.add(Dense(units, activation="relu", input_shape=(n_features,)))
    model.add(Dropout(dropout_rate))
    for _ in range(n_layers - 1):
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")

    # Train
    model.fit(X_train_scaled, y_train_scaled, 
              epochs=50, batch_size=batch_size, verbose=0,
              validation_split=0.2)

    # Predict
    pred = model.predict(X_test_scaled).flatten() * y_sd + y_mean

    # Use RMSE as objective
    score = rmse(y_test, pred)
    tf.keras.backend.clear_session()
    return score

# ===================================================
# Optuna Objective for LSTM
# ===================================================
def objective_lstm(trial):
    # Hyperparameters
    lag = trial.suggest_int("lag", 3, 14)
    units1 = trial.suggest_int("units1", 16, 128, step=16)
    units2 = trial.suggest_int("units2", 8, 64, step=8)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Prepare sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, lag)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, lag)

    # Build model
    model = Sequential()
    model.add(LSTM(units1, return_sequences=True, input_shape=(lag, n_features)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr), loss="mse")

    # Train
    model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=batch_size,
              verbose=0, validation_split=0.2)

    # Predict
    pred = model.predict(X_test_seq).flatten() * y_sd + y_mean
    score = rmse(y_test[lag:], pred)
    tf.keras.backend.clear_session()
    return score

# ===================================================
# Run Optuna
# ===================================================
os.makedirs(os.path.join(data_dir, "output"), exist_ok=True)

# MLP Tuning
study_mlp = optuna.create_study(direction="minimize")
study_mlp.optimize(objective_mlp, n_trials=30)  # adjust n_trials for more thorough search
print("Best MLP params:", study_mlp.best_params)
study_mlp.trials_dataframe().to_csv(os.path.join(data_dir, "output", "hyperparameters","mlp_hyperparameters.csv"), index=False)

# LSTM Tuning
study_lstm = optuna.create_study(direction="minimize")
study_lstm.optimize(objective_lstm, n_trials=30)
print("Best LSTM params:", study_lstm.best_params)
study_lstm.trials_dataframe().to_csv(os.path.join(data_dir, "output", "hyperparameters", "lstm_hyperparameters.csv"), index=False)

print("Hyperparameter tuning completed successfully!")
