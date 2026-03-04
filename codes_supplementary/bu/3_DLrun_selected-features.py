import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.callbacks import EarlyStopping

# Stop training if val_loss doesn't improve for 10 epochs
early_stop = EarlyStopping(
    monitor='val_loss',      # monitor validation loss
    patience=20,             # wait 10 epochs before stopping
    restore_best_weights=True
)


# -----------------------------
# Metrics
# -----------------------------
def rmse(obs, pred):
    return np.sqrt(mean_squared_error(obs, pred))

def nse(obs, pred):
    return 1 - np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2)

def kge(obs, pred):
    r = np.corrcoef(obs, pred)[0, 1]
    alpha = np.std(pred)/np.std(obs)
    beta = np.mean(pred)/np.mean(obs)
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

# -----------------------------
# Load data
# -----------------------------
case_study = "feeagh"
base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "../", case_study)

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
n_holdout = int(round(n*0.15))
train = df.iloc[:n - n_holdout]
test = df.iloc[n - n_holdout:]

X_train = train.drop(columns=["fdom","date"]).values
y_train = train["fdom"].values
X_test = test.drop(columns=["fdom","date"]).values
y_test = test["fdom"].values
dates_train = train["date"].values
dates_test = test["date"].values

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
# Load best hyperparameters from Optuna
# -----------------------------
mlp_params = pd.read_csv(os.path.join(data_dir,"output","hyperparameters","mlp_hyperparameters.csv")).iloc[0].to_dict()
lstm_params = pd.read_csv(os.path.join(data_dir,"output","hyperparameters","lstm_hyperparameters.csv")).iloc[0].to_dict()

# Ensure correct types
# Keep only hyperparameters (keys starting with 'params_')
mlp_params = {k.replace("params_",""): v for k,v in mlp_params.items() if k.startswith("params_")}
lstm_params = {k.replace("params_",""): v for k,v in lstm_params.items() if k.startswith("params_")}

# Ensure correct types (int or float)
for k in mlp_params:
    if isinstance(mlp_params[k], (int, float)):
        continue
    mlp_params[k] = float(mlp_params[k])
    
for k in lstm_params:
    if isinstance(lstm_params[k], (int, float)):
        continue
    lstm_params[k] = float(lstm_params[k])
# -----------------------------
# 1) MLP
# -----------------------------
mlp = Sequential()
mlp.add(Dense(mlp_params.get("units"), activation="relu", input_shape=(n_features,)))
mlp.add(Dropout(mlp_params.get("dropout_rate")))
for _ in range(mlp_params.get("n_layers")-1):
    mlp.add(Dense(mlp_params.get("units"), activation="relu"))
    mlp.add(Dropout(mlp_params.get("dropout_rate")))
mlp.add(Dense(1))
mlp.compile(optimizer=Adam(learning_rate=mlp_params.get("lr")), loss="mse")

mlp.fit(X_train_scaled, y_train_scaled, 
        epochs=120, 
        batch_size=int(mlp_params.get("batch_size",32)), 
        validation_split=0.2, 
        callbacks=[early_stop],
        verbose=1)

pred_mlp = mlp.predict(X_test_scaled).flatten()*y_sd + y_mean
pred_mlp_train = mlp.predict(X_train_scaled).flatten()*y_sd + y_mean

# -----------------------------
# 2) LSTM
# -----------------------------
lag = lstm_params.get("lag",7)

def create_sequences(X,y,lag):
    Xs, ys = [], []
    for i in range(lag, len(X)):
        Xs.append(X[i-lag:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, lag)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, lag)

lstm = Sequential()
lstm.add(LSTM(lstm_params.get("units1"), return_sequences=True, input_shape=(lag,n_features)))
lstm.add(Dropout(lstm_params.get("dropout_rate")))
lstm.add(LSTM(lstm_params.get("units2")))
lstm.add(Dropout(lstm_params.get("dropout_rate")))
lstm.add(Dense(1))
lstm.compile(optimizer=Adam(learning_rate=lstm_params.get("lr")), loss="mse")

lstm.fit(X_train_seq, y_train_seq, 
         epochs=100, 
         batch_size=int(lstm_params.get("batch_size")), 
         validation_split=0.2, 
         callbacks=[early_stop],
         verbose=1)


pred_lstm = lstm.predict(X_test_seq).flatten()*y_sd + y_mean
pred_lstm_train = lstm.predict(X_train_seq).flatten()*y_sd + y_mean

# -----------------------------
# Metrics
# -----------------------------
results = pd.DataFrame({
    "Model":["MLP","LSTM"],
    "R2":[r2_score(y_test,pred_mlp), r2_score(y_test[lag:],pred_lstm)],
    "RMSE":[rmse(y_test,pred_mlp), rmse(y_test[lag:],pred_lstm)],
    "NSE":[nse(y_test,pred_mlp), nse(y_test[lag:],pred_lstm)],
    "KGE":[kge(y_test,pred_mlp), kge(y_test[lag:],pred_lstm)]
})

print("TEST RESULTS")
print(results.round(3))

# -----------------------------
# Save models & predictions
# -----------------------------
out_dir = os.path.join(data_dir,"output")
os.makedirs(os.path.join(out_dir,"models"), exist_ok=True)

mlp.save(os.path.join(out_dir,"models","mlp_model_best.h5"))
lstm.save(os.path.join(out_dir,"models","lstm_model_best.h5"))

results.to_csv(os.path.join(out_dir,"metrics_nn_best.csv"),index=False)
pd.DataFrame({"date":dates_test,"pred_mlp":pred_mlp}).to_csv(os.path.join(out_dir,"pred_mlp_best.csv"),index=False)
pd.DataFrame({"date":dates_test[lag:],"pred_lstm":pred_lstm}).to_csv(os.path.join(out_dir,"pred_lstm_best.csv"),index=False)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(dates_test, y_test, label="Actual", color="black")
plt.plot(dates_test, pred_mlp, label="MLP", color="steelblue")
plt.plot(dates_test[lag:], pred_lstm, label="LSTM", color="magenta")
plt.xlabel("Date")
plt.ylabel("fDOM (QSU)")
plt.title("Neural Network Predictions vs Actual")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir,"nn_predictions.pdf"))
plt.close()

print("Best NN models trained, predictions saved, plot generated successfully!")
