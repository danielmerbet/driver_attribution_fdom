import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def kge(sim, obs):
    """
    Compute Kling-Gupta Efficiency (KGE)
    """
    # Remove NaNs or infs
    idx = np.isfinite(sim) & np.isfinite(obs)
    sim = sim[idx]
    obs = obs[idx]

    if len(sim) == 0:
        return np.nan

    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)

    kge_value = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge_value
  
def r2_score_custom(y_true, y_pred):
    # Convert to numpy arrays if inputs are pandas Series/DataFrame
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Remove NaNs or infs
    valid_idx = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_idx]
    y_pred = y_pred[valid_idx]

    if len(y_true) == 0:
        return np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return np.nan  # Avoid division by zero

    return 1 - (ss_res / ss_tot)

def nse(obs, sim):
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    return 1 - (numerator / denominator)
  
# Evaluate Random Forest
def evaluate(y_true, y_pred, name="Test"):
    y_true = y_true.values  # Convert pandas Series to numpy array if needed
    y_pred = y_pred.flatten()

    r2 = r2_score_custom(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nse_value = nse(y_true, y_pred)
    kge_value = kge(y_true, y_pred)

    print(f"{name} Results:")
    print(f"  R²        = {r2:.4f}")
    print(f"  RMSE      = {rmse:.4f}")
    print(f"  NSE       = {nse_value:.4f}")
    print(f"  KGE       = {kge_value:.4f}")
    print("-" * 40)
    
# Load data
case_study = "sau"  # or "feeagh"
dir_path = f"~/Documents/intoDBP/driver_attribution_fdom/{case_study}/"
data = pd.read_csv(f"{dir_path}data/data.csv")
data['date'] = pd.to_datetime(data['date'])

# Add cos of julian day and random feature
data['cyday'] = np.cos(data['date'].dt.dayofyear * np.pi / 180)
#data['random'] = np.random.rand(len(data))

# Select relevant columns
if case_study == "sau":
    cols = ["v", "st255", "sm100", "sm255", "doc_gwlf", "cyday", "fdom", "date"]
elif case_study == "feeagh":
    cols = ["swt", "sr", "st100", "st255", "sm100", "sm255", "doc_gwlf", "cyday", "fdom", "date"]

data = data[cols]

# Add lag features
#data['fdom_lag1'] = data['fdom'].shift(1)
#data['fdom_lag7'] = data['fdom'].shift(7)

# Add rolling mean
#data['fdom_rolling7'] = data['fdom'].rolling(window=7).mean()

# Fill NA
data = data.dropna()

tscv = TimeSeriesSplit(n_splits=5)

# Train-test split (last 15%)
n = len(data)
n_holdout = int(n * 0.15)
X_train = data.iloc[:-n_holdout].drop(['fdom', 'date'], axis=1)
y_train = data.iloc[:-n_holdout]['fdom']

X_test = data.iloc[-n_holdout:].drop(['fdom', 'date'], axis=1)
y_test = data.iloc[-n_holdout:]['fdom']

rf = RandomForestRegressor(n_estimators=1000,  # same as R's ntree
                           max_features='sqrt', # same as mtry
                           random_state=123,
                           n_jobs=-1)
rf.fit(X_train, y_train)

# Predict with RF
rf_train_pred = rf.predict(X_train)
rf_test_pred = rf.predict(X_test)

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
print("Feature Importance:\n", importances.sort_values(ascending=False))

evaluate(y_train, rf_train_pred, "Train OOB")
evaluate(y_test, rf_test_pred, "Test")

#XGBoost
from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=1000, learning_rate=0.2, max_depth=5, random_state=123)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
evaluate(y_test, y_pred_xgb, "Test")

#LightGBM
from lightgbm import LGBMRegressor

lgb = LGBMRegressor(n_estimators=500, learning_rate=0.1, max_depth=5, random_state=123)
lgb.fit(X_train, y_train)

y_pred_lgbm = lgb.predict(X_test)
evaluate(y_test, y_pred_lgbm, "Test")

#CatBoost
from catboost import CatBoostRegressor

cat = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=0, random_state=123)
cat.fit(X_train, y_train)

y_pred_catb = cat.predict(X_test)
evaluate(y_test, y_pred_catb, "Test")

#support vector regression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# SVR is sensitive to scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train)

y_pred_svr = svr.predict(X_test_scaled)

evaluate(y_test, y_pred_svr, "Test")

#k nearest neighbors
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
evaluate(y_test, y_pred_knn, "Test")

#linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
evaluate(y_test, y_pred_lr, "Test")

#stacjing multiple
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

stacked = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=200, random_state=123)),
        ('catb', CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=0, random_state=123)),
        ('lgb', LGBMRegressor(n_estimators=200, random_state=123))
    ],
    final_estimator=Ridge()
)

stacked.fit(X_train, y_train)
y_pred_stack = stacked.predict(X_test)

evaluate(y_test, y_pred_stack, "Test")

#compare all:

results = []

for name, model in [
    ('RF', rf),
    ('XGB', xgb),
    ('LGB', lgb),
    ('CatBoost', cat),
    ('SVR', svr),
    ('KNN', knn),
    ('Linear', lr)
]:
    y_pred = model.predict(X_test)
    metrics = {
        'Model': name,
        'R2': r2_score_custom(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'NSE': nse(y_test, y_pred),
        'KGE': kge(y_test, y_pred)
    }
    results.append(metrics)

results_df = pd.DataFrame(results)
print(results_df.sort_values(by='KGE', ascending=False))

#to tune parameter of randomf forest:
#get best parameters
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 6),
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=123, oob_score=True)
rs = RandomizedSearchCV(rf, param_dist, n_iter=1000, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
rs.fit(X_train, y_train)

best_rf = rs.best_estimator_
best_rf

def plot_training_testing_with_metrics(
    dates_train, y_train,
    dates_test, y_test,
    y_pred,
    metrics_dict,
    title="Model Performance: Training and Testing",
    filename="model_performance.pdf"
):

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot training data
    ax.plot(dates_train, y_train, label='Training (Actual)', color='blue', linewidth=1)

    # Plot testing actual and predicted
    ax.plot(dates_test, y_test, label='Testing (Actual)', color='black', linewidth=2)
    ax.plot(dates_test, y_pred, label='Testing (Predicted)', color='red', linestyle='--', linewidth=2)

    # Add metrics as text
    metric_text = "\n".join([f"{key}: {value:.2f}" for key, value in metrics_dict.items()])
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.95, metric_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    # Labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("fdom")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()

    # Save to PDF
    with PdfPages(filename) as pdf:
        pdf.savefig(fig, bbox_inches='tight')

    plt.close(fig)
    print(f"Plot saved to {filename}")
    
    
metrics_dict = {
    'R2': r2_score_custom(y_test, rf_test_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, rf_test_pred)),
    'NSE': nse(y_test, rf_test_pred),
    'KGE': kge(y_test, rf_test_pred)
}

dates = data['date'].values
dates_train = dates[:-n_holdout]
dates_test = dates[-n_holdout:]

plot_training_testing_with_metrics(
    dates_train,
    y_train,
    dates_test,
    y_test,
    rf_test_pred,
    metrics_dict,
    title="RF Model Performance",
    filename="rf_model_performance.pdf"
)

