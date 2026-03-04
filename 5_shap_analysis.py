import shap
import xgboost
import pandas as pd
import matplotlib.pyplot as plt
import os
import lightgbm as lgb
from catboost import CatBoostRegressor
from pathlib import Path

# Get the directory where the current script lives
script_dir = Path.cwd()

# Set the directory used in R
site = "feeagh" #feeagh or sau
dir_path = script_dir / ".." / site / "output"
#dir_path = "~/Documents/intoDBP/driver_attribution_fdom/"+site+"/output/"
dir_path = os.path.expanduser(dir_path)

# Load data
X_train = pd.read_csv(os.path.join(dir_path, "shap/X_train.csv"))
X_test = pd.read_csv(os.path.join(dir_path, "shap/X_test.csv"))

# Load model
model = xgboost.Booster()
model.load_model(os.path.join(dir_path, "models/xgb_model.model"))

# SHAP Explainer for tree-based model
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(os.path.join(dir_path, "shap/shap_summary_xgb.pdf"))
plt.close()

# Optional: save SHAP importance table
shap_importance = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": abs(shap_values).mean(axis=0)
}).sort_values(by="mean_abs_shap", ascending=False)

shap_importance.to_csv(os.path.join(dir_path, "shap/shap_importance_xgb.csv"), index=False)

### 1. SHAP for LightGBM
### ============================

print("Loading LightGBM model...")
lgb_model = lgb.Booster(model_file=os.path.join(dir_path, "models/lgb_model.txt"))

print("Explaining LightGBM with SHAP...")
explainer_lgb = shap.TreeExplainer(lgb_model)
shap_values_lgb = explainer_lgb.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values_lgb, X_test, show=False)
plt.tight_layout()
plt.savefig(os.path.join(dir_path, "shap/shap_summary_lgb.pdf"))
plt.close()

# Save SHAP importance
shap_lgb_df = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": abs(shap_values_lgb).mean(axis=0)
}).sort_values(by="mean_abs_shap", ascending=False)

shap_lgb_df.to_csv(os.path.join(dir_path, "shap/shap_importance_lgb.csv"), index=False)

### 2. SHAP for CatBoost
### ============================

print("Loading CatBoost model...")
cat_model = CatBoostRegressor()
cat_model.load_model(os.path.join(dir_path, "models/cat_model.cbm"))

print("Explaining CatBoost with SHAP...")
explainer_cat = shap.TreeExplainer(cat_model)
shap_values_cat = explainer_cat.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values_cat, X_test, show=False)
plt.tight_layout()
plt.savefig(os.path.join(dir_path, "shap/shap_summary_cat.pdf"))
plt.close()

# Save SHAP importance
shap_cat_df = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": abs(shap_values_cat).mean(axis=0)
}).sort_values(by="mean_abs_shap", ascending=False)

shap_cat_df.to_csv(os.path.join(dir_path, "shap/shap_importance_cat.csv"), index=False)

