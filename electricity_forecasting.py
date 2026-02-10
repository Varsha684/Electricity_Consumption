# =========================================================
# ⚡ Electricity Consumption Forecasting using Weather Data
#     + Lag Features (Improved R²)
# =========================================================

# ======================
# 📦 Import Libraries
# ======================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ======================
# 📂 Load Dataset
# ======================
df = pd.read_csv(
    r"C:\Users\ankur\Downloads\archive (9)\electricity_consumption_based_weather_dataset.csv"
)

# ======================
# 🕒 Date Handling
# ======================
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# ======================
# 🧹 Missing Values
# ======================
df = df.ffill()

# ======================
# 🧾 Clean Column Names
# ======================
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace('(', '', regex=False)
    .str.replace(')', '', regex=False)
)

# ======================
# 📆 Date Features
# ======================
df['month'] = df.index.month

# ======================
# 🎯 Target Column
# ======================
target_col = [c for c in df.columns if 'consumption' in c or 'electric' in c]
elec_col = target_col[0]

# ======================
# 🌦 Weather Features (SAFE)
# ======================
possible_weather = ['prcp', 'tmax', 'tmin', 'awnd']
weather_cols = [c for c in possible_weather if c in df.columns]

# =========================================================
# 📊 Exploratory Data Analysis
# =========================================================

# 🔹 Monthly Average Consumption
monthly_avg = df.groupby('month')[elec_col].mean()

plt.figure(figsize=(10,5))
monthly_avg.plot(kind='bar')
plt.title("📊 Average Electricity Consumption by Month")
plt.xlabel("Month")
plt.ylabel("Average Consumption")
plt.grid(axis='y', alpha=0.3)
plt.show()

# 🔹 Correlation Heatmap
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12,8))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    linewidths=0.5
)
plt.title("🔥 Correlation Heatmap")
plt.show()

# =========================================================
# 🔁 Lag Features (MOST IMPORTANT)
# =========================================================
df['lag_1']  = df[elec_col].shift(1)
df['lag_7']  = df[elec_col].shift(7)
df['lag_14'] = df[elec_col].shift(14)

df.dropna(inplace=True)

# =========================================================
# 🧠 Model Data
# =========================================================
X = df[weather_cols + ['month', 'lag_1', 'lag_7', 'lag_14']]
y = df[elec_col]

# ======================
# ✂️ Train-Test Split (TIME SERIES SAFE)
# ======================
split_index = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# =========================================================
# 🌲 Random Forest Model
# =========================================================
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# =========================================================
# 📈 Evaluation Function
# =========================================================
def evaluate_model(model, name):
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n{name} Evaluation")
    print("-" * 35)
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R²  :", round(r2, 3))

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name}: Actual vs Predicted")
    plt.grid(alpha=0.3)
    plt.show()

# ======================
# ✅ Random Forest Result
# ======================
evaluate_model(rf, "Random Forest")

# =========================================================
# 🚀 XGBoost
# =========================================================
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

xgb.fit(X_train, y_train)
evaluate_model(xgb, "XGBoost Regressor")

# =========================================================
# 🌟 LightGBM
# =========================================================
from lightgbm import LGBMRegressor

lgbm = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42
)

lgbm.fit(X_train, y_train)
evaluate_model(lgbm, "LightGBM Regressor")

# =========================================================
# ⚡ Gradient Boosting
# =========================================================
gbr = GradientBoostingRegressor(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)

gbr.fit(X_train, y_train)
evaluate_model(gbr, "Gradient Boosting Regressor")

# =========================================================
# 📌 Feature Importance (RF)
# =========================================================
importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n🔥 Feature Importance\n")
print(importance)

importance.plot(kind='bar', figsize=(10,4), title="Feature Importance")
plt.grid(axis='y', alpha=0.3)
plt.show()
