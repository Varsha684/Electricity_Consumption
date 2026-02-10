# ⚡ Electricity Consumption Forecasting using Weather Data

A complete **End-to-End Machine Learning Time Series Project** that forecasts electricity consumption using **weather data + lag features**, achieving improved **R² performance** 🚀  

---

## 📌 Project Overview

Electricity demand is influenced by:
- 🌦 Weather conditions  
- 📅 Seasonal patterns  
- 🔁 Past consumption behavior  

This project uses **historical electricity consumption (lag features)** along with **weather data** to build accurate forecasting models using ensemble machine learning algorithms.

---

## 🧠 Concepts Covered

- 📈 Time Series Forecasting  
- 🔁 Lag Feature Engineering  
- 🌦 Weather-based Regression  
- 🌲 Ensemble Learning  
- 📊 Model Evaluation & Comparison  

---

## 📂 Dataset Details

**Dataset:** Electricity Consumption based on Weather Data  

**Features Included:**
- 🌧 `prcp` – Precipitation  
- 🌡 `tmax` – Maximum Temperature  
- ❄️ `tmin` – Minimum Temperature  
- 💨 `awnd` – Average Wind Speed  
- ⚡ Electricity Consumption (Target Variable)

---

## 🛠 Tech Stack Used

- 🐍 Python  
- 📦 NumPy, Pandas  
- 📊 Matplotlib, Seaborn  
- 🤖 Scikit-learn  
- 🚀 XGBoost  
- 🌟 LightGBM  

---

## 🔁 Feature Engineering

### ⚡ Lag Features (Key Improvement)

Lag features help the model learn **temporal dependency** in electricity usage.

- lag_1 → Previous day consumption
- lag_7 → Previous week consumption
- lag_14 → Two weeks ago consumption


These features significantly boost model accuracy 📈

---

## 📊 Exploratory Data Analysis (EDA)

- 📅 Monthly average electricity consumption analysis  
- 🔥 Correlation heatmap of numerical features  
- 📈 Trend understanding before modeling  

---

## 🧪 Models Implemented

| Model | Description |
|-----|-------------|
| 🌲 Random Forest | Strong baseline + feature importance |
| 🚀 XGBoost | High-performance gradient boosting |
| 🌟 LightGBM | Fast & efficient boosting |
| ⚡ Gradient Boosting | Stable ensemble regressor |

---

## 📈 Evaluation Metrics

Models are evaluated using:

- 📉 MAE – Mean Absolute Error  
- 📐 RMSE – Root Mean Squared Error  
- 🧮 R² Score – Model goodness of fit  

📊 Actual vs Predicted plots are used for visual validation.

---

## 🏆 Key Results & Insights

- 🔥 Lag features dominate feature importance  
- 🚀 Boosting models outperform traditional regressors  
- 📈 High R² achieved due to temporal learning  
- 🌦 Weather features alone are not sufficient — history matters  

---

## 📌 Feature Importance Highlights

Most influential features:
- 🔁 `lag_1`  
- 🔁 `lag_7`  
- 🔁 `lag_14`  
- 🌡 Temperature-related features  

---

## 📁 Project Structure

Electricity-Consumption-Forecasting/
│
├── electricity_forecasting.py
├── README.md
└── dataset/
└──electricity_consumption_based_weather_dataset.csv


---

## 🚀 How to Run the Project

1️⃣ Clone the repository  

- git clone <your-repository-url>


2️⃣ Install dependencies  

- pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm


3️⃣ Run the script  

- python electricity_forecasting.py


---

## 🔮 Future Enhancements

- 📦 ML Pipelines  
- 📊 SHAP Explainability  
- ⏳ Walk-Forward Validation  
- 🌐 FastAPI Deployment  
- 🐳 Docker & Cloud Hosting  

---

## 👨‍💻 Author

**Varsha Goswami**  
📌 Aspiring Data Scientist | Machine Learning Engineer  
🚀 Focused on End-to-End ML & Time Series Projects  

---

## ⭐ Support

If you found this project useful, don’t forget to **⭐ star the repository** on GitHub 😄  
Happy Forecasting ⚡📈
"# Electricity_Consumption" 

