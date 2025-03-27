import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Set page config
st.set_page_config(page_title="BOD Predictor and CCME WQI Score Calculator", layout="wide", page_icon="💧")

# Load dataset
df = pd.read_csv("Lakes dataset.csv")

# Data Cleaning
df.replace("NIL", np.nan, inplace=True)
df.dropna(subset=["BOD (mg/L)"], inplace=True)

# Convert columns to numeric
df[["DO (mg/L)", "pH", "BOD (mg/L)", "Nitrate (mg/L)", "Turbidity (NTU)", "Chloride (mg/L)", "Ammonia  (mg/L)", "TDS (mg/L)"]] = \
    df[["DO (mg/L)", "pH", "BOD (mg/L)", "Nitrate (mg/L)", "Turbidity (NTU)", "Chloride (mg/L)", "Ammonia  (mg/L)", "TDS (mg/L)"]].apply(pd.to_numeric, errors='coerce')

df.dropna(inplace=True)

# Encode Month column
le = LabelEncoder()
df["Month"] = le.fit_transform(df["Month"])
month_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

# Features and target
features = ["Month", "DO (mg/L)", "pH", "Conductivity (µS/cm)", "Nitrate (mg/L)", "Turbidity (NTU)", "Chloride (mg/L)", "COD (mg/L)", "Ammonia  (mg/L)", "TDS (mg/L)"]
X = df[features]
y = df["BOD (mg/L)"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train BOD model
bod_model = RandomForestRegressor(n_estimators=100, random_state=42)
bod_model.fit(X_scaled, y)

# Save models
with open("bod_model.pkl", "wb") as f:
    pickle.dump(bod_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Water Quality Limits
limits_drinking = {
    "DO (mg/L)": (6.5, 8), "pH": (6.5, 8.5), "Conductivity (µS/cm)": 500,
    "BOD (mg/L)": 1, "Nitrate (mg/L)": 10, "Turbidity (NTU)": 1,
    "Chloride (mg/L)": 250, "COD (mg/L)": 3, "Ammonia  (mg/L)": 0.5, "TDS (mg/L)": 500
}

limits_domestic = {
    "DO (mg/L)": (5, 8), "pH": (6.0, 9.0), "Conductivity (µS/cm)": 1500,
    "BOD (mg/L)": 5, "Nitrate (mg/L)": 50, "Turbidity (NTU)": 5,
    "Chloride (mg/L)": 600, "COD (mg/L)": 10, "Ammonia  (mg/L)": 1, "TDS (mg/L)": 1000
}

limits_agriculture = {
    "DO (mg/L)": (4, 6), "pH": (6.0, 8.5), "Conductivity (µS/cm)": 3000,
    "BOD (mg/L)": 10, "Nitrate (mg/L)": 50, "Turbidity (NTU)": 10,
    "Chloride (mg/L)": 700, "COD (mg/L)": 20, "Ammonia  (mg/L)": 5, "TDS (mg/L)": 2000
}

# CCME WQI Calculation
def get_wqi_category(wqi):
    if wqi >= 95:
        return "Excellent"
    elif wqi >= 80:
        return "Good"
    elif wqi >= 65:
        return "Fair"
    elif wqi >= 45:
        return "Marginal"
    else:
        return "Poor"

def calculate_ccme_wqi(params, limits):
    failed_params, failed_tests, deviations = 0, 0, []
    for param, value in params.items():
        limit = limits.get(param)
        if limit:
            if isinstance(limit, tuple):
                if not (limit[0] <= value <= limit[1]):
                    failed_params += 1
                    failed_tests += 1
                    deviation = min(abs(value - limit[0]), abs(value - limit[1])) / (limit[1] - limit[0]) * 100
                    deviations.append(deviation)
            else:
                if value > limit:
                    failed_params += 1
                    failed_tests += 1
                    deviation = ((value / limit) - 1) * 100
                    deviations.append(deviation)
    F1, F2 = (failed_params / len(limits)) * 100, (failed_tests / len(limits)) * 100
    NSE = np.sum(deviations) / len(limits) if deviations else 0
    F3 = NSE / (0.01 * NSE + 0.01)
    WQI = 100 - (np.sqrt(F1**2 + F2**2 + F3**2) / 1.732)
    return WQI, get_wqi_category(WQI)

# Streamlit UI
st.title("💧 BOD Predictor and CCME WQI Score Calculator")

col1, col2 = st.columns([2, 1])
with col1:
    st.header("Input Parameters")
    input_data = {}
    input_data["Month"] = st.selectbox("Month", options=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
    for feature in features[1:]:
        input_data[feature] = st.number_input(feature, value=0.0)

if st.button("Predict"):
    input_data["Month"] = month_mapping[input_data["Month"]]
    input_scaled = scaler.transform([list(input_data.values())])
    bod_prediction = bod_model.predict(input_scaled)[0]
    input_data["BOD (mg/L)"] = bod_prediction
    
    wqi_drinking, category_drinking = calculate_ccme_wqi(input_data, limits_drinking)
    wqi_domestic, category_domestic = calculate_ccme_wqi(input_data, limits_domestic)
    wqi_agriculture, category_agriculture = calculate_ccme_wqi(input_data, limits_agriculture)
    
    with col2:
        st.header("Predictions")
        st.success(f"Predicted BOD: {bod_prediction:.2f} mg/L")
        st.info(f"WQI Drinking: {wqi_drinking:.2f}", icon="💙")
        st.info(f"Category: {category_drinking}", icon="🟢")
        st.info(f"WQI Domestic: {wqi_domestic:.2f}", icon="🟡")
        st.info(f"Category: {category_domestic}", icon="🔵")
        st.info(f"WQI Agriculture: {wqi_agriculture:.2f}", icon="🟠")
        st.info(f"Category: {category_agriculture}", icon="🔴")