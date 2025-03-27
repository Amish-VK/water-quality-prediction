import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load dataset
dataset = pd.read_csv("C:\\Users\\subha\\Downloads\\Lakes dataset.csv")

# Define input features and target variables
features = ["Month", "DO (mg/L)", "pH", "Conductivity (µS/cm)", "Nitrate (mg/L)",
            "Turbidity (NTU)", "Chloride (mg/L)", "COD (mg/L)", "Ammonia (mg/L)", "TDS (mg/L)"]

target_bod = "BOD (mg/L)"
target_wqi = "CCME WQI"

# Drop rows with missing target values
dataset = dataset.dropna(subset=[target_bod, target_wqi])

# Encode Month (Categorical)
le = LabelEncoder()
dataset["Month"] = le.fit_transform(dataset["Month"])

# Split data into features (X) and targets (y)
X = dataset[features]
y_bod = dataset[target_bod]
y_wqi = dataset[target_wqi]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train_bod, y_test_bod = train_test_split(X_scaled, y_bod, test_size=0.2, random_state=42)
X_train, X_test, y_train_wqi, y_test_wqi = train_test_split(X_scaled, y_wqi, test_size=0.2, random_state=42)

# Train models
bod_model = RandomForestRegressor(n_estimators=100, random_state=42)
wqi_model = RandomForestRegressor(n_estimators=100, random_state=42)

bod_model.fit(X_train, y_train_bod)
wqi_model.fit(X_train, y_train_wqi)

# Save models & preprocessing tools
with open("bod_model.pkl", "wb") as f:
    pickle.dump(bod_model, f)

with open("wqi_model.pkl", "wb") as f:
    pickle.dump(wqi_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Models trained and saved successfully!")
