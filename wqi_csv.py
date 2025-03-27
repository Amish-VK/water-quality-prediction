import pandas as pd
import numpy as np
import re

def clean_value(value):
    if pd.isna(value) or value in ["Nil", "NIL", "Not Analysed"]:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "BDL":
            return 0  # BDL values are considered safe
        if value.startswith("<"):
            return float(value[1:])  # Take upper limit as value
        if "(BDL)" in value:
            return float(value.replace("(BDL)", "").strip())  # Remove (BDL) and use numerical value
        value = re.sub(r'[^0-9.]', '', value)  # Remove any unexpected characters
        try:
            return float(value)
        except ValueError:
            return None  # Ignore invalid text entries
    try:
        return float(value)
    except ValueError:
        return None  # Ignore invalid text entries

def calculate_ccme_wqi(row, limits):
    valid_params = {param: clean_value(row[param]) for param in limits.keys() if param in row and clean_value(row[param]) is not None}
    
    if not valid_params:
        return None  # No valid parameters, return blank
    
    failed_params = 0
    failed_tests = 0
    deviations = []
    num_valid_params = len(valid_params)

    for param, value in valid_params.items():
        limit = limits[param]
        if isinstance(limit, tuple):  # Handle range-based parameters like pH and DO
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

    F1 = (failed_params / num_valid_params) * 100 if num_valid_params > 0 else 0  # Scope
    F2 = (failed_tests / num_valid_params) * 100 if num_valid_params > 0 else 0  # Frequency
    NSE = np.sum(deviations) / num_valid_params if deviations else 0  # Normalized Sum of Excursions
    F3 = NSE / (0.01 * NSE + 0.01)  # Amplitude

    CCME_WQI = 100 - (np.sqrt(F1**2 + F2**2 + F3**2) / 1.732)
    return round(CCME_WQI, 2)

# Define water quality limits for different uses
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

# Load dataset
dataset_path = "C:\\Users\\subha\\Downloads\\Lakes dataset.csv"  # Update path accordingly
df = pd.read_csv(dataset_path)

# Compute WQI for each category
df["WQI Drinking"] = df.apply(lambda row: calculate_ccme_wqi(row, limits_drinking), axis=1)
df["WQI Domestic"] = df.apply(lambda row: calculate_ccme_wqi(row, limits_domestic), axis=1)
df["WQI Agriculture"] = df.apply(lambda row: calculate_ccme_wqi(row, limits_agriculture), axis=1)

# Save the updated dataset
output_path = "Lakes_dataset_with_WQI_cleaned.csv"
df.to_csv(output_path, index=False)
print(f"Updated dataset saved to {output_path}")
