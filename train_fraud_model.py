# train_fraud_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import IsolationForest

# ---- Load your training data (same as premium model’s data) ----
# Expecting columns: age, sex (str), bmi, children, smoker (str), region (str), charges
df = pd.read_csv("insurance.csv")

# ---- Minimal encoding to numeric like your Flask app uses ----
sex_map = {"male": 1, "female": 0}
smoker_map = {"yes": 1, "no": 0}
region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}

X = pd.DataFrame({
    "age": df["age"].astype(float),
    "sex_code": df["sex"].map(sex_map).astype(int),
    "bmi": df["bmi"].astype(float),
    "children": df["children"].astype(float),
    "smoker_code": df["smoker"].map(smoker_map).astype(int),
    "region_code": df["region"].map(region_map).astype(int),
})

# ---- Train Isolation Forest ----
# contamination ~ fraction of expected anomalies; tune 0.03–0.07 if needed
model = IsolationForest(
    n_estimators=200,
    max_samples="auto",
    contamination=0.05,
    random_state=42,
)
model.fit(X)

# ---- Persist model ----
with open("fraud_iforest.pkl", "wb") as f:
    pickle.dump(model, f)

print("Saved fraud_iforest.pkl")
