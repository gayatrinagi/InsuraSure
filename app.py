import os
import pickle
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, render_template

# Ensure sklearn classes are importable for unpickling
from sklearn.ensemble import IsolationForest  # for fraud_iforest.pkl

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# ----------------------------
# Load models
# ----------------------------
PKL_FILENAME = "rf_tuned.pkl"
with open(PKL_FILENAME, 'rb') as file:
    model = pickle.load(file)

FRAUD_MODEL_PATH = "fraud_iforest.pkl"
fraud_model = None
if os.path.exists(FRAUD_MODEL_PATH):
    with open(FRAUD_MODEL_PATH, "rb") as f:
        fraud_model = pickle.load(f)

# ----------------------------
# SHAP explainer (built once)
# ----------------------------
explainer = shap.TreeExplainer(model)

def _compute_top_factors_with_shap(X_df, top_k=3):
    """
    X_df: 1xN pandas DataFrame that goes into model.predict(...)
    Returns: (top_factors: list[dict], base_value: float)
    """
    shap_values = explainer.shap_values(X_df)
    if isinstance(shap_values, list):  # some shap versions return a list
        shap_values = shap_values[0]
    row_shap = np.array(shap_values)[0]  # (n_features,)

    feature_names = list(X_df.columns)
    idx = np.argsort(np.abs(row_shap))[::-1][:top_k]

    def _fmt_currency(x):
        return f"₹{abs(x):,.0f}"

    DISPLAY_MAP = {
        "age": "Age",
        "sex_code": "Sex",
        "bmi": "BMI",
        "children": "No. of children",
        "smoker_code": "Smoking status",
        "region_code": "Region",
    }

    top_factors = []
    for i in idx:
        name = feature_names[i]
        nice = DISPLAY_MAP.get(name, name)
        val = float(row_shap[i])
        sign = "+" if val >= 0 else "−"
        top_factors.append({
            "name": nice,
            "signed_amount": f"{sign}{_fmt_currency(val)}",
            "raw": val,
        })

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.asarray(base_value).ravel()[0])

    return top_factors, float(base_value)

# ----------------------------
# Helpers for safe parsing
# ----------------------------
def _required(name):
    v = (request.form.get(name) or "").strip()
    return v

def _error(msg, pred_text=None):
    # Render the same result page with an error banner
    return render_template(
        "op.html",
        pred=pred_text or "—",
        error_msg=msg,
        top_factors=[],
        base_value=None,
        fraud_flag=False,
        fraud_score=None,
        fraud_msg=None
    )

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1) Read raw values (as strings)
    age_raw      = _required('age')
    sex_raw      = _required('sex')       # 'male'/'female' or 1/0
    bmi_raw      = _required('bmi')
    children_raw = _required('children')
    smoker_raw   = _required('smoker')    # 'yes'/'no' or 1/0
    region_raw   = _required('region')    # 'northeast'/'northwest'/'southeast'/'southwest' or 0..3

    # 2) Validate empties
    missing = [n for n, v in [
        ('age', age_raw),
        ('sex', sex_raw),
        ('bmi', bmi_raw),
        ('children', children_raw),
        ('smoker', smoker_raw),
        ('region', region_raw),
    ] if v == ""]
    if missing:
        return _error(f"Please fill all required fields: {', '.join(missing)}.")

    # 3) Coerce numerics
    try:
        age = float(age_raw)
        bmi = float(bmi_raw)
        children = float(children_raw)
    except ValueError:
        return _error("Age, BMI, and Children must be numeric values (e.g., 30, 24.5, 2).")

    # 4) Encode categoricals -> numeric codes
    def to_lower_str(v): return str(v).strip().lower()

    # sex: male=1, female=0 (fallback to int if already code)
    sex_map = {"male": 1, "m": 1, "female": 0, "f": 0}
    try:
        sex_code = int(sex_raw) if str(sex_raw).isdigit() else sex_map[to_lower_str(sex_raw)]
    except Exception:
        return _error("Invalid value for Sex. Use 'male' or 'female'.")

    # smoker: yes=1, no=0
    smoker_map = {"yes": 1, "y": 1, "true": 1, "no": 0, "n": 0, "false": 0}
    try:
        smoker_code = int(smoker_raw) if str(smoker_raw).isdigit() else smoker_map[to_lower_str(smoker_raw)]
    except Exception:
        return _error("Invalid value for Smoking status. Use 'yes' or 'no'.")

    # region: map to 0..3 (or accept code)
    region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
    try:
        region_code = int(region_raw) if str(region_raw).isdigit() else region_map[to_lower_str(region_raw)]
    except Exception:
        return _error("Invalid Region. Use northeast / northwest / southeast / southwest.")

    # 5) Build model input in the EXACT order the model expects
    cols = ["age", "sex_code", "bmi", "children", "smoker_code", "region_code"]
    row = [[age, sex_code, bmi, children, smoker_code, region_code]]
    X_df = pd.DataFrame(row, columns=cols)

    # 6) Fraud / anomaly score (Isolation Forest)
    fraud_flag = False
    fraud_score = None
    fraud_msg = None
    try:
        if fraud_model is not None:
            # decision_function: higher is more normal; lower is more anomalous
            # predict: 1 = normal, -1 = anomaly
            decision = float(fraud_model.decision_function(X_df.values)[0])
            pred_label = int(fraud_model.predict(X_df.values)[0])  # 1 or -1
            fraud_score = round(decision, 3)
            fraud_flag = (pred_label == -1)
            if fraud_flag:
                fraud_msg = (
                    "Unusual input pattern detected. Please double-check your details "
                    "(e.g., age, BMI, smoker/region codes)."
                )
    except Exception:
        # don’t break premium flow if anomaly scoring fails
        fraud_flag, fraud_score, fraud_msg = False, None, None

    # 7) Predict premium
    try:
        pred_val = float(model.predict(X_df.values)[0])
    except Exception as e:
        return _error(f"Model error: {e}")

    # 8) SHAP explanations (top 3)
    try:
        top_factors, base_value = _compute_top_factors_with_shap(X_df)
    except Exception:
        top_factors, base_value = [], None

    # 9) Render
    if pred_val < 0:
        return render_template(
            'op.html',
            pred='Error calculating Amount!',
            top_factors=top_factors,
            base_value=base_value,
            fraud_flag=fraud_flag,
            fraud_score=fraud_score,
            fraud_msg=fraud_msg
        )
    else:
        return render_template(
            'op.html',
            pred='Expected amount is {0:.3f}'.format(pred_val),
            top_factors=top_factors,
            base_value=base_value,
            fraud_flag=fraud_flag,
            fraud_score=fraud_score,
            fraud_msg=fraud_msg
        )

# ----------------------------
# BMI calculator
# ----------------------------
@app.route('/bmi', methods=['GET', 'POST'])
def bmi():
    bmi_result = None
    category = None
    if request.method == 'POST':
        try:
            weight = float(request.form['weight'])
            height = float(request.form['height']) / 100.0  # cm → meters
            bmi_result = round(weight / (height * height), 2)
            if bmi_result < 18.5:
                category = "Underweight"
            elif 18.5 <= bmi_result < 24.9:
                category = "Normal weight"
            elif 25 <= bmi_result < 29.9:
                category = "Overweight"
            else:
                category = "Obese"
        except Exception:
            bmi_result = "Invalid input"
    return render_template('bmi.html', bmi_result=bmi_result, category=category)

# ----------------------------
# Static pages
# ----------------------------
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/howitworks')
def how_it_works():
    return render_template('howitworks.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)
