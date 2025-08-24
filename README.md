Got it 👍 Here’s the complete polished README.md with your Render live demo link included. Just copy-paste this into your README.md file in GitHub.

⸻

🛡 InsuraSure

Predict. Protect. Prepare.

InsuraSure is a machine learning–powered web application that predicts health insurance premiums based on lifestyle and health metrics.

✅ Try the live app here → https://insurasure.onrender.com

⸻

📖 Overview

Health insurance costs are often complex and opaque. InsuraSure makes them more transparent by using a trained ML model to estimate premiums from inputs such as age, BMI, smoking status, children, and region.

It also highlights why a given premium was predicted (using SHAP explainability) and flags unusual input profiles with an anomaly detector.

⸻

✨ Features
	•	🔮 Premium Prediction → Instant estimates using a trained Random Forest model.
	•	💡 Explainability (SHAP) → Shows top 3 factors that increase/decrease your premium.
	•	🚨 Fraud/Anomaly Detection → Flags suspicious or unrealistic input patterns.
	•	⚖ BMI Calculator → Compute BMI and categories (Underweight, Normal, Overweight, Obese).
	•	🖥 Responsive UI → Built with MaterializeCSS + Jinja2 templates.
	•	📊 Baseline Comparison → Shows model’s expected premium vs. your personalized adjustments.

⸻

🧠 Machine Learning Models

1. Random Forest Regressor (Premium Prediction)
	•	Features: Age, Sex, BMI, Children, Smoker, Region
	•	Target: charges (insurance premiums)
	•	Tuned with GridSearchCV (n_estimators, max_depth, min_samples_split)
	•	Saved as rf_tuned.pkl

2. Isolation Forest (Fraud/Anomaly Detection)
	•	Detects unusual profiles in insurance data
	•	Contamination ≈ 0.05 (5% anomalies assumed)
	•	Saved as fraud_iforest.pkl

3. SHAP (Explainability)
	•	Provides per-prediction attributions for each feature
	•	Example:
	•	↑ Smoking: +₹12,000
	•	↑ BMI: +₹5,000
	•	↓ Younger age: −₹1,500

⸻

🛠 Tech Stack
	•	Backend: Python (Flask, Gunicorn)
	•	ML: scikit-learn (RandomForestRegressor, IsolationForest), SHAP
	•	Frontend: HTML, CSS, MaterializeCSS, Jinja2
	•	Data Handling: pandas, numpy
	•	Deployment: Render

⸻

📂 Project Structure

InsuraSure/
├── app.py                # Flask app
├── wsgi.py               # WSGI entry point
├── rf_tuned.pkl          # Random Forest model
├── fraud_iforest.pkl     # Isolation Forest model
├── requirements.txt      # Python dependencies
├── /templates            # HTML templates
│   ├── home.html
│   ├── op.html
│   ├── bmi.html
│   ├── about.html
│   ├── howitworks.html
│   └── contact.html
└── /static               # CSS, JS, images
    ├── css/materialize.css
    └── js/materialize.js


⸻

⚙ Setup & Installation
	1.	Clone this repo

git clone https://github.com/<your-username>/InsuraSure.git
cd InsuraSure


	2.	Create and activate a virtual environment

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows


	3.	Install dependencies

pip install -r requirements.txt


	4.	Run locally

python app.py

Visit http://127.0.0.1:5000 in your browser.

⸻

▶ Usage
	•	Fill in your details on the Home page → Click Get Premium
	•	Results page displays:
	•	Predicted premium
	•	SHAP top factors
	•	Baseline premium
	•	Fraud/anomaly flag (if any)
	•	Navigate to BMI Calculator for health metrics
	•	Check About / How It Works / Contact tabs for details

⸻

☁ Deployment

Render (already deployed 🎉)
	•	App is live at: https://insurasure.onrender.com
	•	Powered by Flask + Gunicorn on Render free tier.

To deploy yourself
	1.	Push repo to GitHub
	2.	Connect repo on Render → New Web Service
	3.	Use the following Start Command:

gunicorn wsgi:app --workers 2 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT



⸻

🖼 Screenshots

(add your own screenshots here)
	•	🏠 Home Page Form
	•	📊 Prediction Result with SHAP factors
	•	🚨 Fraud/Anomaly Flag Example
	•	⚖ BMI Calculator

⸻

🔮 Future Improvements
	•	📈 Premium projection (simulate 5–10 years into future)
	•	🧮 Scenario builder (compare smoker vs. non-smoker, BMI changes)
	•	🗂 Quote history & trends per user
	•	📑 Downloadable PDF reports
	•	🔐 User authentication & personal dashboards

⸻

🙌 Credits
	•	Dataset: Kaggle — Medical Cost Personal Dataset
	•	ML Libraries: scikit-learn, SHAP
	•	UI: MaterializeCSS
	•	Deployment: Render

⸻

📜 License

This project is licensed under the MIT License — see the LICENSE file for details.

⸻

👉 Pro tip: Add your logo image + tagline at the top of the README for extra polish.

⸻

Do you also want me to generate a LICENSE file (MIT) and a Contributing guide so your repo looks fully professional?
