from flask import Flask, render_template
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__, template_folder='../templates')

# Use absolute paths (important on Vercel/Render)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(BASE_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
model_path = os.path.join(BASE_DIR, "attrition_model.pkl")

# Load data
df = pd.read_csv(csv_path)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df_display = df.copy()

df_encoded = df.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

model = pickle.load(open(model_path, "rb"))

@app.route('/')
def dashboard():
    total_emp = len(df)
    attrition_rate = round(df['Attrition'].mean() * 100, 2)
    avg_salary = round(df['MonthlyIncome'].mean(), 0)

    attrition_counts = df['Attrition'].value_counts().to_dict()
    income_data = df.groupby('Attrition')['MonthlyIncome'].apply(list).to_dict()

    X = df_encoded.drop('Attrition', axis=1)
    df_display['Attrition_Probability'] = model.predict_proba(X)[:, 1]

    high_risk = df_display[df_display['Attrition_Probability'] > 0.6]
    high_risk = high_risk[['Age','Department','MonthlyIncome','Attrition_Probability']].head(20)

    return render_template(
        'index.html',
        total_emp=total_emp,
        attrition_rate=attrition_rate,
        avg_salary=avg_salary,
        high_risk=high_risk.to_dict(orient='records'),
        attrition_counts=attrition_counts,
        income_data=income_data
    )
