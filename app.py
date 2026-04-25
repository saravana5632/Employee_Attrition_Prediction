from flask import Flask, render_template
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder


st.set_page_config(
    page_title="Employee Attrition Dashboard",
    layout="wide"
)


app = Flask(__name__)

# Load dataset

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df_display = df.copy()


df_encoded = df.copy()
le_dict ={}  # store encoders for each column

# Encode categorical data
df_encoded = df.copy()
le_dict = {}


for col in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    le_dict[col] = le



model = pickle.load(open("attrition_model.pkl", "rb"))


st.title("Employee Attrition Analysis Dashboard")


total_emp = len(df)
attrition_rate = df['Attrition'].mean() * 100
avg_salary = df['MonthlyIncome'].mean()


model = pickle.load(open("attrition_model.pkl", "rb"))

@app.route('/')
def dashboard():
    total_emp = len(df)
    attrition_rate = round(df['Attrition'].mean() * 100, 2)
    avg_salary = round(df['MonthlyIncome'].mean(), 0)

    
    high_risk = []
    attrition_counts = {0: 0, 1: 0}
    income_data = {0: [], 1: []}


    try:
        # Charts data
        attrition_counts = df['Attrition'].value_counts().to_dict()
        income_data = df.groupby('Attrition')['MonthlyIncome'].apply(list).to_dict()

        # Prediction
        X = df_encoded.drop('Attrition', axis=1)
        df_display['Attrition_Probability'] = model.predict_proba(X)[:, 1]



col1, col2 = st.columns(2)

        high_risk_df = df_display[df_display['Attrition_Probability'] > 0.6]
        high_risk_df = (
            high_risk_df[['Age', 'Department', 'MonthlyIncome', 'Attrition_Probability']]
            .sort_values(by='Attrition_Probability', ascending=False)
            .head(20)
        )

        high_risk = high_risk_df.to_dict(orient='records')


st.divider()
st.subheader("High Risk Employees")

X = df_encoded.drop('Attrition', axis=1)
df_display['Attrition_Probability'] = model.predict_proba(X)[:, 1]
high_risk = df_display[df_display['Attrition_Probability'] > 0.6]

high_risk = (
    high_risk[['Age', 'Department', 'MonthlyIncome', 'Attrition_Probability']]
    .sort_values(by='Attrition_Probability', ascending=False)
    .head(20)
    .reset_index(drop=True) 
)
high_risk.index = high_risk.index + 1
st.dataframe(high_risk, use_container_width=True)


    except Exception as e:
        print("ERROR:", e)

    return render_template(
        'index.html',
        total_emp=total_emp,
        attrition_rate=attrition_rate,
        avg_salary=avg_salary,
        high_risk=high_risk,
        attrition_counts=attrition_counts,
        income_data=income_data
    )

if __name__ == '__main__':
    app.run(debug=True)

