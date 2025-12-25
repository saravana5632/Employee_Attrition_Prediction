import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Employee Attrition Dashboard",
    layout="wide"
)

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df_display = df.copy()

df_encoded = df.copy()
le_dict ={}  # store encoders for each column

for col in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    le_dict[col] = le  # save encoder if needed later


model = pickle.load(open("attrition_model.pkl", "rb"))


st.title("Employee Attrition Analysis Dashboard")


total_emp = len(df)
attrition_rate = df['Attrition'].mean() * 100
avg_salary = df['MonthlyIncome'].mean()

c1, c2, c3 = st.columns(3)
c1.metric("Total Employees", total_emp)
c2.metric("Attrition Rate (%)", f"{attrition_rate:.2f}")
c3.metric("Average Monthly Income", f"{avg_salary:.0f}")

st.divider()


col1, col2 = st.columns(2)

with col1:
    st.subheader("Attrition Count")
    fig, ax = plt.subplots()
    sns.countplot(x='Attrition', data=df, ax=ax)
    ax.set_xticklabels(['No', 'Yes'])
    st.pyplot(fig)

with col2:
    st.subheader("Monthly Income vs Attrition")
    fig, ax = plt.subplots()
    sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, ax=ax)
    ax.set_xticklabels(['No', 'Yes'])
    st.pyplot(fig)

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

