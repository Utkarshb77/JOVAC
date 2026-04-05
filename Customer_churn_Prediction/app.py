import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.json", "r") as f:
    model_columns = json.load(f)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict churn:")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
Partner = st.selectbox("Partner", ["No", "Yes"])
Dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.slider("Tenure", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 300.0, step=1.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, step=10.0)

input_data = {
    'gender': 1 if gender == "Male" else 0,
    'SeniorCitizen': 1 if SeniorCitizen == "Yes" else 0,
    'Partner': 1 if Partner == "Yes" else 0,
    'Dependents': 1 if Dependents == "Yes" else 0,
    'tenure': tenure,
    'PhoneService': 1 if PhoneService == "Yes" else 0,
    'PaperlessBilling': 1 if PaperlessBilling == "Yes" else 0,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges,

    'MultipleLines_No phone service': 1 if MultipleLines == "No phone service" else 0,
    'MultipleLines_Yes': 1 if MultipleLines == "Yes" else 0,

    'InternetService_Fiber optic': 1 if InternetService == "Fiber optic" else 0,
    'InternetService_No': 1 if InternetService == "No" else 0,

    'OnlineSecurity_Yes': 1 if OnlineSecurity == "Yes" else 0,
    'OnlineSecurity_No internet service': 1 if OnlineSecurity == "No internet service" else 0,

    'OnlineBackup_Yes': 1 if OnlineBackup == "Yes" else 0,
    'OnlineBackup_No internet service': 1 if OnlineBackup == "No internet service" else 0,

    'DeviceProtection_Yes': 1 if DeviceProtection == "Yes" else 0,
    'DeviceProtection_No internet service': 1 if DeviceProtection == "No internet service" else 0,

    'TechSupport_Yes': 1 if TechSupport == "Yes" else 0,
    'TechSupport_No internet service': 1 if TechSupport == "No internet service" else 0,

    'StreamingTV_Yes': 1 if StreamingTV == "Yes" else 0,
    'StreamingTV_No internet service': 1 if StreamingTV == "No internet service" else 0,

    'StreamingMovies_Yes': 1 if StreamingMovies == "Yes" else 0,
    'StreamingMovies_No internet service': 1 if StreamingMovies == "No internet service" else 0,

    'Contract_One year': 1 if Contract == "One year" else 0,
    'Contract_Two year': 1 if Contract == "Two year" else 0,

    'PaymentMethod_Bank transfer (automatic)': 1 if PaymentMethod == "Bank transfer (automatic)" else 0,
    'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == "Credit card (automatic)" else 0,
    'PaymentMethod_Mailed check': 1 if PaymentMethod == "Mailed check" else 0,
}


input_df = pd.DataFrame([input_data])

input_df = input_df.reindex(columns=model_columns, fill_value=0)

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"❌ Churn Likely (Risk: {prob:.2f})")
    else:
        st.success(f"✅ Customer Likely to Stay (Risk: {prob:.2f})")
