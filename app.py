import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('RandomForest.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)
st.title("Heart Disease Prediction System ❤️")
st.subheader("Enter details to predict the risk of heart disease")

# Input

age = st.slider("Age",18,80,23)
gender = st.selectbox("Gender",["Male","Female"])
blood_pressure = st.slider("Blood Pressure (mmHg)",80,180,120)
cholesterol = st.slider("Cholesterol Level (mg/dL)",100,300,180)
exercise = st.selectbox("Exercise Habits",["Low","Medium","High"])
smoking = st.selectbox("Smoking",["Yes","No"])
family_hd = st.selectbox("Family History of Heart Disease",["Yes","No"])
diabetes = st.selectbox("Diabetes",["Yes","No"])
bmi = st.slider("BMI",15,50,22)
high_bp = st.selectbox("High Blood Pressure",["Yes","No"])
low_hdl = st.selectbox("Low HDL Cholesterol",["Yes","No"])
high_ldl = st.selectbox("High LDL Cholesterol",{"Yes","No"})
alcohol = st.selectbox("Alcohol Consumption",["Low","Medium","High"])
stress = st.selectbox("Stress Level",["Low","Medium","High"])
sleep_hours = st.slider("Sleep Hours per Day",2,14,8)
sugar = st.selectbox("Sugar Consumption",["Low","Medium","High"])
triglyceride = st.slider("Triglyceride Level",100,400,250)
fasting_bs = st.slider("Fasting Blood Sugar",60,200,100)
crp = st.slider("CRP Level",0.1,20.0,1.0)
homocysteine = st.slider("Homocysteine Level",5.0,30.0,12.0)

# Encoding

map_level = {"High":0,"Medium":1,"Low":2}
map_yes = {"Yes":1,"No":0}

gender = 1 if gender == "Male" else 0

exercise = map_level[exercise]
alcohol = map_level[alcohol]
stress = map_level[stress]
sugar = map_level[sugar]

smoking = map_yes[smoking]
family_hd = map_yes[family_hd]
diabetes = map_yes[diabetes]
high_bp = map_yes[high_bp]
low_hdl = map_yes[low_hdl]
high_ldl = map_yes[high_ldl]

# Create Input array

input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Blood Pressure": blood_pressure,
    "Cholesterol Level": cholesterol,
    "Exercise Habits": exercise,
    "Smoking": smoking,
    "Family Heart Disease": family_hd,
    "Diabetes": diabetes,
    "BMI": bmi,
    "High Blood Pressure": high_bp,
    "Low HDL Cholesterol": low_hdl,
    "High LDL Cholesterol": high_ldl,
    "Alcohol Consumption": alcohol,
    "Stress Level": stress,
    "Sleep Hours": sleep_hours,
    "Sugar Consumption": sugar,
    "Triglyceride Level": triglyceride,
    "Fasting Blood Sugar": fasting_bs,
    "CRP Level": crp,
    "Homocysteine Level": homocysteine
}])

input_df = input_df[expected_columns]

input_scaled = scaler.transform(input_df)

probability = model.predict_proba(input_scaled)[0][1]
prediction = 1 if probability >= 0.35 else 0


if st.button("🔍 Predict Heart Disease"):    
    if prediction == 1:
        st.error(
            f"⚠️ High Risk of Heart Disease\n\n"
        )
    else:
        st.success(
            f"✅ Low Risk of Heart Disease\n\n"
        )