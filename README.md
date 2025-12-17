# ❤️ Heart Disease Prediction Using Machine Learning

A machine learning–based web application that predicts the likelihood of heart disease based on patient health parameters.  
Built using **Python**, **Scikit-Learn**, and **Streamlit**.

---

## 🌐 Live Demo

👉 **Streamlit App:**  
https://sagar-46-heart-disease-prediction-using-machine-lear-app-eq7ixk.streamlit.app/

---

## 🚀 Project Overview

Heart disease is one of the leading causes of death worldwide. Early detection can significantly reduce risks and improve patient outcomes.  
This project uses supervised machine learning algorithms to predict whether a person is likely to have heart disease based on medical attributes.

The trained model is deployed as a **Streamlit web application** for real-time prediction.

---

## 🧠 Machine Learning Models Used

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Naive Bayes  
- Decision Tree  
- Random Forest  
- XGBoost  

---

## 📊 Model Performance Comparison

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression | 66.42%   |
| KNN                | 80.28%   |
| SVM                | 63.01%   |
| Naive Bayes        | 67.67%   |
| Decision Tree      | 77.32%   |
| **Random Forest**  | **91.24%** |
| XGBoost            | 88.69%   |

---

## 🏆 Best Performing Model

- **Random Forest Classifier**
- **Accuracy:** **91.24%**
- Selected for deployment due to:
  - High predictive accuracy
  - Robustness to overfitting
  - Ability to handle complex feature interactions
  - Strong performance on imbalanced datasets

The trained model was saved using **Joblib** and deployed via **Streamlit**.

---

## 📋 Features Used

- Age  
- Gender  
- Blood Pressure  
- Cholesterol Level  
- BMI  
- Smoking  
- Alcohol Consumption  
- Diabetes  
- Exercise Habits  
- Stress Level  
- Sleep Hours  
- Triglyceride Level  
- Fasting Blood Sugar  
- CRP Level  
- Homocysteine Level  
- Family History of Heart Disease  

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Machine Learning:** Scikit-Learn  
- **Data Processing:** Pandas, NumPy  
- **Model Persistence:** Joblib  
- **Web Framework:** Streamlit  

---

## 📁 Project Structure

heart-disease-prediction-using-machine-learning \
├── app.py # Streamlit application \
├── model.pkl # Trained Random Forest model \
├── scaler.pkl # StandardScaler object \
├── columns.pkl # Feature columns used during training \
├── requirements.txt # Project dependencies \
├── README.md # Project documentation \
└── dataset.csv # Dataset used for training
