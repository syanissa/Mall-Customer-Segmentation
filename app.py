import streamlit as st
import joblib
import pandas as pd

# Load trained models
lin_reg = joblib.load('linear_model.pickle')
tree_reg = joblib.load('tree_model.pickle')
xgb_reg = joblib.load('xgb_model.pickle')
scaler = joblib.load('scaler.pickle')
le = joblib.load('label_encoder.pickle')

def preprocess_input(age, gender, annual_income):
    gender_encoded = le.transform([gender])[0]
    input_data = pd.DataFrame([[age, gender_encoded, annual_income]],
                              columns=['Age', 'Gender', 'Annual Income (k$)'])
    input_data[['Age', 'Annual Income (k$)']] = scaler.transform(
        input_data[['Age', 'Annual Income (k$)']]
    )
    input_data = input_data[['Gender', 'Age', 'Annual Income (k$)']]
    return input_data


def predict_spending_score(age, gender, annual_income, model_name):
    input_data = preprocess_input(age, gender, annual_income)
    if model_name == 'Linear Regression':
        return lin_reg.predict(input_data)[0]
    elif model_name == 'Decision Tree':
        return tree_reg.predict(input_data)[0]
    elif model_name == 'XGBoost':
        return xgb_reg.predict(input_data)[0]
    

st.title("Mall Customer Spending Score Prediction")

st.markdown("""
    Website ini memprediksi **Spending Score** pelanggan berdasarkan 
    **usia, gender**, dan **pendapatan tahunan**. 
    Terdapat 3 model machine learning yang dapat dipilih untuk memprediksi spending score.
            
    Skala spending score berada di antara 1-100 yang menggambarkan 
    customer behaviour serta spending nature.
""")

st.markdown("---")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider('Age', min_value=18, max_value=70, value=30)
annual_income = st.slider('Annual Income (k$)', min_value=0, max_value=150, value=50)
model_name = st.selectbox("Select Model", ["Linear Regression", "Decision Tree", "XGBoost"])

if st.button("Predict"):
    score = predict_spending_score(age, gender, annual_income, model_name)
    st.write(f"Predicted Spending Score: {score:.2f}")
