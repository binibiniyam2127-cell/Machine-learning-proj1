import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load the Model ---
# Using the exact name you provided
try:
    model = joblib.load('xgboost_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")

st.set_page_config(page_title="Fraud Detector", page_icon="🛡️")
st.title("🛡️ Credit Card Fraud Detection")
st.write("Enter transaction details below to check for fraudulent activity.")

# --- 2. Input Fields ---
# Note: Ensure these features match exactly what your model was trained on
st.subheader("Transaction Details")
v1 = st.number_input("Feature V1", value=0.0)
v2 = st.number_input("Feature V2", value=0.0)
amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=10.0)

# --- 3. Prediction Logic ---
if st.button("Run Fraud Analysis"):
    # Create the input array (adjust this if your model needs more features)
    input_data = np.array([[v1, v2, amount]]) 
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("🚨 High Risk: This transaction is flagged as FRAUD.")
    else:
        st.success("✅ Low Risk: This transaction appears legitimate.")