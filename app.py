import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import xgboost as xgb

# --- 1. SMART FILENAME SETTING ---
# This part handles the folder path automatically
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'xgboost_model.pkl')

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Model file not found at {model_path}. Error: {e}")

# --- 2. THE APP INTERFACE ---
st.set_page_config(page_title="Fraud Detector", page_icon="🛡️")
st.title("🛡️ Credit Card Fraud Detection")

v1 = st.number_input("Feature V1", value=0.0)
v2 = st.number_input("Feature V2", value=0.0)
amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=10.0)

if st.button("Run Fraud Analysis"):
    # Ensure this array matches the number of features your model expects
    input_data = np.array([[v1, v2, amount]]) 
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("🚨 High Risk: This transaction is flagged as FRAUD.")
    else:
        st.success("✅ Low Risk: This transaction appears legitimate.")