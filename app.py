import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import xgboost as xgb

# --- 1. Load the Model ---
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'xgboost_model.pkl')

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Could not load model: {e}")

st.title("🛡️ Fraud Detection System (Raw Data Version)")
st.write("Enter the transaction details as they appear in the raw dataset.")

# --- 2. Input Fields (Matching your CSV columns) ---
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Amount", min_value=0.0, value=100.0)
    old_balance = st.number_input("Old Balance", value=5000.0)
    new_balance = st.number_input("New Balance", value=4900.0)
    timestamp = st.number_input("Timestamp (Integer)", value=100)

with col2:
    trans_type = st.selectbox("Transaction Type", ['PAYMENT', 'DEBIT', 'CASH_IN', 'TRANSFER', 'CASH_OUT'])
    region = st.selectbox("Region", ['North', 'South', 'East', 'West', 'Central'])
    device = st.selectbox("Device Type", ['Desktop', 'Tablet', 'Mobile'])
    is_intl = st.selectbox("Is International?", ["No", "Yes"])

# --- 3. Encoding Logic ---
# Note: These mappings must match how you trained your model!
type_map = {'PAYMENT': 0, 'DEBIT': 1, 'CASH_IN': 2, 'TRANSFER': 3, 'CASH_OUT': 4}
region_map = {'North': 0, 'South': 1, 'East': 2, 'West': 3, 'Central': 4}
device_map = {'Desktop': 0, 'Tablet': 1, 'Mobile': 2}
intl_val = 1 if is_intl == "Yes" else 0

# --- 4. Prediction ---
if st.button("Analyze Transaction"):
    # Create the feature array in the EXACT order your model expects.
    # Usually: [Timestamp, Transaction_Type, Amount, Old_Balance, New_Balance, Region, Device_Type, Is_International]
    features = np.array([[
        timestamp, 
        type_map[trans_type], 
        amount, 
        old_balance, 
        new_balance, 
        region_map[region], 
        device_map[device], 
        intl_val
    ]])
    
    try:
        prediction = model.predict(features)
        if prediction[0] == 1:
            st.error("🚨 Warning: Fraudulent Transaction Detected!")
        else:
            st.success("✅ Transaction is Legitimate.")
    except Exception as e:
        st.warning("Feature Mismatch Error.")
        st.info("Check the 'Manage App' logs. It will tell you how many features the model is actually looking for.")
        st.error(f"Details: {e}")