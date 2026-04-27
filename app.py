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
    expected_cols = [
        'Timestamp', 'Amount', 'Old_Balance', 'New_Balance', 'Is_International',
        'Hour', 'Balance_Error', 'Transaction_Type_CASH_OUT', 
        'Transaction_Type_DEBIT', 'Transaction_Type_PAYMENT', 
        'Transaction_Type_TRANSFER', 'Region_East', 'Region_North', 
        'Region_South', 'Region_West', 'Device_Type_Mobile', 'Device_Type_Tablet'
    ]
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("🛡️ Advanced Fraud Detector")

# --- 2. Input Fields ---
col1, col2 = st.columns(2)
with col1:
    timestamp = st.number_input("Timestamp", value=500)
    amount = st.number_input("Transaction Amount", value=5000.0)
    old_balance = st.number_input("Old Balance", value=10000.0)
    new_balance = st.number_input("New Balance", value=10000.0) # Suspicious: Balance didn't drop!

with col2:
    trans_type = st.selectbox("Type", ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'DEBIT', 'CASH_IN'])
    region = st.selectbox("Region", ['North', 'South', 'East', 'West', 'Central'])
    device = st.selectbox("Device", ['Mobile', 'Tablet', 'Desktop'])
    is_intl = st.selectbox("International?", ["Yes", "No"])

# --- 3. Processing ---
if st.button("Analyze Transaction"):
    intl_val = 1 if is_intl == "Yes" else 0
    data = pd.DataFrame([[timestamp, amount, old_balance, new_balance, intl_val]], 
                        columns=['Timestamp', 'Amount', 'Old_Balance', 'New_Balance', 'Is_International'])
    
    # Matching your model's logic
    data['Hour'] = timestamp % 24 
    data['Balance_Error'] = new_balance - (old_balance - amount)
    data['Transaction_Type_' + trans_type] = 1
    data['Region_' + region] = 1
    data['Device_Type_' + device] = 1
    
    final_df = data.reindex(columns=expected_cols, fill_value=0)
    
    try:
        # Get Probability instead of just 0 or 1
        prob = model.predict_proba(final_df)[0][1] 
        risk_pct = round(prob * 100, 2)
        
        st.subheader(f"Risk Score: {risk_pct}%")
        st.progress(prob)
        
        if prob > 0.5:
            st.error("🚨 RESULT: FRAUD DETECTED")
        else:
            st.success("✅ RESULT: LEGITIMATE")
            
    except Exception as e:
        # Fallback if predict_proba isn't supported
        pred = model.predict(final_df)[0]
        if pred == 1: st.error("🚨 RESULT: FRAUD")
        else: st.success("✅ RESULT: LEGITIMATE")