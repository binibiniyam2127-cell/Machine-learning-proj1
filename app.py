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
    # These are the exact names from your error message
    expected_cols = [
        'Hour', 'Balance_Error', 'Transaction_Type_CASH_OUT', 
        'Transaction_Type_DEBIT', 'Transaction_Type_PAYMENT', 
        'Transaction_Type_TRANSFER', 'Region_East', 'Region_North', 
        'Region_South', 'Region_West', 'Device_Type_Mobile', 'Device_Type_Tablet'
    ]
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("🛡️ Fraud Detection System")

# --- 2. Input Fields ---
col1, col2 = st.columns(2)

with col1:
    timestamp = st.number_input("Timestamp (e.g., 0-700)", value=100)
    amount = st.number_input("Amount", value=100.0)
    old_balance = st.number_input("Old Balance", value=1000.0)
    new_balance = st.number_input("New Balance", value=900.0)

with col2:
    trans_type = st.selectbox("Type", ['PAYMENT', 'DEBIT', 'CASH_IN', 'TRANSFER', 'CASH_OUT'])
    region = st.selectbox("Region", ['North', 'South', 'East', 'West', 'Central'])
    device = st.selectbox("Device", ['Desktop', 'Tablet', 'Mobile'])

# --- 3. Preprocessing (Matching the Model's Training) ---
if st.button("Analyze Transaction"):
    # Create a base dataframe
    data = pd.DataFrame([[timestamp, amount, old_balance, new_balance]], 
                        columns=['Timestamp', 'Amount', 'Old_Balance', 'New_Balance'])
    
    # Feature Engineering
    # 1. Hour (assuming timestamp is in hours or needs to be scaled)
    data['Hour'] = timestamp % 24 
    # 2. Balance Error (Common fraud feature: expected vs actual balance)
    data['Balance_Error'] = new_balance - (old_balance - amount)
    
    # 3. One-Hot Encoding (Manual creation to match the expected names)
    data['Transaction_Type_' + trans_type] = 1
    data['Region_' + region] = 1
    data['Device_Type_' + device] = 1
    
    # 4. Final Alignment
    # This creates all missing columns (like Region_South if you picked North) and sets them to 0
    final_df = data.reindex(columns=expected_cols, fill_value=0)
    
    try:
        prediction = model.predict(final_df)
        
        if prediction[0] == 1:
            st.error("🚨 Result: FRAUD DETECTED")
        else:
            st.success("✅ Result: LEGITIMATE")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Current Input Shape:", final_df.columns.tolist())