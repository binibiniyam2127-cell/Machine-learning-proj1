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
    # Get the features the model expects
    expected_features = model.get_booster().feature_names
except Exception as e:
    st.error(f"Error loading model: {e}")
    expected_features = None

st.title("🛡️ Fraud Detection System")

# --- 2. Input Fields ---
col1, col2 = st.columns(2)

with col1:
    timestamp = st.number_input("Timestamp", value=500)
    amount = st.number_input("Transaction Amount", value=100.0)
    old_balance = st.number_input("Old Balance", value=1000.0)
    new_balance = st.number_input("New Balance", value=900.0)

with col2:
    trans_type = st.selectbox("Type", ['PAYMENT', 'DEBIT', 'CASH_IN', 'TRANSFER', 'CASH_OUT'])
    region = st.selectbox("Region", ['North', 'South', 'East', 'West', 'Central'])
    device = st.selectbox("Device", ['Desktop', 'Tablet', 'Mobile'])
    is_intl = st.selectbox("International?", [0, 1])

# --- 3. Encoding ---
# We use a DataFrame here because XGBoost often requires feature names to match
input_df = pd.DataFrame([{
    'Timestamp': timestamp,
    'Transaction_Type': trans_type,
    'Amount': amount,
    'Old_Balance': old_balance,
    'New_Balance': new_balance,
    'Region': region,
    'Device_Type': device,
    'Is_International': is_intl
}])

# Convert categories to numbers (Label Encoding)
input_df['Transaction_Type'] = input_df['Transaction_Type'].map({'PAYMENT':0, 'DEBIT':1, 'CASH_IN':2, 'TRANSFER':3, 'CASH_OUT':4})
input_df['Region'] = input_df['Region'].map({'North':0, 'South':1, 'East':2, 'West':3, 'Central':4})
input_df['Device_Type'] = input_df['Device_Type'].map({'Desktop':0, 'Tablet':1, 'Mobile':2})

# --- 4. Prediction ---
if st.button("Analyze Transaction"):
    try:
        # We ensure the columns are in the exact order the model expects
        if expected_features:
            input_df = input_df[expected_features]
        
        prediction = model.predict(input_df)
        
        if prediction[0] == 1:
            st.error("🚨 Result: Fraudulent")
        else:
            st.success("✅ Result: Legitimate")
            
    except Exception as e:
        st.error(f"Feature Mismatch: {e}")
        st.info("The model expects these features: " + str(expected_features))