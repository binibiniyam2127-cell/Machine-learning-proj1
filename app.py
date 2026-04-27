import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load the Model ---
# Make sure the filename matches EXACTLY what you downloaded
model = joblib.load('model_assets.joblib')

st.set_page_config(page_title="Fraud Detector", page_icon="🛡️")
st.title("🛡️ Credit Card Fraud Detection")
st.write("Enter the transaction features below to verify authenticity.")

# --- 2. Input Fields ---
# Note: Use the feature names your model was trained on (V1, V2, Amount, etc.)
st.subheader("Transaction Details")
col1, col2 = st.columns(2)

with col1:
    v1 = st.number_input("Feature V1", value=0.0)
    v2 = st.number_input("Feature V2", value=0.0)
    # Add more features as per your dataset

with col2:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=10.0)

# --- 3. Prediction Logic ---
if st.button("Run Fraud Analysis"):
    # Organize inputs into the same format the model expects
    input_data = np.array([[v1, v2, amount]]) # Ensure order matches training
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data) # Optional: shows confidence

    if prediction[0] == 1:
        st.error(f"High Risk: This transaction is flagged as FRAUD.")
    else:
        st.success(f"Low Risk: This transaction appears legitimate.")
    
    st.info(f"Model Confidence: {np.max(probability) * 100:.2f}%")