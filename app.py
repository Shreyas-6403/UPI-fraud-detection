import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model and encoders
model = joblib.load("fraud_detection_model.pkl")
encoders = joblib.load("label_encoders.pkl")  # This should be a dictionary of encoders

# Define prediction function
def predict_fraud(sender_upi, receiver_upi, amount, hour, day, month, status):
    # Encode categorical variables safely
    if "sender_upi" in encoders:
        sender_upi = encoders["sender_upi"].transform([sender_upi])[0] if sender_upi in encoders["sender_upi"].classes_ else -1
    else:
        sender_upi = -1  # Default value if encoder is missing

    if "receiver_upi" in encoders:
        receiver_upi = encoders["receiver_upi"].transform([receiver_upi])[0] if receiver_upi in encoders["receiver_upi"].classes_ else -1
    else:
        receiver_upi = -1  # Default value if encoder is missing

    if "status" in encoders:
        status = encoders["status"].transform([status])[0] if status in encoders["status"].classes_ else -1
    else:
        status = -1  # Default value if encoder is missing

    # Convert inputs into a NumPy array for model prediction
    input_data = np.array([[sender_upi, receiver_upi, amount, hour, day, month, status]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return "Fraudulent" if prediction == 1 else "Legitimate"

# Streamlit UI
st.title("UPI Fraud Detection System")

sender_upi = st.text_input("Sender UPI ID")
receiver_upi = st.text_input("Receiver UPI ID")
amount = st.number_input("Transaction Amount (INR)", min_value=0.01)
hour = st.number_input("Transaction Hour (0-23)", min_value=0, max_value=23)
day = st.number_input("Transaction Day (1-31)", min_value=1, max_value=31)
month = st.number_input("Transaction Month (1-12)", min_value=1, max_value=12)
status = st.selectbox("Transaction Status", options=encoders["status"].classes_ if "status" in encoders else ["Pending", "Completed", "Failed"])

if st.button("Predict Fraud"):
    result = predict_fraud(sender_upi, receiver_upi, amount, hour, day, month, status)
    st.write(f"### Prediction: {result}")
