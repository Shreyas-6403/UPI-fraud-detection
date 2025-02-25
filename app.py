import streamlit as st
import joblib
import pandas as pd

# Load the trained model
MODEL_PATH = "fraud_detection_model.pkl"
ENCODER_PATH = "label_encoders.pkl"

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH)
        return model, encoders
    except FileNotFoundError:
        st.error("Model or encoders not found! Please train the model first.")
        return None, None

model, encoders = load_model_and_encoders()

# Streamlit UI
st.title("UPI Fraud Detection")

# Input Fields
transaction_id = st.text_input("Transaction ID")
timestamp = st.text_input("Timestamp")
sender_name = st.text_input("Sender Name")
sender_upi = st.text_input("Sender UPI ID")
receiver_name = st.text_input("Receiver Name")
receiver_upi = st.text_input("Receiver UPI ID")
amount = st.number_input("Transaction Amount (INR)", min_value=0.01)
status = st.selectbox("Transaction Status", ["Success", "Failed"])

if st.button("Detect Fraud"):
    if model is None:
        st.error("Model is not loaded. Please train it first.")
    else:
        # Convert status to binary
        status_encoded = 1 if status == "Success" else 0
        
        # Encode sender/receiver UPI
        try:
            sender_upi_encoded = encoders["sender_upi"].transform([sender_upi])[0]
            receiver_upi_encoded = encoders["receiver_upi"].transform([receiver_upi])[0]
        except KeyError:
            st.warning("New UPI ID detected. Using raw value.")
            sender_upi_encoded = sender_upi
            receiver_upi_encoded = receiver_upi
        
        # Prepare input
        input_data = pd.DataFrame([{
            "Amount (INR)": amount,
            "Status": status_encoded,
            "Sender UPI": sender_upi_encoded,
            "Receiver UPI": receiver_upi_encoded
        }])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "⚠ Fraudulent Transaction!" if prediction == 1 else "✅ Legitimate Transaction"
        
        st.success(result)
