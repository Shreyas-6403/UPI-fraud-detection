import streamlit as st
import joblib
import numpy as np

# Load trained model & encoders
model = joblib.load("fraud_detection_model.pkl")
encoders = joblib.load("label_encoders.pkl")  # Ensure this is a dictionary

# Function to safely encode categorical variables
def encode_category(encoder, value):
    if encoder is None:
        return -1  # Default if encoder is missing
    return encoder.transform([value])[0] if value in encoder.classes_ else -1  # Handle unseen values

# Prediction function
def predict_fraud(sender_upi, receiver_upi, amount, hour, status):
    """Predict fraud using the correct 5 feature input format."""
    sender_upi_encoded = encoders["sender_upi"].transform([sender_upi])[0]
    receiver_upi_encoded = encoders["receiver_upi"].transform([receiver_upi])[0]
    status_encoded = encoders["status"].transform([status])[0]

    # ⚠️ Only 5 features! (Removing day & month)
    input_data = np.array([[sender_upi_encoded, receiver_upi_encoded, amount, hour, status_encoded]])

    # ✅ Fix: Ensure feature count matches the model's requirement
    if input_data.shape[1] != model.n_features_in_:
        st.error(f"Feature mismatch! Model expects {model.n_features_in_} features but received {input_data.shape[1]}.")
        return "Error"

    prediction = model.predict(input_data)[0]
    return "Fraudulent" if prediction == 1 else "Legitimate"

# Streamlit UI
st.title("UPI Fraud Detection System")

sender_upi = st.text_input("Sender UPI ID")
receiver_upi = st.text_input("Receiver UPI ID")
amount = st.number_input("Transaction Amount (INR)", min_value=0.01)
hour = st.number_input("Transaction Hour (0-23)", min_value=0, max_value=23)

# Status Dropdown
status_options = encoders["status"].classes_ if "status" in encoders else ["Pending", "Completed", "Failed"]
status = st.selectbox("Transaction Status", options=status_options)

if st.button("Predict Fraud"):
    result = predict_fraud(sender_upi, receiver_upi, amount, hour, status)
    if result != "Error":
        st.write(f"### Prediction: {result}")

if st.button("Predict Fraud"):
    result = predict_fraud(sender_upi, receiver_upi, amount, hour, day, month, status)
    if result != "Error":
        st.write(f"### Prediction: {result}")
