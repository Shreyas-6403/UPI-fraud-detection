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
def predict_fraud(sender_upi, receiver_upi, amount, hour, day, month, status):
    # Encode categorical inputs safely
    sender_upi_encoded = encode_category(encoders.get("sender_upi"), sender_upi)
    receiver_upi_encoded = encode_category(encoders.get("receiver_upi"), receiver_upi)
    status_encoded = encode_category(encoders.get("status"), status)

    # Ensure input matches model feature count
    input_data = np.array([[sender_upi_encoded, receiver_upi_encoded, amount, hour, day, month, status_encoded]], dtype=np.float64)

    # Check shape before prediction
    if input_data.shape[1] != model.n_features_in_:
        st.error(f"Feature mismatch! Model expects {model.n_features_in_} features but received {input_data.shape[1]}.")
        return "Error"

    # Predict
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
status_options = encoders["status"].classes_ if "status" in encoders else ["Pending", "Completed", "Failed"]
status = st.selectbox("Transaction Status", options=status_options)

if st.button("Predict Fraud"):
    result = predict_fraud(sender_upi, receiver_upi, amount, hour, day, month, status)
    if result != "Error":
        st.write(f"### Prediction: {result}")
