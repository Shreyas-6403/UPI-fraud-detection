import pickle
import numpy as np
import streamlit as st

# Load Model
model = pickle.load(open("fraud_detection_model.pkl", "rb"))

# Load Encoders (Fix: Using correct file name)
with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Debug: Check available encoders
st.write("✅ Available Encoders:", list(encoders.keys()))

# Function to safely encode categorical features
def encode_feature(feature_name, value):
    if feature_name in encoders:
        return encoders[feature_name].transform([value])[0] if value in encoders[feature_name].classes_ else -1
    else:
        st.warning(f"⚠️ Encoder for '{feature_name}' is missing! Using default encoding.")
        return -1  # Default encoding

# Prediction Function
def predict_fraud(sender_upi, receiver_upi, amount, hour, status):
    sender_upi_encoded = encode_feature("sender_upi", sender_upi)
    receiver_upi_encoded = encode_feature("receiver_upi", receiver_upi)
    status_encoded = encode_feature("status", status)

    # Ensure input matches model's expected features
    input_data = np.array([[sender_upi_encoded, receiver_upi_encoded, amount, hour, status_encoded]])

    if input_data.shape[1] != model.n_features_in_:
        st.error(f"⚠️ Feature mismatch! Model expects {model.n_features_in_} features but received {input_data.shape[1]}.")
        return "Error"

    prediction = model.predict(input_data)[0]
    return "Fraudulent" if prediction == 1 else "Legitimate"
