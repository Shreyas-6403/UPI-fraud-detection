import joblib
import streamlit as st
import numpy as np
from PIL import Image

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")

# Load Encoders (Ensure it's a dictionary)
encoders = joblib.load("label_encoders.pkl")

# **Fix: Check if encoders are correctly loaded**
if not isinstance(encoders, dict):
    st.error("❌ `label_encoders.pkl` is not a dictionary! Re-save it using the training script.")
    st.stop()

# Load the image
image = Image.open("image.jpg")

# Display the image with the correct parameter
st.image(image, use_container_width=True)

# Debugging - Show available encoders
#st.write("✅ Available Encoders:", list(encoders.keys()))

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

    feature_6 = 0
    feature_7 = 0
    
    # Ensure input matches model's expected features
    input_data = np.array([[sender_upi_encoded, receiver_upi_encoded, amount, hour, status_encoded, feature_6, feature_7]])

    if input_data.shape[1] != model.n_features_in_:
        st.error(f"⚠️ Feature mismatch! Model expects {model.n_features_in_} features but received {input_data.shape[1]}.")
        return "Error"

    prediction = model.predict(input_data)[0]
    return "Fraud" if prediction == 1 else "Legit"

# Streamlit UI
st.title("UPI Fraud Detection System")

sender_upi = st.text_input("Sender UPI ID")
receiver_upi = st.text_input("Receiver UPI ID")
amount = st.number_input("Transaction Amount (INR)", min_value=0.01)
hour = st.number_input("Transaction Hour (0-23)", min_value=0, max_value=23)
status_options = encoders["status"].classes_ if "status" in encoders else ["Pending", "Completed", "Failed"]
status = st.selectbox("Transaction Status", options=status_options)

if st.button("Predict Fraud"):
    result = predict_fraud(sender_upi, receiver_upi, amount, hour, status)
    if result != "Error":
        st.write(f"### Prediction: {result}")
        
# 🚨 Fraud Prevention Tips
st.markdown("### 🛡️ How to Avoid Fraud Transactions:")
st.markdown("""
- **🔒 Never share your UPI PIN** with anyone.
- **🚫 Avoid clicking on unknown links** asking for payments.
- **🧐 Double-check the receiver's UPI ID** before sending money.
- **⚠️ Be cautious of offers that sound too good to be true.**
- **📞 If in doubt, contact your bank directly** instead of calling unknown numbers.
- **🛑 Don't approve 'Request Money' transactions** unless you trust the sender.
- **✅ Use trusted apps** and keep them updated for security fixes.
""")
