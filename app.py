from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Paths to model and encoder files
MODEL_PATH = "fraud_detection_model.pkl"
ENCODER_PATH = "label_encoders.pkl"

# Load Model and Encoders
def load_model_and_encoders():
    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH)  # Load label encoders
        print("✅ Model and Encoders loaded successfully.")
        return model, encoders
    except FileNotFoundError:
        print("⚠ Model or encoders not found! Train the model first.")
        return None, None

model, encoders = load_model_and_encoders()

@app.route('/')
def home():
    return render_template('index.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Required fields
        required_fields = ["amount", "status", "transaction_id", "timestamp", "sender_name", "sender_upi", "receiver_name", "receiver_upi"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400
        
        # Convert amount to float safely
        amount = float(data["amount"])
        
        # Convert status to binary (1 = Success, 0 = Failed)
        status = 1 if data["status"].strip().lower() == "success" else 0

        # Apply Label Encoders to categorical fields
        sender_upi = encoders["sender_upi"].transform([data["sender_upi"]])[0] if "sender_upi" in encoders else data["sender_upi"]
        receiver_upi = encoders["receiver_upi"].transform([data["receiver_upi"]])[0] if "receiver_upi" in encoders else data["receiver_upi"]

        # Prepare input data
        input_data = pd.DataFrame([{
            "Amount (INR)": amount,
            "Status": status,
            "Sender UPI": sender_upi,
            "Receiver UPI": receiver_upi
        }])
        
        # Ensure model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded. Train the model first."}), 500
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"

        return jsonify({"prediction": result})
    
    except ValueError as ve:
        return jsonify({"error": f"Invalid data format: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
