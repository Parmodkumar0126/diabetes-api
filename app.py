from flask import Flask, request, jsonify
import pickle
import numpy as np
import json
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
with open("diabetes_model_version2.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler safely
try:
    scaler = joblib.load("scaler_version2.pkl")
except:
    with open("scaler_version2.pkl", "rb") as f:
        scaler = pickle.load(f)

# Load feature names
with open("features_version2.json", "r") as f:
    feature_names = json.load(f)


@app.route("/")
def home():
    return "Diabetes Prediction API Running"


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        # Arrange features in correct order
        features = [data[feature] for feature in feature_names]

        features = np.array(features).reshape(1, -1)

        # Apply scaler
        features_scaled = scaler.transform(features)

        # Prediction
        prediction = model.predict(features_scaled)[0]

        # Probability
        prob = float(model.predict_proba(features_scaled)[0][1])

        return jsonify({
           "status": "success",
            "prediction": int(prediction),
            "risk_percentage": round(prob * 100, 2)
        })
        

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
