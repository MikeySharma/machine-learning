from flask import Blueprint, request, jsonify
import joblib
import numpy as np
import os

# Check if the model file exists and load the saved model
model_path = os.path.join(os.path.dirname(__file__), '../trained-model/salary_multiple_model.pkl')
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file not found at path: {model_path}")

# Create a Blueprint for the predict route
predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        if 'features' not in data:
            return jsonify({'error': 'Invalid input format. Expected JSON with "features" key.'}), 400

        # Convert features to a numpy array
        features = np.array(data['features']).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        # Return a 500 error if any exception occurs
        return jsonify({'error': str(e)}), 500
