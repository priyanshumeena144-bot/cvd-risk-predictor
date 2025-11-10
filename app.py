import numpy as np
import joblib
from tensorflow import keras
from flask import Flask, request, jsonify

# 1. Initialize the Flask App
app = Flask(__name__)

# 2. Load Your Model and Scaler
print("--- Loading model and scaler ---")
try:
    # We use joblib.load for the scaler
    scaler = joblib.load('my_scaler.joblib')
    # We use keras.models.load_model for the deep learning model
    model = keras.models.load_model('my_cnn_lstm_model.keras')
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading files: {e}")

# 3. Define Your Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data sent by the user
        data = request.json
        # Extract the list of 15 features
        features_list = data['features']
        
        # 1. Convert the list of 15 features into a 2D numpy array
        features_np = np.array(features_list).reshape(1, -1)
        
        # 2. Apply the scaler (that you saved)
        scaled_features = scaler.transform(features_np)
        
        # 3. Reshape the data for the CNN+LSTM model
        reshaped_features = np.expand_dims(scaled_features, axis=2)
        
        # 4. Make the prediction
        prediction = model.predict(reshaped_features)
        
        # 5. Get the single probability value from the prediction
        probability = float(prediction[0][0])
        
        # Return the result as JSON
        return jsonify({'risk_probability': probability})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# 4. Run the App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)