from flask import Flask, request, jsonify
import numpy as np
import joblib  # Assuming you save your model with joblib

app = Flask(__name__)

# Load your model (replace 'model.pkl' with your actual model file)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)  # Reshape as needed
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
