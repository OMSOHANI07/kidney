import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
with open('kidney.pkl', 'rb') as f:
    kidney_model = pickle.load(f)

with open('scaling.pkl', 'rb') as f:
    scalar = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    # Convert input data to array and reshape
    input_data = np.array(list(data.values())).reshape(1, -1)
    # Scale the input data
    scaled_data = scalar.transform(input_data)
    # Predict using the model
    prediction = kidney_model.predict(scaled_data)
    # Convert prediction to JSON serializable format
    output = {'prediction': prediction.tolist()}  # Convert to list for JSON serialization
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
