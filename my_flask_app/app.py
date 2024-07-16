from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the label encoder for 'Month'
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        year = float(request.form['year'])
        month = request.form['month']
        week = float(request.form['week'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind = float(request.form['wind'])
        solarradiation = float(request.form['solarradiation'])
        dew = float(request.form['dew'])

        # Encode 'Month'
        month_encoded = label_encoder.transform([month])[0]

        # Preprocess the input
        input_features = np.array([[year, month_encoded, week, rainfall, temperature, humidity, wind, solarradiation, dew]])
        
        # Log transformation
        input_features = np.log1p(input_features)
        
        # Scaling
        input_features_scaled = scaler.transform(input_features)

        # Make prediction
        prediction = model.predict(input_features_scaled)
        prediction = np.expm1(prediction)  # Inverse log transformation
        prediction = float(prediction[0])  # Convert NumPy array to Python float

        return render_template('index.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
