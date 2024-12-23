from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')  # Render the UI

@app.route('/predict', methods=['POST'])
def predict_price():
    # Extract form inputs
    area = float(request.form['area'])
    bedrooms = float(request.form['bedrooms'])
    bathrooms = float(request.form['bathrooms'])
    stories = float(request.form['stories'])
    parking = float(request.form['parking'])
    mainroad = request.form.get('mainroad', 'no')
    guestroom = request.form.get('guestroom', 'no')
    basement = request.form.get('basement', 'no')
    prefarea = request.form.get('prefarea', 'no')
    hotwaterheating = request.form.get('hotwaterheating', 'no')
    airconditioning = request.form.get('airconditioning', 'no')
    furnishingstatus = request.form['furnishingstatus']

    # Map categorical inputs to one-hot encoded format
    feature_dict = {
        'mainroad_yes': 1 if mainroad == 'yes' else 0,
        'guestroom_yes': 1 if guestroom == 'yes' else 0,
        'basement_yes': 1 if basement == 'yes' else 0,
        'prefarea_yes': 1 if prefarea == 'yes' else 0,
        'hotwaterheating_yes': 1 if hotwaterheating == 'yes' else 0,
        'airconditioning_yes': 1 if airconditioning == 'yes' else 0,
        'furnishingstatus_semi-furnished': 1 if furnishingstatus == 'semi-furnished' else 0,
        'furnishingstatus_furnished': 1 if furnishingstatus == 'furnished' else 0
    }

    # Create input feature array
    numerical_features = np.array([area, bedrooms, bathrooms, stories, parking]).reshape(1, -1)
    scaled_numerical_features = scaler.transform(numerical_features)

    # Combine numerical and categorical features
    input_features = np.hstack([
        scaled_numerical_features,
        np.array(list(feature_dict.values())).reshape(1, -1)
    ])

    # Predict using the trained model
    predicted_price = model.predict(input_features)

    return render_template('index.html', predicted_price=f"Predicted Price: ₹{predicted_price[0]:,.2f}")

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
