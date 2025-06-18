from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

app = Flask(__name__)

# Load model once
try:
    with open("cvd_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("cvd_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    model = None
    scaler = None
    print(f"❌ Error loading model or scaler: {e}")

@app.route('/')
def index():
    return render_template("index.html", prediction=None)

@app.post('/predict')
def make_prediction():
    if model is None:
        return render_template("index.html", prediction="Error: Model not loaded.")

    try:
        form_data = request.form

        # Extract and process input values from the form
        features = [
            float(form_data['age']),
            int(form_data['gender']),
            float(form_data['chestpain']),
            float(form_data['restingBP']),
            float(form_data['serumcholestrol']),
            int(form_data['fastingbloodsugar']),
            int(form_data['restingrelectro']),
            float(form_data['maxheartrate']),
            int(form_data['exerciseangia']),
            float(form_data['oldpeak']),
            int(form_data['slope']),
            int(form_data['noofmajorvessels'])
        ]

        input_data = np.array([features])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        # Convert prediction to human readable result
        result = "High Risk of Heart Failure!" if prediction == 1 else "Low Risk of Heart Failure!"

    except Exception as e:
        result = f"❌ Error: {str(e)}"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
