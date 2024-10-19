from flask import Flask, request, render_template
import numpy as np
import joblib

# Load pre-trained models
model_dia = joblib.load("ray.joblib")
model_hype = joblib.load("hgbc_model_for_hypertension.joblib")

app = Flask(__name__, template_folder='templete')

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Diabetes prediction route
@app.route('/diabetes', methods=['GET', 'POST'])
def Diabetes():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        # Create an array for the diabetes model
        array = np.array(features).reshape(1,-1)
        
        # Make prediction
        pred = model_dia.predict(array)
        
        if pred == 0.0:
            prediction = "Diabetic"
        elif pred == 1.0:
            prediction = 'Non-Diabetic'
        else:
            prediction = "Pre-Diabetic"
        
        return render_template('index.html', prediction=prediction)
    
    return render_template('index.html')


# Hypertension prediction route
@app.route('/hypertension', methods=['GET', 'POST'])
def Hypertension():
    if request.method == 'POST':
        features = [float(xx) for xx in request.form.values()]
        
        array = np.array(features).reshape(1,-1)
        
        pred = model_hype.predict(array)
        if pred == 0.0:
            prediction = "High Blood Pressure"
       
        else:
            prediction = "Low Blood Pressure"
        
        return render_template('hpe.html', prediction=prediction)
    
    return render_template('hpe.html')

if __name__ == "__main__":
    app.run(debug=True)
