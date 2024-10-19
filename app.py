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
@app.route('/diabetes', methods=['GET', 'POST'])
def Diabetes():
    if request.method == 'POST':
        # Fetch values for the 11 features used in diabetes prediction
        gender = float(request.form['gender'])
        age = float(request.form['age'])
        urea = float(request.form['urea'])
        creatinine = float(request.form['creatinine'])
        haemoglobin = float(request.form['haemoglobin'])
        cholesterol = float(request.form['cholesterol'])
        triglycerides = float(request.form['triglycerides'])
        high_density = float(request.form['high_density'])
        low_density = float(request.form['low_density'])
        very_low_density = float(request.form['very_low_density'])
        bmi = float(request.form['bmi'])

        # Create an array for the diabetes model
        array = np.array([[gender, age, urea, creatinine, haemoglobin, cholesterol, triglycerides, 
                           high_density, low_density, very_low_density, bmi]])
        
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
        
        # Ensure we have the correct number of features (e.g., 12 for hypertension)
        if len(features) != 12:  # Adjust this number based on the model's expected input
            return "Error: The model expects 12 features, but received {}".format(len(features))
        
        array = [np.array(features)]
        
        pred = model_hype.predict(array)
        if pred == 0.0:
            prediction = "High Blood Pressure"
       
        else:
            prediction = "Low Blood Pressure"
        
        return render_template('hpe.html', prediction=prediction)
    
    return render_template('hpe.html')

if __name__ == "__main__":
    app.run(debug=True)
