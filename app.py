from flask import Flask, request, render_template
#import pickle as pkl
import numpy as np
import joblib

model_dia= joblib.load("ray.joblib")
model_hype=joblib.load("hgbc_model_for_hypertension.joblib")

app = Flask(__name__, template_folder='templete')
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def Diabetes():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        array = [np.array(features)]
        
        pred = model_dia.predict(array)
        if pred == 0.0:
            prediction = "Diabetic"
        elif pred == 1.0:
            prediction = 'Non-Diabetic'
        else:
            prediction = "Pre-Diabetic"
        
        # Prepare the data for charting
        colums = ["Gender","Age","Urea","Creatinine","Haemoglobin","Cholestral","Triglycerides","High-Deansity","Low-Density","Very-low Density","BMI"]
        
        return render_template('index.html', prediction=prediction)
    
    # Render the template with the prediction and user input values
    return render_template('index.html')
@app.route('/hypertension',methods=['GET','POST'])
def Hypertension():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        array = [np.array(features)]
        
        pred = model_hype.predict(array)
        if pred == 0.0:
            prediction = "Diabetic"
        elif pred == 1.0:
            prediction = 'Non-Diabetic'
        else:
            prediction = "Pre-Diabetic"
        
        # Prepare the data for charting
        
        return render_template('hpe.html', prediction=prediction)
if __name__ == "__main__":
    app.run(debug=True)
