from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Load database
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

# Load model
with open("models/svc_model.pkl", "rb") as f:
    svc = pickle.load(f)

app = Flask(__name__)

# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])
    
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]
    
    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]
    
    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]
    
    wrkout = workout[workout['disease'] == dis]['workout']
    
    return desc, pre, med, die, wrkout

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        user_symptoms = [s.strip() for s in symptoms.split(',')]
        user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
        predicted_disease = get_predicted_value(user_symptoms)

        desc, pre, med, die, wrkout = helper(predicted_disease)
        my_pre = [i for i in pre[0]]
        return render_template('index.html', predicted_disease=predicted_disease, dis_desc=desc, dis_pre=my_pre, dis_med=med, dis_diet=die, dis_wrkout=wrkout)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/developer')
def developer():
    return render_template('developer.html')

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
