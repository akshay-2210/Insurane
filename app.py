from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn

app=Flask(__name__)
model = pickle.load(open('insurance_logistic_reg.pkl','rb'))
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = int(request.form['Year'])
        Gender = request.form['Gender']
        if (Gender == 'Male'):
            Gender = 1
        else:
            Gender = 0
        BMI = float(request.form['BMI'])
        Smoker = request.form['Smoker']
        if (Smoker == 'Yes'):
            Smoker = 1
        else:
            Smoker = 0
        Childs = int(request.form['childs'])

        prediction=model.predict([[Year, Gender, BMI, Smoker, Childs]])
        probability = float(np.round(model.predict_proba([[Year, Gender, BMI, Smoker, Childs]])[:,1],4))*100
        
        return render_template('index.html',prediction_text="Probablity of buying an insurance is {} %".format(probability))
        
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)