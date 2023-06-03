from flask import Flask, render_template, request, redirect, url_for
import pickle 
import sklearn
import numpy as np
import pandas as pd 

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['POST', 'GET'])

def index():
    if request.method == 'POST':

        with open('knn_pickle', 'rb') as r:
            model = pickle.load(r)

        Age = float(request.form['Age'])
        Sex = float(request.form['Sex'])
        ChestPainType = float(request.form['ChestPainType'])
        RestingBP = float(request.form['RestingBP'])
        Cholesterol = float(request.form['Cholesterol'])
        FastingBS = float(request.form['FastingBS'])
        RestingECG = float(request.form['RestingECG'])
        MaxHR = float(request.form['MaxHR'])
        ExerciseAngina = float(request.form['ExerciseAngina'])
        Oldpeak = float(request.form['Oldpeak'])
        ST_Slope = float(request.form['ST_Slope'])

        datas = np.array((Age, Sex, ChestPainType, RestingBP,Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope))
        datas = np.reshape(datas, (1, -1))

        isJantung = model.predict(datas)

        return render_template('hasil.html', finalData=isJantung)
    else:
        return render_template('index.html')
        
if __name__ == "__main__":
    app.run(debug=True)