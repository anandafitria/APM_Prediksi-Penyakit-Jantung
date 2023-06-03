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

        umur = float(request.form['Age'])
        jenisKelamin = float(request.form['Sex'])
        tipeSakitDada = float(request.form['ChestPainType'])
        kadarKolesterol = float(request.form['Cholesterol'])
        tekananDarah = float(request.form['RestingBP'])
        detakJantung = float(request.form['MaxHR'])
        rasaSakitDiDada = float(request.form['ExerciseAngina'])
        Oldpeak = float(request.form['Oldpeak'])
        ST_Slope = float(request.form['ST_Slope'])
        riwayatJantung = float(request.form['HeartDisease'])

        datas = np.array((umur,jenisKelamin,tipeSakitDada, kadarKolesterol, tekananDarah, detakJantung, rasaSakitDiDada, Oldpeak, ST_Slope, riwayatJantung))
        datas = np.reshape(datas, (1, -1))

        isJantung = model.predict(datas)

        return render_template('hasil.html', finalData=isJantung)
    else:
        return render_template('index.html')
        
if __name__ == "__main__":
    app.run(debug=True)