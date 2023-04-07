# flask, scikit-learn, pandas, pickle-mixin
import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
bangaloreData = pd.read_csv('../Csv/Cleaned_Data.csv')
bangalorePipe = pickle.load(open("../Pickle/RidgeModel.pkl", 'rb'))
puneData = pd.read_csv('../Csv/Pune_House_Data.csv')
punePipe = pickle.load(open("../Pickle/PuneRidgeModel.pkl", 'rb'))


@app.route('/')
def index():
    return render_template('Index.html')


@app.route('/bangalore')
def bangalore():
    bangalore_locations = sorted(bangaloreData['location'].unique())
    return render_template('Bangalore.html', locations=bangalore_locations)


@app.route('/pune')
def pune():
    pune_locations = sorted(puneData['location'].unique())
    return render_template('pune.html', punelocations=pune_locations)


@app.route('/bangalore_predict', methods=['POST'])
def bangalore_predict():
    locations = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    totalsqft = request.form.get('sqft')

    print(locations, bhk, bath, totalsqft)
    dat = pd.DataFrame([[locations, totalsqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    bangalore_prediction = bangalorePipe.predict(dat)[0] * 1e5

    return str(np.round(bangalore_prediction, 2))


@app.route('/punePredict', methods=['POST'])
def pune_predict():
    locations = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    totalsqft = request.form.get('sqft')

    print(locations, bhk, bath, totalsqft)
    dat = pd.DataFrame([[locations, totalsqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    pune_prediction = punePipe.predict(dat)[0] * 1e5

    return str(np.round(pune_prediction, 2))


if __name__ == "__main__":
    app.run(debug=False, port=5000)
