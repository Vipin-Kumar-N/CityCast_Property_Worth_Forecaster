# flask, scikit-learn, pandas, pickle-mixin
import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)
data = pd.read_csv('../Csv/Cleaned_Data.csv')
pipe = pickle.load(open("../Pickle/RidgeModel.pkl", 'rb'))


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('Index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    locations = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    totalsqft = request.form.get('sqft')

    print(locations, bhk, bath, totalsqft)
    dat = pd.DataFrame([[locations, totalsqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(dat)[0] * 1e5

    return str(np.round(prediction, 2))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
