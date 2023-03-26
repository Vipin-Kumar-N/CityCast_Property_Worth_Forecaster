# flask, scikit-learn, pandas, pickle-mixin
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
data = pd.read_csv('../Csv/Cleaned_Data.csv')

@app.route('/')
def index():

    locations = sorted(data['location'].unique())
    return render_template('Index.html', locations=locations)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
