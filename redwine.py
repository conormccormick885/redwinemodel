from flask import Flask
from flask_restplus import Resource, Api
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

app = Flask(__name__)
api = Api(app)

model = joblib.load('redwinemodel.joblib')
@app.route('/api/quality', methods=['GET'])
# http://[ip_addr:5000]/api/quality?volacid=0.7&citacid=0.5&chl=0.1&sul=0.7&alc=9.5
def quality():
    volacid = request.args.get('volacid', type = float)
    citacid = request.args.get('citacid', type = float)
    chl = request.args.get('chl', type = float)
    sul = request.args.get('sul', type = float)
    alc = request.args.get('alc', type = float)
    # array passed to the model has the same order as cols in the orignal dataset
    # result rounded to nearest integer
    qual = model.predict(np.array[volacid, citacid, chl, sul, alc])
    return {'predicted quality': qual[0].round()}

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=True)
