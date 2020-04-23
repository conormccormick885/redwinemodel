from flask import Flask, request
from flask_restplus import Resource, Api
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

app = Flask(__name__)
api = Api(app)

model = joblib.load('redwinemodel.joblib')
@app.route('/api/quality', methods=['GET'])
# http://[ip_addr:5000]/api/quality?volacid=0.7&citacid=0.5&chl=0.1&sul=0.7&alc=9.5

def get():
    print("start of get()")
    volacid = request.args.get('volacid', type = float)
    citacid = request.args.get('citacid', type = float)
    chl = request.args.get('chl', type = float)
    sul = request.args.get('sul', type = float)
    alc = request.args.get('alc', type = float)
    print("volacid=" + str(volacid) + ", citacid=" + str(citacid) + ", chl=" + str(chl) + ", sul=" + str(sul) + ", alc=" + str(alc))
    # array passed to the model has the same order as cols in the orignal dataset
    # result rounded to nearest integer
    qual = model.predict(np.array([0.7, 0.5, 0.1, 0.1, 9.5]).reshape(-1, 1))
    #qual = model.predict(np.array([volacid, citacid, chl, sul, alc], 5).reshape(-1, 1))
    print("model returned " + qual)
    return {'predicted quality': qual}

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
