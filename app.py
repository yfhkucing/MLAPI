from optparse import Values
from urllib import request
from flask import Flask,request
import joblib
import json
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load("randomforest_joblib.pkl")

@app.route('/',methods=['POST'])

def predict():
    event = json.loads(request.data)
    values = event['values']
    values = list(map(np.float,values))
    pre = np.array(values)
    pre = pre.reshape(1,-1)
    res = model.predict(pre)
    print(res)
    return str(res[0])

if __name__=='__main__':
    app.run(debug=True)


