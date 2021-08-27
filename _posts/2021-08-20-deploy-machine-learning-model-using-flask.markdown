---
layout: post
title: "Deploy Machine Learning Model using Flask"
date: 2021-08-20 09:20
categories: Cloud-Computing
---

In this post I explain how to run a machine learning model on a cloud computer and access its predictions via an API.

To do this goal I write a simple API interface using Flask in Python. I have trained a linear regression model on artificial data and saved the model object using Pickle. The code below loads the model and runs an API that receives POST requests for the model predictions.

```python
from flask import Flask, jsonify, abort,request
import numpy as np
import pickle

app = Flask(__name__)

lr=pickle.load(open('lr_model.pkl','rb'))

@app.route('/')
def index():
    return "Linear Regression model API."

@app.route('/model',methods=['POST'])
def model():
    if not request.json or not 'data' in request.json:
        abort(400)
    data=request.json['data']
    x=np.array(data)
    pred=lr.predict(x)

    return jsonify({'prediction': pred.tolist()}), 201

if __name__ == '__main__':
    app.run(debug=True,port=5000,host='0.0.0.0')


```
