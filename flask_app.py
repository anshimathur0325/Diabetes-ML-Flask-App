import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import os
from flask import send_from_directory

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    str=prediction[0]
    
    if prediction[0]==1:
      str="You may have diabetes."
    else:
      str="You probably do not have diabetes."
    
    return render_template("index.html", prediction_text=str)

if __name__ == "__main__":
    app.run(debug=True)
