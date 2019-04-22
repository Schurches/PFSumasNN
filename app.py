# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:41:58 2019

@author: Steven
"""

from flask import Flask, request, jsonify
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib 
import pandas as pd
import keras
from keras.models import load_model
import pickle
import sys
import tensorflow as tf
global graph,model


# Flask
app = Flask(__name__)

# Load model
graph = tf.get_default_graph()
model = load_model('NewPlayerNeuralNet.h5')
sc = joblib.load('scaler.joblib') 

# Handle requests
@app.route('/updatePlayerIntensities', methods=['POST'])
def predictAdjustment():
    return "test"
    

@app.route('/onNewPlayer', methods=['POST'])
def predict():
    playerData = request.get_json(force=True)
    d = {'edad': [playerData['edad']], 'escuela':[playerData['escuela']], 'genero': [playerData['genero']], 'grado': [playerData['grado']]}
    df = pd.DataFrame(data=d)
    X = sc.transform(df)
    with graph.as_default():
        intensities = model.predict(X)
        print(intensities,file=sys.stderr)
        LO0 = "\"LOIN0\":["+str(intensities[0,0])+",1,0],"
        LO1 = "\"LOIN1\":["+str(intensities[0,1])+",1,0],"
        LO2 = "\"LOIN2\":["+str(intensities[0,2])+",1,0],"
        LO3 = "\"LOIN3\":["+str(intensities[0,3])+",1,0],"
        LO4 = "\"LOIN4\":["+str(intensities[0,4])+",1,0]"
        jsonFormatIntensities = "{"+LO0+LO1+LO2+LO3+LO4+"}"
        return jsonFormatIntensities


@app.route('/onNewPlayer', methods=['GET'])
def onWelcome():
    return 'hola'


if __name__ == '__server__':
    app.run(host="localhost", port=5000)
