from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import os
import flask
from flask import Flask, jsonify, request, redirect, url_for, render_template
import json

import pickle
import lime
import lime.lime_text
import lime.explanation
from sklearn.pipeline import make_pipeline

# app = Flask(__name__)
app = flask.Flask(__name__, template_folder='static')
app.config.from_object(__name__)

model = pickle.load(open("models/models.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
pipeline = make_pipeline(vectorizer, model)
labelList = []
proba = 0

def predict(message):
    global labelList   
    explainer = lime.lime_text.LimeTextExplainer(class_names=['Not Sarcasm', 'Sarcasm'])
    vect_msg = vectorizer.transform([message])
    exp = explainer.explain_instance(message, classifier_fn=pipeline.predict_proba, top_labels=1, num_features=10)
    
    labelList = exp.as_list(label=exp.top_labels[0])

    output_file = 'static/sarcasm_explanation.html'.format("sarcasm")
    exp.save_to_file(output_file)

    probs = [model.predict_proba(vect_msg)[0][1]]
    print(probs)
    return probs

@app.route('/')
def main():
    return(flask.render_template('sarcasm.html'))

@app.route('/flat-ui.css')
def send_css():
  return app.send_static_file('flat-ui.css')

@app.route('/handle_data', methods=['POST', 'GET'])
def handle_data():
  if request.method == 'POST':
    msg = request.form['MSG']
    result = predict(msg)
    result = json.dumps(result)
    return result

@app.route('/radar', methods=['POST', 'GET'])
def radar():
  if request.method == 'POST':
    msg = request.form['MSG']
    predict(msg)
    return json.dumps(labelList)

@app.route('/sarcasm')
def sarcasm():
  return app.send_static_file('sarcasm_explanation.html')

if __name__ == '__main__':
  app.run(port='9875')