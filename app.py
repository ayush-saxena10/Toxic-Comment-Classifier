from flask import Flask, render_template, url_for, request, redirect, jsonify
from flask.templating import render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
import numpy as np
import re
import string
import nltk
from sklearn.calibration import CalibratedClassifierCV
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline 

app = Flask(__name__)

# Loading the models for each category
with open(r"D:/ML Projects/Toxic Comment Classifier/model_toxic.pickle", "rb") as f:
    model_toxic = pickle.load(f)

with open(r"D:/ML Projects/Toxic Comment Classifier/model_severe.pickle", "rb") as f:
    model_severe = pickle.load(f)

with open(r"D:/ML Projects/Toxic Comment Classifier/model_obscene.pickle", "rb") as f:
    model_obscene = pickle.load(f)

with open(r"D:/ML Projects/Toxic Comment Classifier/model_threat.pickle", "rb") as f:
    model_threat = pickle.load(f)

with open(r"D:/ML Projects/Toxic Comment Classifier/model_insult.pickle", "rb") as f:
    model_insult = pickle.load(f)

with open(r"D:/ML Projects/Toxic Comment Classifier/model_ide_hate.pickle", "rb") as f:
    model_ide_hate = pickle.load(f)

# Loading the tfidf files for each category
with open(r"D:/ML Projects/Toxic Comment Classifier/tfidf_toxic.pickle", "rb") as f:
    tfidf_toxic = pickle.load(f)

with open(r"D:/ML Projects/Toxic Comment Classifier/tfidf_severe.pickle", "rb") as f:
    tfidf_severe = pickle.load(f)

with open(r"D:/ML Projects/Toxic Comment Classifier/tfidf_obscene.pickle", "rb") as f:
    tfidf_obscene = pickle.load(f)

with open(r"D:/ML Projects/Toxic Comment Classifier/tfidf_threat.pickle", "rb") as f:
    tfidf_threat = pickle.load(f)

with open(r"D:/ML Projects/Toxic Comment Classifier/tfidf_insult.pickle", "rb") as f:
    tfidf_insult = pickle.load(f)

with open(r"D:/ML Projects/Toxic Comment Classifier/tfidf_ide_hate.pickle", "rb") as f:
    tfidf_ide_hate = pickle.load(f)

def comment_cleaner(text):
    
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens_without_sw = [w for w in tokens if not w.lower() in stop_words]
    filtered_sent = (" ").join(tokens_without_sw)
    
    filtered_sent = re.sub('\w*\d\w*', ' ', filtered_sent)
    filtered_sent = re.sub('[%s]' % re.escape(string.punctuation), ' ', filtered_sent.lower())
    filtered_sent = re.sub("\n", ' ', filtered_sent)
    filtered_sent = re.sub(r'[^\x00-\x7f]', r' ', filtered_sent)

    return filtered_sent

def prediction(text):

    text = comment_cleaner(text) 
    text = [text]
    vector_toxic = tfidf_toxic.transform(text)
    vector_severe = tfidf_severe.transform(text)
    vector_obscene = tfidf_obscene.transform(text)
    vector_insult = tfidf_insult.transform(text)
    vector_threat = tfidf_threat.transform(text)
    vector_ide_hate = tfidf_ide_hate.transform(text)

    list_category = ['toxic',
                     'severe_toxic',
                     'obscene',
                     'threat',
                     'insult',
                     'identity_hate']

    list_pred = [model_toxic.predict(vector_toxic)[0],
                 model_severe.predict(vector_severe)[0],
                 model_obscene.predict(vector_obscene)[0],
                 model_threat.predict(vector_threat)[0],
                 model_insult.predict(vector_insult)[0],
                 model_ide_hate.predict(vector_ide_hate)[0]]

    list_proba = ["{:.2f}".format(model_toxic.predict_proba(vector_toxic)[0][1]),
                  "{:.2f}".format(model_severe.predict_proba(vector_severe)[0][1]),
                  "{:.2f}".format(model_obscene.predict_proba(vector_obscene)[0][1]),
                  "{:.2f}".format(model_threat.predict_proba(vector_threat)[0][1]),
                  "{:.2f}".format(model_insult.predict_proba(vector_insult)[0][1]),
                  "{:.2f}".format(model_ide_hate.predict_proba(vector_ide_hate)[0][1])]

    dic = {'category': list_pred,
           'probability': list_proba}

    df = pd.DataFrame(dic, index=list_category)
    
    return df

@app.route("/")
def home():
    return render_template("toxic_home.html")

@app.route("/toxic classifier", methods=['POST'])
def predict():    
    
    input_text = request.form['text']
    df = prediction(input_text)

    # toxic_category = df['category'].to_dict()['toxic']
    # prob_toxic = df['probability'].to_dict()['toxic']
    
    return render_template('toxic_home.html',post=df)

if __name__=='__main__':
    app.run(debug=True)