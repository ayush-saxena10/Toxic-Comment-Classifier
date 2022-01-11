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
model_toxic_f = open('model_toxic.pickle', 'rb')
model_toxic = pickle.load(model_toxic_f)
model_toxic_f.close()

model_severe_f = open('model_severe.pickle', 'rb')
model_severe = pickle.load(model_severe_f)
model_severe_f.close()

model_obscene_f = open('model_obscene.pickle', 'rb')
model_obscene = pickle.load(model_obscene_f)
model_obscene_f.close()

model_threat_f = open('model_threat.pickle', 'rb')
model_threat = pickle.load(model_threat_f)
model_threat_f.close()

model_insult_f = open('model_insult.pickle', 'rb')
model_insult = pickle.load(model_insult_f)
model_insult_f.close()

model_ide_hate_f = open('model_ide_hate.pickle', 'rb')
model_ide_hate = pickle.load(model_ide_hate_f)
model_ide_hate_f.close()

# Loading the tfidf files for each category
tfidf_toxic_f = open('tfidf_toxic.pickle', 'rb')
tfidf_toxic = pickle.load(tfidf_toxic_f)
tfidf_toxic_f.close()

tfidf_severe_f = open('tfidf_severe.pickle', 'rb')
tfidf_severe = pickle.load(tfidf_severe_f)
tfidf_severe_f.close()

tfidf_obscene_f = open('tfidf_obscene.pickle', 'rb')
tfidf_obscene = pickle.load(tfidf_obscene_f)
tfidf_obscene_f.close()

tfidf_threat_f = open('tfidf_threat.pickle', 'rb')
tfidf_threat = pickle.load(tfidf_threat_f)
tfidf_threat_f.close()

tfidf_insult_f = open('tfidf_insult.pickle', 'rb')
tfidf_insult = pickle.load(tfidf_insult_f)
tfidf_insult_f.close()

tfidf_ide_hate_f = open('tfidf_ide_hate.pickle', 'rb')
tfidf_ide_hate = pickle.load(tfidf_ide_hate_f)
tfidf_ide_hate_f.close()

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

@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")

@app.route("/toxic classifier", methods=['GET', 'POST'])
def predict():
    if request.method=='POST':
        input_text = request.form['text']
        df = prediction(input_text)
        
        return render_template('toxic_home.html', df=df)
    else:
        return render_template('toxic_home.html')
        
if __name__=='__main__':
    app.run(debug=True)