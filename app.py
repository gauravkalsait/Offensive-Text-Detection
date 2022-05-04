#scikit-learn==0.22.1
from flask import Flask,request, url_for, redirect, render_template
import joblib
import json
import numpy as np
from sklearn import *
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax

app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained(f"cardiffnlp/twitter-roberta-base-{'sentiment'}")
model = AutoModelForSequenceClassification.from_pretrained(f"cardiffnlp/twitter-roberta-base-{'sentiment'}")

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict' , methods=['POST'])
def prediction():
    data = request.form['first_name']
    encoded_input = tokenizer(data, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    prediction = ranking[::-1][0]

    if prediction == 0:
        return render_template('index.html',pred='Negative Sentiment')
    else:
        return render_template('index.html',pred='Positive Sentiment')

if __name__ == '__main__':
    app.run(debug=True)