import jsonlines
from nltk import ngrams
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB 

from numpy import matlib, mat

TRAIN_PATH = 'data/train_dataset.jsonl'
corpus = []
opt = []
compiler = []
with jsonlines.open(TRAIN_PATH) as reader:
    for elem in reader:
        riga = ""
        for instruction in elem['instructions']:
            riga += instruction + " "
        corpus.append(riga)
        opt.append(elem['opt'])
        compiler.append(elem['compiler'])

count_vec = TfidfVectorizer(ngram_range=(2, 2))
x = count_vec.fit_transform(corpus)
""" 
x_train, x_test, y_train, y_test = train_test_split(x, opt,
                                                    test_size=0.2, random_state=15)
clf = MultinomialNB()
print('MultinomialNB Fitting with params=', clf.get_params())
clf.fit(x_train, y_train)
print("Predicting")
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
 """