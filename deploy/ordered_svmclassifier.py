import jsonlines
from nltk import ngrams
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import *

from numpy import matlib, mat

TRAIN_PATH = 'data/train_dataset.jsonl'
corpus = []
opt = []
compiler = []
with jsonlines.open(TRAIN_PATH) as reader:
    for elem in reader:
        riga = ""
        for instruction in elem['instructions']:
            first = instruction.split(" ")[0]
            riga += first + " "
        corpus.append(riga)
        opt.append(elem['opt'])
        compiler.append(elem['compiler'])

count_vec = CountVectorizer(ngram_range=(2, 2))
x = count_vec.fit_transform(corpus)
x_train, x_test, y_train, y_test = train_test_split(x, opt,
                                                    test_size=0.2, random_state=15)
clf = SVC(gamma='scale', kernel='rbf', C=1)
print('SVM Fitting with params=', clf.get_params())
clf.fit(x_train, y_train)
print("Predicting")
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))