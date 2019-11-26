import jsonlines
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
import numpy as np


TRAIN_PATH = 'train_dataset.jsonl'


def kfold(x, y, name, classifier):
    # print('Doing kfold for classifier', name
    scoring = {'acc': 'accuracy', 'f1': 'f1_macro',
               'prec': 'precision_macro', 'rec': 'recall_macro'}
    cvs = cross_validate(classifier, x, y, scoring=scoring,  n_jobs=-1, cv=3)
    print("---------------", name, "---------------")
    for el in cvs.keys():
        print(el, " average: ", cvs[el].mean())


def main():
    corpus, opt, compiler = read_file()
    count_vec = TfidfVectorizer(ngram_range=(2, 3))
    print('Doing fit transform.')
    x = count_vec.fit_transform(corpus)
    clf1 = MultinomialNB()
    clf2 = RandomForestClassifier(n_estimators=30, n_jobs = -1)
    clf3 = LinearSVC()
    clf = [('multinomial', clf1),
           ('random forest', clf2),
           ('linear', clf3)]

    for name, classifier in clf:
        kfold(x, opt, name + 'opt', classifier)
        kfold(x, compiler, name + 'compiler', classifier)


def read_file():
    print('Reading file.')
    corpus = []
    opt = []
    compiler = []
    with jsonlines.open(TRAIN_PATH) as reader:
        for elem in reader:
            riga = ""
            for instruction in elem['instructions']:
                first = instruction
                riga += first + " "
            corpus.append(riga)
            opt.append(elem['opt'])
            compiler.append(elem['compiler'])
    return corpus, opt, compiler


main()
