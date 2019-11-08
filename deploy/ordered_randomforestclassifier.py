import jsonlines
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np

TRAIN_PATH = 'train_dataset.jsonl'
TEST_PATH = 'test_dataset_blind.jsonl'


def kfold(x, y):
    random_clf = RandomForestClassifier(n_jobs=1,n_estimators=50)
    return cross_val_score(random_clf, x, y, n_jobs=-1, cv=3, verbose=1)


def main():
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

    count_vec = TfidfVectorizer(ngram_range=(2, 2))
    x = count_vec.fit_transform(corpus)
    # print("Accuracy for opt:", predict(x, opt))
    # print("Accuracy for compiler:", predict(x, compiler))
    kf = kfold(x, opt)
    print("KFOLD opt:", kf)
    print("KFOLD opt mean:", kf.mean())
    kf = kfold(x, compiler)
    print("KFOLD compiler:", kf)
    print("KFOLD compiler mean:", kf.mean())

def predict(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2, random_state=15)
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    print('RandomForest Fitting with params=', clf.get_params())
    clf.fit(x_train, y_train)
    print("Predicting")
    y_pred = clf.predict(x_test)
    return accuracy_score(y_test, y_pred)

main()
