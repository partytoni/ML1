import jsonlines
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import numpy as np
import collections

TRAIN_PATH = 'train_dataset.jsonl'
TEST_PATH = 'test_dataset_blind.jsonl'


def main():
    print("Doing train corpus.")
    corpus_train, opt, compiler = read_json(TRAIN_PATH)
    print("Doing test corpus.")
    corpus_test = read_json(TEST_PATH)
    corpus = corpus_train+corpus_test
    tf_vec = TfidfVectorizer(ngram_range=(2, 3))
    print("Doing fit transform.")
    x = tf_vec.fit_transform(corpus)
    opt_pred=predict(x, opt)
    compiler_pred = predict(x, compiler)

    print("Writing to file results.csv")
    write_results_to_csv(opt_pred, compiler_pred)

def write_results_to_csv(opt_pred,compiler_pred):
    print(collections.Counter(opt_pred))
    print(collections.Counter(compiler_pred))
    import os
    
    path=os.path.dirname(__file__)

    with open('/home/antonio/svm_results.csv','w') as file:
        for i in range(len(opt_pred)):
            line = opt_pred[i]+","+compiler_pred[i]+"\n"
            file.write(line)

def read_json(path):
    corpus = []
    opt = []
    compiler = []
    with jsonlines.open(path) as reader:
        for elem in reader:
            riga = ""
            for instruction in elem['instructions']:
                riga += instruction + " "
            corpus.append(riga)
            if path==TRAIN_PATH:
                opt.append(elem['opt'])
                compiler.append(elem['compiler'])
    if path==TRAIN_PATH:
        return corpus, opt, compiler
    else:
        return corpus

def predict(x, y):
    x_train = x[:30000]
    x_test = x[-3000:]
    clf = LinearSVC(random_state=123)
    print('Doing LinearSVC fitting.')
    clf.fit(x_train, y)
    print("Predicting")
    y_pred = clf.predict(x_test)
    return y_pred

main()