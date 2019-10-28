import jsonlines
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import collections

TRAIN_PATH = 'data/train_dataset.jsonl'
TEST_PATH = 'data/test_dataset_blind.jsonl'


def main():
    print("Doing train corpus.")
    corpus_train, opt, compiler = read_json(TRAIN_PATH)
    print("Doing test corpus.")
    corpus_test = read_json(TEST_PATH)
    corpus = corpus_train+corpus_test
    count_vec = CountVectorizer(ngram_range=(2, 2))
    print("Doing fit transform.")
    x = count_vec.fit_transform(corpus)
    opt_pred=predict(x, opt)
    compiler_pred = predict(x, compiler)

    print("Writing to file results.csv")
    write_results_to_csv(opt_pred, compiler_pred)

    


def write_results_to_csv(opt_pred,compiler_pred):
    print(collections.Counter(opt_pred))
    print(collections.Counter(compiler_pred))
    import os
    
    with open(os.path.abspath(__file__)+'random_forest_results.csv','w') as file:
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
                first = instruction.split(" ")[0]
                riga += first + " "
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
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=123)
    print('Doing RandomForest fitting.')
    clf.fit(x_train, y)
    print("Predicting")
    y_pred = clf.predict(x_test)
    return y_pred

main()

