import jsonlines
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

TRAIN_PATH = 'data/train_dataset.jsonl'
TEST_PATH = 'data/test_dataset_blind.jsonl'


def main():
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
    print("Accuracy for opt:", predict(x, opt))
    print("Accuracy for compiler:", predict(x, compiler))


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
