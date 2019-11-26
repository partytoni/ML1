import jsonlines
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


TRAIN_PATH = 'train_dataset.jsonl'


def main():
    print("Reading file")
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

    count_vec = TfidfVectorizer(ngram_range=(2, 3))
    print("Doing fit transfrom")
    x = count_vec.fit_transform(corpus)
    print("Accuracy for opt:", predict(x, opt))
    print("Accuracy for compiler:", predict(x, compiler))


def predict(x, y):
    clf = LinearSVC()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print('SVM Fitting with params=', clf.get_params())
    clf.fit(x_train, y_train)
    print("Predicting")
    y_pred = clf.predict(x_test)
    return accuracy_score(y_test, y_pred)


main()
