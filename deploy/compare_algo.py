import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
import jsonlines


TRAIN_PATH = 'deploy/train_dataset.jsonl'


def cross_validation(X, Y, seed):
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('SVM', LinearSVC()))
    models.append(('RFC', RandomForestClassifier()))
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(
            model, X, Y, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


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
    cross_validation(x, opt, 7)

main()
