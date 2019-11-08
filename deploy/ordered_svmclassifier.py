import jsonlines
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_val_score


def kfold(x, y):
    svm_clf = LinearSVC()
    return cross_val_score(svm_clf, x, y, n_jobs=-1, cv=3, verbose=1)

TRAIN_PATH = 'train_dataset.jsonl'
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
x = count_vec.fit_transform(corpus)

clf = LinearSVC(loss='hinge')
kfopt = kfold(x, opt)
kfcomp = kfold(x, compiler)
print("kfold opt:", kfopt, kfopt.mean())
print("kfold compiler:",kfcomp, kfcomp.mean())
""" 
x_train, x_test, y_train, y_test = train_test_split(x, compiler, test_size=0.2)
print('SVM Fitting with params=', clf.get_params())
clf.fit(x_train, y_train)
print("Predicting")
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
 """