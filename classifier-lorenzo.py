import jsonlines
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
N_LINES = 8000

def read_file():
    matrix = []
    i = 0
    with jsonlines.open('train_dataset.jsonl') as reader:
        for line in reader:
            program = list()
            for instruction in line['instructions']:
                program.append(instruction.split(" ")[0])
            matrix.append([" ".join(program), line['opt']])
            if i == N_LINES:
                break
            else:
                i+=1
    return matrix


def svm_model(x_train,y_train,x_test):
    print("Starting SVM Model")

    svm_model = svm.SVC(kernel="linear")
    svm_model.fit(x_train, y_train)

    print("SVM Results")
    y_pred = svm_model.predict(x_test)
    print(metrics.accuracy_score(y_test,y_pred))

def gaussian_bayes(x_train,y_train,x_test):
    print("Starting Gaussian Model")
    gaussian_model = GaussianNB()
    gaussian_model.fit(x_train,y_train)
    print("Gaussian Results")
    y_pred = gaussian_model.predict(x_test)
    print(metrics.accuracy_score(y_test,y_pred))

def decision_tree(x_train,y_train,x_test):
    print("Starting Decision Tree Model")
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(x_train,y_train)
    print("Decision Tree Results")
    y_pred = decision_tree_model.predict(x_test)
    print(metrics.accuracy_score(y_test,y_pred))


matrix = read_file()
vectorizer = CountVectorizer()
vector = np.array(matrix)

matrix_flat = np.squeeze(np.asarray(vector[:,0]))
print("Matrix flat done!")
classes = np.squeeze(np.asarray(vector[:,1]))

for i in range(len(classes)):
    if classes[i] == 'L':
        classes[i] = 0
    else:
        classes[i] = 1
            
count_matrix = vectorizer.fit_transform(matrix_flat).todense()

print("Count matrix done!")

x_train, x_test, y_train, y_test = train_test_split(count_matrix, classes, test_size=0.3)
print("Split text done")

print("Train ", x_train.shape[0], " Test : ", x_test.shape[0])

decision_tree(x_train,y_train,x_test)

gaussian_bayes(x_train,y_train,x_test)

svm_model(x_train,y_train,x_test)
