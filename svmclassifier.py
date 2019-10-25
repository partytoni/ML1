import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

SEED = 123


def main():
    dataframe = pd.read_pickle('matrix_zero_ones.pkl')
    x = dataframe.drop('opt', axis=1)
    x = x.drop('compiler', axis=1)

    opt_target = dataframe['opt']
    compiler_target = dataframe['compiler']

    print("CLASSIFICATION REPORT FOR OPTIMIZATION PREDICTION:")
    print(prediction(x, opt_target))
    print("CLASSIFICATION REPORT FOR COMPILER PREDICTION:")
    print(prediction(x, compiler_target))


def prediction(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    clf = SVC(gamma='auto', kernel='rbf', C=1, verbose=True)
    print('SVM Fitting with params=', clf.get_params())
    clf.fit(x_train, y_train)
    print('Predicting.')
    y_pred = clf.predict(x_test)
    return classification_report(y_test, y_pred)


main()
