import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

SEED = 123


def main():
    dataframe = pd.read_pickle('data/matrix_zero_ones.pkl')
    X = dataframe.drop('opt', axis=1)
    X = X.drop('compiler', axis=1)

    opt_target = dataframe['opt']
    compiler_target = dataframe['compiler']

    print("CLASSIFICATION REPORT FOR OPTIMIZATION PREDICTION:")
    print(prediction(X, opt_target))
    print("CLASSIFICATION REPORT FOR COMPILER PREDICTION:")
    print(prediction(X, compiler_target))


def prediction(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.20)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    print('RandomForest Fitting with params=', clf.get_params())
    clf.fit(X_train, y_train)
    print("Predicting")
    y_pred = clf.predict(X_test)
    return classification_report(y_test, y_pred)


main()
