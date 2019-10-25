import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

NUM = 30000

dataframe = pd.read_pickle('matrix2.pkl')
first_n = dataframe.head(NUM)

X = first_n.drop('opt', axis=1)
X = X.drop('compiler', axis=1)

y1 = first_n['opt']
y2 = first_n['compiler']

y = pd.concat([y1, y2], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.20)
svclassifier = SVC(gamma='scale',kernel='linear', C=1)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(classification_report(y_test, y_pred))
