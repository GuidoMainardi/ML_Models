from KNearestNeighbor import KNearestNeighbor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv('column_data.csv')
print(data.head())
X = data.drop(['class'], axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

knn = KNearestNeighbor(neighbors=3)

knn.fit(X_train, y_train)

preds = knn.predict(np.array(X_test))

print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))


data = pd.read_csv('column_data.csv')
print(data.head())
X = data.drop(['class'], axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

knn = KNearestNeighbor(policy='KDTree')

knn.fit(X_train, y_train)

preds = knn.predict(np.array(X_test))

print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))