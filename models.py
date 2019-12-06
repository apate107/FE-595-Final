from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import pandas as pd


def KNN(data):
    Y = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    scores = {}
    for k in range(1, len(y_train)+1):  # Check accuracy for range of values for k
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        scores[k] = knn.score(X_test, y_test)

    return pd.DataFrame(scores.items(), columns=['k', 'Accuracy'])


def LinearReg(data):
    return pd.DataFrame()
