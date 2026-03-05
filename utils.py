import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def load_preprocess(path):

    df = pd.read_csv(path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if y.dtype == object:
        y = y.astype("category").cat.codes

    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=None
    )

    return X_train, X_test, y_train, y_test


def evaluate_model(solution, X_train, X_test, y_train, y_test):

    C = float(solution[0])

    model = LogisticRegression(
        C=C,
        max_iter=500,
        solver="liblinear"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return rmse
