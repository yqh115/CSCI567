import numpy as np
import pandas as pd
from sklearn import linear_model

from sklearn.linear_model import Ridge

white = pd.read_csv('winequality-white.csv', low_memory=False, sep=';').values

[N, d] = white.shape

np.random.seed(3)
# prepare data
ridx = np.random.permutation(N)
ntr = int(np.round(N * 0.8))
nval = int(np.round(N * 0.1))
ntest = N - ntr - nval

# spliting training, validation, and test

Xtrain = np.hstack([np.ones([ntr, 1]), white[ridx[0:ntr], 0:-1]])

ytrain = white[ridx[0:ntr], -1]

Xval = np.hstack([np.ones([nval, 1]), white[ridx[ntr:ntr + nval], 0:-1]])
yval = white[ridx[ntr:ntr + nval], -1]

Xtest = np.hstack([np.ones([ntest, 1]), white[ridx[ntr + nval:], 0:-1]])
ytest = white[ridx[ntr + nval:], -1]


def linear_regression_noreg(X, y):
    w1 = np.linalg.lstsq(X, y, rcond=None)[0]
    X_ = np.linalg.inv(X.T.dot(X))
    w2 = X_.dot(X.T).dot(y)

    return w1, w2


def regularized_linear_regression(X, y, lambd):
    I = np.eye(len(X.T))
    X_ = np.linalg.inv(X.T.dot(X) + lambd * I)
    X1 = X_.dot(X.T)
    w = X1.dot(y)
    return w


def tune_lambda(Xtrain, ytrain, Xval, yval, lambds):
    i = 0
    w = []
    err = []
    for lambd in lambds:
        w.append(regularized_linear_regression(Xtrain, ytrain, lambd))
        err.append(test_error(w[i], Xval, yval))
        i = i + 1
    # print(w)
    num = err.index(min(err))
    bestlambda = w[num]
    print(bestlambda)
    print(err[num])
    print(err)
    return bestlambda


def test_error(w, X, y):
    err = np.linalg.norm(X.dot(w) - y)
    return err


lambds = [0, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 10 ** 2]
tune_lambda(Xtrain, ytrain, Xval, yval, lambds)

