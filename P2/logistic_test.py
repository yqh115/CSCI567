from __future__ import division, print_function

import numpy as np


#######################################################################
# Replace TODO with your code
#######################################################################

def multinomial_train(X, y, C,
                      w0=None,
                      b0=None,
                      step_size=0.5,
                      max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - C: number of classes in the data
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes

    Implement multinomial logistic regression for multiclass
    classification. Again use the *average* of the gradients for all training
	examples multiplied by the step_size to update parameters.

	You may find it useful to use a special (one-hot) representation of the labels,
	where each label y_i is represented as a row of zeros with a single 1 in
    the column, that corresponds to the class y_i.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    ""
    w = np.zeros((C, D + 1))
    X0 = np.ones(N)
    X = np.c_[X0, X]
    y_example = list(range(C))

    for i in range(max_iterations):
        for k in range(0, N):
            #kk = np.random.randint(0, N, 1)

            #k = int(k)
            #print(k)
            y_k = y[k]
            x_k = X[k] #- max(X[k])
            x_k = x_k.reshape(1, D+1)
            sum = 0
            for j in range(C):
                sum += np.exp(w[j].dot(x_k.T))


            grad = np.zeros((C, 1))
            for j in range(C):
                if y_k != y_example[j]:
                    grad[j] = np.exp(w[j].dot(x_k.T)) / sum
                else:
                    grad[j] = (np.exp(w[j].dot(x_k.T)) / sum) - 1

            w -= step_size * grad.dot(x_k) / N

    w = w.T
    b = w[0]
    w = w[1:]
    w = w.T
    ""

    assert w.shape == (C, D)
    assert b.shape == (C,)
    return w, b


def multinomial_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier
    - b: bias terms of the trained multinomial classifier

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes

    Make predictions for multinomial classifier.
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N)

    ""
    y_example = list(range(C))
    for i in range(N):
        h = list(w.dot(X[i]) + b)
        preds[i] = y_example[h.index(max(h))]
    ""

    assert preds.shape == (N,)
    return preds


def OVR_train(X, y, C, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array,
    indicating the labels of each training point
    - C: number of classes in the data
    - w0: initial value of weight matrix
    - b0: initial value of bias term
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: a C-by-D weight matrix of OVR logistic regression
    - b: bias vector of length C

    Implement multiclass classification using one-versus-rest with binary logistic
	regression as the black-box. Recall that the one-versus-rest classifier is
    trained by training C different classifiers.
    """
    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    ""
    X0 = np.ones(N)
    X = np.c_[X0, X]
    # w_ = np.zeros(D + 1)

    for i in range(C):
        w_ = np.zeros(D + 1)
        y_i = np.zeros(N)
        for j in range(N):
            if y[j] == i:
                y_i[j] = 1
            else:
                y_i[j] = 0

        for k in range(max_iterations):
            h = sigmoid(X.dot(w_))
            #print(h)
            dw_ = step_size * X.T.dot((h - y_i)) / N
            w_ -= dw_
        b[i] = w_[0]
        w[i] = w_[1:]


    ""
    assert w.shape == (C, D), 'wrong shape of weights matrix'
    assert b.shape == (C,), 'wrong shape of bias terms vector'
    return w, b


def OVR_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained OVR model
    - b: bias terms of the trained OVR model

    Returns:
    - preds: vector of class label predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes.

    Make predictions using OVR strategy and probability predictions from binary
    classifiers.
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N)

    ""
    y_example = list(range(C))
    for i in range(N):
        h = list(w.dot(X[i].T) + b)
        preds[i] = y_example[h.index(max(h))]
    ""

    assert preds.shape == (N,)
    return preds


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def accuracy_score(true, preds):
    return np.sum(true == preds).astype(float) / len(true)


from data_loader import toy_data_multiclass_3_classes_non_separable, \
    toy_data_multiclass_5_classes, \
    data_loader_mnist

datasets = [(toy_data_multiclass_3_classes_non_separable(),
             'Synthetic data', 3),
            (toy_data_multiclass_5_classes(), 'Synthetic data', 5),
            (data_loader_mnist(), 'MNIST', 10)]

for data, name, num_classes in datasets:
    print('%s: %d class classification' % (name, num_classes))
    X_train, X_test, y_train, y_test = data

    print('One-versus-rest:')
    w, b = OVR_train(X_train, y_train, C=num_classes)
    train_preds = OVR_predict(X_train, w=w, b=b)
    preds = OVR_predict(X_test, w=w, b=b)
    print('train acc: %f, test acc: %f' %
          (accuracy_score(y_train, train_preds),
           accuracy_score(y_test, preds)))


