import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod


class Boosting(Classifier):
    # Boosting from pre-defined classifiers
    def __init__(self, clfs: Set[Classifier], T=0):
        self.clfs = clfs  # set of weak classifiers to be considered
        self.num_clf = len(clfs)
        if T < 1:
            self.T = self.num_clf
        else:
            self.T = T

        self.clfs_picked = []  # list of classifiers h_t for t=0,...,T-1
        self.betas = []  # list of weights beta_t for t=0,...,T-1
        return

    @abstractmethod
    def train(self, features: List[List[float]], labels: List[int]):
        return

    def predict(self, features: List[List[float]]) -> List[int]:
        '''
        Inputs:
        - features: the features of all test examples

        Returns:
        - the prediction (-1 or +1) for each example (in a list)
        '''
        ########################################################
        # TODO: implement "predict"
        # sum = 0
        # pred = np.zeros(len(features))
        # for j in range(len(features)):
        #     for i in range(len(self.clfs_picked)):
        #         temp = self.betas[i] * self.clfs_picked[i].predict(features[j])
        #         sum += temp
        #     if sum > 0:
        #         pred[j] = 1
        #     else:
        #         pred[j] = -1
        # pred.tolist()
        # return pred

        h = np.zeros(len(features))
        h = np.sum([beta * np.array(clf.predict(features)) for clf, beta in zip(self.clfs_picked, self.betas)], axis=0)
        for i in range(len(h)):
            if h[i] > 0:
                h[i] = 1
            else:
                h[i] = -1

        return h.tolist()
########################################################


class AdaBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "AdaBoost"
        return

    def train(self, features: List[List[float]], labels: List[int]):
        D = len(labels)
        w = np.full(D, 1 / D)

        for t in range(self.T):
            e_t = len(features) + 1

            for clf in self.clfs:
                h = clf.predict(features)
                err = np.sum(w * (np.array(labels) != np.array(h)))
                if err < e_t:
                    h_t = clf
                    e_t = err
                    htx = h
            self.clfs_picked.append(h_t)

            beta = 0.5 * np.log((1 - e_t) / e_t)
            self.betas.append(beta)

            for n in range(D):
                w[n] *= np.exp(-beta) if labels[n] == htx[n] else np.exp(beta)
            w_sum = np.sum(w)
            w = w / w_sum

    ############################################################

    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)
