#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import copy
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


# This is an experimental classifier
# It assumes using a binary classifier which has predic_prob()
# Please cite me if you use this code and/or idea.
class ClusteredClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_clusterizer, base_classifier, num_iter=10, verbose=False, *args, **keywords):
        self.base_clusterizer = base_clusterizer
        self.base_classifier = base_classifier
        self.num_iter = num_iter
        self.verbose = verbose
        self.models = None

    def fit(self, x_train, y_train):
        models = list()
        for itr in xrange(self.num_iter):
            models.append(self.fit_sub(x_train, y_train, itr))
        # todo: pick up top n models
        self.models = models
        return models

    def fit_sub(self, x_train, y_train, itr):
        if self.verbose:
            print('running clusutering #{0}'.format(itr), file=sys.stderr)
        clusterizer = copy.deepcopy(self.base_clusterizer)
        clusterizer.fit(x_train)
        cluster_numbers = clusterizer.predict(x_train)
        num_clusters = len(set(cluster_numbers))
        classifiers = list()
        for num in xrange(num_clusters):
            if self.verbose:
                print('running classification #{0}-{1}'.format(iter, num), file=sys.stderr)
            x_train_sub, y_train_sub = extract(x_train, y_train, cluster_numbers, num)
            classifier = copy.deepcopy(self.base_classifier)
            classifier.fit(x_train_sub, y_train_sub)
            accuracy = classifier.score(x_train_sub, y_train_sub)
            classifiers.append((classifier, accuracy,))
        return clusterizer, classifiers

    def predict_probs(self, x_test):
        probs_list = list()
        for features in x_test:
            x = np.array(features)
            probs = list()
            for model in self.models:
                clusterizer, classifiers = model
                num = clusterizer.predict(x)
                prob = classifiers[num][0].predict_prob(x)
                probs.append(prob)
            probs_list.append(probs)
        return probs_list

    def predict_prob(self, x_test, remove_outliers=False):
        def calc(ps):
            h = max(ps)
            l = max(ps)
            s = sum(ps)
            return (s-h-l)/(len(ps)-2) if remove_outliers else s/len(ps)
        probs_list = self.predict_probs(x_test)
        prob_list = [calc(ps) for ps in probs_list]
        return np.array(prob_list)

    def predict(self, x_test, remove_outliers=False):
        prob_list = self.predict_prob(x_test, remove_outliers)
        return np.array([p > 0.5 for p in prob_list])

    # investigate the worst performing clusters
    def investigate(self, x_test, y_test):
        pass


def extract(x_train, y_train, numbers, num):
    conditions = [num == x for x in numbers]
    return np.extract(conditions, x_train), np.extract(conditions, y_train)

if True:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import datasets
    iris = datasets.load_iris()
    x_train = iris.data
    y_train = iris.target

    cc = ClusteredClassifier(KMeans(n_clusters=3),
                             RandomForestClassifier(n_estimators=100, class_weight='balanced'))
    cc.fit(x_train, y_train)

