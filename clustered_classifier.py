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
        self.minimum_samples_per_cluster = 3
        self.minimum_train_accuracy = 0.5
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
            print('running clusutering    #{0}'.format(itr+1), file=sys.stderr)
        clusterizer = copy.deepcopy(self.base_clusterizer)
        clusterizer.fit(x_train)
        cluster_numbers = clusterizer.predict(x_train)
        num_clusters = len(set(cluster_numbers))
        num_labels = len(set(y_train))
        classifiers = list()
        for num in xrange(num_clusters):
            if self.verbose:
                print('running classification #{0}-{1}'.format(itr+1, num), file=sys.stderr)
            x_train_sub, y_train_sub = extract(x_train, y_train, cluster_numbers, num)
            if len(x_train_sub) > self.minimum_samples_per_cluster and \
               len(set(y_train_sub)) == num_labels:
                classifier = copy.deepcopy(self.base_classifier)
                classifier.fit(x_train_sub, y_train_sub)
                accuracy = classifier.score(x_train_sub, y_train_sub)
            else:
                classifier = None
                accuracy = np.NaN
            classifiers.append((classifier, accuracy,))
            if self.verbose:
                print('training accuracy      #{0}-{1}: {2}'.format(itr+1, num, accuracy), file=sys.stderr)
        return clusterizer, classifiers

    def predict_probas(self, x_test):
        probas_list = list()
        for features in x_test:
            x = np.array([features])
            probas = list()
            for model in self.models: # slow!
                clusterizer, classifiers = model
                num = clusterizer.predict(x)
                classifier = classifiers[num][0]
                print(classifiers[num])
                if not (classifier is None):
                    proba_set = classifier.predict_proba(x)
                    probas.append(proba_set)
            probas_list.append(probas)
        return probas_list

    def predict_proba(self, x_test, remove_outliers=False):
        def calc(ps):
            print(ps)
            h = max(ps)
            l = max(ps)
            s = sum(ps)
            return (s-h-l)/(len(ps)-2) if remove_outliers else s/len(ps)
        def calc_probas(ps):
            if ps is None or len(ps) == 0:
                return None
            print(ps)
            num_labels = len(ps[0][0])
            print(num_labels)
        probas_list = self.predict_probas(x_test)
        proba_list = [calc_probas(ps) for ps in probas_list]
        return proba_list

    def predict(self, x_test, remove_outliers=False):
        proba_list = self.predict_proba(x_test, remove_outliers)
        return np.array([p > 0.5 for p in proba_list])

    # investigate the worst performing clusters
    def investigate(self, x_test, y_test):
        pass


def extract(x_train, y_train, numbers, num):
    selected = [i for i, x in enumerate(numbers) if num == x]
    conditions = [num == x for x in numbers]
    x_train_sub = x_train[selected, :]
    y_train_sub = np.extract(conditions, y_train)
    return x_train_sub, y_train_sub


if True:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import datasets

    COSINE_SIMILARITY = False
    if COSINE_SIMILARITY:
        from sklearn.cluster import k_means_
        from scipy.spatial.distance import cdist
        def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
            return cdist(X, Y, 'cosine')
        # MONKEY PATCH
        k_means_.euclidean_distances = new_euclidean_distances

    iris = datasets.load_iris()
    x_train = iris.data
    y_train = iris.target

    cc = ClusteredClassifier(KMeans(n_clusters=1),
                             RandomForestClassifier(n_estimators=50, class_weight='balanced'),
                             verbose=True)
    cc.fit(x_train, y_train)
    preds = cc.predict(x_train)
    print(preds)
