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
        self.labels = list()
        self.models = list()

    def fit(self, x_train, y_train):
        self.labels = sorted(list(set(y_train)))
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
        classifiers = list()
        for num in xrange(num_clusters):
            if self.verbose:
                print('running classification #{0}-{1}'.format(itr+1, num), file=sys.stderr)
            x_train_sub, y_train_sub = extract(x_train, y_train, cluster_numbers, num)
            if len(x_train_sub) > self.minimum_samples_per_cluster:
                classifier = copy.deepcopy(self.base_classifier)
                classifier.fit(x_train_sub, y_train_sub)
                classes = classifier.classes_
                accuracy = classifier.score(x_train_sub, y_train_sub)
            else:
                classifier = None
                classes = []
                accuracy = np.NaN
            classifiers.append((classifier, classes, accuracy,))
            if self.verbose:
                print('training classe labels #{0}-{1}: {2}'.format(itr+1, num, classes), file=sys.stderr)
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
                classes = classifiers[num][1]
                if not (classifier is None):
                    proba_set = (classifier.predict_proba(x), classes)
                    probas.append(proba_set)
            probas_list.append(probas)
        return probas_list

    def predict_proba(self, x_test, remove_outliers=False):
        #
        def calc(px):
            if len(px) == 0:
                return 0.0
            h = max(px)
            l = min(px)
            s = np.sum(px)
            return (s-h-l)/(len(px)-2) if remove_outliers else s/len(px)
        #
        def calc_probas(ps):
            if ps is None or len(ps) == 0:
                return None
            pred_dics = dict()
            for c in self.labels:
                pred_dics[c] = list()
            for preds_, classes in ps:
                for p, c in zip(preds_[0], classes):
                    pred_dics[c].append(p)
            # print(pred_dics)
            raw_result = list()
            for c in self.labels:
                raw_result.append(calc(pred_dics[c]))
            s = np.sum(raw_result)
            normalized_probas = [x/s for x in raw_result]
            return np.array(normalized_probas)
        probas_list = self.predict_probas(x_test)
        proba_list = [calc_probas(ps) for ps in probas_list]
        return np.array(proba_list)

    def predict(self, x_test, remove_outliers=False):
        proba_list = self.predict_proba(x_test, remove_outliers)
        return np.array([self.labels[np.argmax(p)] for p in proba_list])

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
    from sklearn import datasets
    from sklearn.cross_validation import train_test_split
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    COSINE_SIMILARITY = True
    if COSINE_SIMILARITY:
        from sklearn.cluster import k_means_
        from scipy.spatial.distance import cdist
        def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
            return cdist(X, Y, 'cosine')
        # MONKEY PATCH
        k_means_.euclidean_distances = new_euclidean_distances

    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5)

    print("ClusteredClassifier")
    cc = ClusteredClassifier(KMeans(n_clusters=6),
                             RandomForestClassifier(n_estimators=50, class_weight='balanced'),
                             num_iter=3,
                             verbose=True)
    cc.fit(x_train, y_train)
    y_pred_train = cc.predict(x_train)
    print(classification_report(y_train, y_pred_train))
    y_pred_test = cc.predict(x_test)
    print(classification_report(y_test, y_pred_test))

    print("RandomForest")
    rf = RandomForestClassifier(n_estimators=50, class_weight='balanced')
    rf.fit(x_train, y_train)
    y_pred_train = rf.predict(x_train)
    print(classification_report(y_train, y_pred_train))
    y_pred_test = rf.predict(x_test)
    print(classification_report(y_test, y_pred_test))
