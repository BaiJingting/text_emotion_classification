# -*- coding: utf-8 -*-
'''
Created on 2017年7月19日

@author: baijingting
'''

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import pickle
import constant


def load_model():
    model = pickle.load(constant.model_path)
    return model


def train(train_x, train_y):

    # model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=9,
    #                                    subsample=0.8, max_features=0.5, loss='deviance')
    model = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=None, n_jobs=8,
                    min_samples_split=2,max_features='auto', class_weight="balanced")
    # model = MultinomialNB()

    model.fit(train_x, train_y)
    with open(constant.classification_model_path, "wb") as f:
        pickle.dump(model, f)
    return model


def predict(model, test_x):
    pred_Y = model.predict(test_x)
    return pred_Y


def classification_evaluate(test_Y, pred_Y):
    n_total = 0
    n11 = n10 = n1_1 = 0
    n01 = n00 = n0_1 = 0
    n_11 = n_10 = n_1_1 = 0
    for i in range(len(test_Y)):
        if test_Y[i] == 1:
            if pred_Y[i] == 1:
                n11 += 1
            elif pred_Y[i] == 0:
                n10 += 1
            else:
                n1_1 += 1
        elif test_Y[i] == 0:
            if pred_Y[i] == 1:
                n01 += 1
            elif pred_Y[i] == 0:
                n00 += 1
            else:
                n0_1 += 1
        else:
            if pred_Y[i] == 1:
                n_11 += 1
            elif pred_Y[i] == 0:
                n_10 += 1
            else:
                n_1_1 += 1
        n_total += 1
    test_n1 = n11 + n10 + n1_1
    test_n0 = n01 + n00 + n0_1
    test_n_1 = n_11 + n_10 + n_1_1
    pred_n1 = n11 + n01 + n_11
    pred_n0 = n10 + n00 + n_10
    pred_n_1 = n1_1 + n0_1 + n_1_1
    accuracy = (n11 + n00 + n_1_1) * 1.0 / n_total
    recall1 = (n11 + 1) * 1.0 / (n11 + n10 + n1_1 + 3)
    recall0 = (n00 + 1) * 1.0 / (n01 + n00 + n0_1 + 3)
    recall_1 = (n_1_1 + 1) * 1.0 / (n_11 + n_10 + n_1_1 + 3)
    precision1 = (n11 + 1) * 1.0 / (n11 + n01 + n_11 + 3)
    precision0 = (n00 + 1) * 1.0 / (n10 + n00 + n_10 + 3)
    precision_1 = (n_1_1 + 1) * 1.0 / (n1_1 + n0_1 + n_1_1 + 3)
    print "test data: 1: %d, 0: %d, -1: %d" % (test_n1, test_n0, test_n_1)
    print "predict: 1: %d, 0: %d, -1: %d" % (pred_n1, pred_n0, pred_n_1)
    print "accuracy: %f" % accuracy
    print "precision 1: %f, precision 0: %f, precision -1: %f" % (precision1, precision0, precision_1)
    print "recall 1: %f, recall 0: %f, recall -1: %f" % (recall1, recall0, recall_1)