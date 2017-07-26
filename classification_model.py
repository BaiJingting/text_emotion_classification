# -*- coding: utf-8 -*-
"""
Created on 2017年7月19日

@author: baijingting
"""
import os
import sys
import pickle
from sklearn.ensemble import RandomForestClassifier

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)


def train(train_x, train_y, path):
    """

    :param train_x:
    :param train_y:
    :param path:
    :return:
    """

    model = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=None, n_jobs=8,
                    min_samples_split=2,max_features='auto', class_weight="balanced")

    model.fit(train_x, train_y)
    with open(path, "w") as f:
        pickle.dump(model, f)
    return model


def predict(model, test_x):
    """

    :param model:
    :param test_x:
    :return:
    """
    pred_Y = model.predict(test_x)
    return pred_Y
