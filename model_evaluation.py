# -*- coding: utf-8 -*-
"""
Created on 2017年7月19日

@author: baijingting
"""


def classification_evaluate(test_Y, pred_Y):
    """

    :param test_Y:
    :param pred_Y:
    :return:
    """
    n_total = len(test_Y)
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


def evaluate(data_y, pred_y, pos_pred_y, neg_pred_y):
    """

    :param data_y:
    :param pred_y:
    :param pos_pred_y:
    :param neg_pred_y:
    :return:
    """
    n_total = len(data_y)
    n11 = n10 = n1_1 = 0
    n01 = n00 = n0_1 = 0
    n_11 = n_10 = n_1_1 = 0
    j = 0
    k = 0
    for i in range(len(data_y)):
        if (pred_y[i] == 1 and pos_pred_y[j] == 1) or (pred_y[i] == -1 and neg_pred_y[k] == 1):
            if data_y[i] == 1:
                n11 += 1
            elif data_y[i] == 0:
                n01 += 1
            else:
                n_11 += 1
            if pred_y[i] == 1:
                j += 1
            else:
                k += 1
        elif (pred_y[i] == 1 and pos_pred_y[j] == -1) or (pred_y[i] == -1 and neg_pred_y[k] == -1):
            if data_y[i] == 1:
                n1_1 += 1
            elif data_y[i] == 0:
                n0_1 += 1
            else:
                n_1_1 += 1
            if pred_y[i] == 1:
                j += 1
            else:
                k += 1
        else:
            if data_y[i] == 1:
                n10 += 1
            elif data_y[i] == 0:
                n00 += 1
            else:
                n_10 += 1
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
    print "after reclassify ------------------------------------------------------------"
    print "test data: 1: %d, 0: %d, -1: %d" % (test_n1, test_n0, test_n_1)
    print "predict: 1: %d, 0: %d, -1: %d" % (pred_n1, pred_n0, pred_n_1)
    print "accuracy: %f" % accuracy
    print "precision 1: %f, precision 0: %f, precision -1: %f" % (precision1, precision0, precision_1)
    print "recall 1: %f, recall 0: %f, recall -1: %f" % (recall1, recall0, recall_1)
