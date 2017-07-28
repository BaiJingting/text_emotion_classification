# -*- coding: utf-8 -*-

def classification_evaluate(test_Y, pred_Y):
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
    accuracy = (n11 + n00 + n_1_1) * 1.0 / n_total
    recall1 = (n11 + 1) * 1.0 / (n11 + n10 + n1_1 + 3)
    recall0 = (n00 + 1) * 1.0 / (n01 + n00 + n0_1 + 3)
    recall_1 = (n_1_1 + 1) * 1.0 / (n_11 + n_10 + n_1_1 + 3)
    precision1 = (n11 + 1) * 1.0 / (n11 + n01 + n_11 + 3)
    precision0 = (n00 + 1) * 1.0 / (n10 + n00 + n_10 + 3)
    precision_1 = (n_1_1 + 1) * 1.0 / (n1_1 + n0_1 + n_1_1 + 3)

    precision = [precision1, precision0, precision_1]
    recall = [recall1, recall0, recall_1]

    return accuracy, precision, recall

