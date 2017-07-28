# -*- coding: utf-8 -*-

import os
import pickle
import sys
from gensim import models
from gensim.corpora import Dictionary

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

import apply_preprogress as ap
import aux.classification_model as cm
from aux import constant


def main(data, need_segment):
    if need_segment:
        data = ap.get_segment(data)

    lda_model = models.LdaModel.load(constant.lda_model_path)
    dictionary = Dictionary.load(constant.dictionary_path)
    x_vecs = ap.lda_vecs(dictionary, lda_model, data)

    with open(constant.classification_model_path, "r") as f:
        classification_model = pickle.load(f)

    pred_first_step = cm.predict(classification_model, x_vecs)

    ret = ap.reclassify_data(x_vecs, pred_first_step)
    pos_x = ret[0]
    pos_x_id = ret[1]
    neg_x = ret[2]
    neg_x_id = ret[3]

    with open(constant.reclassify_pos_model_path, "r") as f:
        reclassify_pos_model = pickle.load(f)
    pos_pred_y = reclassify_pos_model.predict(pos_x)

    with open(constant.reclassify_neg_model_path, "r") as f:
        reclassify_neg_model = pickle.load(f)
    neg_pred_y = reclassify_neg_model.predict(neg_x)

    pred_second_step = pred_first_step[:]
    for i in range(len(pos_x_id)):
        pred_second_step[pos_x_id[i]] = pos_pred_y[i]
    for i in range(len(neg_x_id)):
        pred_second_step[neg_x_id[i]] = neg_pred_y[i]

    with open(constant.train_first_step_result) as f:
        pricision = [float(x) for x in f.readline().strip("\n").split()]
        recall = [float(x) for x in f.readline().strip("\n").split()]

    effect = [pricision, recall]

    return pred_first_step, pred_second_step, effect
