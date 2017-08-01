#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2017年7月19日

@author: baijingting
"""
import os
import sys
import pickle
from gensim import models
from gensim.corpora import Dictionary
from sklearn.model_selection import train_test_split

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

from src.datafactory.weibo_emotion_classification import apply_preprogress as ap
import src.datafactory.weibo_emotion_classification.classification_model as cm
from src.datafactory.weibo_emotion_classification import constant
import src.datafactory.weibo_emotion_classification.train_preprogress as tp
import src.datafactory.weibo_emotion_classification.model_evaluation as me

if __name__ == "__main__":

    if constant.OBJECTIVE == "train_overall":

        data_x, data_y = tp.get_train_dataset(constant.POS_START_DATE, constant.NEG_START_DATE,
                                              constant.NEU_START_DATE, constant.TRAIN_END_DATE)

        ## lda
        dictionary = Dictionary(data_x)
        dictionary.filter_extremes(no_below=5, no_above=0.7)
        dictionary.save(constant.dictionary_path)

        corpus = [dictionary.doc2bow(text) for text in data_x]
        lda_model = models.LdaModel(corpus, id2word=dictionary,
                                    num_topics=constant.topic_nums, iterations=500)
        lda_model.save(constant.lda_model_path)
        data_x = tp.lda_vecs(lda_model, corpus)

        ### classify

        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

        model = cm.train(x_train, y_train, constant.classification_model_path)
        pred_y = cm.predict(model, x_test)
        me.classification_evaluate(y_test, pred_y)

    elif constant.OBJECTIVE in ["apply_reclassify", "train_reclassify"]:
        if constant.OBJECTIVE == "train_reclassify":
            data_x, data_y = ap.get_apply_dataset(constant.RECLASSIFY_TRAIN_START_DATE,
                                                  constant.RECLASSIFY_TRAIN_END_DATE)
        else:
            data_x, data_y = ap.get_apply_dataset(constant.RECLASSIFY_APPLY_START_DATE,
                                                  constant.RECLASSIFY_APPLY_END_DATE)

        lda_model = models.LdaModel.load(constant.lda_model_path)
        dictionary = Dictionary.load(constant.dictionary_path)
        x_vecs = ap.lda_vecs(dictionary, lda_model, data_x)

        with open(constant.classification_model_path, "r") as f:
            classification_model = pickle.load(f)

        pred_y = cm.predict(classification_model, x_vecs)

        me.classification_evaluate(data_y, pred_y)

        ## 在召回基础上提升精确率
        if constant.OBJECTIVE == "train_reclassify":

            pos_x, pos_y, neg_x, neg_y = tp.reclassify_data(x_vecs, data_y, pred_y)
            tp.retrain(pos_x, pos_y, neg_x, neg_y)

        else:

            pos_x, pos_y, neg_x, neg_y = tp.reclassify_data(x_vecs, data_y, pred_y)

            with open(constant.reclassify_pos_model_path, "r") as f:
                reclassify_pos_model = pickle.load(f)
            pos_pred_y = reclassify_pos_model.predict(pos_x)
            print "reclassify_pos ---------------------------------------------"
            me.classification_evaluate(pos_y, pos_pred_y)

            with open(constant.reclassify_neg_model_path, "r") as f:
                reclassify_neg_model = pickle.load(f)
            neg_pred_y = reclassify_neg_model.predict(neg_x)
            print "reclassify_neg ---------------------------------------------"
            me.classification_evaluate(neg_y, neg_pred_y)

            ## 综合评估
            me.evaluate(data_y, pred_y, pos_pred_y, neg_pred_y)


    elif constant.OBJECTIVE == "apply_overall":

        data_x, data_y = ap.get_apply_dataset(constant.APPLY_START_DATE, constant.APPLY_END_DATE)

        lda_model = models.LdaModel.load(constant.lda_model_path)
        dictionary = Dictionary.load(constant.dictionary_path)
        x_vecs = ap.lda_vecs(dictionary, lda_model, data_x)

        with open(constant.classification_model_path, "rb") as f:
            classification_model = pickle.load(f)

        pred_y = cm.predict(classification_model, x_vecs)
        cm.classification_evaluate(data_y, pred_y)
