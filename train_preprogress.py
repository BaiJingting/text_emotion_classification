# -*- coding: utf-8 -*-
"""
Created on 2017年7月19日

@author: baijingting
"""

import os
import pickle
import sys
from multiprocessing import Pool
import numpy as np
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import Doc2Vec
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

from src.datafactory.weibo_emotion_classification import constant
from src.datafactory.weibo_emotion_classification import segment_process
from src.datafactory.weibo_emotion_classification import tb_weibo_content_for_classification as twc
from src.datafactory.weibo_emotion_classification import apply_preprogress as ap
from src.datafactory.weibo_emotion_classification import model_evaluation as me
from src.datafactory.weibo_emotion_classification import classification_model as cm


def get_train_dataset(startdate1, startdate2, startdate3, enddate):
    """
    :param startdate1:
    :param startdate2:
    :param startdate3:
    :param enddate:
    :return:
    """
    pos_data = read_weibo_emotion_data(startdate1, enddate, emotion=1)
    neg_data = read_weibo_emotion_data(startdate2, enddate, emotion=-1)
    neu_data = read_weibo_emotion_data(startdate3, enddate, emotion=0)
    pos_data = get_segment(pos_data)
    neg_data = get_segment(neg_data)
    neu_data = get_segment(neu_data)

    total_data_X = pos_data + neg_data + neu_data
    total_data_Y = np.concatenate((np.ones(len(pos_data)), -1 * np.ones(len(neg_data)),
                                   np.zeros(len(neu_data))))
    return total_data_X, total_data_Y


def read_weibo_emotion_data(start_date, end_date, emotion, data_source_ids=1):
    """
    :param start_date:
    :param end_date:
    :param emotion:
    :param data_source_ids:
    :return:
    """
    my_filter = twc.Filter()
    my_filter.set_content_datetime(start_date, end_date)
    my_filter.set_emotion(emotion)
    my_filter.set_data_source_ids(data_source_ids)
    my_filter.set_return_feilds(["content_detail"])
    query = twc.query(my_filter)
    print "number of emotion %d data: %d" % (emotion, len(query))
    return query


def get_segment(result):
    """
    :param result:
    :return:
    """
    aux = []
    pool = Pool(processes=constant.process_num)
    try:
        for line in result:
            seg_res = pool.apply_async(single_segment, args=(line,))
            aux.append(seg_res)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    ret = []
    for line in aux:
        if line.successful() and line.get() is not None:
            ret.append(line.get())
    return ret

# ## 过滤部分词性
# def single_segment(line):
#     aux = line['content_detail'].replace("\\", "\\\\").replace("\"", "\\\"") \
#         .replace("\'", "\\\"").replace("%", "").replace("`", "")
#     aux = postag_process.get_postag(aux)
#     if len(aux) == 0:
#         return
#     ret = ap.filter_postag(aux)
#     return aux


def single_segment(line):
    """
    :param line:
    :return:
    """
    aux = line['content_detail'].replace("\\", "\\\\").replace("\"", "\\\"") \
        .replace("\'", "\\\"").replace("%", "").replace("`", "")
    aux = segment_process.get_segment(aux)
    if len(aux) == 0:
        return
    return aux


def lda_vecs(lda_model, data):
    """
    :param lda_model:
    :param data:
    :return:
    """
    ret = []
    for line in data:
        arr = [0 for i in range(constant.topic_nums)]
        for item in lda_model[line]:
            arr[item[0]] = item[1]
        ret.append(arr)
    return ret


def reclassify_data(x_vecs, data_y, pred_y):
    """
    :param x_vecs:
    :param data_y:
    :param pred_y:
    :return:
    """
    pos_x = []
    pos_y = []
    neg_x = []
    neg_y = []
    for i in range(len(pred_y)):
        if pred_y[i] == 1:
            pos_x.append(x_vecs[i])
            pos_y.append(data_y[i])

        if pred_y[i] == -1:
            neg_x.append(x_vecs[i])
            neg_y.append(data_y[i])
    return pos_x, pos_y, neg_x, neg_y


def retrain(pos_x, pos_y, neg_x, neg_y):
    """
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(pos_x, pos_y, test_size=0.2)
    model = cm.train(x_train, y_train, constant.reclassify_pos_model_path)
    pred_y = cm.predict(model, x_test)
    print "reclassify_pos_test -------------------------------------------"
    me.classification_evaluate(y_test, pred_y)

    x_train, x_test, y_train, y_test = train_test_split(neg_x, neg_y, test_size=0.2)
    model = cm.train(x_train, y_train, constant.reclassify_neg_model_path)
    pred_y = cm.predict(model, x_test)
    print "reclassify_neg_test -------------------------------------------"
    me.classification_evaluate(y_test, pred_y)
