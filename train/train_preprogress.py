# -*- coding: utf-8 -*-

import os
import sys
from multiprocessing import Pool
import numpy as np

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

from aux import classification_model as cm
from aux import constant
from aux import segment_process
from aux import tb_weibo_content_for_classification as twc


def get_train_dataset(startdate1, startdate2, startdate3, enddate):
    pos_data = read_weibo_emotion_data(startdate1, enddate, emotion=1)
    neg_data = read_weibo_emotion_data(startdate2, enddate, emotion=-1)
    neu_data = read_weibo_emotion_data(startdate3, enddate, emotion=0)

    pos_data = get_segment(pos_data, single_segment_1)
    neg_data = get_segment(neg_data, single_segment_1)
    neu_data = get_segment(neu_data, single_segment_1)

    pos_data = deparser(pos_data)
    neg_data = deparser(neg_data)
    neu_data = deparser(neu_data)

    total_data_X = pos_data + neg_data + neu_data
    total_data_Y = np.concatenate((np.ones(len(pos_data)), -1 * np.ones(len(neg_data)),
                                   np.zeros(len(neu_data))))
    return total_data_X, total_data_Y


def deparser(data):
    ret = []
    for line in data:
        if line.successful() and line.get() is not None:
            ret.append(line.get())
    return ret


def read_weibo_emotion_data(start_date, end_date, emotion, data_source_ids=1):
    my_filter = twc.Filter()
    my_filter.set_content_datetime(start_date, end_date)
    my_filter.set_emotion(emotion)
    my_filter.set_data_source_ids(data_source_ids)
    my_filter.set_return_feilds(["content_detail"])
    query = twc.query(my_filter)
    print "number of emotion %d data: %d" % (emotion, len(query))
    return query


def get_segment(result, function):
    aux = []
    pool = Pool(processes=constant.process_num)
    try:
        for line in result:
            seg_res = pool.apply_async(function, args=(line,))
            aux.append(seg_res)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    return aux


def single_segment_1(line):
    aux = line['content_detail'].replace("\\", "\\\\").replace("\"", "\\\"") \
        .replace("\'", "\\\"").replace("%", "").replace("`", "")
    aux = segment_process.get_segment(aux)
    if len(aux) == 0:
        return
    return aux


def single_segment_2(line):
    aux = line['content_detail'].replace("\\", "\\\\").replace("\"", "\\\"") \
        .replace("\'", "\\\"").replace("%", "").replace("`", "")
    line['content_detail'] = segment_process.get_segment(aux)
    if len(aux) == 0:
        return
    return line


def lda_vecs(lda_model, data):
    ret = []
    for line in data:
        arr = [0 for i in range(constant.topic_nums)]
        for item in lda_model[line]:
            arr[item[0]] = item[1]
        ret.append(arr)
    return ret


def reclassify_data(x_vecs, data_y, pred_y):
    ret = []
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
    ret.append(pos_x)
    ret.append(pos_y)
    ret.append(neg_x)
    ret.append(neg_y)
    return ret


def retrain(pos_x, pos_y, neg_x, neg_y):
    cm.train(pos_x, pos_y, constant.reclassify_pos_model_path)
    cm.train(neg_x, neg_y, constant.reclassify_neg_model_path)


def get_apply_dataset(startdate, enddate):
    result = read_weibo_data(startdate, enddate)
    data = get_segment(result, single_segment_2)

    data_x = []
    data_y = []
    for line in data:
        if line.successful() and line.get() is not None:
            data_x.append(line.get()['content_detail'])
            data_y.append(line.get()['emotion_type'])
    return data_x, data_y


def read_weibo_data(start_date, end_date, data_source_ids=1):
    my_filter = twc.Filter()
    my_filter.set_content_datetime(start_date, end_date)
    my_filter.set_data_source_ids(data_source_ids)
    query = twc.query(my_filter)
    print "total number of data: %d" % len(query)
    return query
