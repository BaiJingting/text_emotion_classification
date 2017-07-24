# -*- coding: utf-8 -*-
'''
Created on 2017年7月19日

@author: baijingting
'''

import os
import sys
import pickle
import numpy as np
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from sklearn.utils import shuffle

import constant
from multiprocessing import Pool
import tb_weibo_content_for_classification as twc
import apply_preprogress as ap
from segment import postag_process
from segment import segment_process

ROOT_PATH = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)


def save_data(path, data):
    s = '\n'.join([str(x) for x in data])
    with open(path, "wb") as f:
        f.write(s)


def load_data_x(path):
    data = []
    with open(path, "rb") as f:
        for line in f.readlines():
            line = line.strip("\n").split(" ")
            data.append(line)
    return data


def load_data_y(path):
    data = []
    with open(path, "rb") as f:
        for line in f.readlines():
            line = int(float(line.strip("\n")))
            data.append(line)
    return data


def get_train_dataset(startdate1, startdate2, enddate):

    pos_data = read_weibo_emotion_data(startdate1, enddate, emotion=1)
    neg_data = read_weibo_emotion_data(startdate1, enddate, emotion=-1)
    neu_data = read_weibo_emotion_data(startdate2, enddate, emotion=0)
    pos_data = get_segment(pos_data)
    neg_data = get_segment(neg_data)
    neu_data = get_segment(neu_data)

    total_data_X = pos_data + neg_data + neu_data
    total_data_Y = np.concatenate((np.ones(len(pos_data)), -1 * np.ones(len(neg_data)),
                                   np.zeros(len(neu_data))))
    return total_data_X, total_data_Y


def read_weibo_emotion_data(start_date, end_date, emotion, data_source_ids=1):
    my_filter = twc.Filter()
    my_filter.set_content_datetime(start_date, end_date)
    my_filter.set_emotion(emotion)
    my_filter.set_data_source_ids(data_source_ids)
    my_filter.set_return_feilds(["content_detail"])
    query = twc.query(my_filter)
    print "number of emotion %d data: %d" % (emotion, len(query))
    return query


def get_segment(result):
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
    aux = line['content_detail'].replace("\\", "\\\\").replace("\"", "\\\"") \
        .replace("\'", "\\\"").replace("%", "").replace("`", "")
    aux = segment_process.get_segment(aux)
    if len(aux) == 0:
        return
    return aux


def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


def train(x_train, x_test, size, epoch_num=10):

    model_dm = Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    model_dbow = Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

    total_data = x_train + x_test

    model_dm.build_vocab(total_data)
    model_dbow.build_vocab(total_data)

    for epoch in range(epoch_num):
        model_dm.train(shuffle(total_data))
        model_dbow.train(shuffle(total_data))

    with open(constant.dm_model_path, "wb") as f:
        pickle.dump(model_dm, f)
    with open(constant.dbow_model_path, "wb") as f:
        pickle.dump(model_dbow, f)

    return model_dm, model_dbow


def getVecs(model, corpus, size):
    print np.array(model.docvecs[corpus[0].tags[0]]).shape
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)


def get_vectors(model_dm, model_dbow, data, size):
    print len(model_dm.docvecs)
    print model_dm.docvecs.shape
    vecs_dm = getVecs(model_dm, data, size)
    vecs_dbow = getVecs(model_dbow, data, size)
    vecs = np.hstack((vecs_dm, vecs_dbow))
    return vecs


def lda_vecs(lda_model, data):
    ret = []
    for line in data:
        arr = [0 for i in range(constant.topic_nums)]
        for item in lda_model[line]:
            arr[item[0]] = item[1]
        ret.append(arr)
    return ret
