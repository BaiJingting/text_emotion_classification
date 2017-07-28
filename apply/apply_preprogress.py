# -*- coding: utf-8 -*-

import os
import sys
from multiprocessing import Pool

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

from aux import constant
from aux import segment_process

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
        print "Caught KeyboardInterrupt, terminating workers"
        pool.terminate()
        pool.join()

    ret = []
    for line in aux:
        if line.successful() and line.get() is not None:
            ret.append(line.get())
    return ret


def single_segment(line):
    aux = line.replace("\\", "\\\\").replace("\"", "\\\"") \
        .replace("\'", "\\\"").replace("%", "").replace("`", "")
    aux = segment_process.get_segment(aux)
    if len(aux) == 0:
        return
    return aux


def lda_vecs(dictionary, lda_model, data):
    ret = []
    for line in data:
        line = dictionary.doc2bow(line)
        arr = [0 for i in range(constant.topic_nums)]
        for item in lda_model[line]:
            arr[item[0]] = item[1]
        ret.append(arr)
    return ret


def reclassify_data(x_vecs, pred_y):
    ret = []
    pos_x = []
    pos_x_id = []
    neg_x = []
    neg_x_id = []
    for i in range(len(pred_y)):
        if pred_y[i] == 1:
            pos_x.append(x_vecs[i])
            pos_x_id.append(i)
        if pred_y[i] == -1:
            neg_x.append(x_vecs[i])
            neg_x_id.append(i)
    ret.append(pos_x)
    ret.append(pos_x_id)
    ret.append(neg_x)
    ret.append(neg_x_id)
    return ret
