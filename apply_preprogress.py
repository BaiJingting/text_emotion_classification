# -*- coding: utf-8 -*-
"""
Created on 2017年7月19日

@author: baijingting
"""

import os
import sys
from multiprocessing import Pool

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

from src.datafactory.weibo_emotion_classification import constant
from src.datafactory.weibo_emotion_classification import segment_process
from src.datafactory.weibo_emotion_classification import tb_weibo_content_for_classification as twc


def get_apply_dataset(startdate, enddate):
    """

    :param startdate:
    :param enddate:
    :return:
    """
    result = read_weibo_data(startdate, enddate)
    data = get_segment(result)

    data_x = []
    data_y = []
    for line in data:
        if line.successful() and line.get() is not None:
            data_x.append(line.get()['content_detail'])
            data_y.append(line.get()['emotion_type'])
    return data_x, data_y


def read_weibo_data(start_date, end_date, data_source_ids=1):
    """

    :param start_date:
    :param end_date:
    :param data_source_ids:
    :return:
    """
    my_filter = twc.Filter()
    my_filter.set_content_datetime(start_date, end_date)
    my_filter.set_data_source_ids(data_source_ids)
    query = twc.query(my_filter)
    print "total number of data: %d" % len(query)
    return query


def get_segment(result):
    """

    :param result:
    :return:
    """
    ret = []
    pool = Pool(processes=constant.process_num)
    try:
        for line in result:
            seg_res = pool.apply_async(single_segment, args=(line,))
            ret.append(seg_res)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print "Caught KeyboardInterrupt, terminating workers"
        pool.terminate()
        pool.join()
    return ret


# ## 过滤部分词性
# def single_segment(line):
#     aux = line['content_detail'].replace("\\", "\\\\").replace("\"", "\\\"") \
#         .replace("\'", "\\\"").replace("%", "").replace("`", "")
#     aux = postag_process.get_postag(aux)
#     if len(aux) == 0:
#         return
#     line['content_detail'] = filter_postag(aux)
#     return line


def single_segment(line):
    """
    :param line:
    :return:
    """
    aux = line['content_detail'].replace("\\", "\\\\").replace("\"", "\\\"") \
        .replace("\'", "\\\"").replace("%", "").replace("`", "")
    line['content_detail'] = segment_process.get_segment(aux)
    if len(aux) == 0:
        return
    return line


def filter_postag(list):
    """
    :param list:
    :return:
    """
    ret = []
    for i in list:
        if i[1] in constant.KEEP_POS:
            ret.append(i[0])
    return ret


def lda_vecs(dictionary, lda_model, data):
    """
    :param dictionary:
    :param lda_model:
    :param data:
    :return:
    """
    ret = []
    for line in data:
        line = dictionary.doc2bow(line)
        arr = [0 for i in range(constant.topic_nums)]
        for item in lda_model[line]:
            arr[item[0]] = item[1]
        ret.append(arr)
    return ret
