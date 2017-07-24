# -*- coding: utf-8 -*-
'''
Created on 2017年7月19日

@author: baijingting
'''

OBJECTIVE = "train"
# OBJECTIVE = "apply"

train_data_x_path =  "/home/search/baijingting/yuanfang/nlp/yuanfang_nlp/src/weibo/train_data_x.txt"
train_data_y_path =  "/home/search/baijingting/yuanfang/nlp/yuanfang_nlp/src/weibo/train_data_y.txt"

train_data_vecs_path = "/home/search/baijingting/yuanfang/nlp/yuanfang_nlp/src/weibo/train_data_vecs.txt"

apply_data_x_path =  "/home/search/baijingting/yuanfang/nlp/yuanfang_nlp/src/weibo/apply_data_x.txt"
apply_data_y_path =  "/home/search/baijingting/yuanfang/nlp/yuanfang_nlp/src/weibo/apply_data_y.txt"

WORD_POS_MAP = {
    0: ("Me", "合并词"),  # 无法判断词性
    1: ("Ag", "形语素"),    2: ("Dg", "副语素"),    3: ("Ng", "名语素"),    4: ("Tg", "时语素"),
    5: ("Vg", "动语素"),    6: ("a", "形容词"),     7: ("ad", "副形词"),    8: ("an", "名形词"),
    9: ("b", "区别词"),     10: ("c", "连词"),      11: ("d", "副词"),      12: ("e", "叹词"),
    13: ("f", "方位词"),    14: ("g", "语素"),      15: ("h", "前接成分"),  16: ("i", "成语"),
    17: ("j", "简称略语"),  18: ("k", "后接成分"),   19: ("l", "习用语"),    20: ("m", "数词"),
    21: ("n", "名词"),      22: ("nr", "人名"),     23: ("ns", "地名"),     24: ("nt", "机构团体"),
    25: ("nx", "外文专名"),  26: ("nz", "其他专名"), 27: ("o", "拟声词"),    28: ("p", "介词"),
    29: ("q", "量词"),      30: ("r", "代词"),      31: ("s", "处所词"),    32: ("t", "时间词"),
    33: ("u", "助词"),      34: ("v", "动词"),      35: ("vd", "副动词"),   36: ("vn", "名动词"),
    37: ("w", "标点符号"),   38: ("y", "语气词"),    39: ("z", "状态词")
}

KEEP_POS = [1, 5, 6, 7, 8, 11, 12, 16, 19, 34, 35, 36, 38, 39]

process_num = 31

dm_model_path = "/home/search/baijingting/yuanfang/nlp/yuanfang_nlp/src/weibo/dm_model.pkl"
dbow_model_path = "/home/search/baijingting/yuanfang/nlp/yuanfang_nlp/src/weibo/dbow_model.pkl"

dictionary_path = "/home/search/baijingting/yuanfang/nlp/yuanfang_nlp/src/weibo/dictionary.dict"
lda_model_path = "/home/search/baijingting/yuanfang/nlp/yuanfang_nlp/src/weibo/lda_model.pkl"

classification_model_path = "/home/search/baijingting/yuanfang/nlp/yuanfang_nlp/src/weibo/classification_model.pkl"

## train args
POS_NEG_DATA_START_DATE = 20170615
POS_NEG_DATA_END_DATE = 20170721
NEU_DATA_START_DATE = 20170720

DM_DBOW_VEC_LEN = 500
epoch_num = 10

topic_nums = 200

## apply args
START_DATE = 20170722
END_DATE = 20170723