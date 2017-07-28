# -*- coding: utf-8 -*-

import os
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


process_num = 31

dictionary_path = ROOT_PATH + "/model/dictionary.dict"
lda_model_path = ROOT_PATH + "/model/lda_model.pkl"

classification_model_path = ROOT_PATH + "/model/classification_model.pkl"
reclassify_pos_model_path = ROOT_PATH + "/model/reclassify_pos_model.pkl"
reclassify_neg_model_path = ROOT_PATH + "/model/reclassify_neg_model.pkl"

train_first_step_result = ROOT_PATH + "/model/train_first_step_result.txt"

topic_nums = 500
