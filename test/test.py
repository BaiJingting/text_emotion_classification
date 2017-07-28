# -*- coding: utf-8 -*-

import os
import sys
import pickle
from gensim import models
from gensim.corpora import Dictionary

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

import apply.apply_preprogress as ap
import aux.classification_model as cm
import train.train_preprogress as tp
import aux.model_evaluation as me
from aux import constant

from src.common.util import log
USER_HOME_PATH = os.path.expanduser('~')
logging = log.get_logger('weibo_emotion_classification_logger')


def usage():
    print "Usage: Input args with form 'yymmdd': TEST_START_DATE, TEST_END_DATE"


if __name__ == "__main__":

    args = sys.argv

    if len(args) < 3:
        usage()
        sys.exit(1)

    TEST_START_DATE = args[1]
    TEST_END_DATE = args[2]

    data_x, data_y = tp.get_apply_dataset(TEST_START_DATE, TEST_END_DATE)

    lda_model = models.LdaModel.load(constant.lda_model_path)
    dictionary = Dictionary.load(constant.dictionary_path)
    x_vecs = ap.lda_vecs(dictionary, lda_model, data_x)

    with open(constant.classification_model_path, "r") as f:
        classification_model = pickle.load(f)

    pred_y = cm.predict(classification_model, x_vecs)

    accuracy, precision, recall = me.classification_evaluate(data_y, pred_y)
    logging.info("========================================================")
    logging.info("test_first_step")
    logging.info("========================================================")
    logging.info("accuracy: %f") % accuracy
    logging.info("precision: 1: %f, 0: %f, -1: %f") % (precision[0], precision[1], precision[2])
    logging.info("recall: 1: %f, 0: %f, -1: %f") % (recall[0], recall[1], recall[2])

    ret = tp.reclassify_data(x_vecs, data_y, pred_y)
    pos_x = ret[0]
    pos_y = ret[1]
    neg_x = ret[2]
    neg_y = ret[3]

    with open(constant.reclassify_pos_model_path, "r") as f:
        reclassify_pos_model = pickle.load(f)
    pos_pred_y = reclassify_pos_model.predict(pos_x)
    accuracy, precision, recall = me.classification_evaluate(pos_y, pos_pred_y)
    logging.info("========================================================")
    logging.info("test_second_step_pos")
    logging.info("========================================================")
    logging.info("accuracy: %f") % accuracy
    logging.info("precision: 1: %f, 0: %f, -1: %f") % (precision[0], precision[1], precision[2])
    logging.info("recall: 1: %f, 0: %f, -1: %f") % (recall[0], recall[1], recall[2])

    with open(constant.reclassify_neg_model_path, "r") as f:
        reclassify_neg_model = pickle.load(f)
    neg_pred_y = reclassify_neg_model.predict(neg_x)
    accuracy, precision, recall = me.classification_evaluate(neg_y, neg_pred_y)
    logging.info("========================================================")
    logging.info("test_second_step_neg")
    logging.info("========================================================")
    logging.info("accuracy: %f") % accuracy
    logging.info("precision: 1: %f, 0: %f, -1: %f") % (precision[0], precision[1], precision[2])
    logging.info("recall: 1: %f, 0: %f, -1: %f") % (recall[0], recall[1], recall[2])
