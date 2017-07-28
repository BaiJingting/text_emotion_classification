# -*- coding: utf-8 -*-

import os
import pickle
import sys
from gensim import models
from gensim.corpora import Dictionary

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

from aux import constant
from apply import apply_preprogress as ap
import aux.classification_model as cm
import train.train_preprogress as tp
import aux.model_evaluation as me

from src.common.util import log
USER_HOME_PATH = os.path.expanduser('~')
logging = log.get_logger('weibo_emotion_classification_logger')


def usage():
    """ notify the usage information """
    print "Usage: Input args with form 'yymmdd': TRAIN_START_DATE, TRAIN_END_DATE"
    print "time span: 5 days"


if __name__ == "__main__":

    args = sys.argv

    if len(args) < 3:
        usage()
        sys.exit(1)

    TRAIN_START_DATE = args[1]
    TRAIN_END_DATE = args[2]

    data_x, data_y = tp.get_apply_dataset(TRAIN_START_DATE, TRAIN_END_DATE)

    lda_model = models.LdaModel.load(constant.lda_model_path)
    dictionary = Dictionary.load(constant.dictionary_path)
    x_vecs = ap.lda_vecs(dictionary, lda_model, data_x)

    with open(constant.classification_model_path, "r") as f:
        classification_model = pickle.load(f)

    pred_y = cm.predict(classification_model, x_vecs)

    accuracy, precision, recall = me.classification_evaluate(data_y, pred_y)
    logging.info("========================================================")
    logging.info("first_step_classify")
    logging.info("========================================================")
    logging.info("accuracy: %f") % accuracy
    logging.info("precision: 1: %f, 0: %f, -1: %f") % (precision[0], precision[1], precision[2])
    logging.info("recall: 1: %f, 0: %f, -1: %f") % (recall[0], recall[1], recall[2])

    with open(constant.train_first_step_result) as f:
        f.write(" ".join(str(x) for x in precision))
        f.write("\n")
        f.write(" ".join(str(x) for x in recall))

    ## reclassify
    ret = tp.reclassify_data(x_vecs, data_y, pred_y)
    pos_x = ret[0]
    pos_y = ret[1]
    neg_x = ret[2]
    neg_y = ret[3]
    tp.retrain(pos_x, pos_y, neg_x, neg_y)