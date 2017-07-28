# -*- coding: utf-8 -*-

import os
import sys
from gensim import models
from gensim.corpora import Dictionary

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

import aux.classification_model as cm
import train.train_preprogress as tp
from aux import constant


def usage():
    print "Usage: Input args with form 'yymmdd': " \
          "POS_START_DATE, NEG_START_DATE, NEU_START_DATE, TRAIN_END_DATE"
    print "ratio of POS, NEG, NEU --> 27:40:1"


if __name__ == "__main__":

    args = sys.argv

    if len(args) < 5:
        usage()
        sys.exit(1)

    POS_START_DATE = args[1]
    NEG_START_DATE = args[2]
    NEU_START_DATE = args[3]
    TRAIN_END_DATE = args[4]

    data_x, data_y = tp.get_train_dataset(POS_START_DATE, NEG_START_DATE,
                                          NEU_START_DATE, TRAIN_END_DATE)

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

    cm.train(data_x, data_y, constant.classification_model_path)
