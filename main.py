# -*- coding: utf-8 -*-
'''
Created on 2017年7月19日

@author: baijingting
'''

import pickle
from datetime import datetime


import train_preprogress as tp
import apply_preprogress as ap
import classification_model as cm
import constant

from gensim.corpora import Dictionary
from gensim import models
from sklearn.cross_validation import train_test_split

if __name__ == "__main__":

    if constant.OBJECTIVE == "train":

        # data_x, data_y = tp.get_train_dataset\
        #     (constant.POS_NEG_DATA_START_DATE, constant.NEU_DATA_START_DATE, constant.POS_NEG_DATA_END_DATE)
        #
        # tp.save_data(constant.train_data_x_path, data_x)
        # tp.save_data(constant.train_data_y_path, data_y)

        data_x = tp.load_data_x(constant.train_data_x_path)

        ## doc2vec
        # x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
        # x_train = tp.labelizeReviews(x_train, 'TRAIN')
        # x_test = tp.labelizeReviews(x_test, 'TEST')
        # model_dm, model_dbow = tp.train(x_train, x_test, constant.DM_DBOW_VEC_LEN, epoch_num=constant.epoch_num)
        # x_train = tp.get_vectors(model_dm, model_dbow, x_train, constant.DM_DBOW_VEC_LEN)
        # x_test = tp.get_vectors(model_dm, model_dbow, x_test, constant.DM_DBOW_VEC_LEN)

        ## lda
        # dictionary = Dictionary(data_x)
        # dictionary.filter_extremes(no_below=5, no_above=0.7)
        # dictionary.save(constant.dictionary_path)

        dictionary = Dictionary.load(constant.dictionary_path)
        lda_model = models.LdaModel.load(constant.lda_model_path)
        data_x = ap.lda_vecs(dictionary, lda_model, data_x)

        tp.save_data(constant.train_data_vecs_path, data_x)


        data_x = tp.load_data_x(constant.train_data_vecs_path)
        data_y = tp.load_data_y(constant.train_data_y_path)


        # corpus = [dictionary.doc2bow(text) for text in data_x]
        # lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=constant.topic_nums, iterations=500)
        # lda_model.save(constant.lda_model_path)

        time3 = datetime.now()
        print "time consuming for dictionary and model: ", (time3 - time2).seconds


        ### classify

        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

        model = cm.train(x_train, y_train)
        pred_y = cm.predict(model, x_test)
        cm.classification_evaluate(y_test, pred_y)

    elif constant.OBJECTIVE == "apply":

        # data_x, data_y = ap.get_apply_dataset(constant.START_DATE, constant.END_DATE)
        #
        # tp.save_data(constant.apply_data_x_path, data_x)
        # tp.save_data(constant.apply_data_y_path, data_y)

        data_x = tp.load_data_x(constant.apply_data_x_path)
        data_y = tp.load_data_y(constant.apply_data_y_path)

        ## doc2vec
        # data_x = tp.labelizeReviews(data_x, 'APPLY')
        # model_dm, model_dbow = ap.update_model(data_x)
        # x_vecs = tp.get_vectors(model_dm, model_dbow, data_x, constant.DM_DBOW_VEC_LEN)

        # lda
        time1 = datetime.now()

        lda_model = models.LdaModel.load(constant.lda_model_path)
        dictionary = Dictionary.load(constant.dictionary_path)
        x_vecs = ap.lda_vecs(dictionary, lda_model, data_x)

        time2 = datetime.now()
        print "time consuming for dictionary and model: ", (time2 - time1).seconds

        ## predict

        with open(constant.classification_model_path, "rb") as f:
            classification_model = pickle.load(f)

        pred_y = cm.predict(classification_model, x_vecs)
        cm.classification_evaluate(data_y, pred_y)