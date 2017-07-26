# -*- coding: utf-8 -*-
"""
Created on 2017年7月19日

@author: baijingting
"""

import os
import sys
import logging
import MySQLdb
import MySQLdb.cursors

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

from src.datafactory.config import database

class Filter(object):
    """
    filter conditions for table tb_content_for_mark
    """

    def __init__(self):
        self.product_ids = None
        self.start_date = None
        self.end_date = None
        self.data_source_ids = None
        self.return_fields = ["content_detail", "emotion_type"]
        self.emotion = None

    def set_product_ids(self, product_ids):
        """
        set product_id list
        """
        self.product_ids = product_ids

    def set_content_datetime(self, start_date, end_date):
        """
        set start_date and end_date
        """
        self.start_date = start_date
        self.end_date = end_date

    def set_data_source_ids(self, data_source_ids):
        """
        set data_source_id list
        """
        self.data_source_ids = data_source_ids

    def set_emotion(self, emotion):
        """
        set data_source_id list
        """
        self.emotion = emotion

    def set_return_feilds(self, return_fields):
        """
        set data_source_id list
        """
        self.return_fields = return_fields

    def convert_2_sql(self):
        """
        generate query sql

        """
        filters = []
        if self.product_ids is not None:
            product_ids = [str(product_id) for product_id in self.product_ids]
            filters.append("product_id in (%s)" % ",".join(product_ids))
        if self.start_date is not None:
            filters.append("content_datetime>='%s'" % self.start_date)
        if self.end_date is not None:
            filters.append("content_datetime<='%s'" % self.end_date)
        if self.data_source_ids is not None:
            filters.append("data_source_id=%s" % self.data_source_ids)
        if self.emotion is not None:
            filters.append("emotion_type=%s" % self.emotion)
        filters.append("is_emotion_marked=2")
        filters.append("is_related=1")
        filters.append("emotion_type in (1,0,-1)")
        if 0 == len(filters):
            filters_str = "limit 0,10000"
        else:
            filters_str = "where %s" % " and ".join(filters)
        return "SELECT %s FROM tb_content_for_mark %s" % (",".join(self.return_fields), filters_str)


def query(filter_instance):
    """
    query tb_content_for_mark table

    Parameters
    ---------
    filter_instance: Filter instance

    Returns
    -------
    contents: dict
    """
    sql_query = filter_instance.convert_2_sql()
    conn = get_connect_yuanfangdb()
    if conn is None:
        logging.critical("get connect of yuanfang db failed")
        return None
    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        cursor.close()
        return result
    except MySQLdb.Error as e:
        logging.critical("execute sql failed: %s" % e)
    except UnicodeEncodeError as e:
        logging.critical("execute sql failed: %s" % e)
    return None


def get_connect_yuanfangdb():
    """

    :return:
    """
    conn = None
    try:
        conn = MySQLdb.connect(host=database.YUANFANG_HOSTNAME,
                               user=database.YUANFANG_USERNAME,
                               passwd=database.YUANFANG_PASSWORD,
                               db=database.YUANFANG_DB,
                               port=database.YUANFANG_PORT,
                               charset="utf8",
                               cursorclass=MySQLdb.cursors.DictCursor)
    except MySQLdb.Error as e:
        logging.critical("create yuanfang db connect failed: %s" % e)
    return conn



if __name__ == "__main__":
    product_ids = [75, 73]
    start_date = "2016-07-26 00:00:00"
    end_date = "2016-07-26 23:59:59"
    data_source_ids = [1, 2]
    my_filter = Filter()
    my_filter.set_product_ids(product_ids)
    my_filter.set_content_datetime(start_date, end_date)
    my_filter.set_data_source_ids(data_source_ids)
    query_result = query(my_filter)
