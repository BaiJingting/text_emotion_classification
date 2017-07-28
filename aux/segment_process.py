# -*- coding: utf-8 -*-

import os
import logging
import subprocess
import json
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

from src.datafactory.common import json_util
from src.datafactory.config import constant


class SegmentProcess(object):
    """
    function for segment
    """

    def do_nlp_seg(self, sentence):
        """
        connect nlp wordseg
        """
        cmd = "curl -d '{\"lang_id\":1,\"lang_para\":0,\"query\":\"%s\"}" \
              "' %s?username=%s\&app=%s\&encoding=utf8" % (
            sentence,
            constant.SEGMENT_URL,
            constant.SEGMENT_USERNAME,
            constant.SEGMENT_APP
        )
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            retn = p.communicate()[0]
        except Exception as e:
            logging.critical("segment(%s) failed and try again:%s" % (sentence, e))
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            retn = p.communicate()[0]

        return retn

    def deparser(self, segment_result_str):
        segment_result = []
        try:
            segment_result_dict = json.loads(segment_result_str,
                                             object_hook=json_util._decode_dict)
            if "scw_out" in segment_result_dict and "wordsepbuf" in segment_result_dict["scw_out"]:
                wordsepbuf = segment_result_dict["scw_out"]["wordsepbuf"]
                wordsepbuf_split = wordsepbuf.strip("\t").split("\t")
                for word in wordsepbuf_split:

                    segment_result.append(word)
            else:
                logging.critical("segment result(%s) error without wordsepbuf"
                                 % segment_result_str)
        except ValueError as e:
            logging.critical("deparser segment result(%s) failed: %s" % (segment_result_str, e))
        return segment_result


def get_segment(ori_data):
    seg = SegmentProcess()
    result = seg.do_nlp_seg(ori_data)
    segment_result = seg.deparser(result)
    return segment_result


if __name__ == "__main__":
    print get_segment("同意 写论文的时候 用百度查一个庭园方面的术语\\\"ll")
