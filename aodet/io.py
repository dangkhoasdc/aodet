"""
Writer/Reader for Confusion Matrix
"""

import pandas as pd
from collections import defaultdict
import re
from .cfm import DetectionConfusionMatrix
from .common import index, intersect_list
import pandas as pd
import warnings

# TODO: whether support the mapping features ??
def load_mapping_file(filepath):
    """
    Load mapping file with following format:
        ref_exp, test_exp
        labelA, label1
        labelB, label2
    It will map  labelA into label1 and labelB into label2, etc.
    Arguments:
        :param filepath: path to the file
        :type filepath: str
        :return: (list of headers, dict of mapping)
    """
    mapping_df = pd.read_csv(filepath)
    headers = list(mapping_df)

    mapping = pd.read_csv(filepath, index_col=0, squeeze=True).to_dict()
    print(mapping)

    return headers, mapping

def get_mapping_index(df, mapping, is_detection_system=True):
    """
    given a mapping, find the corresponding index in the confusion matrix
    Arguments:
        :param df: input data frame
        :type df: DataFrame
        :param mapping: a map whose index is the source label and its value is
        the destination label
        :type mapping: map

    Returns:
        :return: a map whose index is the destination label and its value is
        a list of index in the confusion matrix
        :rtype: map
    """

    headers = list(df)
    mapping_index = defaultdict(list)

    def check_label(txt, lbl):
        if is_detection_system:
            lbl_regex = r"(\w+)"
        else:
            lbl_regex = r"^(\w+)\s\((\d+)/\d+\)$"
        m = re.search(lbl_regex, txt)
        return m and m.group(1) == lbl

    for src_lbl, dst_lbl in mapping.items():
        try:
            idx = index(headers, lambda x: check_label(x, src_lbl))
        except Exception as e:
            continue

        mapping_index[dst_lbl].append(idx)

    return mapping_index

class Reader:

    @staticmethod
    def from_search_sys(fpath, mapping_file):
        pass


    @staticmethod
    def from_detection_sys(fpath, mapping_file):
        """
        compute statistic from the confusion matrix, the format is from the
        detection system.
        Paramters:
            :param df: DataFrame of confusion matrix
            :param mapping: label mapping
        """

        if mapping_file is not None:
            exps, mapping = load_mapping_file(mapping_file)
        else:
            # use default mapping
            pass

        df = pd.read_csv(fpath)
        zero_rows = [idx for idx, r in df.iterrows() if int(r[-1]) == 0]

        df.drop(zero_rows, inplace=True)
        mapping_indexes = get_mapping_index(df, mapping)
        detcfm = DetectionConfusionMatrix(list(set(mapping.values())))
        lbls = set(mapping.keys())
        for idx, row in df.iterrows():
            if row[0] in lbls:
                dstgt = mapping[row[0]]
                # TODO: refactor this
                for prdlbl in list(df):
                    if prdlbl in lbls:
                        dstpred = mapping[prdlbl]
                        v = detcfm.get(dstgt, dstpred)
                        detcfm.set_value(dstgt, dstpred, v + int(row[prdlbl]))
                    elif prdlbl == "*NO_MATCH*":
                        v = detcfm.get_nomatch(dstgt)
                        detcfm.set_nomatch(dstgt, v + int(row[prdlbl]))
                    elif prdlbl == "*DUPLICATE*":
                        v = detcfm.get_duplicate(dstgt)
                        detcfm.set_duplicate(dstgt, v + int(row[prdlbl]))
                    else:
                        warnings.warn("Skip {}".format(prdlbl), UserWarning)


            elif row[0] == "*NO_MATCH*":
                for lbl in intersect_list(lbls, list(df)):
                    dstpred = mapping[lbl]
                    v = detcfm.get_nogt_spec(dstpred)
                    detcfm.set_nogt_spec(dstpred, v + int(row[lbl]))

        return detcfm


class Writer:
    @staticmethod
    def to_json(cfmat):
        """ convert to json format  """
        pass
