#!/usr/bin/python
import re
from functools import partial


def make_list(text_list):
    if '\n' in text_list:
        tabline = text_list.split('\n')
    elif ';' in text_list:
        tabline = text_list.split(';')
    else:
        tabline = text_list.split(',')
    return tabline


def clean_element(element):
    element = re.sub("[\"'«»“”    ]+$", "", re.sub("^[\"'«»“”    ]+", "", element))
    element = element.strip()
    return element


def clean_up_list_elements(list_to_clean):
    return [x for x in set([clean_element(x) for x in list_to_clean]) if x != '']


def score(scoring_func, items):
    """
    Calculate the accuracy of a list of predictions against the list of references.
    The format of either can be comma-, semi-colon- or newline-separated and the order
    should not matter.
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    tot_score = 0
    tot_num = 0
    for ref, pred in zip(refs, preds):
        ref_list = clean_up_list_elements(make_list(ref[0])) # N.B. Assumes a single reference
        pred_list = clean_up_list_elements(make_list(pred))
        num_examples, score = scoring_func(ref_list, pred_list)
        tot_score += score
        tot_num += num_examples
    return tot_score/tot_num


def fscore(precision_func, recall_func, items):
    p = score(precision_func, items)
    r = score(recall_func, items)
    if p + r == 0:
        return float("nan")
    else:
        return (2 * p * r) / (p + r)


def sent_precision(ref_list, pred_list):
    if len(pred_list) == 0:
        return 0, 0
    else:
        return 1, float(len([x for x in pred_list if x in ref_list]))/len(pred_list)


def sent_recall(ref_list, pred_list):
    if len(ref_list) == 0:
        return 0, 0
    else:
        return 1, float(len([x for x in ref_list if x in pred_list]))/len(ref_list)


def precision(ref_list, pred_list):
    if len(pred_list) == 0:
        return 0, 0
    else:
        return len(pred_list), float(len([x for x in pred_list if x in ref_list]))


def recall(ref_list, pred_list):
    if len(ref_list) == 0:
        return 0, 0
    else:
        return len(ref_list), float(len([x for x in ref_list if x in pred_list]))
