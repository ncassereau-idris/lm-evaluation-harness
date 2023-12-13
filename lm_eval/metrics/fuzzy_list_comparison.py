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
    if tot_num == 0:
        return float("nan")
    else:
        return tot_score / tot_num


def fscore(precision_func, recall_func, items):
    p = score(precision_func, items)
    r = score(recall_func, items)
    if p + r == 0:
        return float("nan")
    else:
        return (2 * p * r) / (p + r)


def precision(ref_list, pred_list):
    return len(pred_list), float(len([x for x in pred_list if x in ref_list]))


def recall(ref_list, pred_list):
    return len(ref_list), float(len([x for x in ref_list if x in pred_list]))


if __name__ == '__main__':
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file')
    args = parser.parse_args()

    refs, preds = [], []
    with open(args.output_file) as fp:
        for line in fp:
            dict_output = json.loads(line)
            preds.append(dict_output['pred'])
            assert len(dict_output['target']) == 1
            refs.append(dict_output['target'])
    p = score(precision, list(zip(refs, preds)))
    r = score(recall, list(zip(refs, preds)))
    f = fscore(precision, recall, list(zip(refs, preds)))
    print('Precision  =', p)
    print('Recall  =', r)
    print('F-score  =', f)
    
