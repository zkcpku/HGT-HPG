import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import subprocess
import argparse
import numpy as np


from collections import OrderedDict, Counter
# from tqdm import tqdm
from c2nl.inputters.timer import AverageMeter, Timer

from c2nl.eval.bleu import corpus_bleu
from c2nl.eval.rouge import Rouge
from c2nl.eval.meteor import Meteor


def normalize_answer(s):
    """Lower text and remove extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))
def eval_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    precision, recall, f1 = 0, 0, 0
    if len(ground_truth) == 0:
        if len(prediction) == 0:
            precision, recall, f1 = 1, 1, 1
    else:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same != 0:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def compute_eval_score(prediction, ground_truths):
    assert isinstance(prediction, str)
    precision, recall, f1 = 0, 0, 0
    for gt in ground_truths:
        _prec, _rec, _f1 = eval_score(prediction, gt)
        if _f1 > f1:
            precision, recall, f1 = _prec, _rec, _f1
    return precision, recall, f1


def eval_accuracies(hypotheses, references, copy_info, sources=None,
                    filename=None, print_copy_info=False, mode='dev'):
    """An unofficial evalutation helper.
     Arguments:
        hypotheses: A mapping from instance id to predicted sequences.
        references: A mapping from instance id to ground truth sequences.
        copy_info: Map of id --> copy information.
        sources: Map of id --> input text sequence.
        filename:
        print_copy_info:
    """
    assert (sorted(references.keys()) == sorted(hypotheses.keys()))

    # Compute BLEU scores
    # bleu_scorer = Bleu(n=4)
    # _, _, bleu = bleu_scorer.compute_score(references, hypotheses, verbose=0)
    # bleu = compute_bleu(references, hypotheses, max_order=4)['bleu']
    # _, bleu, ind_bleu = nltk_corpus_bleu(hypotheses, references)
    try:
        _, bleu, ind_bleu = corpus_bleu(hypotheses, references)
    except Exception as e:
        print("bleu error: " + str(e))
        _,bleu, ind_bleu = 0, 0, 0

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    if mode == 'test':
        meteor_calculator = Meteor()
        meteor, _ = meteor_calculator.compute_score(references, hypotheses)
    else:
        meteor = 0

    f1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()

    fw = open(filename, 'w') if filename else None
    for key in references.keys():
        _prec, _rec, _f1 = compute_eval_score(hypotheses[key][0],
                                              references[key])
        precision.update(_prec)
        recall.update(_rec)
        f1.update(_f1)
        if fw:
            if copy_info is not None and print_copy_info:
                prediction = hypotheses[key][0].split()
                pred_i = [word + ' [' + str(copy_info[key][j]) + ']'
                          for j, word in enumerate(prediction)]
                pred_i = [' '.join(pred_i)]
            else:
                pred_i = hypotheses[key]

            logobj = OrderedDict()
            logobj['id'] = key
            if sources is not None:
                logobj['code'] = sources[key]
            logobj['predictions'] = pred_i
            logobj['references'] = references[key][0] if args.print_one_target \
                else references[key]
            logobj['bleu'] = ind_bleu[key]
            logobj['rouge_l'] = ind_rouge[key]
            fw.write(json.dumps(logobj) + '\n')

    if fw: fw.close()
    return bleu * 100, rouge_l * 100, meteor * 100, precision.avg * 100, \
           recall.avg * 100, f1.avg * 100

def list2seq(l):
    return ''.join([e +' ' for e in l])

def unit_test():
    # test bleu

    references = [['np', '.', 'concatenate', '(', '(', 'A', ',', 'B', ')', ')'],['df', '=', 'df', '[', '(', 'df', '[', "'", 'closing_price', "'", ']', '>=', '99', ')', '&', '(', 'df', '[', "'", 'closing_price', "'", ']', '<=', '101', ')', ']']]
    hypotheses = [['numpy', '.', 'linspace', '(', '1', ',', '2', ',', '3', ')', '.', 'transpose', '(', ')'],['df', '.', 'groupby', '(', '~', 'df', '.', 'columns', ')']]
    # references = [['a', 'b', 'c'], ['a', 'b', 'd']]
    ref = {i: [list2seq(references[i])] for i in range(len(references))}
    # hypotheses = [['a', 'b', 'c'], ['a', 'b', 'd']]
    pred = {i: [list2seq(hypotheses[i])] for i in range(len(hypotheses))}
    bleu, rouge_l, meteor, precision, recall, f1 = eval_accuracies(pred, ref, None)

    return {'bleu': bleu, 'rouge_l': rouge_l, 'meteor': meteor, 'precision': precision, 'recall': recall, 'f1': f1}
    # print("bleu", bleu, '\t', "rouge_l", rouge_l, '\t', "meteor", meteor, '\t')
    # print("precision", precision, '\t', "recall", recall, '\t', "f1", f1)
    # assert bleu == 100

    # ref = {i:[list2seq(train_data[0][i][0]),list2seq(train_data[0][i][1])] for i in range(len(train_data[0]))}
#     ref = {}
#     for i in range(len(train_data[0])):
#         now_l = []
#         if len(train_data[0][i][1]) != 0:
#             now_l.append(list2seq(train_data[0][i][1]))
#         else:
#             now_l.append(list2seq(train_data[0][i][0]))
#         ref[i] = now_l
#     pred = {i:[list2seq(train_data[1][i])] for i in range(len(train_data[1]))}
# #     print(ref[:2])
# #     print(pred[:2])
#     print(len(pred))
#     assert(len(pred) == len(ref))
#     eval_rst = eval_accuracies(pred,ref,None)
#     print(eval_rst)


def corpus_test(all_labels, all_preds):
    references = all_labels

    hypotheses = []
    for hyp in all_preds:
        # print(hyp)
        non_zero_hyp = [e for e in hyp if len(e) > 0]
        if len(non_zero_hyp) > 0:
            hypotheses.append(non_zero_hyp[0])
        else:
            hypotheses.append(['<s>'])
    # hypotheses = [e[0] if len(e) > 0 else [] for e in all_preds]
    
    ref = {i: [list2seq(references[i])] for i in range(len(references))}
    # hypotheses = [['a', 'b', 'c'], ['a', 'b', 'd']]
    pred = {i: [list2seq(hypotheses[i])] for i in range(len(hypotheses))}
    bleu, rouge_l, meteor, precision, recall, f1 = eval_accuracies(pred, ref, None)

    return {'bleu': bleu, 'rouge_l': rouge_l, 'meteor': meteor, 'precision': precision, 'recall': recall, 'f1': f1}



if __name__ == "__main__":
    unit_test()
