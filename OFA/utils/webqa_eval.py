import numpy as np
import os
import json, time, copy
import math

from tqdm import tqdm
import random
import pickle
from datetime import datetime
from pytz import timezone
from word2number import w2n
import string, re
from collections import Counter, defaultdict
from pprint import pprint
import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat", "parser"])
np.set_printoptions(precision=4)

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--Qcate_breakdown", type=str, default='["all"]')
# parser.add_argument("--file", type=str)
# parser.add_argument('--no_norm', action='store_true')
# parser.add_argument('--dir', type=str)
# parser.add_argument('--output_idx', type=int, default=0)
# args = parser.parse_args()

import sys

sys.path.append("/mnt/inspurfs/user-fs/yangqian/MMQA/OFA/utils/BARTScore")
from OFA.utils.BARTScore.bart_score import BARTScorer

# bart_scorer_ParaBank = BARTScorer(device='cuda', checkpoint='/mnt/inspurfs/user-fs/yangqian/bart-large-cnn')
# bart_scorer_ParaBank.load(path='/mnt/inspurfs/user-fs/yangqian/MMQA/OFA/utils/BARTScore/bart_score.pth') # Please change the path to bart.pth


def detectNum(l):
    result = []
    for w in l:
        try:
            result.append(str(int(w)))
        except:
            pass
    return result


def toNum(word):
    if word == 'point': return word
    try:
        return w2n.word_to_num(word)
    except:
        return word


def normalize_text(s):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):  # additional: converting numbers to digit form
        return " ".join([str(toNum(w)) for w in text.split()])

    def remove_punc(text):
        exclude = set(string.punctuation) - set(['.'])
        text1 = "".join(ch for ch in text if ch not in exclude)
        return re.sub(r"\.(?!\d)", "", text1)  # remove '.' if it's not a decimal point

    def lower(text):
        return text.lower()

    def lemmatization(text):
        return " ".join([token.lemma_ for token in nlp(text)])

    if len(s.strip()) == 1:
        # accept article and punc if input is a single char
        return white_space_fix(lower(s))
    elif len(s.strip().split()) == 1:
        # accept article if input is a single word
        return lemmatization(white_space_fix(remove_punc(lower(s))))

    return lemmatization(white_space_fix(remove_articles(remove_punc(lower(s)))))


color_set = {'orangebrown', 'spot', 'yellow', 'blue', 'rainbow', 'ivory', 'brown', 'gray', 'teal', 'bluewhite',
             'orangepurple', 'black', 'white', 'gold', 'redorange', 'pink', 'blonde', 'tan', 'turquoise', 'grey',
             'beige', 'golden', 'orange', 'bronze', 'maroon', 'purple', 'bluere', 'red', 'rust', 'violet',
             'transparent', 'yes', 'silver', 'chrome', 'green', 'aqua'}
shape_set = {'globular', 'octogon', 'ring', 'hoop', 'octagon', 'concave', 'flat', 'wavy', 'shamrock', 'cross',
             'cylinder', 'cylindrical', 'pentagon', 'point', 'pyramidal', 'crescent', 'rectangular', 'hook', 'tube',
             'cone', 'bell', 'spiral', 'ball', 'convex', 'square', 'arch', 'h', 'cuboid', 'step', 'rectangle', 'dot',
             'oval', 'circle', 'star', 'crosse', 'crest', 'octagonal', 'cube', 'triangle', 'semicircle', 'domeshape',
             'obelisk', 'corkscrew', 'curve', 'circular', 'xs', 'slope', 'pyramid', 'round', 'bow', 'straight',
             'triangular', 'heart', 'fork', 'teardrop', 'fold', 'curl', 'spherical', 'diamond', 'keyhole', 'conical',
             'dome', 'sphere', 'bellshaped', 'rounded', 'hexagon', 'flower', 'globe', 'torus'}
yesno_set = {'yes', 'no'}

TABLE = str.maketrans(dict.fromkeys(string.punctuation))


def normalize_text_for_bart(x):  # Light text normalization for WebQA eval: white space fix + punctuation removal
    return " ".join(x.translate(TABLE).split())


def compute_bartscore_ParaBank(c, a, switch=False):
    c_removepunc = [normalize_text_for_bart(x) for x in c]
    a_removepunc = [normalize_text_for_bart(x) for x in a]
    if switch:
        score = np.exp(bart_scorer_ParaBank.score(c_removepunc, a_removepunc))
    else:
        score = np.exp(bart_scorer_ParaBank.score(a_removepunc, c_removepunc))
    return score


def _webqa_acc_approx(predction, ground_truth, domain=None):
    """VQA Eval (SQuAD style EM, F1)"""
    bow_pred = normalize_text(predction).split()
    bow_target = normalize_text(ground_truth).split()
    if domain == {"NUMBER"}:
        bow_pred = detectNum(bow_pred)
        bow_target = detectNum(bow_target)
    elif domain is not None:
        bow_pred = list(domain.intersection(bow_pred))
        bow_target = list(domain.intersection(bow_target))
    else:
        # TODO: fine-grained evaluation (e.g., content words) for text question types
        bow_pred = bow_pred
        bow_target = bow_target

    common = Counter(bow_target) & Counter(bow_pred)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = num_same / len(bow_pred)
    recall = num_same / len(bow_target)
    f1 = 2 * precision * recall / (precision + recall)

    return f1, recall, precision


def webqa_metrics_approx(prediction, ground_truth, Qcate="text"):
    f1, recall, precision = _webqa_acc_approx(
        prediction,
        ground_truth,
        domain={
            "color": color_set,
            "shape": shape_set,
            "YesNo": yesno_set,
            "number": {"NUMBER"},
            "text": None,
            "Others": None,
            "choose": None,
        }[Qcate],
    )
    if Qcate in ["color", "shape", "number", "YesNo"]:
        accuracy = f1
    else:
        accuracy = recall
    return {"acc_approx": accuracy}
