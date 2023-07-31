import numpy as np
import os
import json, time, copy
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--Qcate_breakdown", type=str, default='["all"]')
parser.add_argument("--file", type=str, default='dataset/webqa/WebQA_train_val.json')
parser.add_argument('--no_norm', action='store_true')
parser.add_argument('--dir', type=str, default='/mnt/inspurfs/user-fs/yangqian/MMQA/')
parser.add_argument('--output_idx', type=int, default=0)
args = parser.parse_args()

import sys

sys.path.append("/mnt/inspurfs/user-fs/yangqian/MMQA/OFA/utils/BARTScore")

from bart_score import BARTScorer

bart_scorer_ParaBank = BARTScorer(device='cuda', checkpoint='/mnt/inspurfs/user-fs/yangqian/bart-large-cnn')
bart_scorer_ParaBank.load(
    path='/mnt/inspurfs/user-fs/yangqian/MMQA/OFA/utils/BARTScore/bart_score.pth')


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


# VQA Eval (SQuAD style EM, F1)
def compute_vqa_metrics(cands, a, exclude="", domain=None):
    if len(cands) == 0: return (0, 0, 0)
    bow_a = normalize_text(a).split()
    F1 = []
    EM = 0
    RE = []
    PR = []
    e = normalize_text(exclude).split()
    for c in cands:
        bow_c = [w for w in normalize_text(c).split() if not w in e]
        if domain == {"NUMBER"}:
            bow_c = detectNum(bow_c)
            bow_a = detectNum(bow_a)
        elif domain is not None:
            bow_c = list(domain.intersection(bow_c))
            bow_a = list(domain.intersection(bow_a))

        # print(bow_c)
        # print(bow_a)
        if bow_c == bow_a:
            EM = 1
        common = Counter(bow_a) & Counter(bow_c)
        num_same = sum(common.values())
        if num_same == 0:
            return (0, 0, 0, 0, 0)
        precision = 1.0 * num_same / len(bow_c)
        recall = 1.0 * num_same / len(bow_a)
        RE.append(recall)
        PR.append(precision)

        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        F1.append(f1)

    PR_avg = np.mean(PR)
    RE_avg = np.mean(RE)
    F1_avg = np.mean(F1)
    F1_max = np.max(F1)
    return (F1_avg, F1_max, EM, RE_avg, PR_avg)


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


Qcate_breakdown = json.loads(args.Qcate_breakdown)
print("Use categories: ", Qcate_breakdown)
print("Use normalization = ", not args.no_norm)
print("Output_idx = ", args.output_idx)
# Please change the path to your output folder
with open(os.path.join(args.dir, args.file), "r") as fp:
    # lines = fp.readlines()
    # header = lines[0].strip().split('\t')
    # rows = lines[1:]
    rows = json.load(fp)
split_input_data = {}
for key, value in rows.items():
    if value['split'] == 'val':
        split_input_data[key] = value
pred_path = 'OFA/outputs/webqa/VLP_x101_combinedTraining_val_end2end_predictions.json'
with open(os.path.join(args.dir, pred_path), "r") as fp:
    pred = json.load(fp)
F1_avg_scores = {'All': [], 'number': [], 'YesNo': [], 'choose': [], 'color': [], 'shape': [], 'Others': [], 'text': []}
F1_max_scores = {'All': [], 'number': [], 'YesNo': [], 'choose': [], 'color': [], 'shape': [], 'Others': [], 'text': []}
EM_scores = {'All': [], 'number': [], 'YesNo': [], 'choose': [], 'color': [], 'shape': [], 'Others': [], 'text': []}
RE_scores = {'All': [], 'number': [], 'YesNo': [], 'choose': [], 'color': [], 'shape': [], 'Others': [], 'text': []}
PR_scores = {'All': [], 'number': [], 'YesNo': [], 'choose': [], 'color': [], 'shape': [], 'Others': [], 'text': []}
fluency_scores = {'All': [], 'number': [], 'YesNo': [], 'choose': [], 'color': [], 'shape': [], 'Others': [],
                  'text': []}
acc_scores = {'All': [], 'number': [], 'YesNo': [], 'choose': [], 'color': [], 'shape': [], 'Others': [], 'text': []}
mul_scores = {'All': [], 'number': [], 'YesNo': [], 'choose': [], 'color': [], 'shape': [], 'Others': [], 'text': []}

output_Q = []
output_A = []
output_N = []
output_O = []
output_KA = []
output_G = []
output_QC = []
# Guid	Qcate	Q	A	Keywords_A	Output_conf	Output
for r in tqdm(split_input_data.items()):

    # datum = r.strip().split('\t')
    Qcate = r[1]['Qcate']
    if (not 'all' in Qcate_breakdown) and (not Qcate in Qcate_breakdown): continue
    O = pred[r[1]['Guid']]['predicted answers'][0]
    # C是对应的context
    C = [O]

    # A是answer
    A = r[1]['A']
    Keywords_A = A[0]
    # normalizer = guid2norm[datum[key['Guid']]]
    normalizer = compute_bartscore_ParaBank(A, A)

    output_Q.append(r[1]['Q'])
    output_A.append(A)
    output_N.append(normalizer)
    output_O.append(O)
    output_KA.append(Keywords_A)
    # Guide是key
    output_G.append(r[1]['Guid'])
    output_QC.append(Qcate)

    if args.no_norm:
        score = min(1, np.max(compute_bartscore_ParaBank(C * len(A), A)))
    else:
        score = min(1, np.max(compute_bartscore_ParaBank(C * len(A), A) / np.array(normalizer)))

    if Qcate == 'color':
        F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics(C, Keywords_A, "", color_set)
    elif Qcate == 'shape':
        F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics(C, Keywords_A, "", shape_set)
    elif Qcate == 'YesNo':
        F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics(C, Keywords_A, "", yesno_set)
    elif Qcate == 'number':
        F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics(C, Keywords_A, "", {"NUMBER"})
    else:
        F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics(C, Keywords_A)
    fluency_scores['All'].append(score)
    fluency_scores[Qcate].append(score)
    if Qcate in ['color', 'shape', 'number', 'YesNo']:
        acc_scores['All'].append(F1_avg)
        acc_scores[Qcate].append(F1_avg)
        mul_scores['All'].append(F1_avg * score)
        mul_scores[Qcate].append(F1_avg * score)
    else:
        acc_scores['All'].append(RE_avg)
        acc_scores[Qcate].append(RE_avg)
        mul_scores['All'].append(RE_avg * score)
        mul_scores[Qcate].append(RE_avg * score)

    F1_avg_scores['All'].append(F1_avg)
    F1_max_scores['All'].append(F1_max)
    EM_scores['All'].append(EM)
    RE_scores['All'].append(RE_avg)
    PR_scores['All'].append(PR_avg)

    F1_avg_scores[Qcate].append(F1_avg)
    F1_max_scores[Qcate].append(F1_max)
    EM_scores[Qcate].append(EM)
    RE_scores[Qcate].append(RE_avg)
    PR_scores[Qcate].append(PR_avg)

assert len(F1_avg_scores) == len(F1_max_scores) == len(EM_scores) == len(RE_scores) == len(PR_scores) == len(
    fluency_scores) == len(acc_scores) == len(mul_scores)
assert len(output_Q) == len(output_A) == len(output_N) == len(output_O) == len(output_KA) == len(output_G) == len(
    output_QC) == len(mul_scores['All'])

print("#eval samples = ", len(mul_scores['All']))
F1_avg = np.mean(F1_avg_scores['All'])
F1_max = np.mean(F1_max_scores['All'])
EM = np.mean(EM_scores['All'])
RE_avg = np.mean(RE_scores['All'])
PR_avg = np.mean(PR_scores['All'])

fluency_avg = np.mean(fluency_scores['All'])
acc_avg = np.mean(acc_scores['All'])
mul_avg = np.mean(mul_scores['All'])

print("F1_avg = {}".format(F1_avg))
# print("F1_max = {}".format(F1_max))
# print("EM = {}".format(EM))
print("RE_avg = {}".format(RE_avg))
# print("PR_avg = {}".format(PR_avg))
print("fluency_avg = {}".format(fluency_avg))
print("acc_avg = {}".format(acc_avg))
print("mul_avg = {}".format(mul_avg))
print(" ------------------------------------------------------------------------------------ \n")