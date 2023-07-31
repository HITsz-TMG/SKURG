import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import json
import string
from BARTScore.bart_score import BARTScorer
from webqa_eval import webqa_metrics_approx
import numpy as np
from tqdm import tqdm

TABLE = str.maketrans(dict.fromkeys(string.punctuation))


def normalize_text_for_bart(x):  # Light text normalization for WebQA eval: white space fix + punctuation removal
    return " ".join(x.translate(TABLE).split())


def compute_bartscore_ParaBank(c, a, switch=False):
    bart_scorer_ParaBank = BARTScorer(device='cuda', checkpoint='/./bart-large-cnn')
    bart_scorer_ParaBank.load(
        path='/./MMQA/OFA/utils/BARTScore/bart_score.pth')  # Please change the path to bart.pth
    c_removepunc = [normalize_text_for_bart(x) for x in c]
    a_removepunc = [normalize_text_for_bart(x) for x in a]
    if switch:
        score = np.exp(bart_scorer_ParaBank.score(c_removepunc, a_removepunc))
    else:
        score = np.exp(bart_scorer_ParaBank.score(a_removepunc, c_removepunc))
    return score


path = 'file.json'
# path to pred-file

pred = json.load(open(path, 'r'))
n_examples = 0
total_correct = 0
total_re_num = 0
total_sc_num = 0
acc_scores = []
golden_res_list = []
gen_res_list = []
golden = json.load(open('/./MMQA/dataset/webqa/WebQA_train_val.json', 'r'))
# path to WebQA validation set

pbar = tqdm(total=len(pred))
for key, value in pred.items():
    pos_source = []
    golden_ans = golden[key]['A'][0].lower()
    if golden_ans[0] == '"':
        golden_ans = golden_ans[1:-1]
    pos_source = []
    for item in golden[key]['img_posFacts']:
        pos_source.append(item['image_id'])
    for item in golden[key]['txt_posFacts']:
        pos_source.append(item['snippet_id'])
    n_examples += 1
    pred_ans = value['gen_ans']
    answer = ''
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    exclude = list(string.punctuation)
    for idx, char in enumerate(pred_ans):
        if char in exclude:
            if pred_ans[idx - 1] == ' ':
                answer = answer[:-1]
            answer += char
        else:
            answer += char
    pred_ans = answer.lower()

    Qcate = golden[key]['Qcate']
    res_dict = webqa_metrics_approx(pred_ans.lower(), golden_ans, Qcate)
    accuracy = res_dict['acc_approx']
    acc_scores.append(accuracy)
    golden_res_list.append(golden_ans)
    gen_res_list.append(pred_ans)
    total_sc_num += len(value['gen_id'])
    total_re_num += len(value['golden_id'])
    total_correct += len(list(set(value['gen_id']).intersection(set(value['golden_id']))))
    pbar.update(1)
print('Re pre:', total_correct / total_sc_num)
print('Re recall:', total_correct / total_re_num)
pre = total_correct / total_sc_num
recall = total_correct / total_re_num
f1 = 2 * pre * recall / (pre + recall)
print('Re F1:', f1)
normalizer = compute_bartscore_ParaBank(golden_res_list, golden_res_list)
BARTscore = compute_bartscore_ParaBank(gen_res_list, golden_res_list) / np.array(normalizer)
bart_scores = np.where(BARTscore > 1, 1, BARTscore)
acc_scores = np.array(acc_scores)
QA_scores = bart_scores * acc_scores
bart_scores = bart_scores.mean()
acc_scores = acc_scores.mean()
QA_scores = QA_scores.mean()
print(path)
print('acc_score:', acc_scores)
print('bart_scores:', bart_scores)
print('QA:', QA_scores)
