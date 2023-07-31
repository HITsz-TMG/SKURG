import json
from evaluate import evaluate_predictions_example
import jsonlines

Qtype = ['TextQ',
         'TableQ',
         'ImageQ', 'Compose(TextQ,TableQ)',
         'ImageListQ',
         'Compose(TableQ,ImageListQ)', 'Compose(ImageQ,TableQ)', 'Compare(Compose(TableQ,ImageQ),TableQ)',
         'Compose(TableQ,TextQ)', 'Compare(TableQ,Compose(TableQ,TextQ))', 'Intersect(TableQ,TextQ)',
         'Compose(TextQ,ImageListQ)', 'Intersect(ImageListQ,TableQ)',
         'Compose(ImageQ,TextQ)',
         'Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ))',
         'Intersect(ImageListQ,TextQ)'
         ]

single = ['TextQ',
          'TableQ',
          'ImageQ']


def read_jsonlines(file_name):
    lines = []
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


path = 'file.json'

# path to the pred-file

golden_path = '/./MMQA/dataset/MMQA_dev.jsonl'
# path to MultimodalQA validation set

input_data = read_jsonlines(golden_path)

data = json.load(open(path, 'r'))
em_scores_single = 0
f1_scores_single = 0
text_total = 0
em_scores_multi = 0
f1_scores_multi = 0
image_total = 0
Retr_correct_single = 0
Retr_pred_single = 0
Retr_golden_single = 0
Retr_correct_multi = 0
Retr_pred_multi = 0
Retr_golden_multi = 0
for item in input_data:
    type = item['metadata']['type']
    id = item['qid']
    try:
        gen_ans_str = data[id]['gen_ans'].lower()
    except:
        print('miss')
        continue
    if type in single:
        id = item['qid']
        gen_ans_str = data[id]['gen_ans'].lower()
        golden_ans = data[id]['golden_ans']
        eval_scores = evaluate_predictions_example(gen_ans_str, golden_ans)
        em_scores_single += eval_scores['list_em'] * 100
        f1_scores_single += eval_scores['list_f1'] * 100
        gen_index = data[id]['gen_id']
        golden_index = data[id]['golden_id']
        Retr_correct_single += len(list(set(gen_index).intersection(set(golden_index))))
        Retr_golden_single += len(golden_index)
        Retr_pred_single += len(gen_index)
        text_total += 1
    else:
        id = item['qid']
        gen_ans_str = data[id]['gen_ans'].lower()
        golden_ans = data[id]['golden_ans']
        eval_scores = evaluate_predictions_example(gen_ans_str, golden_ans)
        em_scores_multi += eval_scores['list_em'] * 100
        f1_scores_multi += eval_scores['list_f1'] * 100
        gen_index = data[id]['gen_id']
        golden_index = data[id]['golden_id']
        Retr_correct_multi += len(list(set(gen_index).intersection(set(golden_index))))
        Retr_golden_multi += len(golden_index)
        Retr_pred_multi += len(gen_index)
        image_total += 1

print(path)
print('em_scores_single:', em_scores_single / text_total)
print('f1_scores_single:', f1_scores_single / text_total)
pre = Retr_correct_single / Retr_pred_single
recall = Retr_correct_single / Retr_golden_single
f1 = 2 * pre * recall / (pre + recall)
print('Re_scores_pre:', pre)
print('Re_scores_recall:', recall)
print('Re_scores_f1:', f1)
print('em_scores_multi:', em_scores_multi / image_total)
print('f1_scores_multi:', f1_scores_multi / image_total)
pre = Retr_correct_multi / Retr_pred_multi
recall = Retr_correct_multi / Retr_golden_multi
f1 = 2 * pre * recall / (pre + recall)
print('Re_scores_pre:', pre)
print('Re_scores_recall:', recall)
print('Re_scores_f1:', f1)
print('ALL EM:', (em_scores_single + em_scores_multi) / (text_total + image_total))
print('ALL F1:', (f1_scores_single + f1_scores_multi) / (text_total + image_total))
pre = (Retr_correct_multi + Retr_correct_single) / (Retr_pred_single + Retr_pred_multi)
recall = (Retr_correct_multi + Retr_correct_single) / (Retr_golden_multi + Retr_golden_single)
f1 = 2 * pre * recall / (pre + recall)
print('Re_scores_pre:', pre)
print('Re_scores_recall:', recall)
print('Re_scores_f1:', f1)
