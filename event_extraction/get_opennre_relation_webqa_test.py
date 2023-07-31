import opennre
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from tqdm import tqdm
import re
import random
from threading import Thread

entities = json.load(
    open('../dataset/webqa/WebQA_entities_dict.json', 'r'))
# file that contains the entities in WebQA

ws_lists = json.load(open('../dataset/webqa/WebQA_facts_ws_test.json', 'r'))
# file that contains facts using Coreference Resolution

file_path = '../dataset/webqa/WebQA_test.json'
# WebQA test set

data = json.load(open(file_path, 'r'))

relations = {}


def get_rel(keys):
    pbar = tqdm(total=len(keys))
    model = opennre.get_model('wiki80_bert_softmax')
    model = model.cuda(device=4)
    for key in keys:
        value = data[key]
        txt_facts = value['txt_Facts']
        img_facts = value['img_Facts']
        for fact in txt_facts:
            id = fact['snippet_id']
            relations[id] = {}
            ws = ws_lists[id]
            entity = entities[id]
            for title_ent in entity['title']:
                try:
                    start = [substr.start() for substr in re.finditer(title_ent, ws)]
                except:
                    print('Do not find the entity!')
                    continue
                tmp_rel = {}
                if len(start) == 0:
                    relations[id][title_ent] = tmp_rel
                    continue
                pos_start = start[0]
                pos_end = pos_start + len(title_ent)
                for fact_ent in entity['facts']:
                    if fact_ent == title_ent:
                        continue
                    try:
                        start = [substr.start() for substr in re.finditer(fact_ent, ws[pos_end:])]
                    except:
                        print('Do not find the entity!')
                        continue
                    if len(start) == 0:
                        print()
                        continue
                    start = random.sample(start, 1)
                    t_start = start[0] + pos_end
                    t_end = t_start + len(fact_ent)
                    rel = model.infer({'text': ws, 'h': {'pos': (pos_start, pos_end)}, 't': {'pos': (t_start, t_end)}})
                    tmp_rel[fact_ent] = rel[0]
                relations[id][title_ent] = tmp_rel
        for fact in img_facts:
            id = str(fact['image_id'])
            relations[id] = {}
            if id in ws_lists.keys():
                ws = ws_lists[id]
            else:
                ws = fact['caption']
            entity = entities[id]
            for title_ent in entity['title']:
                try:
                    start = [substr.start() for substr in re.finditer(title_ent, ws)]
                except:
                    print('Do not find the entity!')
                    continue
                tmp_rel = {}
                if len(start) == 0:
                    relations[id][title_ent] = tmp_rel
                    continue
                pos_start = start[0]
                pos_end = pos_start + len(title_ent)
                for fact_ent in entity['title']:
                    if fact_ent == title_ent:
                        continue
                    try:
                        start = [substr.start() for substr in re.finditer(fact_ent, ws)]
                    except:
                        print('Do not find the entity!')
                        continue
                    if len(start) == 0:
                        print()
                        continue
                    start = random.sample(start, 1)
                    t_start = start[0]
                    t_end = t_start + len(fact_ent)
                    rel = model.infer({'text': ws, 'h': {'pos': (pos_start, pos_end)}, 't': {'pos': (t_start, t_end)}})
                    tmp_rel[fact_ent] = rel[0]
                relations[id][title_ent] = tmp_rel
        pbar.update(1)


def split_df():
    # multiprocessing
    thres_list = []
    count = 0
    keys = list(data.keys())
    split_count = 500
    times = len(keys) // split_count + 1
    for item in range(times):
        data_spilt = keys[count:count + split_count]
        count += split_count
        thread = Thread(target=get_rel, args=(data_spilt,))
        thres_list.append(thread)
        thread.start()
    for item in thres_list:
        item.join()


if __name__ == '__main__':
    y = split_df()
    res = json.dumps(relations, indent=4)
    with open('../dataset/webqa/WebQA_entities_relation_opennre_test.json',
              'w') as f:
        f.write(res)
    print('Write to ../dataset/webqa/WebQA_entities_relation_opennre_test.json')
