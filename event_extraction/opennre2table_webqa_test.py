# 建立entities_table并链接各模态
import sys

sys.path.append('../')
import json
import pickle
from tqdm import tqdm
from OFA.ofa import OFATokenizer
import random

tokenizer = OFATokenizer.from_pretrained('./OFA')
tokenizer.add_special_tokens(
    {'additional_special_tokens': ['<title>', '</title>', 'ROW', '[b_ans]', '[e_ans]', '[b_source]', '[e_source]']})

MAX_LENGTH = 760

opennre_data = json.load(
    open('./MMQA/dataset/webqa/WebQA_entities_relation_opennre_test.json', 'r'))
# file that contains entities and relations.

file_path = './MMQA/dataset/webqa/WebQA_test.json'
# WebQA test set

data = json.load(open(file_path, 'r'))
pbar = tqdm(total=len(data))
table_connect = {}
tables_feats = {}
tables_spans = {}

for key, value in data.items():
    table_context = "ROW 1 :"
    txt_facts = value['txt_Facts']
    img_facts = value['img_Facts']
    entities = {}
    for item in txt_facts:
        try:
            tmp_entities = opennre_data[item['snippet_id']]
            entities.update(tmp_entities)
        except:
            print('Do not find entities.')
            continue
    for item in img_facts:
        try:
            tmp_entities = opennre_data[str(item['image_id'])]
            entities.update(tmp_entities)
        except:
            print('Do not find entities.')
            continue
    index = 2
    connect_info = []
    span_info = []
    dict_key_ls = list(entities.keys())
    random.shuffle(dict_key_ls)
    tmp_index = 0
    source_has_relation = {}

    for title_ent in dict_key_ls:
        fact_ent_rel = entities[title_ent]
        if title_ent == '':
            continue
        start_index = len(tokenizer.tokenize(table_context))
        if start_index > MAX_LENGTH:
            break
        table_context += title_ent
        table_context += ' ;'
        end_index = len(tokenizer.tokenize(table_context))
        span_info.append([start_index, end_index])
        if start_index > MAX_LENGTH:
            break
        tmp_connect_info = {}
        tmp_connect_info['span'] = [start_index, end_index]
        tmp_connect_info['connect'] = {}
        tmp_connect_info['connect']['text'] = []
        tmp_connect_info['connect']['image'] = []
        for fact in txt_facts:
            if title_ent.lower() in fact['fact'].lower() or title_ent in fact['title'].lower():
                id = fact['snippet_id']
                if id not in source_has_relation.keys():
                    tmp_connect_info['connect']['text'].append(id)
                    source_has_relation[id] = tmp_index
                else:
                    if random.random() < 0.5:
                        # If a source contains several entities, we randomly choose one entity.
                        tmp_connect_info['connect']['text'].append(id)
                        # add the aligned source ID to the dictionary of the new head entity
                        connect_info[source_has_relation[id]]['connect']['text'].remove(id)
                        # remove the source ID from the dictionary of the former head entity
                        source_has_relation[id] = tmp_index
                        # save the new alignment between the source ID and head entity index

        for fact in img_facts:
            if title_ent.lower() in fact['caption'].lower():
                id = fact['image_id']
                if id not in source_has_relation.keys():
                    tmp_connect_info['connect']['image'].append(id)
                    source_has_relation[id] = tmp_index
                else:
                    if random.random() < 0.5:
                        # If a source contains several entities, we randomly choose one entity.
                        tmp_connect_info['connect']['image'].append(id)
                        connect_info[source_has_relation[id]]['connect']['image'].remove(id)
                        source_has_relation[id] = tmp_index

        connect_info.append(tmp_connect_info)
        tmp_index += 1
        for fact_ent, rel in fact_ent_rel.items():
            if fact_ent == '':
                continue
            table_context += ' ' + rel + ' ,'
            table_context += ' ' + fact_ent + ' ;'
        table_context = table_context[:-2]
        table_context += ' . ROW ' + str(index) + ' :'
        index += 1
    table_context = table_context[:-(8 + len(str(index - 1)))]
    table_tokens = tokenizer.tokenize(table_context)
    table_tokens = tokenizer.convert_tokens_to_ids(table_tokens)
    tables_feats[key] = {'ids': table_tokens}
    table_connect[key] = connect_info
    tables_spans[key] = span_info
    pbar.update(1)

pickle.dump(tables_feats,
            open('./MMQA/dataset/webqa/WebQA_opennre_tables_ofa_feats_rel_test.pkl',
                 'wb'))
print('save to', './MMQA/dataset/webqa/WebQA_opennre_tables_ofa_feats_rel_test.pkl')
# tokenized ids to the linearized KG

pickle.dump(table_connect,
            open('./MMQA/dataset/webqa/WebQA_opennre_tables_connect_rel_test.pkl',
                 'wb'))
print('save to', './MMQA/dataset/webqa/WebQA_opennre_tables_connect_rel_test.pkl')
# alignment between sources and head entities

pickle.dump(tables_spans,
            open('./MMQA/dataset/webqa/WebQA_opennre_tables_spans_rel_test.pkl',
                 'wb'))
print('save to', './MMQA/dataset/webqa/WebQA_opennre_tables_spans_rel_test.pkl')
# spans of the head entities in the linearized KGs
