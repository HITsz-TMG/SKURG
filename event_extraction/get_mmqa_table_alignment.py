import sys

sys.path.append('/')
import torch
import os
import numpy as np
import json
from tqdm import tqdm
from ofa import OFATokenizer
import jsonlines
import pickle


def read_jsonlines(file_name):
    lines = []
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


tokenizer = OFATokenizer.from_pretrained('./OFA')
tokenizer.add_special_tokens(
    {'additional_special_tokens': ['<title>', '</title>', 'ROW', '[b_ans]', '[e_ans]', '[b_source]', '[e_source]']})


def main():
    texts = read_jsonlines('./mmqa/dataset/MMQA_texts.jsonl')
    images = read_jsonlines('./mmqa/dataset/MMQA_images.jsonl')
    tables = read_jsonlines('./mmqa/dataset/MMQA_tables.jsonl')

    examples = read_jsonlines('./mmqa/dataset/MMQA_train.jsonl')
    texts_dict = {doc["id"]: doc for doc in texts}
    images_dict = {doc["id"]: doc for doc in images}
    tables_dict = {doc["id"]: doc for doc in tables}
    pbar = tqdm(total=len(examples))
    table_connect = {}
    has_feat_image_ids = os.listdir('./mmqa/dataset/MMQA_images_ofa_feats')
    for example in examples:
        table = tables_dict[example['metadata']['table_id']]
        texts_url = {}
        for key in example['metadata']['text_doc_ids']:
            texts_url[texts_dict[key]['url']] = key
        images_url = {}
        for key in example['metadata']['image_doc_ids']:
            if key + '.pkl' in has_feat_image_ids:
                images_url[images_dict[key]['url']] = key
            else:
                continue
        connect_info = []
        rows = table['table']['table_rows']
        title = table['title']
        title = tokenizer.tokenize(' ' + title)
        title = ['<title>'] + title + ['</title>']
        table_context = "ROW 1 :"
        index = 2
        for row_data in rows:
            for cell in row_data:
                text = cell['text']
                if text == '':
                    continue
                start_index = len(tokenizer.tokenize(table_context)) + len(title)
                table_context += ' ' + text
                end_index = len(tokenizer.tokenize(table_context)) + len(title)
                table_context += ' ;'
                links = cell['links']
                connect = []
                for link in links:
                    url = link['url']
                    if url in list(texts_url.keys()):
                        connect.append(['text', texts_url[url]])
                    if url in list(images_url.keys()):
                        connect.append(['image', images_url[url]])
                if len(connect) != 0:
                    connect_info.append({'span': [start_index, end_index], 'connect': connect})
            table_context = table_context[:-2]
            table_context += ' . ROW ' + str(index) + ' :'
            index += 1
        table_connect[example['qid']] = connect_info
        pbar.update(1)
    pickle.dump(table_connect, open('./mmqa/dataset/MMQA_tables_connect.pkl', 'wb'))
    print('save to', './mmqa/dataset/MMQA_tables_connect.pkl')
    # alignment between sources and table cells


if __name__ == "__main__":
    main()
