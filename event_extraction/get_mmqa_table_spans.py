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
    tables = read_jsonlines('./mmqa/dataset/MMQA_tables.jsonl')

    pbar = tqdm(total=len(tables))
    table_connect = {}
    for table in tables:
        connect_info = []
        rows = table['table']['table_rows']
        title = table['title']
        title = tokenizer.tokenize(' ' + title)
        title = ['<title>'] + title + ['</title>']
        table_context = "ROW 1 :"
        # table_context = tokenizer.tokenize("ROW 1 : ")
        index = 2
        title_len = len(title)
        for row_data in rows:
            for cell in row_data:
                text = cell['text']
                if text == '':
                    continue
                start_index = len(tokenizer.tokenize(table_context)) + title_len
                table_context += ' ' + text
                end_index = len(tokenizer.tokenize(table_context)) + title_len
                table_context += ' ;'
                connect_info.append([start_index, end_index])
            table_context = table_context[:-2]
            table_context += ' . ROW ' + str(index) + ' :'
            index += 1
        table_connect[table['id']] = connect_info
        pbar.update(1)
    pickle.dump(table_connect, open('./mmqa/dataset/MMQA_cell_spans.pkl', 'wb'))
    print('save to', './mmqa/dataset/MMQA_cell_spans.pkl')
    # spans of the cells in the tables in mmqa


if __name__ == "__main__":
    main()
