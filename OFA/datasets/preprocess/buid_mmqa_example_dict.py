import sys

sys.path.append('../')
import torch
import os
import numpy as np
import json
from tqdm import tqdm
from ofa import OFATokenizer
from torch.utils.data import TensorDataset
import jsonlines
from PIL import Image
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import pickle

MAX_TEXT_LEN = 256
MAX_LENGTH = 1026
MAX_IMAGE_TITLE_LENTH = 64
MAX_DEC_LEN = 64
IMAGE_PATH = './mmqa/final_dataset_images'
TEXT_NUM = 10
IMAGE_NUM = 14
MAX_SC_NUM = 3


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
    # Get tokenized texts
    texts = read_jsonlines('./mmqa/dataset/MMQA_texts.jsonl')
    texts_dict = {doc["id"]: doc for doc in texts}
    pbar = tqdm(total=len(texts_dict))
    texts_feats = {}
    for key, text_dict in texts_dict.items():
        pbar.update(1)
        title = text_dict['title']
        text = text_dict['text']
        title = tokenizer.tokenize(' ' + title)
        text = tokenizer.tokenize(' ' + text)
        context_text = ['<title>'] + title + ['</title>'] + text
        attn_mask = [1] * len(context_text)
        context_text = tokenizer.convert_tokens_to_ids(context_text)
        texts_feats[key] = {'ids': context_text, 'masks': attn_mask}
    pickle.dump(texts_feats, open('./mmqa/dataset/MMQA_texts_ofa_feats.pkl', 'wb'))
    print('save to', './mmqa/dataset/MMQA_texts_ofa_feats.pkl')

    # get tokenized tables
    tables = read_jsonlines('./mmqa/dataset/MMQA_tables.jsonl')
    tables_dict = {doc["id"]: doc for doc in tables}
    tables_feats = {}
    pbar = tqdm(total=len(tables_dict))
    for key, table_dict in tables_dict.items():
        pbar.update(1)
        title = table_dict['title']
        title = tokenizer.tokenize(' ' + title)
        title = ['<title>'] + title + ['</title>']
        table_context = "ROW 1 : "
        rows = table_dict['table']['table_rows']
        index = 2
        for row_data in rows:
            for cell in row_data:
                text = cell['text']
                if text == '':
                    continue
                table_context += text
                table_context += ' ; '
            table_context = table_context[:-3]
            table_context += ' . ROW ' + str(index) + ' : '
            index += 1
        table_context = table_context[:-(8 + len(str(index - 1)))]
        table_tokens = tokenizer.tokenize(table_context)
        table_tokens = title + table_tokens
        attn_mask = [1] * len(table_tokens)
        table_tokens = tokenizer.convert_tokens_to_ids(table_tokens)
        tables_feats[key] = {'ids': table_tokens, 'masks': attn_mask}
    pickle.dump(tables_feats, open('./mmqa/dataset/MMQA_tables_ofa_feats.pkl', 'wb'))
    print('save to', './mmqa/dataset/MMQA_tables_ofa_feats.pkl')

    # Get image feats
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 256
    patch_resize_transform = transforms.Compose([lambda image: image.convert("RGB"),
                                                 transforms.Resize((resolution, resolution),
                                                                   interpolation=Image.BICUBIC),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=mean, std=std)
                                                 ])
    images = read_jsonlines('./mmqa/dataset/MMQA_images.jsonl')
    images_dict = {doc["id"]: doc for doc in images}
    pbar = tqdm(total=len(images_dict))
    for key, image_info in images_dict.items():
        pbar.update(1)
        title = image_info['title']
        image_path = os.path.join(IMAGE_PATH, image_info['path'])
        try:
            img = Image.open(image_path)
            patch_img = patch_resize_transform(img)
        except:
            continue
        title_tokens = tokenizer.tokenize(' ' + title)
        attn_mask = [1] * len(title_tokens)
        title_tokens = tokenizer.convert_tokens_to_ids(title_tokens)
        tmp_feats = {'ids': title_tokens, 'masks': attn_mask, 'img_feat': patch_img}
        save_path = os.path.join('./mmqa/dataset/MMQA_images_ofa_feats', key + '.pkl')
        pickle.dump(tmp_feats, open(save_path, 'wb'))
    print('save to', './mmqa/dataset/MMQA_images_ofa_feats')


if __name__ == "__main__":
    main()
