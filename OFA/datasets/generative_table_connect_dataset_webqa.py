"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MLM datasets
"""
import json
import random
from cytoolz import concat
import os
from PIL import ImageFile
import timeit
import base64
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
import logging
from multiprocessing import Pool, Manager
from torch.utils.data import Dataset
from tqdm import tqdm
import jsonlines
import pickle

logger = logging.getLogger(__name__)


def read_jsonlines(file_name):
    lines = []
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


class generative_webqa_dataset(Dataset):
    def __init__(self, tokenizer, input_example_dir, text_dir, img_dir, table_dir, connect_dir, span_dir, prefix_path,
                 doc_stride, max_img_length, max_query_len, max_seq_len, is_val=False):
        super().__init__()
        self.is_val = is_val
        self.tokenizer = tokenizer
        self.doc_stride = doc_stride
        self.max_query_len = max_query_len
        self.max_seq_len = max_seq_len
        self.max_caption_len = max_img_length
        self.paragraphs = pickle.load(open(os.path.join(prefix_path, text_dir), 'rb'))
        self.tables = pickle.load(open(os.path.join(prefix_path, table_dir), 'rb'))
        self.connect = pickle.load(open(os.path.join(prefix_path, connect_dir), 'rb'))
        self.cell_span = pickle.load(open(os.path.join(prefix_path, span_dir), 'rb'))
        self.imgs = os.path.join(prefix_path, img_dir)
        self.input_examples = self.read_examples(os.path.join(prefix_path, input_example_dir))

    def __len__(self):
        return len(self.input_examples)

    def example_process(self, input_data):
        pbar = tqdm(total=len(input_data))
        examples = []
        for key in tqdm(input_data.keys()):
            example = input_data[key]
            qid = example["Guid"]
            question_text = example['Q']
            if question_text[0] == '"':
                question_text = question_text[1:-1]
            ques_tokens = self.tokenizer.tokenize(question_text)
            ques_ids = self.tokenizer.convert_tokens_to_ids(ques_tokens)
            input_texts = example["txt_posFacts"] + example["txt_negFacts"]
            random.shuffle(input_texts)
            if len(input_texts) == 0:
                text_context_dict = {
                    '123456': {'ids': torch.tensor([0]), 'mask': torch.tensor([0]), 'token_type': torch.tensor([0])}}
            else:
                text_context_dict = {item['snippet_id']: self.text2id(self.paragraphs[item['snippet_id']], ques_ids) for
                                     item in input_texts}
            input_images = example["img_posFacts"] + example["img_negFacts"]
            random.shuffle(input_images)
            image_context = self.image2id(input_images, ques_ids)
            table_context = self.table2id(self.tables[key], ques_ids)

            text_id_list = list(text_context_dict.keys())
            image_id_list = list(image_context.keys())
            golden_source_index = []
            for golden_source in example["txt_posFacts"]:
                gs_id = golden_source['snippet_id']
                try:
                    golden_source_index.append(text_id_list.index(gs_id))
                except:
                    continue
            for golden_source in example["img_posFacts"]:
                gs_id = golden_source['image_id']
                try:
                    # 注意+1，因为引入了table
                    golden_source_index.append(image_id_list.index(gs_id) + len(text_id_list) + 1)
                except:
                    continue

            if len(golden_source_index) == 0:
                # 没有source,直接跳过
                if self.is_val == True:
                    golden_source_index = [0]
                else:
                    continue
            connect_info = self.connect[key].copy()
            table_connect_span = []
            table_connect_index = []
            ques_len = len(ques_ids)
            for info in connect_info:
                span = [item + ques_len + 2 for item in info['span']]
                span[1] = span[1] - 1

                if span[1] >= self.max_seq_len - 1:
                    # 超长
                    break
                sc_index = []
                for sc in info['connect']['text']:
                    try:
                        sc_index.append(text_id_list.index(sc))
                    except:
                        continue
                for sc in info['connect']['image']:
                    try:
                        #  注意不需要+1,因为需要计算align的source中没有table
                        sc_index.append(image_id_list.index(sc) + len(text_id_list))
                    except:
                        continue
                if len(sc_index) != 0:
                    table_connect_span.append(torch.tensor(span))
                    table_connect_index.append(torch.tensor(sc_index))
            table_cell_span = []
            cell_span_info = self.cell_span[key].copy()
            for info in cell_span_info:
                span = [item + ques_len + 2 for item in info]
                span[1] = span[1] - 1
                if span[1] >= self.max_seq_len - 1:
                    # 超长
                    break
                table_cell_span.append(torch.tensor(span))
            if table_cell_span == []:
                if self.is_val == True:
                    table_cell_span = [torch.tensor([0, 0])]
                else:
                    continue
            answer_text = example['A'][0]
            if answer_text[0] == '"':
                answer_text = answer_text[1:-1]
            ans_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
            ques_ids = self.tokenizer.convert_tokens_to_ids(ques_tokens)
            gpt_ids = [self.tokenizer.bos_token_id] + ques_ids + self.tokenizer.encode('[b_source]',
                                                                                       add_special_tokens=False) + [
                          self.tokenizer.mask_token_id] * len(golden_source_index) + self.tokenizer.encode(
                '[e_source]', add_special_tokens=False) + ans_ids + [self.tokenizer.eos_token_id]
            if len(gpt_ids) > self.max_query_len and self.is_val == False:
                continue
            attn_mask = [1] * len(gpt_ids)
            ans_loss_mask = [0] * len(attn_mask)
            ans_loss_mask[-(len(ans_ids) + 1):] = [1] * (len(ans_ids) + 1)
            sc_loss_mask = [0] * len(attn_mask)
            sc_loss_mask[-(len(ans_ids) + len(golden_source_index) + 2):-(len(ans_ids) + 1)] = [1] * (
                    len(golden_source_index) + 1)
            gpt_ids = torch.tensor(gpt_ids)
            attn_mask = torch.tensor(attn_mask)
            ans_loss_mask = torch.tensor(ans_loss_mask)
            sc_loss_mask = torch.tensor(sc_loss_mask)
            ques_ids = torch.tensor(ques_ids)
            golden_source_index = torch.tensor(golden_source_index)
            input = {
                "qid": qid,
                "question": {'ques': example["Q"], 'ques_tensor': ques_ids},
                "text_context": text_context_dict,
                "image_context": image_context,
                'table_context': table_context,
                "answers": {'ids': gpt_ids, 'attn_mask': attn_mask,
                            'ans_loss_mask': ans_loss_mask, 'sc_loss_mask': sc_loss_mask},
                "answer_list": [answer_text],
                'Qcate': example['Qcate'],
                "golden_source_index": golden_source_index,
                'table_connect_span': table_connect_span,
                'table_connect_index': table_connect_index,
                'table_cell_span': table_cell_span,
            }
            examples.append(input)
            pbar.update(1)
        return examples

    def read_examples(self, dir):
        input_data = json.load(open(dir, 'r'))
        split_input_data = {}
        if self.is_val == True:
            for key, value in input_data.items():
                if value['split'] == 'val':
                    split_input_data[key] = value
        else:
            for key, value in input_data.items():
                if value['split'] == 'train':
                    split_input_data[key] = value

        exampels = self.example_process(split_input_data)
        return exampels

    def text2id(self, text_dict, ques_ids):
        ques_len = len(ques_ids)
        ids = text_dict['ids']
        text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + ids + [
            self.tokenizer.eos_token_id]
        if len(text) > self.doc_stride:
            text = text[:self.doc_stride - 1]
            text += [self.tokenizer.eos_token_id]
            token_type_id = [0] * (2 + ques_len) + [1] * (self.doc_stride - 2 - ques_len)
        else:
            token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 1)
        text = torch.tensor(text)
        text_mask = torch.ones_like(text)
        token_type_id = torch.tensor(token_type_id)
        return {'ids': text, 'mask': text_mask, 'token_type': token_type_id}

    def table2id(self, text_dict, ques_ids):
        ques_len = len(ques_ids)
        ids = text_dict['ids']
        text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + ids + [
            self.tokenizer.eos_token_id]
        if len(text) > self.max_seq_len:
            text = text[:self.max_seq_len - 1]
            text += [self.tokenizer.eos_token_id]
            token_type_id = [0] * (2 + ques_len) + [1] * (self.max_seq_len - 2 - ques_len)
        else:
            token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 1)
        text = torch.tensor(text)
        text_mask = torch.ones_like(text)
        token_type_id = torch.tensor(token_type_id)
        return {'ids': text, 'mask': text_mask, 'token_type': token_type_id}

    def image2id(self, image_dicts, ques_ids):
        examples = {}
        ques_len = len(ques_ids)
        for item in image_dicts:
            caption = item['caption']
            caption_tokens = self.tokenizer.tokenize(caption)
            ids = self.tokenizer.convert_tokens_to_ids(caption_tokens)
            text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + [
                self.tokenizer.convert_tokens_to_ids('<title>')] + ids + [
                       self.tokenizer.convert_tokens_to_ids('</title>')] + [
                       self.tokenizer.eos_token_id]
            if len(text) > self.max_caption_len:
                text = text[:self.max_caption_len - 1]
                text += [self.tokenizer.eos_token_id]
                token_type_id = [0] * (2 + ques_len) + [1] * (self.max_caption_len - 2 - ques_len)
            else:
                token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 3)
            text = torch.tensor(text)
            text_mask = torch.ones_like(text)
            token_type_id = torch.tensor(token_type_id)
            examples[item['image_id']] = {'ids': text, 'mask': text_mask, 'token_type': token_type_id}
        return examples

    def __getitem__(self, i):
        input = self.input_examples[i]
        text_context_dict = input['text_context']
        input_text = [text_context_dict[key]['ids'] for key in text_context_dict.keys()]
        input_mask = [text_context_dict[key]['mask'] for key in text_context_dict.keys()]
        input_text_keys = [key for key in text_context_dict.keys()]
        input_token_type = [text_context_dict[key]['token_type'] for key in text_context_dict.keys()]
        input_text = pad_sequence(input_text, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        input_token_type = pad_sequence(input_token_type, batch_first=True, padding_value=0)

        table_context_dict = input['table_context']
        table_text = table_context_dict['ids']
        table_one_mask = table_context_dict['mask']
        table_token_type = table_context_dict['token_type']

        image_context_dict = input['image_context']
        images = []
        images_keys = []
        for key in image_context_dict.keys():
            path = os.path.join(self.imgs, str(key) + '.pkl')
            feat = pickle.load(open(path, 'rb'))
            feat = feat['feats']
            images.append(feat)
            images_keys.append(key)

        image_titles = [image_context_dict[key]['ids'] for key in image_context_dict.keys()]
        title_masks = [image_context_dict[key]['mask'] for key in image_context_dict.keys()]
        image_token_types = [image_context_dict[key]['token_type'] for key in image_context_dict.keys()]

        if len(images) == 0:
            images = [torch.zeros((3, 256, 256), dtype=torch.float32)]
            image_titles = [input["question"]['ques_tensor']]
            title_masks = [torch.ones(input["question"]['ques_tensor'].size(0), dtype=torch.int64)]
            image_token_types = [torch.zeros(input["question"]['ques_tensor'].size(0), dtype=torch.int64)]
        images = torch.stack(images)
        image_titles = pad_sequence(image_titles, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        title_masks = pad_sequence(title_masks, batch_first=True, padding_value=0)
        image_token_types = pad_sequence(image_token_types, batch_first=True, padding_value=0)
        gpt_ids = input['answers']['ids']
        attn_mask = input['answers']['attn_mask']
        ans_loss_mask = input['answers']['ans_loss_mask']
        sc_loss_mask = input['answers']['sc_loss_mask']
        Qcate = input['Qcate']
        golden_index = input['golden_source_index']

        table_connect_span = input['table_connect_span']
        if len(table_connect_span) == 0:
            table_connect_span = [torch.tensor([0, 0])]
        table_connect_span = torch.stack(table_connect_span)
        table_connect_index = input['table_connect_index']
        if len(table_connect_index) == 0:
            table_connect_index = [torch.tensor([-1])]
        table_connect_index = pad_sequence(table_connect_index, batch_first=True, padding_value=-1)

        table_cell_span = input['table_cell_span']
        gather_index = torch.zeros_like(table_text)
        span_lable = torch.zeros((input_text.size(0) + images.size(0)), dtype=torch.long)
        for idx, span in enumerate(table_cell_span):
            gather_index[span[0]:span[1]] = idx + 1
            connect_index = torch.nonzero(table_connect_span == span)
            if connect_index.size(0) >= 2:
                connect_source_index = table_connect_index[connect_index[0, 0]]
                for item in connect_source_index:
                    if item == -1:
                        break
                    # 第0位用于冗余编码：question/ROW/；/PAD等符号
                    span_lable[item] = idx + 1
        table_cell_span = torch.stack(table_cell_span)
        return [(input['qid'], input_text, input_mask, input_token_type, table_text, table_one_mask,
                 table_token_type, images, image_titles, title_masks, image_token_types, gpt_ids, attn_mask,
                 ans_loss_mask, sc_loss_mask, input['answer_list'], Qcate, golden_index, table_connect_span,
                 table_connect_index, table_cell_span, gather_index, span_lable, images_keys, input_text_keys)]

    def mmqa_collate(self, inputs):
        (ques_id, input_ids, input_mask, input_token_type, table_text, table_one_mask,
         table_token_type, image_feat, titles, title_masks, image_token_types, gpt_ids,
         attn_mask, ans_loss_mask, sc_loss_mask, answer_list, Qcate, golden_index, table_connect_span,
         table_connect_index, table_cell_span, gather_index, span_lable, images_keys,
         input_text_keys) = map(list, unzip(concat(inputs)))
        batch_text_len = []
        flatten_input_ids = []
        flatten_input_mask = []
        flatten_input_token_type = []
        for idx, item in enumerate(input_ids):
            batch_text_len.append([item.size(0) * item.size(1), item.size(1)])
            item = item.view(-1)
            flatten_input_ids.append(item)
            flatten_input_mask.append(input_mask[idx].reshape(-1))
            flatten_input_token_type.append(input_token_type[idx].reshape(-1))
        batch_text_len = torch.tensor(batch_text_len)
        flatten_input_ids = pad_sequence(flatten_input_ids, batch_first=True, padding_value=0)
        flatten_input_mask = pad_sequence(flatten_input_mask, batch_first=True, padding_value=0)
        flatten_input_token_type = pad_sequence(flatten_input_token_type, batch_first=True, padding_value=0)

        table_text = pad_sequence(table_text, batch_first=True, padding_value=0)
        table_token_type = pad_sequence(table_token_type, batch_first=True, padding_value=0)
        table_one_mask = pad_sequence(table_one_mask, batch_first=True, padding_value=0)

        batch_image_len = []
        image_feat_padded = []
        image_mask_padded = []
        image_ques_padded = []
        image_token_type_padded = []
        for idx, item in enumerate(image_feat):
            batch_image_len.append([item.size(0), title_masks[idx].size(1)])
            item = item.reshape(-1)
            image_feat_padded.append(item)
            image_mask_padded.append(title_masks[idx].reshape(-1))
            image_ques_padded.append(titles[idx].reshape(-1))
            image_token_type_padded.append(image_token_types[idx].reshape(-1))

        image_feat_padded = pad_sequence(image_feat_padded, batch_first=True, padding_value=0)
        image_mask_padded = pad_sequence(image_mask_padded, batch_first=True, padding_value=0)
        batch_image_len = torch.tensor(batch_image_len)
        image_ques_padded = pad_sequence(image_ques_padded, batch_first=True, padding_value=0)
        image_token_type_padded = pad_sequence(image_token_type_padded, batch_first=True, padding_value=0)

        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        ans_loss_mask = pad_sequence(ans_loss_mask, batch_first=True, padding_value=0)
        sc_loss_mask = pad_sequence(sc_loss_mask, batch_first=True, padding_value=0)
        golden_index = pad_sequence(golden_index, batch_first=True, padding_value=-1)

        table_connect_num = [[item.size(0), item.size(1)] for item in table_connect_index]
        flatten_connect_spans = []
        flatten_connect_index = []
        for idx, item in enumerate(table_connect_index):
            item = item.reshape(-1)
            flatten_connect_index.append(item)
            flatten_connect_spans.append(table_connect_span[idx].reshape(-1))
        table_connect_num = torch.tensor(table_connect_num)
        flatten_connect_index = pad_sequence(flatten_connect_index, batch_first=True, padding_value=-1)
        flatten_connect_spans = pad_sequence(flatten_connect_spans, batch_first=True, padding_value=-1)
        table_cell_num = [item.size(0) for item in table_cell_span]
        flatten_cell_span = []
        for idx, item in enumerate(table_cell_span):
            item = item.reshape(-1)
            flatten_cell_span.append(item)
        table_cell_num = torch.tensor(table_cell_num)
        flatten_span_label = pad_sequence(span_lable, batch_first=True, padding_value=0)
        flatten_cell_span = pad_sequence(flatten_cell_span, batch_first=True, padding_value=-1)
        gather_index = pad_sequence(gather_index, batch_first=True, padding_value=0)

        batch = {'ques_id': ques_id, 'input_ids': flatten_input_ids, 'input_mask': flatten_input_mask,
                 'batch_text_len': batch_text_len, 'input_token_type': flatten_input_token_type,
                 'table_text': table_text, 'table_mask': table_one_mask, 'table_token_type': table_token_type,
                 'image_feat': image_feat_padded, 'batch_image_len': batch_image_len,
                 'image_ques_id': image_ques_padded, 'image_token_type': image_token_type_padded,
                 'image_mask': image_mask_padded, 'sc_loss_mask': sc_loss_mask,
                 'ans_ids': gpt_ids, 'attn_mask': attn_mask, 'golden_index': golden_index,
                 'ans_loss_mask': ans_loss_mask, 'answer_list': answer_list, 'Qcate': Qcate,
                 'table_connect_num': table_connect_num, 'flatten_connect_spans': flatten_connect_spans,
                 'flatten_connect_index': flatten_connect_index, 'span_lable': span_lable,
                 'flatten_cell_span': flatten_cell_span, 'table_cell_num': table_cell_num,
                 'gather_index': gather_index, 'flatten_span_label': flatten_span_label,
                 'input_text_keys': input_text_keys, 'images_keys': images_keys,
                 }

        return batch


class generative_webqa_dataset_test(Dataset):
    def __init__(self, tokenizer, input_example_dir, text_dir, img_dir, table_dir, connect_dir, span_dir, prefix_path,
                 doc_stride, max_query_len, max_seq_len, is_val=False):
        super().__init__()
        self.is_val = is_val
        self.tokenizer = tokenizer
        self.doc_stride = doc_stride
        self.max_query_len = max_query_len
        self.max_seq_len = max_seq_len
        self.paragraphs = pickle.load(open(os.path.join(prefix_path, text_dir), 'rb'))
        self.tables = pickle.load(open(os.path.join(prefix_path, table_dir), 'rb'))
        self.connect = pickle.load(open(os.path.join(prefix_path, connect_dir), 'rb'))
        self.cell_span = pickle.load(open(os.path.join(prefix_path, span_dir), 'rb'))
        self.imgs = os.path.join(prefix_path, img_dir)
        self.input_examples = self.read_examples(os.path.join(prefix_path, input_example_dir))

    def __len__(self):
        return len(self.input_examples)

    def example_process(self, input_data):
        pbar = tqdm(total=len(input_data))
        examples = []
        for key in tqdm(input_data.keys()):
            example = input_data[key]
            qid = example["Guid"]
            question_text = example['Q']
            if question_text[0] == '"':
                question_text = example['Q'][1:-1]
            ques_tokens = self.tokenizer.tokenize(question_text)
            ques_ids = self.tokenizer.convert_tokens_to_ids(ques_tokens)
            input_texts = example["txt_Facts"]
            if len(input_texts) == 0:
                text_context_dict = {
                    '123456': {'ids': torch.tensor([0]), 'mask': torch.tensor([0]), 'token_type': torch.tensor([0])}}
            else:
                text_context_dict = {item['snippet_id']: self.text2id(self.paragraphs[item['snippet_id']], ques_ids) for
                                     item in input_texts}
            input_images = example["img_Facts"]
            image_context = self.image2id(input_images, ques_ids)
            table_context = self.table2id(self.tables[key], ques_ids)
            text_id_list = list(text_context_dict.keys())
            image_id_list = list(image_context.keys())
            connect_info = self.connect[key].copy()
            table_connect_span = []
            table_connect_index = []
            ques_len = len(ques_ids)
            for info in connect_info:
                span = [item + ques_len + 2 for item in info['span']]
                span[1] = span[1] - 1

                if span[1] >= self.max_seq_len - 1:
                    # 超长
                    break
                sc_index = []
                for sc in info['connect']['text']:
                    try:
                        sc_index.append(text_id_list.index(sc))
                    except:
                        continue
                for sc in info['connect']['image']:
                    try:
                        #  注意不需要+1,因为需要计算align的source中没有table
                        sc_index.append(image_id_list.index(sc) + len(text_id_list))
                    except:
                        continue
                if len(sc_index) != 0:
                    table_connect_span.append(torch.tensor(span))
                    table_connect_index.append(torch.tensor(sc_index))
            table_cell_span = []
            cell_span_info = self.cell_span[key].copy()
            for info in cell_span_info:
                span = [item + ques_len + 2 for item in info]
                span[1] = span[1] - 1
                if span[1] >= self.max_seq_len - 1:
                    # 超长
                    break
                table_cell_span.append(torch.tensor(span))
            if table_cell_span == []:
                table_cell_span = [torch.tensor([0, 0])]
            ques_ids = self.tokenizer.convert_tokens_to_ids(ques_tokens)
            gpt_ids = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.convert_tokens_to_ids('[b_source]')]
            attn_mask = [1] * len(gpt_ids)
            gpt_ids = torch.tensor(gpt_ids)
            attn_mask = torch.tensor(attn_mask)
            ques_ids = torch.tensor(ques_ids)
            input = {
                "qid": qid,
                "question": {'ques': example["Q"], 'ques_tensor': ques_ids},
                "text_context": text_context_dict,
                'table_context': table_context,
                "image_context": image_context,
                "answers": {'ids': gpt_ids, 'attn_mask': attn_mask},
                "answer_list": example['A'],
                'table_connect_span': table_connect_span,
                'table_connect_index': table_connect_index,
                'table_cell_span': table_cell_span,
            }
            examples.append(input)
            pbar.update(1)
        return examples

    def read_examples(self, dir):
        input_data = json.load(open(dir, 'r'))

        exampels = self.example_process(input_data)
        return exampels

    def text2id(self, text_dict, ques_ids):
        ques_len = len(ques_ids)
        ids = text_dict['ids']
        text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + ids + [
            self.tokenizer.eos_token_id]
        if len(text) > self.doc_stride:
            text = text[:self.doc_stride - 1]
            text += [self.tokenizer.eos_token_id]
            token_type_id = [0] * (2 + ques_len) + [1] * (self.doc_stride - 2 - ques_len)
        else:
            token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 1)
        text = torch.tensor(text)
        text_mask = torch.ones_like(text)
        token_type_id = torch.tensor(token_type_id)
        return {'ids': text, 'mask': text_mask, 'token_type': token_type_id}

    def table2id(self, text_dict, ques_ids):
        ques_len = len(ques_ids)
        ids = text_dict['ids']
        text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + ids + [
            self.tokenizer.eos_token_id]
        if len(text) > self.max_seq_len:
            text = text[:self.max_seq_len - 1]
            text += [self.tokenizer.eos_token_id]
            token_type_id = [0] * (2 + ques_len) + [1] * (self.max_seq_len - 2 - ques_len)
        else:
            token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 1)
        text = torch.tensor(text)
        text_mask = torch.ones_like(text)
        token_type_id = torch.tensor(token_type_id)
        return {'ids': text, 'mask': text_mask, 'token_type': token_type_id}

    def image2id(self, image_dicts, ques_ids):
        examples = {}
        ques_len = len(ques_ids)
        for item in image_dicts:
            caption = item['caption']
            caption_tokens = self.tokenizer.tokenize(caption)
            ids = self.tokenizer.convert_tokens_to_ids(caption_tokens)
            text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + [
                self.tokenizer.convert_tokens_to_ids('<title>')] + ids + [
                       self.tokenizer.convert_tokens_to_ids('</title>')] + [
                       self.tokenizer.eos_token_id]
            if len(text) > self.max_seq_len:
                text = text[:self.max_seq_len - 1]
                text += [self.tokenizer.eos_token_id]
                token_type_id = [0] * (2 + ques_len) + [1] * (self.max_seq_len - 2 - ques_len)
            else:
                token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 3)
            text = torch.tensor(text)
            text_mask = torch.ones_like(text)
            token_type_id = torch.tensor(token_type_id)
            examples[item['image_id']] = {'ids': text, 'mask': text_mask, 'token_type': token_type_id}
        return examples

    def __getitem__(self, i):
        input = self.input_examples[i]
        text_context_dict = input['text_context']
        input_text = [text_context_dict[key]['ids'] for key in text_context_dict.keys()]
        input_mask = [text_context_dict[key]['mask'] for key in text_context_dict.keys()]
        input_token_type = [text_context_dict[key]['token_type'] for key in text_context_dict.keys()]
        input_text = pad_sequence(input_text, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        input_token_type = pad_sequence(input_token_type, batch_first=True, padding_value=0)
        table_context_dict = input['table_context']
        table_text = table_context_dict['ids']
        table_one_mask = table_context_dict['mask']
        table_token_type = table_context_dict['token_type']

        image_context_dict = input['image_context']
        images = []
        for key in image_context_dict.keys():
            path = os.path.join(self.imgs, str(key) + '.pkl')
            feat = pickle.load(open(path, 'rb'))
            feat = feat['feats']
            images.append(feat)
        image_titles = [image_context_dict[key]['ids'] for key in image_context_dict.keys()]
        title_masks = [image_context_dict[key]['mask'] for key in image_context_dict.keys()]
        image_token_types = [image_context_dict[key]['token_type'] for key in image_context_dict.keys()]
        guide_keys = list(text_context_dict.keys()) + ['0000'] + list(image_context_dict.keys())
        if len(images) == 0:
            images = [torch.zeros((3, 256, 256), dtype=torch.float32)]
            image_titles = [input["question"]['ques_tensor']]
            title_masks = [torch.ones(input["question"]['ques_tensor'].size(0), dtype=torch.int64)]
            image_token_types = [torch.zeros(input["question"]['ques_tensor'].size(0), dtype=torch.int64)]
        images = torch.stack(images)
        image_titles = pad_sequence(image_titles, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        title_masks = pad_sequence(title_masks, batch_first=True, padding_value=0)
        image_token_types = pad_sequence(image_token_types, batch_first=True, padding_value=0)
        gpt_ids = input['answers']['ids']
        attn_mask = input['answers']['attn_mask']
        table_connect_span = input['table_connect_span']
        if len(table_connect_span) == 0:
            table_connect_span = [torch.tensor([0, 0])]
        table_connect_span = torch.stack(table_connect_span)
        table_connect_index = input['table_connect_index']
        if len(table_connect_index) == 0:
            table_connect_index = [torch.tensor([-1])]
        table_connect_index = pad_sequence(table_connect_index, batch_first=True, padding_value=-1)

        table_cell_span = input['table_cell_span']
        gather_index = torch.zeros_like(table_text)
        span_lable = torch.zeros((input_text.size(0) + images.size(0)), dtype=torch.long)
        for idx, span in enumerate(table_cell_span):
            gather_index[span[0]:span[1]] = idx + 1
            connect_index = torch.nonzero(table_connect_span == span)
            if connect_index.size(0) >= 2:
                connect_source_index = table_connect_index[connect_index[0, 0]]
                for item in connect_source_index:
                    if item == -1:
                        break
                    # 第0位用于冗余编码：question/ROW/；/PAD等符号
                    span_lable[item] = idx + 1
        table_cell_span = torch.stack(table_cell_span)
        return [(input['qid'], input_text, input_mask, input_token_type, table_text, table_one_mask,
                 table_token_type,
                 images, image_titles, title_masks, image_token_types, gpt_ids, attn_mask, guide_keys,
                 table_connect_span,
                 table_connect_index, table_cell_span, gather_index, span_lable)]

    def mmqa_collate(self, inputs):
        (ques_id, input_ids, input_mask, input_token_type, table_text, table_one_mask,
         table_token_type, image_feat, titles, title_masks, image_token_types, gpt_ids,
         attn_mask, guide_keys, table_connect_span, table_connect_index, table_cell_span, gather_index,
         span_lable) = map(list, unzip(concat(inputs)))
        batch_text_len = []
        flatten_input_ids = []
        flatten_input_mask = []
        flatten_input_token_type = []
        for idx, item in enumerate(input_ids):
            batch_text_len.append([item.size(0) * item.size(1), item.size(1)])
            item = item.view(-1)
            flatten_input_ids.append(item)
            flatten_input_mask.append(input_mask[idx].reshape(-1))
            flatten_input_token_type.append(input_token_type[idx].reshape(-1))
        batch_text_len = torch.tensor(batch_text_len)
        flatten_input_ids = pad_sequence(flatten_input_ids, batch_first=True, padding_value=0)
        flatten_input_mask = pad_sequence(flatten_input_mask, batch_first=True, padding_value=0)
        flatten_input_token_type = pad_sequence(flatten_input_token_type, batch_first=True, padding_value=0)
        table_text = pad_sequence(table_text, batch_first=True, padding_value=0)
        table_token_type = pad_sequence(table_token_type, batch_first=True, padding_value=0)
        table_one_mask = pad_sequence(table_one_mask, batch_first=True, padding_value=0)

        batch_image_len = []
        image_feat_padded = []
        image_mask_padded = []
        image_ques_padded = []
        image_token_type_padded = []
        for idx, item in enumerate(image_feat):
            batch_image_len.append([item.size(0), title_masks[idx].size(1)])
            item = item.reshape(-1)
            image_feat_padded.append(item)
            image_mask_padded.append(title_masks[idx].reshape(-1))
            image_ques_padded.append(titles[idx].reshape(-1))
            image_token_type_padded.append(image_token_types[idx].reshape(-1))

        image_feat_padded = pad_sequence(image_feat_padded, batch_first=True, padding_value=0)
        image_mask_padded = pad_sequence(image_mask_padded, batch_first=True, padding_value=0)
        batch_image_len = torch.tensor(batch_image_len)
        image_ques_padded = pad_sequence(image_ques_padded, batch_first=True, padding_value=0)
        image_token_type_padded = pad_sequence(image_token_type_padded, batch_first=True, padding_value=0)

        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        table_connect_num = [[item.size(0), item.size(1)] for item in table_connect_index]
        flatten_connect_spans = []
        flatten_connect_index = []
        for idx, item in enumerate(table_connect_index):
            item = item.reshape(-1)
            flatten_connect_index.append(item)
            flatten_connect_spans.append(table_connect_span[idx].reshape(-1))
        table_connect_num = torch.tensor(table_connect_num)
        flatten_connect_index = pad_sequence(flatten_connect_index, batch_first=True, padding_value=-1)
        flatten_connect_spans = pad_sequence(flatten_connect_spans, batch_first=True, padding_value=-1)
        table_cell_num = [item.size(0) for item in table_cell_span]
        flatten_cell_span = []
        for idx, item in enumerate(table_cell_span):
            item = item.reshape(-1)
            flatten_cell_span.append(item)
        table_cell_num = torch.tensor(table_cell_num)
        flatten_span_label = pad_sequence(span_lable, batch_first=True, padding_value=0)
        flatten_cell_span = pad_sequence(flatten_cell_span, batch_first=True, padding_value=-1)
        gather_index = pad_sequence(gather_index, batch_first=True, padding_value=0)

        batch = {'ques_id': ques_id, 'input_ids': flatten_input_ids, 'input_mask': flatten_input_mask,
                 'batch_text_len': batch_text_len, 'input_token_type': flatten_input_token_type,
                 'table_text': table_text, 'table_mask': table_one_mask, 'table_token_type': table_token_type,
                 'image_feat': image_feat_padded, 'batch_image_len': batch_image_len,
                 'image_ques_id': image_ques_padded, 'image_token_type': image_token_type_padded,
                 'image_mask': image_mask_padded, 'ans_ids': gpt_ids, 'attn_mask': attn_mask,
                 'guide_keys': guide_keys, 'table_connect_num': table_connect_num,
                 'flatten_connect_spans': flatten_connect_spans,
                 'flatten_connect_index': flatten_connect_index, 'span_lable': span_lable,
                 'flatten_cell_span': flatten_cell_span, 'table_cell_num': table_cell_num,
                 'gather_index': gather_index, 'flatten_span_label': flatten_span_label}

        return batch
