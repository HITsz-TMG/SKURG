"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MLM datasets
"""
import random
from cytoolz import concat
import os
from PIL import ImageFile
import timeit

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


class generative_mmqa_dataset(Dataset):
    def __init__(self, tokenizer, input_example_dir, text_dir, img_dir, table_dir, doc_stride,
                 max_query_len, max_seq_len, connect_path, span_dir, is_debug=False, is_test=False):
        super().__init__()
        self.is_debug = is_debug
        self.is_test = is_test
        self.tokenizer = tokenizer
        self.doc_stride = doc_stride
        self.max_query_len = max_query_len
        self.max_seq_len = max_seq_len
        self.connect = pickle.load(open(connect_path, 'rb'))
        self.cell_span = pickle.load(open(span_dir, 'rb'))
        self.paragraphs = pickle.load(open(text_dir, 'rb'))
        self.tables = pickle.load(open(table_dir, 'rb'))
        self.imgs = pickle.load(open(img_dir, 'rb'))
        self.input_examples = self.read_examples(input_example_dir)

    def __len__(self):
        return len(self.input_examples)

    def example_process(self, input_data):
        pbar = tqdm(total=len(input_data))
        examples = []
        for example in tqdm(input_data):
            qid = example["qid"]
            question_text = example['question']
            ques_tokens = self.tokenizer.tokenize(question_text)
            ques_ids = self.tokenizer.convert_tokens_to_ids(ques_tokens)
            text_context_dict = {doc_id: self.text2id(self.paragraphs[doc_id], ques_ids) for doc_id in
                                 example["metadata"]["text_doc_ids"]}
            table_context = self.tabel2id(self.tables[example["metadata"]["table_id"]], ques_ids)
            image_context = self.image2id(example["metadata"]["image_doc_ids"], ques_ids)

            text_id_list = list(text_context_dict.keys())
            image_id_list = list(image_context.keys())
            golden_source_index = []
            for golden_source in example['supporting_context']:
                gs_id = golden_source['doc_id']
                if golden_source['doc_part'] == 'text':
                    try:
                        golden_source_index.append(text_id_list.index(gs_id))
                    except:
                        continue
                elif golden_source['doc_part'] == 'image':
                    try:
                        golden_source_index.append(image_id_list.index(gs_id) + len(text_id_list) + 1)
                    except:
                        continue
                else:
                    golden_source_index.append(len(text_id_list))
            if len(golden_source_index) == 0:
                if self.is_test == True:
                    golden_source_index = [0]
                else:
                    continue

            # random.shuffle(golden_source_index)
            connect_info = self.connect[example["qid"]].copy()
            table_connect_span = []
            table_connect_index = []
            ques_len = len(ques_ids)
            for info in connect_info:
                span = [item + ques_len + 2 for item in info['span']]
                if span[1] >= self.max_seq_len - 1:
                    # 超长
                    break
                sc_index = []
                for sc in info['connect']:
                    if sc[0] == 'text':
                        try:
                            sc_index.append(text_id_list.index(sc[1]))
                        except:
                            continue
                    else:
                        try:
                            #  注意不需要+1
                            sc_index.append(image_id_list.index(sc[1]) + len(text_id_list))
                        except:
                            continue
                if len(sc_index) != 0:
                    table_connect_span.append(torch.tensor(span))
                    table_connect_index.append(torch.tensor(sc_index))
            table_cell_span = []
            cell_span_info = self.cell_span[example["metadata"]["table_id"]].copy()
            for info in cell_span_info:
                span = [item + ques_len + 2 for item in info]
                if span[1] >= self.max_seq_len - 1:
                    # 超长
                    break
                table_cell_span.append(torch.tensor(span))

            source_index = []
            source_span = []
            source_ids = None
            answer = example['answers'][0]
            answer_id = self.tokenizer.encode(' ' + str(answer['answer']), add_special_tokens=False)
            if answer['type'] == 'string':
                if answer['modality'] == 'table':
                    source_index.append(len(text_id_list))
                    source_ids = table_context['ids']
                elif answer['modality'] == 'text':
                    doc_id = answer['text_instances'][0]['doc_id']
                    text_index = text_id_list.index(doc_id)
                    source_index.append(text_index)
                    source_ids = text_context_dict[doc_id]['ids']
                else:
                    doc_id = answer['image_instances'][0]['doc_id']
                    try:
                        image_index = image_id_list.index(doc_id)
                    except:
                        continue
                    source_index.append(len(text_id_list) + 1 + image_index)
                    source_ids = image_context[doc_id]['ids']
            if source_ids is not None:
                for span in range(len(ques_ids), len(source_ids) - len(answer_id)):
                    if list(source_ids[span:span + len(answer_id)]) == answer_id:
                        source_span.append([span, span + len(answer_id)])

            answer_text = str(example['answers'][0]['answer'])
            ans_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
            ques_ids = self.tokenizer.convert_tokens_to_ids(ques_tokens)
            gpt_ids = [self.tokenizer.bos_token_id] + ques_ids + self.tokenizer.encode('[b_source]',
                                                                                       add_special_tokens=False) + [
                          self.tokenizer.mask_token_id] * len(golden_source_index) + self.tokenizer.encode(
                '[e_source]', add_special_tokens=False) + ans_ids + [self.tokenizer.eos_token_id]
            if len(gpt_ids) > self.max_query_len and self.is_test == False:
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
            input = {
                "qid": qid,
                "question": {'ques': example["question"], 'ques_tensor': ques_ids},
                "text_context": text_context_dict,
                "table_context": table_context,
                "image_context": image_context,
                "golden_source_index": golden_source_index,
                "answers": {'ids': gpt_ids, 'attn_mask': attn_mask, 'sc_loss_mask': sc_loss_mask,
                            'ans_loss_mask': ans_loss_mask},
                "answer_list": example['answers'],
                'source_index': source_index,
                'source_span': source_span,
                'table_connect_span': table_connect_span,
                'table_connect_index': table_connect_index,
                'table_cell_span': table_cell_span
            }
            examples.append(input)
            pbar.update(1)
        return examples

    def read_examples(self, dir):
        input_data = read_jsonlines(dir)
        if self.is_debug:
            input_data = input_data[:100]
        exampels = self.example_process(input_data)
        return exampels

    def text2id(self, text_dict, ques_ids):
        ques_len = len(ques_ids)
        ids = text_dict['ids']
        mask = text_dict['masks']
        text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + ids + [
            self.tokenizer.eos_token_id]
        text_mask = mask + [1] * (ques_len + 3)
        if len(text) > self.doc_stride:
            text = text[:self.doc_stride - 1]
            text += [self.tokenizer.eos_token_id]
            text_mask = text_mask[:self.doc_stride]
            token_type_id = [0] * (2 + ques_len) + [1] * (self.doc_stride - 2 - ques_len)
        else:
            token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 1)
        text = torch.tensor(text)
        text_mask = torch.tensor(text_mask)
        token_type_id = torch.tensor(token_type_id)
        return {'ids': text, 'mask': text_mask, 'token_type': token_type_id}

    def tabel2id(self, text_dict, ques_ids):
        ques_len = len(ques_ids)
        ids = text_dict['ids']
        mask = text_dict['masks']
        text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + ids + [
            self.tokenizer.eos_token_id]
        text_mask = mask + [1] * (ques_len + 3)
        if len(text) > self.max_seq_len:
            text = text[:self.max_seq_len - 1]
            text += [self.tokenizer.eos_token_id]
            text_mask = text_mask[:self.max_seq_len]
            token_type_id = [0] * (2 + ques_len) + [1] * (self.max_seq_len - 2 - ques_len)
        else:
            token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 1)
        text = torch.tensor(text)
        text_mask = torch.tensor(text_mask)
        token_type_id = torch.tensor(token_type_id)
        return {'ids': text, 'mask': text_mask, 'token_type': token_type_id}

    def image2id(self, image_ids, ques_ids):
        examples = {}
        ques_len = len(ques_ids)
        for id in image_ids:
            try:
                image_dict = self.imgs[id]
            except:
                continue
            ids = image_dict['ids']
            mask = image_dict['masks']
            # feat = image_dict['img_feat']
            text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + [
                self.tokenizer.convert_tokens_to_ids('<title>')] + ids + [
                       self.tokenizer.convert_tokens_to_ids('</title>')] + [
                       self.tokenizer.eos_token_id]
            text_mask = mask + [1] * (ques_len + 5)
            if len(text) > self.max_seq_len:
                text = text[:self.max_seq_len - 1]
                text += [self.tokenizer.eos_token_id]
                text_mask = text_mask[:self.max_seq_len]
                token_type_id = [0] * (2 + ques_len) + [1] * (self.max_seq_len - 2 - ques_len)
            else:
                token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 3)
            text = torch.tensor(text)
            text_mask = torch.tensor(text_mask)
            token_type_id = torch.tensor(token_type_id)
            examples[id] = {'ids': text, 'mask': text_mask, 'token_type': token_type_id}
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
        table_mask = table_context_dict['mask']
        table_token_type = table_context_dict['token_type']

        image_context_dict = input['image_context']
        images = [self.imgs[key]['img_feat'] for key in image_context_dict.keys()]
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
        golden_source_index = torch.tensor(input['golden_source_index'])

        source_index = input['source_index']
        if len(source_index) > 1:
            source_index = [source_index[-1]]
        elif len(source_index) == 0:
            source_index = [-1]
        source_span = input['source_span']
        if len(source_span) > 1:
            # 取最后一个
            source_span = source_span[-1]
        elif len(source_span) == 0:
            source_span = [0, 1]
        else:
            source_span = source_span[0]
        source_index = torch.tensor(source_index, dtype=torch.int)
        source_span = torch.tensor(source_span, dtype=torch.int)

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
        span_lable = torch.zeros((len(table_cell_span), input_text.size(0) + images.size(0)))
        for idx, span in enumerate(table_cell_span):
            gather_index[span[0]:span[1]] = idx + 1
            connect_index = torch.nonzero(table_connect_span == span)
            if connect_index.size(0) == 2:
                connect_source_index = table_connect_index[connect_index[0, 0]]
                span_lable[idx][connect_source_index] = 1
        table_cell_span = torch.stack(table_cell_span)

        return [(input['qid'], input_text, input_mask, input_token_type, table_text, table_mask, table_token_type,
                 images, image_titles, title_masks, image_token_types, gpt_ids, attn_mask, golden_source_index,
                 ans_loss_mask, sc_loss_mask, input['answer_list'], source_index, source_span, table_connect_span,
                 table_connect_index, table_cell_span, gather_index, span_lable)]

    def mmqa_collate(self, inputs):
        (ques_id, input_ids, input_mask, input_token_type, tabel_text, tabel_mask, tabel_token_type, image_feat,
         titles, title_masks, image_token_types, gpt_ids, attn_mask, golden_source_index, ans_loss_mask,
         sc_loss_mask, answer_list, source_index, source_span, table_connect_span, table_connect_index,
         table_cell_span, gather_index, span_lable) = map(list, unzip(concat(inputs)))
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

        tabel_text = pad_sequence(tabel_text, batch_first=True, padding_value=0)
        tabel_mask = pad_sequence(tabel_mask, batch_first=True, padding_value=0)
        tabel_token_type = pad_sequence(tabel_token_type, batch_first=True, padding_value=0)

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
        batch_image_len = torch.tensor(batch_image_len).cuda()
        image_ques_padded = pad_sequence(image_ques_padded, batch_first=True, padding_value=0)
        image_token_type_padded = pad_sequence(image_token_type_padded, batch_first=True, padding_value=0)

        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        ans_loss_mask = pad_sequence(ans_loss_mask, batch_first=True, padding_value=0)
        sc_loss_mask = pad_sequence(sc_loss_mask, batch_first=True, padding_value=0)
        golden_source_index = pad_sequence(golden_source_index, batch_first=True, padding_value=-1)

        source_index = torch.stack(source_index)
        source_span = torch.stack(source_span)

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

        tabel_cell_num = [[item.size(0), item.size(1)] for item in span_lable]
        flatten_cell_span = []
        flatten_span_label = []
        for idx, item in enumerate(table_cell_span):
            item = item.reshape(-1)
            flatten_cell_span.append(item)
            flatten_span_label.append(span_lable[idx].reshape(-1))
        tabel_cell_num = torch.tensor(tabel_cell_num)
        flatten_span_label = pad_sequence(flatten_span_label, batch_first=True, padding_value=-1)
        flatten_cell_span = pad_sequence(flatten_cell_span, batch_first=True, padding_value=-1)
        gather_index = pad_sequence(gather_index, batch_first=True, padding_value=0)

        batch = {'ques_id': ques_id, 'input_ids': flatten_input_ids, 'input_mask': flatten_input_mask,
                 'batch_text_len': batch_text_len, 'input_token_type': flatten_input_token_type,
                 'tabel_text': tabel_text, 'tabel_mask': tabel_mask, 'tabel_token_type': tabel_token_type,
                 'image_feat': image_feat_padded, 'batch_image_len': batch_image_len,
                 'image_ques_id': image_ques_padded, 'image_token_type': image_token_type_padded,
                 'image_mask': image_mask_padded,
                 'ans_ids': gpt_ids, 'attn_mask': attn_mask,
                 'ans_loss_mask': ans_loss_mask, 'sc_loss_mask': sc_loss_mask,
                 'golden_source_index': golden_source_index, 'answer_list': answer_list,
                 'source_index': source_index, 'source_span': source_span,
                 'table_connect_num': table_connect_num, 'flatten_connect_spans': flatten_connect_spans,
                 'flatten_connect_index': flatten_connect_index, 'span_lable': span_lable,
                 'flatten_cell_span': flatten_cell_span, 'tabel_cell_num': tabel_cell_num,
                 'gather_index': gather_index, 'flatten_span_label': flatten_span_label
                 }

        return batch


class generative_mmqa_dataset_wo_retrieve(Dataset):
    def __init__(self, tokenizer, input_example_dir, text_dir, img_dir, table_dir, doc_stride,
                 max_query_len, max_seq_len, connect_path, span_dir, is_debug=False, is_test=False):
        super().__init__()
        self.is_debug = is_debug
        self.is_test = is_test
        self.tokenizer = tokenizer
        self.doc_stride = doc_stride
        self.max_query_len = max_query_len
        self.max_seq_len = max_seq_len
        self.connect = pickle.load(open(connect_path, 'rb'))
        self.cell_span = pickle.load(open(span_dir, 'rb'))
        self.paragraphs = pickle.load(open(text_dir, 'rb'))
        self.tables = pickle.load(open(table_dir, 'rb'))
        self.imgs = pickle.load(open(img_dir, 'rb'))
        self.input_examples = self.read_examples(input_example_dir)

    def __len__(self):
        return len(self.input_examples)

    def example_process(self, input_data):
        pbar = tqdm(total=len(input_data))
        examples = []
        for example in tqdm(input_data):
            qid = example["qid"]
            question_text = example['question']
            ques_tokens = self.tokenizer.tokenize(question_text)
            ques_ids = self.tokenizer.convert_tokens_to_ids(ques_tokens)
            text_context_dict = {doc_id: self.text2id(self.paragraphs[doc_id], ques_ids) for doc_id in
                                 example["metadata"]["text_doc_ids"]}
            table_context = self.tabel2id(self.tables[example["metadata"]["table_id"]], ques_ids)
            image_context = self.image2id(example["metadata"]["image_doc_ids"], ques_ids)

            text_id_list = list(text_context_dict.keys())
            image_id_list = list(image_context.keys())
            golden_source_index = []
            for golden_source in example['supporting_context']:
                gs_id = golden_source['doc_id']
                if golden_source['doc_part'] == 'text':
                    try:
                        golden_source_index.append(text_id_list.index(gs_id))
                    except:
                        continue
                elif golden_source['doc_part'] == 'image':
                    try:
                        golden_source_index.append(image_id_list.index(gs_id) + len(text_id_list) + 1)
                    except:
                        continue
                else:
                    golden_source_index.append(len(text_id_list))
            if len(golden_source_index) == 0:
                if self.is_test == True:
                    golden_source_index = [0]
                else:
                    continue
            connect_info = self.connect[example["qid"]].copy()
            table_connect_span = []
            table_connect_index = []
            ques_len = len(ques_ids)
            for info in connect_info:
                span = [item + ques_len + 2 for item in info['span']]
                if span[1] >= self.max_seq_len - 1:
                    # 超长
                    break
                sc_index = []
                for sc in info['connect']:
                    if sc[0] == 'text':
                        try:
                            sc_index.append(text_id_list.index(sc[1]))
                        except:
                            continue
                    else:
                        try:
                            #  注意不需要+1
                            sc_index.append(image_id_list.index(sc[1]) + len(text_id_list))
                        except:
                            continue
                if len(sc_index) != 0:
                    table_connect_span.append(torch.tensor(span))
                    table_connect_index.append(torch.tensor(sc_index))
            table_cell_span = []
            cell_span_info = self.cell_span[example["metadata"]["table_id"]].copy()
            for info in cell_span_info:
                span = [item + ques_len + 2 for item in info]
                if span[1] >= self.max_seq_len - 1:
                    # 超长
                    break
                table_cell_span.append(torch.tensor(span))

            source_index = []
            source_span = []
            source_ids = None
            answer = example['answers'][0]
            answer_id = self.tokenizer.encode(' ' + str(answer['answer']), add_special_tokens=False)
            if answer['type'] == 'string':
                if answer['modality'] == 'table':
                    source_index.append(len(text_id_list))
                    source_ids = table_context['ids']
                elif answer['modality'] == 'text':
                    doc_id = answer['text_instances'][0]['doc_id']
                    text_index = text_id_list.index(doc_id)
                    source_index.append(text_index)
                    source_ids = text_context_dict[doc_id]['ids']
                else:
                    doc_id = answer['image_instances'][0]['doc_id']
                    try:
                        image_index = image_id_list.index(doc_id)
                    except:
                        continue
                    source_index.append(len(text_id_list) + 1 + image_index)
                    source_ids = image_context[doc_id]['ids']
            if source_ids is not None:
                for span in range(len(ques_ids), len(source_ids) - len(answer_id)):
                    if list(source_ids[span:span + len(answer_id)]) == answer_id:
                        source_span.append([span, span + len(answer_id)])

            answer_text = str(example['answers'][0]['answer'])
            ans_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
            ques_ids = self.tokenizer.convert_tokens_to_ids(ques_tokens)
            gpt_ids = [self.tokenizer.bos_token_id] + ques_ids + self.tokenizer.encode(
                '[e_source]', add_special_tokens=False) + ans_ids + [self.tokenizer.eos_token_id]
            if len(gpt_ids) > self.max_query_len and self.is_test == False:
                continue
            attn_mask = [1] * len(gpt_ids)
            ans_loss_mask = [0] * len(attn_mask)
            ans_loss_mask[-(len(ans_ids) + 1):] = [1] * (len(ans_ids) + 1)

            gpt_ids = torch.tensor(gpt_ids)
            attn_mask = torch.tensor(attn_mask)
            ans_loss_mask = torch.tensor(ans_loss_mask)
            ques_ids = torch.tensor(ques_ids)
            input = {
                "qid": qid,
                "question": {'ques': example["question"], 'ques_tensor': ques_ids},
                "text_context": text_context_dict,
                "table_context": table_context,
                "image_context": image_context,
                "golden_source_index": golden_source_index,
                "answers": {'ids': gpt_ids, 'attn_mask': attn_mask,
                            'ans_loss_mask': ans_loss_mask},
                "answer_list": example['answers'],
                'source_index': source_index,
                'source_span': source_span,
                'table_connect_span': table_connect_span,
                'table_connect_index': table_connect_index,
                'table_cell_span': table_cell_span
            }
            examples.append(input)
            pbar.update(1)
        return examples

    def read_examples(self, dir):
        input_data = read_jsonlines(dir)
        if self.is_debug:
            input_data = input_data[:100]
        exampels = self.example_process(input_data)
        return exampels

    def text2id(self, text_dict, ques_ids):
        ques_len = len(ques_ids)
        ids = text_dict['ids']
        mask = text_dict['masks']
        text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + ids + [
            self.tokenizer.eos_token_id]
        text_mask = mask + [1] * (ques_len + 3)
        if len(text) > self.doc_stride:
            text = text[:self.doc_stride - 1]
            text += [self.tokenizer.eos_token_id]
            text_mask = text_mask[:self.doc_stride]
            token_type_id = [0] * (2 + ques_len) + [1] * (self.doc_stride - 2 - ques_len)
        else:
            token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 1)
        text = torch.tensor(text)
        text_mask = torch.tensor(text_mask)
        token_type_id = torch.tensor(token_type_id)
        return {'ids': text, 'mask': text_mask, 'token_type': token_type_id}

    def tabel2id(self, text_dict, ques_ids):
        ques_len = len(ques_ids)
        ids = text_dict['ids']
        mask = text_dict['masks']
        text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + ids + [
            self.tokenizer.eos_token_id]
        text_mask = mask + [1] * (ques_len + 3)
        if len(text) > self.max_seq_len:
            text = text[:self.max_seq_len - 1]
            text += [self.tokenizer.eos_token_id]
            text_mask = text_mask[:self.max_seq_len]
            token_type_id = [0] * (2 + ques_len) + [1] * (self.max_seq_len - 2 - ques_len)
        else:
            token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 1)
        text = torch.tensor(text)
        text_mask = torch.tensor(text_mask)
        token_type_id = torch.tensor(token_type_id)
        return {'ids': text, 'mask': text_mask, 'token_type': token_type_id}

    def image2id(self, image_ids, ques_ids):
        examples = {}
        ques_len = len(ques_ids)
        for id in image_ids:
            try:
                image_dict = self.imgs[id]
            except:
                continue
            ids = image_dict['ids']
            mask = image_dict['masks']
            # feat = image_dict['img_feat']
            text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + [
                self.tokenizer.convert_tokens_to_ids('<title>')] + ids + [
                       self.tokenizer.convert_tokens_to_ids('</title>')] + [
                       self.tokenizer.eos_token_id]
            text_mask = mask + [1] * (ques_len + 5)
            if len(text) > self.max_seq_len:
                text = text[:self.max_seq_len - 1]
                text += [self.tokenizer.eos_token_id]
                text_mask = text_mask[:self.max_seq_len]
                token_type_id = [0] * (2 + ques_len) + [1] * (self.max_seq_len - 2 - ques_len)
            else:
                token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 3)
            text = torch.tensor(text)
            text_mask = torch.tensor(text_mask)
            token_type_id = torch.tensor(token_type_id)
            examples[id] = {'ids': text, 'mask': text_mask, 'token_type': token_type_id}
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
        table_mask = table_context_dict['mask']
        table_token_type = table_context_dict['token_type']

        image_context_dict = input['image_context']
        images = [self.imgs[key]['img_feat'] for key in image_context_dict.keys()]
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
        golden_source_index = torch.tensor(input['golden_source_index'])

        source_index = input['source_index']
        if len(source_index) > 1:
            source_index = [source_index[-1]]
        elif len(source_index) == 0:
            source_index = [-1]
        source_span = input['source_span']
        if len(source_span) > 1:
            # 取最后一个
            source_span = source_span[-1]
        elif len(source_span) == 0:
            source_span = [0, 1]
        else:
            source_span = source_span[0]
        source_index = torch.tensor(source_index, dtype=torch.int)
        source_span = torch.tensor(source_span, dtype=torch.int)

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
        span_lable = torch.zeros((len(table_cell_span), input_text.size(0) + images.size(0)))
        for idx, span in enumerate(table_cell_span):
            gather_index[span[0]:span[1]] = idx + 1
            connect_index = torch.nonzero(table_connect_span == span)
            if connect_index.size(0) == 2:
                connect_source_index = table_connect_index[connect_index[0, 0]]
                span_lable[idx][connect_source_index] = 1
        table_cell_span = torch.stack(table_cell_span)

        return [(input['qid'], input_text, input_mask, input_token_type, table_text, table_mask, table_token_type,
                 images, image_titles, title_masks, image_token_types, gpt_ids, attn_mask, golden_source_index,
                 ans_loss_mask, input['answer_list'], source_index, source_span, table_connect_span,
                 table_connect_index, table_cell_span, gather_index, span_lable)]

    def mmqa_collate(self, inputs):
        (ques_id, input_ids, input_mask, input_token_type, tabel_text, tabel_mask, tabel_token_type, image_feat,
         titles, title_masks, image_token_types, gpt_ids, attn_mask, golden_source_index, ans_loss_mask,
         answer_list, source_index, source_span, table_connect_span, table_connect_index,
         table_cell_span, gather_index, span_lable) = map(list, unzip(concat(inputs)))
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

        tabel_text = pad_sequence(tabel_text, batch_first=True, padding_value=0)
        tabel_mask = pad_sequence(tabel_mask, batch_first=True, padding_value=0)
        tabel_token_type = pad_sequence(tabel_token_type, batch_first=True, padding_value=0)

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
        batch_image_len = torch.tensor(batch_image_len).cuda()
        image_ques_padded = pad_sequence(image_ques_padded, batch_first=True, padding_value=0)
        image_token_type_padded = pad_sequence(image_token_type_padded, batch_first=True, padding_value=0)

        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        ans_loss_mask = pad_sequence(ans_loss_mask, batch_first=True, padding_value=0)
        golden_source_index = pad_sequence(golden_source_index, batch_first=True, padding_value=-1)

        source_index = torch.stack(source_index)
        source_span = torch.stack(source_span)

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

        tabel_cell_num = [[item.size(0), item.size(1)] for item in span_lable]
        flatten_cell_span = []
        flatten_span_label = []
        for idx, item in enumerate(table_cell_span):
            item = item.reshape(-1)
            flatten_cell_span.append(item)
            flatten_span_label.append(span_lable[idx].reshape(-1))
        tabel_cell_num = torch.tensor(tabel_cell_num)
        flatten_span_label = pad_sequence(flatten_span_label, batch_first=True, padding_value=-1)
        flatten_cell_span = pad_sequence(flatten_cell_span, batch_first=True, padding_value=-1)
        gather_index = pad_sequence(gather_index, batch_first=True, padding_value=0)

        batch = {'ques_id': ques_id, 'input_ids': flatten_input_ids, 'input_mask': flatten_input_mask,
                 'batch_text_len': batch_text_len, 'input_token_type': flatten_input_token_type,
                 'tabel_text': tabel_text, 'tabel_mask': tabel_mask, 'tabel_token_type': tabel_token_type,
                 'image_feat': image_feat_padded, 'batch_image_len': batch_image_len,
                 'image_ques_id': image_ques_padded, 'image_token_type': image_token_type_padded,
                 'image_mask': image_mask_padded,
                 'ans_ids': gpt_ids, 'attn_mask': attn_mask,
                 'ans_loss_mask': ans_loss_mask,
                 'golden_source_index': golden_source_index, 'answer_list': answer_list,
                 'source_index': source_index, 'source_span': source_span,
                 'table_connect_num': table_connect_num, 'flatten_connect_spans': flatten_connect_spans,
                 'flatten_connect_index': flatten_connect_index, 'span_lable': span_lable,
                 'flatten_cell_span': flatten_cell_span, 'tabel_cell_num': tabel_cell_num,
                 'gather_index': gather_index, 'flatten_span_label': flatten_span_label
                 }

        return batch


class generative_source2cell_mmqa_dataset(Dataset):
    def __init__(self, tokenizer, input_example_dir, text_dir, img_dir, table_dir, doc_stride,
                 max_query_len, max_seq_len, connect_path, span_dir, is_debug=False, is_test=False):
        super().__init__()
        self.is_debug = is_debug
        self.is_test = is_test
        self.tokenizer = tokenizer
        self.doc_stride = doc_stride
        self.max_query_len = max_query_len
        self.max_seq_len = max_seq_len
        self.connect = pickle.load(open(connect_path, 'rb'))
        self.cell_span = pickle.load(open(span_dir, 'rb'))
        self.paragraphs = pickle.load(open(text_dir, 'rb'))
        self.tables = pickle.load(open(table_dir, 'rb'))
        self.imgs = pickle.load(open(img_dir, 'rb'))
        self.ROW_id = self.tokenizer.convert_tokens_to_ids('ROW')
        self.input_examples = self.read_examples(input_example_dir)

    def __len__(self):
        return len(self.input_examples)

    def example_process(self, input_data):
        pbar = tqdm(total=len(input_data))
        examples = []
        # e_title_id = self.tokenizer.convert_tokens_to_ids('</title>')
        for example in tqdm(input_data):
            qid = example["qid"]
            question_text = example['question']
            ques_tokens = self.tokenizer.tokenize(question_text)
            ques_ids = self.tokenizer.convert_tokens_to_ids(ques_tokens)
            text_context_dict = {doc_id: self.text2id(self.paragraphs[doc_id], ques_ids) for doc_id in
                                 example["metadata"]["text_doc_ids"]}
            table_context = self.tabel2id(self.tables[example["metadata"]["table_id"]], ques_ids)
            image_context = self.image2id(example["metadata"]["image_doc_ids"], ques_ids)

            text_id_list = list(text_context_dict.keys())
            image_id_list = list(image_context.keys())
            golden_source_index = []
            modal_label = []
            for golden_source in example['supporting_context']:
                gs_id = golden_source['doc_id']
                if golden_source['doc_part'] == 'text':
                    try:
                        golden_source_index.append(text_id_list.index(gs_id))
                        modal_label.append(1)
                    except:
                        continue
                elif golden_source['doc_part'] == 'image':
                    try:
                        golden_source_index.append(image_id_list.index(gs_id) + len(text_id_list) + 1)
                        modal_label.append(3)

                    except:
                        continue
                else:
                    golden_source_index.append(len(text_id_list))
                    modal_label.append(2)

            if len(golden_source_index) == 0:
                if self.is_test == True:
                    golden_source_index = [0]
                    modal_label.append(1)
                else:
                    continue
            modal_label.append(0)
            connect_info = self.connect[example["qid"]].copy()
            table_connect_span = []
            table_connect_index = []
            ques_len = len(ques_ids)
            for info in connect_info:
                span = [item + ques_len + 2 for item in info['span']]
                if span[1] >= self.max_seq_len - 1:
                    # 超长
                    break
                sc_index = []
                for sc in info['connect']:
                    if sc[0] == 'text':
                        try:
                            sc_index.append(text_id_list.index(sc[1]))
                        except:
                            continue
                    else:
                        try:
                            #  注意不需要+1
                            sc_index.append(image_id_list.index(sc[1]) + len(text_id_list))
                        except:
                            continue
                if len(sc_index) != 0:
                    table_connect_span.append(torch.tensor(span))
                    table_connect_index.append(torch.tensor(sc_index))
            table_cell_span = []
            cell_span_info = self.cell_span[example["metadata"]["table_id"]].copy()
            for info in cell_span_info:
                span = [item + ques_len + 2 for item in info]
                if span[1] >= self.max_seq_len - 1:
                    # 超长
                    break
                table_cell_span.append(torch.tensor(span))

            source_index = []
            source_span = []
            source_ids = None
            answer = example['answers'][0]
            answer_id = self.tokenizer.encode(' ' + str(answer['answer']), add_special_tokens=False)
            if answer['type'] == 'string':
                if answer['modality'] == 'table':
                    source_index.append(len(text_id_list))
                    source_ids = table_context['ids']
                elif answer['modality'] == 'text':
                    doc_id = answer['text_instances'][0]['doc_id']
                    text_index = text_id_list.index(doc_id)
                    source_index.append(text_index)
                    source_ids = text_context_dict[doc_id]['ids']
                else:
                    doc_id = answer['image_instances'][0]['doc_id']
                    try:
                        image_index = image_id_list.index(doc_id)
                    except:
                        continue
                    source_index.append(len(text_id_list) + 1 + image_index)
                    source_ids = image_context[doc_id]['ids']
            if source_ids is not None:
                for span in range(len(ques_ids), len(source_ids) - len(answer_id)):
                    if list(source_ids[span:span + len(answer_id)]) == answer_id:
                        source_span.append([span, span + len(answer_id)])

            answer_text = str(example['answers'][0]['answer'])
            ans_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
            ques_ids = self.tokenizer.convert_tokens_to_ids(ques_tokens)
            gpt_ids = [self.tokenizer.bos_token_id] + ques_ids + self.tokenizer.encode('[b_source]',
                                                                                       add_special_tokens=False) + [
                          self.tokenizer.mask_token_id] * len(golden_source_index) + self.tokenizer.encode(
                '[e_source]', add_special_tokens=False) + ans_ids + [self.tokenizer.eos_token_id]
            if len(gpt_ids) > self.max_query_len and self.is_test == False:
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
            # modal_label = torch.tensor(modal_label)

            input = {
                "qid": qid,
                "question": {'ques': example["question"], 'ques_tensor': ques_ids},
                "text_context": text_context_dict,
                "table_context": table_context,
                "image_context": image_context,
                "golden_source_index": golden_source_index,
                "answers": {'ids': gpt_ids, 'attn_mask': attn_mask, 'sc_loss_mask': sc_loss_mask,
                            'ans_loss_mask': ans_loss_mask},
                "answer_list": example['answers'],
                'source_index': source_index,
                'source_span': source_span,
                'table_connect_span': table_connect_span,
                'table_connect_index': table_connect_index,
                'table_cell_span': table_cell_span,
                'modal_label': modal_label
            }
            examples.append(input)
            pbar.update(1)
        return examples

    def read_examples(self, dir):
        input_data = read_jsonlines(dir)
        if self.is_debug:
            input_data = input_data[:100]
        exampels = self.example_process(input_data)
        return exampels

    def text2id(self, text_dict, ques_ids):
        ques_len = len(ques_ids)
        ids = text_dict['ids']
        mask = text_dict['masks']
        text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + ids + [
            self.tokenizer.eos_token_id]
        text_mask = mask + [1] * (ques_len + 3)
        if len(text) > self.doc_stride:
            text = text[:self.doc_stride - 1]
            text += [self.tokenizer.eos_token_id]
            text_mask = text_mask[:self.doc_stride]
            token_type_id = [0] * (2 + ques_len) + [1] * (self.doc_stride - 2 - ques_len)
        else:
            token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 1)
        text = torch.tensor(text)
        text_mask = torch.tensor(text_mask)
        token_type_id = torch.tensor(token_type_id)
        return {'ids': text, 'mask': text_mask, 'token_type': token_type_id}

    def tabel2id(self, text_dict, ques_ids):
        ques_len = len(ques_ids)
        ids = text_dict['ids']
        mask = text_dict['masks']
        text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + ids + [
            self.tokenizer.eos_token_id]
        text_mask = mask + [1] * (ques_len + 3)
        if len(text) > self.max_seq_len:
            text = text[:self.max_seq_len - 1]
            text += [self.tokenizer.eos_token_id]
            text_mask = text_mask[:self.max_seq_len]
            token_type_id = [0] * (2 + ques_len) + [1] * (self.max_seq_len - 2 - ques_len)
        else:
            token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 1)
        structure_mask = torch.eye(len(text_mask))
        start = 0
        end = text.index(self.ROW_id)
        structure_mask[start:end, :] = 1
        for idx, id in enumerate(text):
            if id == self.ROW_id:
                start = end
                end = idx
                structure_mask[start:end, start:end] = 1
        structure_mask[end:, end:] = 1
        text = torch.tensor(text)
        text_mask = torch.tensor(text_mask)
        token_type_id = torch.tensor(token_type_id)
        return {'ids': text, 'mask': text_mask, 'token_type': token_type_id, 'structure_mask': structure_mask}

    def image2id(self, image_ids, ques_ids):
        examples = {}
        ques_len = len(ques_ids)
        for id in image_ids:
            try:
                image_dict = self.imgs[id]
            except:
                continue
            ids = image_dict['ids']
            mask = image_dict['masks']
            # feat = image_dict['img_feat']
            text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + [
                self.tokenizer.convert_tokens_to_ids('<title>')] + ids + [
                       self.tokenizer.convert_tokens_to_ids('</title>')] + [
                       self.tokenizer.eos_token_id]
            text_mask = mask + [1] * (ques_len + 5)
            if len(text) > self.max_seq_len:
                text = text[:self.max_seq_len - 1]
                text += [self.tokenizer.eos_token_id]
                text_mask = text_mask[:self.max_seq_len]
                token_type_id = [0] * (2 + ques_len) + [1] * (self.max_seq_len - 2 - ques_len)
            else:
                token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 3)
            text = torch.tensor(text)
            text_mask = torch.tensor(text_mask)
            token_type_id = torch.tensor(token_type_id)
            examples[id] = {'ids': text, 'mask': text_mask, 'token_type': token_type_id}
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
        table_mask = table_context_dict['structure_mask']
        table_one_mask = table_context_dict['mask']

        table_token_type = table_context_dict['token_type']

        image_context_dict = input['image_context']
        images = [self.imgs[key]['img_feat'] for key in image_context_dict.keys()]
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
        golden_source_index = torch.tensor(input['golden_source_index'])

        source_index = input['source_index']
        if len(source_index) > 1:
            source_index = [source_index[-1]]
        elif len(source_index) == 0:
            source_index = [-1]
        source_span = input['source_span']
        if len(source_span) > 1:
            # 取最后一个
            source_span = source_span[-1]
        elif len(source_span) == 0:
            source_span = [0, 1]
        else:
            source_span = source_span[0]
        source_index = torch.tensor(source_index, dtype=torch.int)
        source_span = torch.tensor(source_span, dtype=torch.int)

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
            if connect_index.size(0) == 2:
                connect_source_index = table_connect_index[connect_index[0, 0]]
                for item in connect_source_index:
                    # 第0位用于冗余编码：question/ROW/；/PAD等符号
                    span_lable[item] = idx + 1
        table_cell_span = torch.stack(table_cell_span)
        modal_label = input['modal_label']
        modal_label = torch.tensor(modal_label)

        return [(input['qid'], input_text, input_mask, input_token_type, table_text, table_mask, table_one_mask,
                 table_token_type,
                 images, image_titles, title_masks, image_token_types, gpt_ids, attn_mask, golden_source_index,
                 ans_loss_mask, sc_loss_mask, input['answer_list'], source_index, source_span, table_connect_span,
                 table_connect_index, table_cell_span, gather_index, span_lable, modal_label)]

    def mmqa_collate(self, inputs):
        (ques_id, input_ids, input_mask, input_token_type, tabel_text, tabel_mask, table_one_mask, tabel_token_type,
         image_feat, titles, title_masks, image_token_types, gpt_ids, attn_mask, golden_source_index, ans_loss_mask,
         sc_loss_mask, answer_list, source_index, source_span, table_connect_span, table_connect_index,
         table_cell_span, gather_index, span_lable, modal_label) = map(list, unzip(concat(inputs)))
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

        tabel_text = pad_sequence(tabel_text, batch_first=True, padding_value=0)
        tabel_token_type = pad_sequence(tabel_token_type, batch_first=True, padding_value=0)
        table_one_mask = pad_sequence(table_one_mask, batch_first=True, padding_value=0)
        max_table_len = tabel_text.size(1)
        padd_table_mask = []
        for item in tabel_mask:
            item = item.cuda()
            pad_matrix = torch.zeros((max_table_len - item.size(0), item.size(1))).cuda()
            item = torch.cat((item, pad_matrix), dim=0)
            pad_matrix = torch.zeros((item.size(0), max_table_len - item.size(1))).cuda()
            item = torch.cat((item, pad_matrix), dim=1)
            padd_table_mask.append(item)
        padd_table_mask = torch.stack(padd_table_mask)

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
        batch_image_len = torch.tensor(batch_image_len).cuda()
        image_ques_padded = pad_sequence(image_ques_padded, batch_first=True, padding_value=0)
        image_token_type_padded = pad_sequence(image_token_type_padded, batch_first=True, padding_value=0)

        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        ans_loss_mask = pad_sequence(ans_loss_mask, batch_first=True, padding_value=0)
        sc_loss_mask = pad_sequence(sc_loss_mask, batch_first=True, padding_value=0)
        golden_source_index = pad_sequence(golden_source_index, batch_first=True, padding_value=-1)

        source_index = torch.stack(source_index)
        source_span = torch.stack(source_span)

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

        tabel_cell_num = [item.size(0) for item in table_cell_span]
        flatten_cell_span = []
        for idx, item in enumerate(table_cell_span):
            item = item.reshape(-1)
            flatten_cell_span.append(item)
        tabel_cell_num = torch.tensor(tabel_cell_num)
        flatten_span_label = pad_sequence(span_lable, batch_first=True, padding_value=0)
        flatten_cell_span = pad_sequence(flatten_cell_span, batch_first=True, padding_value=-1)
        gather_index = pad_sequence(gather_index, batch_first=True, padding_value=0)
        modal_label = pad_sequence(modal_label, batch_first=True, padding_value=-1)

        batch = {'ques_id': ques_id, 'input_ids': flatten_input_ids, 'input_mask': flatten_input_mask,
                 'batch_text_len': batch_text_len, 'input_token_type': flatten_input_token_type,
                 'tabel_text': tabel_text, 'tabel_mask': padd_table_mask, 'tabel_token_type': tabel_token_type,
                 'table_one_mask': table_one_mask,
                 'image_feat': image_feat_padded, 'batch_image_len': batch_image_len,
                 'image_ques_id': image_ques_padded, 'image_token_type': image_token_type_padded,
                 'image_mask': image_mask_padded,
                 'ans_ids': gpt_ids, 'attn_mask': attn_mask,
                 'ans_loss_mask': ans_loss_mask, 'sc_loss_mask': sc_loss_mask,
                 'golden_source_index': golden_source_index, 'answer_list': answer_list,
                 'source_index': source_index, 'source_span': source_span,
                 'table_connect_num': table_connect_num, 'flatten_connect_spans': flatten_connect_spans,
                 'flatten_connect_index': flatten_connect_index, 'span_lable': span_lable,
                 'flatten_cell_span': flatten_cell_span, 'tabel_cell_num': tabel_cell_num,
                 'gather_index': gather_index, 'flatten_span_label': flatten_span_label, 'modal_label': modal_label
                 }

        return batch


class generative_source2cell_sep_mmqa_dataset(Dataset):
    def __init__(self, tokenizer, input_example_dir, text_dir, img_dir, table_dir, doc_stride,
                 max_query_len, max_seq_len, connect_path, span_dir, is_debug=False, is_test=False):
        super().__init__()
        self.is_debug = is_debug
        self.is_test = is_test
        self.tokenizer = tokenizer
        self.doc_stride = doc_stride
        self.max_query_len = max_query_len
        self.max_seq_len = max_seq_len
        self.connect = pickle.load(open(connect_path, 'rb'))
        self.cell_span = pickle.load(open(span_dir, 'rb'))
        self.paragraphs = pickle.load(open(text_dir, 'rb'))
        self.tables = pickle.load(open(table_dir, 'rb'))
        self.imgs = pickle.load(open(img_dir, 'rb'))
        self.ROW_id = self.tokenizer.convert_tokens_to_ids('ROW')
        self.input_examples = self.read_examples(input_example_dir)

    def __len__(self):
        return len(self.input_examples)

    def example_process(self, input_data):
        pbar = tqdm(total=len(input_data))
        examples = []
        # e_title_id = self.tokenizer.convert_tokens_to_ids('</title>')
        for example in tqdm(input_data):
            qid = example["qid"]
            question_text = example['question']
            ques_tokens = self.tokenizer.tokenize(question_text)
            ques_ids = self.tokenizer.convert_tokens_to_ids(ques_tokens)
            text_context_dict = {doc_id: self.text2id(self.paragraphs[doc_id], ques_ids) for doc_id in
                                 example["metadata"]["text_doc_ids"]}
            table_context = self.tabel2id(self.tables[example["metadata"]["table_id"]], ques_ids)
            image_context = self.image2id(example["metadata"]["image_doc_ids"], ques_ids)

            text_id_list = list(text_context_dict.keys())
            image_id_list = list(image_context.keys())
            golden_source_index = []
            modal_label = []
            for golden_source in example['supporting_context']:
                gs_id = golden_source['doc_id']
                if golden_source['doc_part'] == 'text':
                    try:
                        golden_source_index.append(text_id_list.index(gs_id))
                        modal_label.append(0)
                    except:
                        continue
                elif golden_source['doc_part'] == 'image':
                    try:
                        golden_source_index.append(image_id_list.index(gs_id) + len(text_id_list) + 1)
                        modal_label.append(2)

                    except:
                        continue
                else:
                    golden_source_index.append(len(text_id_list))
                    modal_label.append(1)

            if len(golden_source_index) == 0:
                if self.is_test == True:
                    golden_source_index = [0]
                    modal_label.append(1)
                else:
                    continue
            connect_info = self.connect[example["qid"]].copy()
            table_connect_span = []
            table_connect_index = []
            ques_len = len(ques_ids)
            for info in connect_info:
                span = [item + ques_len + 2 for item in info['span']]
                if span[1] >= self.max_seq_len - 1:
                    # 超长
                    break
                sc_index = []
                for sc in info['connect']:
                    if sc[0] == 'text':
                        try:
                            sc_index.append(text_id_list.index(sc[1]))
                        except:
                            continue
                    else:
                        try:
                            #  注意不需要+1
                            sc_index.append(image_id_list.index(sc[1]) + len(text_id_list))
                        except:
                            continue
                if len(sc_index) != 0:
                    table_connect_span.append(torch.tensor(span))
                    table_connect_index.append(torch.tensor(sc_index))
            table_cell_span = []
            cell_span_info = self.cell_span[example["metadata"]["table_id"]].copy()
            for info in cell_span_info:
                span = [item + ques_len + 2 for item in info]
                if span[1] >= self.max_seq_len - 1:
                    # 超长
                    break
                table_cell_span.append(torch.tensor(span))

            source_index = []
            source_span = []
            source_ids = None
            answer = example['answers'][0]
            answer_id = self.tokenizer.encode(' ' + str(answer['answer']), add_special_tokens=False)
            if answer['type'] == 'string':
                if answer['modality'] == 'table':
                    source_index.append(len(text_id_list))
                    source_ids = table_context['ids']
                elif answer['modality'] == 'text':
                    doc_id = answer['text_instances'][0]['doc_id']
                    text_index = text_id_list.index(doc_id)
                    source_index.append(text_index)
                    source_ids = text_context_dict[doc_id]['ids']
                else:
                    doc_id = answer['image_instances'][0]['doc_id']
                    try:
                        image_index = image_id_list.index(doc_id)
                    except:
                        continue
                    source_index.append(len(text_id_list) + 1 + image_index)
                    source_ids = image_context[doc_id]['ids']
            if source_ids is not None:
                for span in range(len(ques_ids), len(source_ids) - len(answer_id)):
                    if list(source_ids[span:span + len(answer_id)]) == answer_id:
                        source_span.append([span, span + len(answer_id)])

            answer_text = str(example['answers'][0]['answer'])
            ans_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
            ques_ids = self.tokenizer.convert_tokens_to_ids(ques_tokens)
            gpt_ids = [self.tokenizer.bos_token_id] + ques_ids + self.tokenizer.encode('[b_source]',
                                                                                       add_special_tokens=False) + [
                          self.tokenizer.mask_token_id] * len(golden_source_index) + self.tokenizer.encode(
                '[e_source]', add_special_tokens=False) + ans_ids + [self.tokenizer.eos_token_id]
            if len(gpt_ids) > self.max_query_len and self.is_test == False:
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
            # modal_label = torch.tensor(modal_label)

            input = {
                "qid": qid,
                "question": {'ques': example["question"], 'ques_tensor': ques_ids},
                "text_context": text_context_dict,
                "table_context": table_context,
                "image_context": image_context,
                "golden_source_index": golden_source_index,
                "answers": {'ids': gpt_ids, 'attn_mask': attn_mask, 'sc_loss_mask': sc_loss_mask,
                            'ans_loss_mask': ans_loss_mask},
                "answer_list": example['answers'],
                'source_index': source_index,
                'source_span': source_span,
                'table_connect_span': table_connect_span,
                'table_connect_index': table_connect_index,
                'table_cell_span': table_cell_span,
                'modal_label': modal_label
            }
            examples.append(input)
            pbar.update(1)
        return examples

    def read_examples(self, dir):
        input_data = read_jsonlines(dir)
        if self.is_debug:
            input_data = input_data[:100]
        exampels = self.example_process(input_data)
        return exampels

    def text2id(self, text_dict, ques_ids):
        ques_len = len(ques_ids)
        ids = text_dict['ids']
        mask = text_dict['masks']
        text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + ids + [
            self.tokenizer.eos_token_id]
        text_mask = mask + [1] * (ques_len + 3)
        if len(text) > self.doc_stride:
            text = text[:self.doc_stride - 1]
            text += [self.tokenizer.eos_token_id]
            text_mask = text_mask[:self.doc_stride]
            token_type_id = [0] * (2 + ques_len) + [1] * (self.doc_stride - 2 - ques_len)
        else:
            token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 1)
        text = torch.tensor(text)
        text_mask = torch.tensor(text_mask)
        token_type_id = torch.tensor(token_type_id)
        return {'ids': text, 'mask': text_mask, 'token_type': token_type_id}

    def tabel2id(self, text_dict, ques_ids):
        ques_len = len(ques_ids)
        ids = text_dict['ids']
        mask = text_dict['masks']
        text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + ids + [
            self.tokenizer.eos_token_id]
        text_mask = mask + [1] * (ques_len + 3)
        if len(text) > self.max_seq_len:
            text = text[:self.max_seq_len - 1]
            text += [self.tokenizer.eos_token_id]
            text_mask = text_mask[:self.max_seq_len]
            token_type_id = [0] * (2 + ques_len) + [1] * (self.max_seq_len - 2 - ques_len)
        else:
            token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 1)
        structure_mask = torch.eye(len(text_mask))
        start = 0
        end = text.index(self.ROW_id)
        structure_mask[start:end, :] = 1
        for idx, id in enumerate(text):
            if id == self.ROW_id:
                start = end
                end = idx
                structure_mask[start:end, start:end] = 1
        structure_mask[end:, end:] = 1
        text = torch.tensor(text)
        text_mask = torch.tensor(text_mask)
        token_type_id = torch.tensor(token_type_id)
        return {'ids': text, 'mask': text_mask, 'token_type': token_type_id, 'structure_mask': structure_mask}

    def image2id(self, image_ids, ques_ids):
        examples = {}
        ques_len = len(ques_ids)
        for id in image_ids:
            try:
                image_dict = self.imgs[id]
            except:
                continue
            ids = image_dict['ids']
            mask = image_dict['masks']
            # feat = image_dict['img_feat']
            text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + [
                self.tokenizer.convert_tokens_to_ids('<title>')] + ids + [
                       self.tokenizer.convert_tokens_to_ids('</title>')] + [
                       self.tokenizer.eos_token_id]
            text_mask = mask + [1] * (ques_len + 5)
            if len(text) > self.max_seq_len:
                text = text[:self.max_seq_len - 1]
                text += [self.tokenizer.eos_token_id]
                text_mask = text_mask[:self.max_seq_len]
                token_type_id = [0] * (2 + ques_len) + [1] * (self.max_seq_len - 2 - ques_len)
            else:
                token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 3)
            text = torch.tensor(text)
            text_mask = torch.tensor(text_mask)
            token_type_id = torch.tensor(token_type_id)
            examples[id] = {'ids': text, 'mask': text_mask, 'token_type': token_type_id}
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
        table_mask = table_context_dict['structure_mask']
        table_one_mask = table_context_dict['mask']

        table_token_type = table_context_dict['token_type']

        image_context_dict = input['image_context']
        images = [self.imgs[key]['img_feat'] for key in image_context_dict.keys()]
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
        golden_source_index = torch.tensor(input['golden_source_index'])

        source_index = input['source_index']
        if len(source_index) > 1:
            source_index = [source_index[-1]]
        elif len(source_index) == 0:
            source_index = [-1]
        source_span = input['source_span']
        if len(source_span) > 1:
            # 取最后一个
            source_span = source_span[-1]
        elif len(source_span) == 0:
            source_span = [0, 1]
        else:
            source_span = source_span[0]
        source_index = torch.tensor(source_index, dtype=torch.int)
        source_span = torch.tensor(source_span, dtype=torch.int)

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
            if connect_index.size(0) == 2:
                connect_source_index = table_connect_index[connect_index[0, 0]]
                for item in connect_source_index:
                    # 第0位用于冗余编码：question/ROW/；/PAD等符号
                    span_lable[item] = idx + 1
        table_cell_span = torch.stack(table_cell_span)
        modal_label = input['modal_label']
        modal_label = torch.tensor(modal_label)

        return [(input['qid'], input_text, input_mask, input_token_type, table_text, table_mask, table_one_mask,
                 table_token_type,
                 images, image_titles, title_masks, image_token_types, gpt_ids, attn_mask, golden_source_index,
                 ans_loss_mask, sc_loss_mask, input['answer_list'], source_index, source_span, table_connect_span,
                 table_connect_index, table_cell_span, gather_index, span_lable, modal_label)]

    def mmqa_collate(self, inputs):
        (ques_id, input_ids, input_mask, input_token_type, tabel_text, tabel_mask, table_one_mask, tabel_token_type,
         image_feat, titles, title_masks, image_token_types, gpt_ids, attn_mask, golden_source_index, ans_loss_mask,
         sc_loss_mask, answer_list, source_index, source_span, table_connect_span, table_connect_index,
         table_cell_span, gather_index, span_lable, modal_label) = map(list, unzip(concat(inputs)))
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

        tabel_text = pad_sequence(tabel_text, batch_first=True, padding_value=0)
        tabel_token_type = pad_sequence(tabel_token_type, batch_first=True, padding_value=0)
        table_one_mask = pad_sequence(table_one_mask, batch_first=True, padding_value=0)
        max_table_len = tabel_text.size(1)
        padd_table_mask = []
        for item in tabel_mask:
            item = item.cuda()
            pad_matrix = torch.zeros((max_table_len - item.size(0), item.size(1))).cuda()
            item = torch.cat((item, pad_matrix), dim=0)
            pad_matrix = torch.zeros((item.size(0), max_table_len - item.size(1))).cuda()
            item = torch.cat((item, pad_matrix), dim=1)
            padd_table_mask.append(item)
        padd_table_mask = torch.stack(padd_table_mask)

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
        batch_image_len = torch.tensor(batch_image_len).cuda()
        image_ques_padded = pad_sequence(image_ques_padded, batch_first=True, padding_value=0)
        image_token_type_padded = pad_sequence(image_token_type_padded, batch_first=True, padding_value=0)

        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        ans_loss_mask = pad_sequence(ans_loss_mask, batch_first=True, padding_value=0)
        sc_loss_mask = pad_sequence(sc_loss_mask, batch_first=True, padding_value=0)
        golden_source_index = pad_sequence(golden_source_index, batch_first=True, padding_value=-1)

        source_index = torch.stack(source_index)
        source_span = torch.stack(source_span)

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

        tabel_cell_num = [item.size(0) for item in table_cell_span]
        flatten_cell_span = []
        for idx, item in enumerate(table_cell_span):
            item = item.reshape(-1)
            flatten_cell_span.append(item)
        tabel_cell_num = torch.tensor(tabel_cell_num)
        flatten_span_label = pad_sequence(span_lable, batch_first=True, padding_value=0)
        flatten_cell_span = pad_sequence(flatten_cell_span, batch_first=True, padding_value=-1)
        gather_index = pad_sequence(gather_index, batch_first=True, padding_value=0)
        modal_label = pad_sequence(modal_label, batch_first=True, padding_value=-1)

        batch = {'ques_id': ques_id, 'input_ids': flatten_input_ids, 'input_mask': flatten_input_mask,
                 'batch_text_len': batch_text_len, 'input_token_type': flatten_input_token_type,
                 'tabel_text': tabel_text, 'tabel_mask': padd_table_mask, 'tabel_token_type': tabel_token_type,
                 'table_one_mask': table_one_mask,
                 'image_feat': image_feat_padded, 'batch_image_len': batch_image_len,
                 'image_ques_id': image_ques_padded, 'image_token_type': image_token_type_padded,
                 'image_mask': image_mask_padded,
                 'ans_ids': gpt_ids, 'attn_mask': attn_mask,
                 'ans_loss_mask': ans_loss_mask, 'sc_loss_mask': sc_loss_mask,
                 'golden_source_index': golden_source_index, 'answer_list': answer_list,
                 'source_index': source_index, 'source_span': source_span,
                 'table_connect_num': table_connect_num, 'flatten_connect_spans': flatten_connect_spans,
                 'flatten_connect_index': flatten_connect_index, 'span_lable': span_lable,
                 'flatten_cell_span': flatten_cell_span, 'tabel_cell_num': tabel_cell_num,
                 'gather_index': gather_index, 'flatten_span_label': flatten_span_label, 'modal_label': modal_label
                 }

        return batch


class generative_source2cell_wo_re_mmqa_dataset(Dataset):
    def __init__(self, tokenizer, input_example_dir, text_dir, img_dir, table_dir, doc_stride,
                 max_query_len, max_seq_len, connect_path, span_dir, is_debug=False, is_test=False):
        super().__init__()
        self.is_debug = is_debug
        self.is_test = is_test
        self.tokenizer = tokenizer
        self.doc_stride = doc_stride
        self.max_query_len = max_query_len
        self.max_seq_len = max_seq_len
        self.connect = pickle.load(open(connect_path, 'rb'))
        self.cell_span = pickle.load(open(span_dir, 'rb'))
        self.paragraphs = pickle.load(open(text_dir, 'rb'))
        self.tables = pickle.load(open(table_dir, 'rb'))
        self.imgs = pickle.load(open(img_dir, 'rb'))
        self.ROW_id = self.tokenizer.convert_tokens_to_ids('ROW')
        self.input_examples = self.read_examples(input_example_dir)

    def __len__(self):
        return len(self.input_examples)

    def example_process(self, input_data):
        pbar = tqdm(total=len(input_data))
        examples = []
        # e_title_id = self.tokenizer.convert_tokens_to_ids('</title>')
        for example in tqdm(input_data):
            qid = example["qid"]
            question_text = example['question']
            ques_tokens = self.tokenizer.tokenize(question_text)
            ques_ids = self.tokenizer.convert_tokens_to_ids(ques_tokens)
            text_context_dict = {doc_id: self.text2id(self.paragraphs[doc_id], ques_ids) for doc_id in
                                 example["metadata"]["text_doc_ids"]}
            table_context = self.tabel2id(self.tables[example["metadata"]["table_id"]], ques_ids)
            image_context = self.image2id(example["metadata"]["image_doc_ids"], ques_ids)

            text_id_list = list(text_context_dict.keys())
            image_id_list = list(image_context.keys())
            golden_source_index = []
            modal_label = []
            for golden_source in example['supporting_context']:
                gs_id = golden_source['doc_id']
                if golden_source['doc_part'] == 'text':
                    try:
                        golden_source_index.append(text_id_list.index(gs_id))
                        modal_label.append(1)
                    except:
                        continue
                elif golden_source['doc_part'] == 'image':
                    try:
                        golden_source_index.append(image_id_list.index(gs_id) + len(text_id_list) + 1)
                        modal_label.append(3)

                    except:
                        continue
                else:
                    golden_source_index.append(len(text_id_list))
                    modal_label.append(2)

            if len(golden_source_index) == 0:
                if self.is_test == True:
                    golden_source_index = [0]
                    modal_label.append(1)
                else:
                    continue
            modal_label.append(0)
            connect_info = self.connect[example["qid"]].copy()
            table_connect_span = []
            table_connect_index = []
            ques_len = len(ques_ids)
            for info in connect_info:
                span = [item + ques_len + 2 for item in info['span']]
                if span[1] >= self.max_seq_len - 1:
                    # 超长
                    break
                sc_index = []
                for sc in info['connect']:
                    if sc[0] == 'text':
                        try:
                            sc_index.append(text_id_list.index(sc[1]))
                        except:
                            continue
                    else:
                        try:
                            #  注意不需要+1
                            sc_index.append(image_id_list.index(sc[1]) + len(text_id_list))
                        except:
                            continue
                if len(sc_index) != 0:
                    table_connect_span.append(torch.tensor(span))
                    table_connect_index.append(torch.tensor(sc_index))
            table_cell_span = []
            cell_span_info = self.cell_span[example["metadata"]["table_id"]].copy()
            for info in cell_span_info:
                span = [item + ques_len + 2 for item in info]
                if span[1] >= self.max_seq_len - 1:
                    # 超长
                    break
                table_cell_span.append(torch.tensor(span))

            source_index = []
            source_span = []
            source_ids = None
            answer = example['answers'][0]
            answer_id = self.tokenizer.encode(' ' + str(answer['answer']), add_special_tokens=False)
            if answer['type'] == 'string':
                if answer['modality'] == 'table':
                    source_index.append(len(text_id_list))
                    source_ids = table_context['ids']
                elif answer['modality'] == 'text':
                    doc_id = answer['text_instances'][0]['doc_id']
                    text_index = text_id_list.index(doc_id)
                    source_index.append(text_index)
                    source_ids = text_context_dict[doc_id]['ids']
                else:
                    doc_id = answer['image_instances'][0]['doc_id']
                    try:
                        image_index = image_id_list.index(doc_id)
                    except:
                        continue
                    source_index.append(len(text_id_list) + 1 + image_index)
                    source_ids = image_context[doc_id]['ids']
            if source_ids is not None:
                for span in range(len(ques_ids), len(source_ids) - len(answer_id)):
                    if list(source_ids[span:span + len(answer_id)]) == answer_id:
                        source_span.append([span, span + len(answer_id)])

            answer_text = str(example['answers'][0]['answer'])
            ans_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
            ques_ids = self.tokenizer.convert_tokens_to_ids(ques_tokens)
            gpt_ids = [self.tokenizer.bos_token_id] + ques_ids + self.tokenizer.encode(
                '[e_source]', add_special_tokens=False) + ans_ids + [self.tokenizer.eos_token_id]
            if len(gpt_ids) > self.max_query_len and self.is_test == False:
                continue
            attn_mask = [1] * len(gpt_ids)
            ans_loss_mask = [0] * len(attn_mask)
            ans_loss_mask[-(len(ans_ids) + 1):] = [1] * (len(ans_ids) + 1)

            gpt_ids = torch.tensor(gpt_ids)
            attn_mask = torch.tensor(attn_mask)
            ans_loss_mask = torch.tensor(ans_loss_mask)
            ques_ids = torch.tensor(ques_ids)
            # modal_label = torch.tensor(modal_label)

            input = {
                "qid": qid,
                "question": {'ques': example["question"], 'ques_tensor': ques_ids},
                "text_context": text_context_dict,
                "table_context": table_context,
                "image_context": image_context,
                "golden_source_index": golden_source_index,
                "answers": {'ids': gpt_ids, 'attn_mask': attn_mask,
                            'ans_loss_mask': ans_loss_mask},
                "answer_list": example['answers'],
                'source_index': source_index,
                'source_span': source_span,
                'table_connect_span': table_connect_span,
                'table_connect_index': table_connect_index,
                'table_cell_span': table_cell_span,
                'modal_label': modal_label
            }
            examples.append(input)
            pbar.update(1)
        return examples

    def read_examples(self, dir):
        input_data = read_jsonlines(dir)
        if self.is_debug:
            input_data = input_data[:100]
        exampels = self.example_process(input_data)
        return exampels

    def text2id(self, text_dict, ques_ids):
        ques_len = len(ques_ids)
        ids = text_dict['ids']
        mask = text_dict['masks']
        text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + ids + [
            self.tokenizer.eos_token_id]
        text_mask = mask + [1] * (ques_len + 3)
        if len(text) > self.doc_stride:
            text = text[:self.doc_stride - 1]
            text += [self.tokenizer.eos_token_id]
            text_mask = text_mask[:self.doc_stride]
            token_type_id = [0] * (2 + ques_len) + [1] * (self.doc_stride - 2 - ques_len)
        else:
            token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 1)
        text = torch.tensor(text)
        text_mask = torch.tensor(text_mask)
        token_type_id = torch.tensor(token_type_id)
        return {'ids': text, 'mask': text_mask, 'token_type': token_type_id}

    def tabel2id(self, text_dict, ques_ids):
        ques_len = len(ques_ids)
        ids = text_dict['ids']
        mask = text_dict['masks']
        text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + ids + [
            self.tokenizer.eos_token_id]
        text_mask = mask + [1] * (ques_len + 3)
        if len(text) > self.max_seq_len:
            text = text[:self.max_seq_len - 1]
            text += [self.tokenizer.eos_token_id]
            text_mask = text_mask[:self.max_seq_len]
            token_type_id = [0] * (2 + ques_len) + [1] * (self.max_seq_len - 2 - ques_len)
        else:
            token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 1)
        structure_mask = torch.eye(len(text_mask))
        start = 0
        end = text.index(self.ROW_id)
        structure_mask[start:end, :] = 1
        for idx, id in enumerate(text):
            if id == self.ROW_id:
                start = end
                end = idx
                structure_mask[start:end, start:end] = 1
        structure_mask[end:, end:] = 1
        text = torch.tensor(text)
        text_mask = torch.tensor(text_mask)
        token_type_id = torch.tensor(token_type_id)
        return {'ids': text, 'mask': text_mask, 'token_type': token_type_id, 'structure_mask': structure_mask}

    def image2id(self, image_ids, ques_ids):
        examples = {}
        ques_len = len(ques_ids)
        for id in image_ids:
            try:
                image_dict = self.imgs[id]
            except:
                continue
            ids = image_dict['ids']
            mask = image_dict['masks']
            # feat = image_dict['img_feat']
            text = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id] + [
                self.tokenizer.convert_tokens_to_ids('<title>')] + ids + [
                       self.tokenizer.convert_tokens_to_ids('</title>')] + [
                       self.tokenizer.eos_token_id]
            text_mask = mask + [1] * (ques_len + 5)
            if len(text) > self.max_seq_len:
                text = text[:self.max_seq_len - 1]
                text += [self.tokenizer.eos_token_id]
                text_mask = text_mask[:self.max_seq_len]
                token_type_id = [0] * (2 + ques_len) + [1] * (self.max_seq_len - 2 - ques_len)
            else:
                token_type_id = [0] * (2 + ques_len) + [1] * (len(ids) + 3)
            text = torch.tensor(text)
            text_mask = torch.tensor(text_mask)
            token_type_id = torch.tensor(token_type_id)
            examples[id] = {'ids': text, 'mask': text_mask, 'token_type': token_type_id}
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
        table_mask = table_context_dict['structure_mask']
        table_one_mask = table_context_dict['mask']

        table_token_type = table_context_dict['token_type']

        image_context_dict = input['image_context']
        images = [self.imgs[key]['img_feat'] for key in image_context_dict.keys()]
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
        golden_source_index = torch.tensor(input['golden_source_index'])

        source_index = input['source_index']
        if len(source_index) > 1:
            source_index = [source_index[-1]]
        elif len(source_index) == 0:
            source_index = [-1]
        source_span = input['source_span']
        if len(source_span) > 1:
            # 取最后一个
            source_span = source_span[-1]
        elif len(source_span) == 0:
            source_span = [0, 1]
        else:
            source_span = source_span[0]
        source_index = torch.tensor(source_index, dtype=torch.int)
        source_span = torch.tensor(source_span, dtype=torch.int)

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
            if connect_index.size(0) == 2:
                connect_source_index = table_connect_index[connect_index[0, 0]]
                for item in connect_source_index:
                    # 第0位用于冗余编码：question/ROW/；/PAD等符号
                    span_lable[item] = idx + 1
        table_cell_span = torch.stack(table_cell_span)
        modal_label = input['modal_label']
        modal_label = torch.tensor(modal_label)

        return [(input['qid'], input_text, input_mask, input_token_type, table_text, table_mask, table_one_mask,
                 table_token_type,
                 images, image_titles, title_masks, image_token_types, gpt_ids, attn_mask, golden_source_index,
                 ans_loss_mask, input['answer_list'], source_index, source_span, table_connect_span,
                 table_connect_index, table_cell_span, gather_index, span_lable, modal_label)]

    def mmqa_collate(self, inputs):
        (ques_id, input_ids, input_mask, input_token_type, tabel_text, tabel_mask, table_one_mask, tabel_token_type,
         image_feat, titles, title_masks, image_token_types, gpt_ids, attn_mask, golden_source_index, ans_loss_mask,
         answer_list, source_index, source_span, table_connect_span, table_connect_index,
         table_cell_span, gather_index, span_lable, modal_label) = map(list, unzip(concat(inputs)))
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

        tabel_text = pad_sequence(tabel_text, batch_first=True, padding_value=0)
        tabel_token_type = pad_sequence(tabel_token_type, batch_first=True, padding_value=0)
        table_one_mask = pad_sequence(table_one_mask, batch_first=True, padding_value=0)
        max_table_len = tabel_text.size(1)
        padd_table_mask = []
        for item in tabel_mask:
            item = item.cuda()
            pad_matrix = torch.zeros((max_table_len - item.size(0), item.size(1))).cuda()
            item = torch.cat((item, pad_matrix), dim=0)
            pad_matrix = torch.zeros((item.size(0), max_table_len - item.size(1))).cuda()
            item = torch.cat((item, pad_matrix), dim=1)
            padd_table_mask.append(item)
        padd_table_mask = torch.stack(padd_table_mask)

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
        batch_image_len = torch.tensor(batch_image_len).cuda()
        image_ques_padded = pad_sequence(image_ques_padded, batch_first=True, padding_value=0)
        image_token_type_padded = pad_sequence(image_token_type_padded, batch_first=True, padding_value=0)

        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        ans_loss_mask = pad_sequence(ans_loss_mask, batch_first=True, padding_value=0)
        golden_source_index = pad_sequence(golden_source_index, batch_first=True, padding_value=-1)

        source_index = torch.stack(source_index)
        source_span = torch.stack(source_span)

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

        tabel_cell_num = [item.size(0) for item in table_cell_span]
        flatten_cell_span = []
        for idx, item in enumerate(table_cell_span):
            item = item.reshape(-1)
            flatten_cell_span.append(item)
        tabel_cell_num = torch.tensor(tabel_cell_num)
        flatten_span_label = pad_sequence(span_lable, batch_first=True, padding_value=0)
        flatten_cell_span = pad_sequence(flatten_cell_span, batch_first=True, padding_value=-1)
        gather_index = pad_sequence(gather_index, batch_first=True, padding_value=0)
        modal_label = pad_sequence(modal_label, batch_first=True, padding_value=-1)

        batch = {'ques_id': ques_id, 'input_ids': flatten_input_ids, 'input_mask': flatten_input_mask,
                 'batch_text_len': batch_text_len, 'input_token_type': flatten_input_token_type,
                 'tabel_text': tabel_text, 'tabel_mask': padd_table_mask, 'tabel_token_type': tabel_token_type,
                 'table_one_mask': table_one_mask,
                 'image_feat': image_feat_padded, 'batch_image_len': batch_image_len,
                 'image_ques_id': image_ques_padded, 'image_token_type': image_token_type_padded,
                 'image_mask': image_mask_padded,
                 'ans_ids': gpt_ids, 'attn_mask': attn_mask,
                 'ans_loss_mask': ans_loss_mask,
                 'golden_source_index': golden_source_index, 'answer_list': answer_list,
                 'source_index': source_index, 'source_span': source_span,
                 'table_connect_num': table_connect_num, 'flatten_connect_spans': flatten_connect_spans,
                 'flatten_connect_index': flatten_connect_index, 'span_lable': span_lable,
                 'flatten_cell_span': flatten_cell_span, 'tabel_cell_num': tabel_cell_num,
                 'gather_index': gather_index, 'flatten_span_label': flatten_span_label, 'modal_label': modal_label
                 }

        return batch
