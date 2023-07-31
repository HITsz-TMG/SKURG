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
    def __init__(self, tokenizer, input_example_dir, doc_stride,
                 max_query_len, max_seq_len, is_test=False):
        super().__init__()
        self.is_test = is_test
        self.tokenizer = tokenizer
        self.doc_stride = doc_stride
        self.max_query_len = max_query_len
        self.max_seq_len = max_seq_len
        self.input_examples = self.read_examples(input_example_dir)

    def __len__(self):
        return len(self.input_examples)

    def example_process(self, input_data):
        pbar = tqdm(total=len(input_data))
        examples = []
        for item in tqdm(input_data):
            title = item['title']
            title = self.tokenizer.tokenize(title)
            for content in item['paragraphs']:
                context = self.tokenizer.tokenize(' ' + content['context'])
                context = ['<title>'] + title + ['</title>'] + context
                context_ids = self.tokenizer.convert_tokens_to_ids(context)
                for example in content['qas']:
                    qid = example['id']
                    if example['is_impossible'] == False:
                        # We only use the examples that have an answer.
                        question_text = example['question']
                        ques_tokens = self.tokenizer.tokenize(question_text)
                        ques_ids = self.tokenizer.convert_tokens_to_ids(ques_tokens)
                        text = [self.tokenizer.bos_token_id] + ques_ids + [
                            self.tokenizer.eos_token_id] + context_ids + [
                                   self.tokenizer.eos_token_id]
                        text_mask = [1] * len(text)
                        ques_len = len(ques_ids)
                        if len(text) > self.doc_stride:
                            text = text[:self.doc_stride - 1]
                            text += [self.tokenizer.eos_token_id]
                            text_mask = text_mask[:self.doc_stride]
                            token_type_id = [0] * (2 + ques_len) + [1] * (self.doc_stride - 2 - ques_len)
                        else:
                            token_type_id = [0] * (2 + ques_len) + [1] * (len(context_ids) + 1)
                        text = torch.tensor(text, dtype=torch.long)
                        token_type_id = torch.tensor(token_type_id, dtype=torch.long)
                        text_mask = torch.tensor(text_mask, dtype=torch.long)
                        answer_text = str(example['answers'][0]['text'])
                        ans_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
                        gpt_ids = [self.tokenizer.bos_token_id] + ques_ids + self.tokenizer.encode(
                            '[e_source]', add_special_tokens=False) + ans_ids + [self.tokenizer.eos_token_id]
                        if len(gpt_ids) > self.max_query_len and self.is_test == False:
                            continue
                        attn_mask = [1] * len(gpt_ids)
                        ans_loss_mask = [0] * len(attn_mask)
                        ans_loss_mask[-(len(ans_ids) + 1):] = [1] * (len(ans_ids) + 1)

                        gpt_ids = torch.tensor(gpt_ids, dtype=torch.long)
                        attn_mask = torch.tensor(attn_mask, dtype=torch.long)
                        ans_loss_mask = torch.tensor(ans_loss_mask, dtype=torch.long)
                        ques_ids = torch.tensor(ques_ids, dtype=torch.long)
                        input = {
                            "qid": qid,
                            "question": {'ques': example["question"], 'ques_tensor': ques_ids},
                            "text_mask": text_mask,
                            "text": text,
                            "token_type_id": token_type_id,
                            "answers": {'ids': gpt_ids, 'attn_mask': attn_mask, 'ans_loss_mask': ans_loss_mask},
                            "answer_list": example['answers'],
                        }
                        examples.append(input)
            pbar.update(1)
        return examples

    def read_examples(self, dir):
        input_data = json.load(open(dir, 'r'))
        input_data = input_data['data']
        exampels = self.example_process(input_data)
        return exampels

    def __getitem__(self, i):
        input = self.input_examples[i]
        input_text = input['text']
        input_mask = input['text_mask']
        input_token_type = input['token_type_id']
        gpt_ids = input['answers']['ids']
        attn_mask = input['answers']['attn_mask']
        ans_loss_mask = input['answers']['ans_loss_mask']

        return [(input['qid'], input_text, input_mask, input_token_type, gpt_ids, attn_mask,
                 ans_loss_mask, input['answer_list'])]

    def mmqa_collate(self, inputs):
        (ques_id, input_ids, input_mask, input_token_type, gpt_ids, attn_mask, ans_loss_mask, answer_list) \
            = map(list, unzip(concat(inputs)))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        input_token_type = pad_sequence(input_token_type, batch_first=True, padding_value=0)

        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        ans_loss_mask = pad_sequence(ans_loss_mask, batch_first=True, padding_value=0)

        batch = {'ques_id': ques_id, 'input_ids': input_ids, 'input_mask': input_mask,
                 'input_token_type': input_token_type,
                 'ans_ids': gpt_ids, 'attn_mask': attn_mask,
                 'ans_loss_mask': ans_loss_mask, 'answer_list': answer_list}

        return batch
