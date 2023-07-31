# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BART model."""
import copy
import math
import random
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import string
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, LogitsProcessorList, MinLengthLogitsProcessor,
                          BeamScorer)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

# logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/bart-base"
_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 768]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "valhalla/bart-large-sst2"
_SEQ_CLASS_EXPECTED_LOSS = 0.0
_SEQ_CLASS_EXPECTED_OUTPUT = "'POSITIVE'"

# QuestionAsnwering docstring
_CHECKPOINT_FOR_QA = "valhalla/bart-large-finetuned-squadv1"
_QA_EXPECTED_LOSS = 0.59
_QA_EXPECTED_OUTPUT = "' nice puppet'"

BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-large",
    # see all BART models at https://huggingface.co/models?filter=bart
]
from .generation_utils import BeamSearchScorer, BeamSearchAccScorer


class OFABart_AnsAcc(nn.Module):
    def __init__(self, OFA, BART, BARTTable, tokenizer, length_penalty=1.0, constrained=1.0):
        super().__init__()

        self.OFA_encoder = OFA.encoder
        self.BART_encoder = BART.model.encoder
        self.BART_table_encoder = BARTTable.model.encoder

        self.tokenizer = tokenizer
        self.decoder = BART.model.decoder
        self.lm_head = BART.lm_head
        self.final_logits_bias = BART.final_logits_bias
        self.retreive_num_head = nn.Linear(BART.config.d_model, 1, bias=False)

        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        self.sc_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        self.cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.b_sc = torch.tensor(self.tokenizer.convert_tokens_to_ids(["[b_source]"])).cuda()
        self.e_sc = torch.tensor(self.tokenizer.convert_tokens_to_ids(["[e_source]"])).cuda()
        self.max_gen_len = 50
        self.max_source_len = 10
        self.beam_size = 5
        self.length_penalty = length_penalty
        self.constrained = constrained

        self.dim = BART.config.d_model

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_list(self, input_ids, input_mask, table_text, table_mask, image_feat, image_mask, image_ques_id,
                       batch_text_len, batch_image_len, input_token_type, table_token_type, image_token_type):
        batch_size = input_ids.size(0)
        input_ids = input_ids.type(torch.int64).cuda()
        input_mask = input_mask.type(torch.int64).cuda()
        input_token_type = input_token_type.type(torch.int64).cuda()
        table_text = table_text.cuda()
        table_mask = table_mask.cuda()
        table_token_type = table_token_type.cuda()
        image_feat = image_feat.cuda()
        image_mask = image_mask.type(torch.int64).cuda()
        image_token_type = image_token_type.type(torch.int64).cuda()
        image_ques_id = image_ques_id.type(torch.int64).cuda()

        input_ids_new = None
        input_mask_new = None
        input_token_type_new = None
        image_feat_new = None
        image_mask_new = None
        image_ques_id_new = None
        image_token_type_new = None
        max_token_len = max([lenth[1].item() for lenth in batch_text_len])
        max_title_len = max([lenth[1].item() for lenth in batch_image_len])

        for i in range(batch_size):
            tmp_input_ids = input_ids[i, :batch_text_len[i][0].item()]
            tmp_input_mask = input_mask[i, :batch_text_len[i][0].item()]
            tmp_input_token_type = input_token_type[i, :batch_text_len[i][0].item()]
            tmp_input_ids = tmp_input_ids.reshape(-1, batch_text_len[i][1].item())
            tmp_input_mask = tmp_input_mask.reshape(-1, batch_text_len[i][1].item())
            tmp_input_token_type = tmp_input_token_type.reshape(-1, batch_text_len[i][1].item())
            if tmp_input_ids.size(1) < max_token_len:
                pad_matrix = torch.zeros((tmp_input_ids.size(0), max_token_len - tmp_input_ids.size(1)),
                                         dtype=torch.long).cuda()
                tmp_input_ids = torch.cat((tmp_input_ids, pad_matrix), dim=-1)
                tmp_input_mask = torch.cat((tmp_input_mask, pad_matrix), dim=-1)
                tmp_input_token_type = torch.cat((tmp_input_token_type, pad_matrix), dim=-1)
            if input_ids_new is None:
                input_ids_new = tmp_input_ids
                input_mask_new = tmp_input_mask
                input_token_type_new = tmp_input_token_type
            else:
                input_ids_new = torch.cat((input_ids_new, tmp_input_ids), dim=0)
                input_mask_new = torch.cat((input_mask_new, tmp_input_mask), dim=0)
                input_token_type_new = torch.cat((input_token_type_new, tmp_input_token_type), dim=0)

            tmp_image_feat = image_feat[i, :batch_image_len[i][0].item() * 3 * 256 * 256]
            tmp_image_mask = image_mask[i, :batch_image_len[i][0].item() * batch_image_len[i][1].item()]
            tmp_image_ques_id = image_ques_id[i, :batch_image_len[i][0].item() * batch_image_len[i][1].item()]
            tmp_image_token_type = image_token_type[i, :batch_image_len[i][0].item() * batch_image_len[i][1].item()]
            tmp_image_feat = tmp_image_feat.reshape(batch_image_len[i][0].item(), 3, 256, 256)
            tmp_image_mask = tmp_image_mask.reshape(batch_image_len[i][0].item(), -1)
            tmp_image_token_type = tmp_image_token_type.reshape(batch_image_len[i][0].item(), -1)
            tmp_image_ques_id = tmp_image_ques_id.reshape(batch_image_len[i][0].item(), -1)
            if tmp_image_ques_id.size(1) < max_title_len:
                pad_matrix = torch.zeros((tmp_image_ques_id.size(0), max_title_len - tmp_image_ques_id.size(1)),
                                         dtype=torch.long).cuda()
                tmp_image_ques_id = torch.cat((tmp_image_ques_id, pad_matrix), dim=-1)
                tmp_image_mask = torch.cat((tmp_image_mask, pad_matrix), dim=-1)
                tmp_image_token_type = torch.cat((tmp_image_token_type, pad_matrix), dim=-1)

            if image_ques_id_new is None:
                image_ques_id_new = tmp_image_ques_id
                image_mask_new = tmp_image_mask
                image_token_type_new = tmp_image_token_type
                image_feat_new = tmp_image_feat
            else:
                image_ques_id_new = torch.cat((image_ques_id_new, tmp_image_ques_id), dim=0)
                image_mask_new = torch.cat((image_mask_new, tmp_image_mask), dim=0)
                image_token_type_new = torch.cat((image_token_type_new, tmp_image_token_type), dim=0)
                image_feat_new = torch.cat((image_feat_new, tmp_image_feat), dim=0)

        return input_ids_new, input_mask_new, table_text, table_mask, image_feat_new, image_mask_new, image_ques_id_new, \
               input_token_type_new, table_token_type, image_token_type_new

    def find_golden_sc(self, encoder_hidden_states, golden_source_index):
        if isinstance(golden_source_index, list):
            tmp_hidden = encoder_hidden_states[golden_source_index]
            return tmp_hidden
        else:
            gs_hidden = []
            for i in range(golden_source_index.size(0)):
                index_list = golden_source_index[i].cpu().tolist()
                for j in range(len(index_list)):
                    if index_list[j] == -1:
                        index_list = index_list[:j]
                        break
                tmp_hidden = encoder_hidden_states[i][index_list]
                gs_hidden.append(tmp_hidden)
            return gs_hidden

    def find_ans_sc_index(self, loss_mask):
        gs_index = []
        for i in range(loss_mask.size(0)):
            nonzero = torch.nonzero(loss_mask[i])
            gs_index.append([nonzero[0].item(), nonzero[-1].item()])
        return gs_index

    def forward(self, input_ids, input_mask, table_text, table_mask, image_feat, image_mask, image_ques_id,
                batch_text_len, batch_image_len, ans_ids, attn_mask, ans_loss_mask, sc_loss_mask,
                golden_index, input_token_type, table_token_type, image_token_type, table_connect_num,
                flatten_connect_spans, flatten_connect_index, table_cell_num, flatten_cell_span, gather_index,
                span_label, encoder_outputs=None):
        input_ids, input_mask, table_text, table_mask, image_feat, image_mask, image_ques_id, \
        input_token_type, table_token_type, image_token_type = self.get_input_list(input_ids, input_mask, table_text,
                                                                                   table_mask, image_feat, image_mask,
                                                                                   image_ques_id, batch_text_len,
                                                                                   batch_image_len, input_token_type,
                                                                                   table_token_type, image_token_type)

        output_attentions = False
        output_hidden_states = True
        use_cache = False
        text_outputs = self.BART_encoder(input_ids=input_ids, attention_mask=input_mask,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        text_seq = text_outputs[0]

        image_mask_invert = torch.where(image_mask == 0,
                                        torch.full(image_mask.size(), fill_value=True).cuda(),
                                        torch.full(image_mask.size(), fill_value=False).cuda())
        image_outputs = self.OFA_encoder(input_ids=image_ques_id, attention_mask=image_mask_invert,
                                         token_type_ids=image_token_type, patch_images=image_feat,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        image_seq = image_outputs[0]

        seq_hidden, connect_spans, connect_index, span_label, cell_span = self.get_table_connet(
            table_connect_num, flatten_connect_spans,
            flatten_connect_index, text_seq[:, 0],
            image_seq[:, 0], batch_text_len,
            batch_image_len, table_cell_num, span_label, flatten_cell_span)
        table_outputs, connect_loss, similarity_loss = self.BART_table_encoder(input_ids=table_text,
                                                                               attention_mask=table_mask,
                                                                               output_attentions=output_attentions,
                                                                               output_hidden_states=output_hidden_states,
                                                                               connect_span=connect_spans,
                                                                               connect_index=connect_index,
                                                                               cell_span=cell_span,
                                                                               gather_index=gather_index,
                                                                               span_label=span_label,
                                                                               is_train=True,
                                                                               seq_hidden=seq_hidden)
        table_seq = table_outputs[0]
        # image first
        # image_mask = torch.cat((torch.ones((image_mask.size(0), 16 * 16), dtype=torch.long).cuda(), image_mask), 1)
        padding_mask = image_outputs.padding_mask
        image_mask = torch.where(padding_mask == False,
                                 torch.full(padding_mask.size(), fill_value=1).cuda(),
                                 torch.full(padding_mask.size(), fill_value=0).cuda())
        encoder_hidden_states, encoder_attention_mask, encoder_pooled_states, pointer_masks, copy_masks, batch_ids, image_total_len = self.flatten_to_batch(
            text_seq, table_seq, image_seq, input_mask, table_mask, image_mask,
            batch_text_len, batch_image_len, golden_index, input_ids, table_text, image_ques_id)
        del text_seq
        del text_outputs
        del image_seq
        del image_outputs
        sc_loss_mask = sc_loss_mask.cuda()
        gs_hidden = self.find_golden_sc(encoder_pooled_states, golden_index)
        gs_index = self.find_ans_sc_index(sc_loss_mask)
        encoder_hidden_states = pad_sequence(encoder_hidden_states, batch_first=True, padding_value=0)
        encoder_attention_mask = pad_sequence(encoder_attention_mask, batch_first=True, padding_value=0)
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1).repeat(1, ans_ids.size(1), 1)
        pointer_masks = pad_sequence(pointer_masks, batch_first=True, padding_value=0)
        copy_masks = pad_sequence(copy_masks, batch_first=True, padding_value=0)
        for i in range(len(gs_index)):
            encoder_attention_mask[i, gs_index[i][0] - 1:gs_index[i][1]] = pointer_masks[i].unsqueeze(0)
            encoder_attention_mask[i, gs_index[i][1]:] = copy_masks[i].unsqueeze(0)

        ans_ids = ans_ids.cuda()
        attn_mask = attn_mask.cuda()
        decoder_outputs = self.decoder(
            input_ids=ans_ids,
            attention_mask=attn_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            gs_hidden=gs_hidden, gs_index=gs_index)
        lm_logits = self.lm_head(decoder_outputs[0]) + self.final_logits_bias.cuda()
        re_logits = self.retreive_num_head(decoder_outputs.last_hidden_state)
        cross_attention_weights = decoder_outputs.cross_attentions
        del decoder_outputs
        cross_attention_weights = torch.stack(cross_attention_weights).cuda().sum(dim=0)
        cross_attention_weights = cross_attention_weights.sum(dim=1)
        ans_loss_mask = ans_loss_mask.cuda()
        ans_labels = ans_ids * ans_loss_mask
        ans_shift_labels = ans_labels[..., 1:].contiguous()
        shift_logits = lm_logits[..., :-1, :].contiguous()

        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), ans_shift_labels.view(-1))
        pointer_loss = []
        re_loss = []
        for i in range(ans_ids.size(0)):
            pointer_weight = cross_attention_weights[i, gs_index[i][0] - 1:gs_index[i][1] - 1]
            # Note -1
            cls_index = torch.nonzero(pointer_weight[-1]).squeeze(-1)
            # 同一样例label相同
            pointer_weight = pointer_weight[:, cls_index]
            label = golden_index[i].cuda()
            label = label[:pointer_weight.size(0)]
            # pointer_weight = pointer_weight.float()
            tmp_loss = self.sc_criterion(pointer_weight, label)
            pointer_loss.append(tmp_loss)
            tmp_re_logits = re_logits[i, gs_index[i][0] - 1:gs_index[i][1]]
            cls_len = tmp_re_logits.size(0)
            label = torch.cat(
                (torch.ones(cls_len - 1, dtype=torch.float32), torch.zeros(1, dtype=torch.float32))).cuda().to(
                dtype=tmp_re_logits.dtype)
            # re_logits = re_logits.float()
            tmp_loss = self.cls_criterion(tmp_re_logits.view(-1), label)
            re_loss.append(tmp_loss)
        pointer_loss = torch.stack(pointer_loss).mean()
        re_loss = torch.stack(re_loss).mean()
        return gen_loss, pointer_loss, re_loss, connect_loss, similarity_loss

    def flatten_to_batch(self, text_seq, table_seq, image_seq, input_mask, table_mask, image_mask, batch_text_len,
                         batch_image_len, golden_source_index,
                         input_ids, table_text, image_ques_id):
        encoder_pooled_hidden_states = []
        encoder_hidden_states = []
        encoder_masks = []
        tmp_text_index = 0
        tmp_image_index = 0
        pointer_masks = []
        copy_masks = []
        batch_ids = []
        image_total_len = []
        for i in range(len(batch_text_len)):
            text_num = batch_text_len[i, 0] // batch_text_len[i, 1].item()
            tmp_hidden_states = text_seq[tmp_text_index:tmp_text_index + text_num]
            tmp_pooled_states = tmp_hidden_states[:, 0]
            tmp_masks_text = input_mask[tmp_text_index:tmp_text_index + text_num]
            text_ids = input_ids[tmp_text_index:tmp_text_index + text_num]
            pad_text_len = tmp_masks_text.size(1)
            tmp_pointer_mask_text = torch.tensor([[1] + [0] * (pad_text_len - 1)]).repeat(text_num, 1).cuda()
            tmp_text_index += text_num
            image_num = batch_image_len[i, 0].item()
            tmp_image_states = image_seq[tmp_image_index:tmp_image_index + image_num]
            tmp_image_pooled_hiddens = tmp_image_states[:, 0]
            tmp_image_masks = image_mask[tmp_image_index:tmp_image_index + image_num]
            pad_image_len = tmp_image_masks.size(1)
            tmp_image_ques_id = image_ques_id[tmp_image_index:tmp_image_index + image_num]
            tmp_pointer_mask_image = torch.tensor([[1] + [0] * (pad_image_len - 1)]).repeat(image_num, 1).cuda()
            tmp_image_index += image_num
            image_total_len.append(tmp_pointer_mask_image.reshape(-1).size(0))

            table_pooled = table_seq[i][0].unsqueeze(0)
            tmp_hidden_pooled_states = torch.cat(
                (tmp_pooled_states, table_pooled, tmp_image_pooled_hiddens), 0)
            tmp_hidden_states = torch.cat(
                (tmp_hidden_states.reshape(-1, tmp_hidden_states.size(-1)), table_seq[i],
                 tmp_image_states.reshape(-1, tmp_image_states.size(-1))), 0)
            tmp_masks = torch.cat((tmp_masks_text.reshape(-1), table_mask[i], tmp_image_masks.reshape(-1)), 0)

            table_len = table_mask[i].size(0)
            tmp_pointer_mask_table = torch.tensor([[1] + [0] * (table_len - 1)]).cuda()
            tmp_pointer_mask = torch.cat((tmp_pointer_mask_text.reshape(-1), tmp_pointer_mask_table.reshape(-1),
                                          tmp_pointer_mask_image.reshape(-1)), 0)
            tmp_copy_mask = torch.zeros_like(tmp_masks).cuda()
            if isinstance(golden_source_index, list):
                index_list = golden_source_index[i]
            else:
                index_list = golden_source_index[i].cpu().tolist()
                for j in range(len(index_list)):
                    if index_list[j] == -1:
                        index_list = index_list[:j]
                        break
            for index in index_list:
                if index < text_num:
                    start = index * pad_text_len
                    tmp_copy_mask[start:start + pad_text_len] = tmp_masks[start:start + pad_text_len]
                else:
                    start = text_num * pad_text_len + table_len + (index - text_num - 1) * pad_image_len
                    tmp_copy_mask[start:start + pad_image_len] = tmp_masks[start:start + pad_image_len]
            # 保证能够看到information
            start = text_num * pad_text_len
            tmp_copy_mask[start:start + table_len] = tmp_masks[start:start + table_len]
            image_ids = []
            image_pad_ids = torch.zeros(tmp_image_states.size(1) - tmp_image_ques_id.size(1), dtype=torch.int64).cuda()
            for idx in range(tmp_image_ques_id.size(0)):
                tmp_ids = torch.cat((tmp_image_ques_id[idx], image_pad_ids), dim=0)
                image_ids.append(tmp_ids)
            image_ids = torch.stack(image_ids)
            tmp_ids = torch.cat((text_ids.reshape(-1), table_text[i], image_ids.reshape(-1)), 0)
            encoder_pooled_hidden_states.append(tmp_hidden_pooled_states)
            encoder_hidden_states.append(tmp_hidden_states)
            encoder_masks.append(tmp_masks)
            pointer_masks.append(tmp_pointer_mask)
            copy_masks.append(tmp_copy_mask)
            batch_ids.append(tmp_ids)
        return encoder_hidden_states, encoder_masks, encoder_pooled_hidden_states, pointer_masks, copy_masks, batch_ids, image_total_len

    def get_pointer_hidden(self, encoder_hidden, cross_attention, text_num, text_len, tabel_len, image_len):
        cross_attention = torch.stack(cross_attention).cuda().sum(dim=0)
        cross_attention = cross_attention.sum(dim=1)
        cross_attention = cross_attention[:, -1:]
        pointer_weight = torch.where(cross_attention == 0., torch.full_like(cross_attention, fill_value=-10000),
                                     cross_attention)
        pointer_weight = torch.softmax(pointer_weight, dim=-1)
        index = torch.argmax(pointer_weight, -1).item()
        hidden = encoder_hidden[:, index, :].unsqueeze(1)

        total_text_len = text_len * text_num
        if index < total_text_len:
            index = index // text_len
        elif index == total_text_len:
            index = text_num
        else:
            index = text_num + 1 + (index - total_text_len - tabel_len) // image_len
        return hidden, index

    def get_table_connet(self, table_connect_num, flatten_connect_spans, flatten_connect_index, text_pool, image_pool,
                         batch_text_len, batch_image_len, table_cell_num, flatten_span_label, flatten_cell_span):
        seq_hidden, connect_span, connect_index = [], [], []
        span_label, cell_span = [], []

        tmp_text_index = 0
        tmp_image_index = 0
        for i in range(len(table_connect_num)):
            connect_num = table_connect_num[i, 0].item()
            entity_num = table_connect_num[i, 1].item()
            tmp_flatten_connect_spans = flatten_connect_spans[i, :connect_num * 2]
            tmp_flatten_connect_index = flatten_connect_index[i, :connect_num * entity_num]
            tmp_connect_spans = tmp_flatten_connect_spans.reshape(connect_num, 2)
            tmp_connect_index = tmp_flatten_connect_index.reshape(connect_num, entity_num)
            connect_span.append(tmp_connect_spans)
            connect_index.append(tmp_connect_index)

            text_num = batch_text_len[i, 0] // batch_text_len[i, 1].item()
            tmp_pool_text_states = text_pool[tmp_text_index:tmp_text_index + text_num]
            tmp_text_index += text_num
            image_num = batch_image_len[i, 0].item()
            tmp_pool_image_states = image_pool[tmp_image_index:tmp_image_index + image_num]
            tmp_image_index += image_num
            tmp_hidden_pooled_states = torch.cat((tmp_pool_text_states, tmp_pool_image_states), 0)
            seq_hidden.append(tmp_hidden_pooled_states)

            cell_num = table_cell_num[i]
            # source_num = table_cell_num[i][1]
            tmp_span_label = flatten_span_label[i, :text_num + image_num]
            span_label.append(tmp_span_label)
            tmp_cell_span = flatten_cell_span[i, :cell_num * 2].reshape(cell_num, 2)
            cell_span.append(tmp_cell_span)
        return seq_hidden, connect_span, connect_index, span_label, cell_span

    def evaluate(self, input_ids, input_mask, table_text, table_mask, image_feat, image_mask, image_ques_id,
                 batch_text_len, batch_image_len, ans_ids, attn_mask, ans_loss_mask, sc_loss_mask,
                 golden_index, input_token_type, table_token_type, image_token_type, table_connect_num,
                 flatten_connect_spans, flatten_connect_index, table_cell_num, flatten_cell_span, gather_index,
                 span_label, encoder_outputs=None):
        input_ids, input_mask, table_text, table_mask, image_feat, image_mask, image_ques_id, \
        input_token_type, table_token_type, image_token_type = self.get_input_list(input_ids, input_mask, table_text,
                                                                                   table_mask, image_feat, image_mask,
                                                                                   image_ques_id, batch_text_len,
                                                                                   batch_image_len, input_token_type,
                                                                                   table_token_type, image_token_type)
        output_attentions = False
        output_hidden_states = True

        text_outputs = self.BART_encoder(input_ids=input_ids, attention_mask=input_mask,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        text_seq = text_outputs[0]

        image_mask_invert = torch.where(image_mask == 0,
                                        torch.full(image_mask.size(), fill_value=True).cuda(),
                                        torch.full(image_mask.size(), fill_value=False).cuda())
        image_outputs = self.OFA_encoder(input_ids=image_ques_id, attention_mask=image_mask_invert,
                                         token_type_ids=image_token_type, patch_images=image_feat,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        image_seq = image_outputs[0]

        seq_hidden, connect_span, connect_index, span_label, cell_span = self.get_table_connet(
            table_connect_num, flatten_connect_spans,
            flatten_connect_index, text_seq[:, 0],
            image_seq[:, 0], batch_text_len,
            batch_image_len, table_cell_num, span_label, flatten_cell_span)

        table_outputs, has_connect, similarity = self.BART_table_encoder(input_ids=table_text,
                                                                         attention_mask=table_mask,
                                                                         output_attentions=output_attentions,
                                                                         output_hidden_states=output_hidden_states,
                                                                         connect_span=connect_span,
                                                                         connect_index=connect_index,
                                                                         seq_hidden=seq_hidden, cell_span=cell_span,
                                                                         gather_index=gather_index,
                                                                         span_label=span_label,
                                                                         is_train=False)
        table_seq = table_outputs[0]

        image_mask = torch.cat((torch.ones((image_mask.size(0), 16 * 16), dtype=torch.long).cuda(), image_mask), 1)
        encoder_hidden_states, encoder_attention_mask, encoder_pooled_states, pointer_masks, copy_masks, batch_ids, image_total_len = self.flatten_to_batch(
            text_seq, table_seq, image_seq, input_mask, table_mask, image_mask,
            batch_text_len, batch_image_len, golden_index, input_ids, table_text, image_ques_id)
        batch_size = len(encoder_hidden_states)
        outputs = torch.full((batch_size, self.max_gen_len), fill_value=self.tokenizer.pad_token_id,
                             dtype=torch.int64).cuda()
        gen_index = []
        total_correct = 0
        total_hidden_num = 0
        ans_ids = ans_ids.cuda()
        for i in range(batch_size):
            encoder_hidden = encoder_hidden_states[i].unsqueeze(0)
            encoder_mask = encoder_attention_mask[i].unsqueeze(0)
            pointer_mask = pointer_masks[i].unsqueeze(0)
            decoder_input = ans_ids[i]
            b_souce_index = torch.nonzero(decoder_input == self.b_sc.unsqueeze(0)).to(torch.int64)[0, -1]
            dec_input_id = decoder_input[:b_souce_index].unsqueeze(0)
            gpt_outputs = self.decoder(input_ids=dec_input_id,
                                       encoder_hidden_states=encoder_hidden,
                                       encoder_attention_mask=encoder_mask,
                                       use_cache=True,
                                       past_key_values=None)
            past_key_values = gpt_outputs.past_key_values
            dec_input_id = self.b_sc.unsqueeze(0)
            gpt_outputs = self.decoder(input_ids=dec_input_id,
                                       encoder_hidden_states=encoder_hidden,
                                       encoder_attention_mask=pointer_mask,
                                       use_cache=True,
                                       past_key_values=past_key_values, output_attentions=True)
            past_key_values = gpt_outputs.past_key_values
            gpt_out = gpt_outputs.last_hidden_state
            cls_logits = F.sigmoid(self.retreive_num_head(gpt_out))
            text_num = (batch_text_len[i, 0] // batch_text_len[i, 1]).item()
            text_len = input_ids.size(1)
            table_len = table_text.size(1)
            image_len = image_seq.size(1)
            pointer_hidden, source_index = self.get_pointer_hidden(encoder_hidden, gpt_outputs.cross_attentions,
                                                                   text_num, text_len, table_len, image_len)
            tmp_source_hidden = []
            tmp_source_index = []
            for index in range(self.max_source_len):
                if cls_logits.item() < 0.5:
                    break
                tmp_source_hidden.append(pointer_hidden.squeeze(0))
                if source_index not in tmp_source_index:
                    tmp_source_index.append(source_index)
                gpt_outputs = self.decoder(inputs_embeds=pointer_hidden,
                                           encoder_hidden_states=encoder_hidden,
                                           encoder_attention_mask=pointer_mask,
                                           use_cache=True,
                                           past_key_values=past_key_values, output_attentions=True)
                past_key_values = gpt_outputs.past_key_values
                gpt_out = gpt_outputs.last_hidden_state
                cls_logits = F.sigmoid(self.retreive_num_head(gpt_out))
                pointer_hidden, source_index = self.get_pointer_hidden(encoder_hidden, gpt_outputs.cross_attentions,
                                                                       text_num, text_len, table_len, image_len)

            sc_correct_num = len(list(set(tmp_source_index).intersection(set(golden_index[i].tolist()))))
            total_correct += sc_correct_num
            total_hidden_num += len(tmp_source_index)
            # source_hidden.append(tmp_source_hidden)
            gen_index.append(tmp_source_index)
            cur_unfinished = outputs.new(1).fill_(1)
            cur_len = 0
            tokens_to_add = self.e_sc
            copy_mask = torch.zeros_like(encoder_mask)
            for index in tmp_source_index:
                if index < text_num:
                    start = index * text_len
                    copy_mask[:, start:start + text_len] = encoder_mask[:, start:start + text_len]
                else:
                    start = text_num * text_len + table_len + (index - text_num - 1) * image_len
                    copy_mask[:, start:start + image_len] = encoder_mask[:, start:start + image_len]
            # 保证能够看到entity information
            start = text_num * text_len
            copy_mask[:, start:start + table_len] = encoder_mask[:, start:start + table_len]
            for index in range(self.max_gen_len - 1):
                dec_input_id = tokens_to_add.unsqueeze(0)
                gpt_outputs = self.decoder(input_ids=dec_input_id,
                                           encoder_hidden_states=encoder_hidden,
                                           encoder_attention_mask=copy_mask,
                                           use_cache=True,
                                           past_key_values=past_key_values)
                lm_logits = self.lm_head(gpt_outputs[0]) + self.final_logits_bias.cuda()

                past_key_values = gpt_outputs[1]
                gen_label = torch.argmax(lm_logits, dim=-1).squeeze(-1)
                tokens_to_add = gen_label * cur_unfinished + self.tokenizer.pad_token_id * (1 - cur_unfinished)
                outputs[i, index] = tokens_to_add
                cur_len += 1
                cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.tokenizer.eos_token_id).long())
                if cur_unfinished.max() == 0:
                    break
            if cur_len == self.max_gen_len:
                outputs[i, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.tokenizer.eos_token_id)
        return outputs, total_correct, total_hidden_num, gen_index, has_connect, similarity

    def evaluate_golden_source(self, input_ids, input_mask, table_text, table_mask, image_feat, image_mask,
                               image_ques_id, batch_text_len, batch_image_len, ans_ids, attn_mask, ans_loss_mask,
                               sc_loss_mask,
                               golden_index, input_token_type, table_token_type, image_token_type, table_connect_num,
                               flatten_connect_spans, flatten_connect_index, table_cell_num, flatten_cell_span,
                               gather_index, span_label, encoder_outputs=None):
        input_ids, input_mask, table_text, table_mask, image_feat, image_mask, image_ques_id, \
        input_token_type, table_token_type, image_token_type = self.get_input_list(input_ids, input_mask, table_text,
                                                                                   table_mask, image_feat, image_mask,
                                                                                   image_ques_id, batch_text_len,
                                                                                   batch_image_len, input_token_type,
                                                                                   table_token_type, image_token_type)
        output_attentions = False
        output_hidden_states = True

        text_outputs = self.BART_encoder(input_ids=input_ids, attention_mask=input_mask,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        text_seq = text_outputs[0]

        image_mask_invert = torch.where(image_mask == 0,
                                        torch.full(image_mask.size(), fill_value=True).cuda(),
                                        torch.full(image_mask.size(), fill_value=False).cuda())
        image_outputs = self.OFA_encoder(input_ids=image_ques_id, attention_mask=image_mask_invert,
                                         token_type_ids=image_token_type, patch_images=image_feat,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        image_seq = image_outputs[0]

        seq_hidden, connect_span, connect_index, span_label, cell_span = self.get_table_connet(
            table_connect_num, flatten_connect_spans,
            flatten_connect_index, text_seq[:, 0],
            image_seq[:, 0], batch_text_len,
            batch_image_len, table_cell_num, span_label, flatten_cell_span)

        table_outputs, has_connect, similarity = self.BART_table_encoder(input_ids=table_text,
                                                                         attention_mask=table_mask,
                                                                         output_attentions=output_attentions,
                                                                         output_hidden_states=output_hidden_states,
                                                                         connect_span=connect_span,
                                                                         connect_index=connect_index,
                                                                         seq_hidden=seq_hidden, cell_span=cell_span,
                                                                         gather_index=gather_index,
                                                                         span_label=span_label,
                                                                         is_train=False)
        table_seq = table_outputs[0]

        image_mask = torch.cat((torch.ones((image_mask.size(0), 16 * 16), dtype=torch.long).cuda(), image_mask), 1)
        encoder_hidden_states, encoder_attention_mask, encoder_pooled_states, pointer_masks, copy_masks, batch_ids, image_total_len = self.flatten_to_batch(
            text_seq, table_seq, image_seq, input_mask, table_mask, image_mask,
            batch_text_len, batch_image_len, golden_index, input_ids, table_text, image_ques_id)
        batch_size = len(encoder_hidden_states)
        gs_hidden = self.find_golden_sc(encoder_pooled_states, golden_index)
        gs_index = self.find_ans_sc_index(sc_loss_mask)
        outputs = torch.full((batch_size, self.max_gen_len), fill_value=self.tokenizer.pad_token_id,
                             dtype=torch.int64).cuda()
        ans_ids = ans_ids.cuda()
        for i in range(batch_size):
            encoder_hidden = encoder_hidden_states[i].unsqueeze(0)
            encoder_mask = encoder_attention_mask[i].unsqueeze(0).unsqueeze(0)
            pointer_mask = pointer_masks[i].unsqueeze(0)
            copy_mask = copy_masks[i].unsqueeze(0)
            decoder_input = ans_ids[i]
            b_souce_index = torch.nonzero(decoder_input == self.e_sc.unsqueeze(0)).to(torch.int64)[0, -1]
            dec_input_id = decoder_input[:b_souce_index].unsqueeze(0)
            encoder_mask = encoder_mask.repeat(1, dec_input_id.size(1), 1)
            encoder_mask[:, gs_index[i][0] - 1:gs_index[i][1]] = pointer_mask

            gpt_outputs = self.decoder(
                input_ids=dec_input_id,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=encoder_mask,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=output_hidden_states,
                gs_hidden=gs_hidden[i].unsqueeze(0), gs_index=[gs_index[i]])
            past_key_values = gpt_outputs.past_key_values
            tokens_to_add = self.e_sc
            cur_unfinished = outputs.new(1).fill_(1)
            cur_len = 0
            for index in range(self.max_gen_len - 1):
                dec_input_id = tokens_to_add.unsqueeze(0)
                gpt_outputs = self.decoder(input_ids=dec_input_id,
                                           encoder_hidden_states=encoder_hidden,
                                           encoder_attention_mask=copy_mask,
                                           use_cache=True,
                                           past_key_values=past_key_values)
                lm_logits = self.lm_head(gpt_outputs[0]) + self.final_logits_bias.cuda()

                past_key_values = gpt_outputs[1]
                gen_label = torch.argmax(lm_logits, dim=-1).squeeze(-1)
                tokens_to_add = gen_label * cur_unfinished + self.tokenizer.pad_token_id * (1 - cur_unfinished)
                outputs[i, index] = tokens_to_add
                cur_len += 1
                cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.tokenizer.eos_token_id).long())
                if cur_unfinished.max() == 0:
                    break
            if cur_len == self.max_gen_len:
                outputs[i, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.tokenizer.eos_token_id)
        return outputs

    def evaluate_golden_source2entity(self, input_ids, input_mask, table_text, table_mask, image_feat, image_mask,
                                      image_ques_id,
                                      batch_text_len, batch_image_len, ans_ids, attn_mask, ans_loss_mask, sc_loss_mask,
                                      golden_index, input_token_type, table_token_type, image_token_type,
                                      table_connect_num,
                                      flatten_connect_spans, flatten_connect_index, table_cell_num, flatten_cell_span,
                                      gather_index,
                                      span_label, encoder_outputs=None):
        input_ids, input_mask, table_text, table_mask, image_feat, image_mask, image_ques_id, \
        input_token_type, table_token_type, image_token_type = self.get_input_list(input_ids, input_mask, table_text,
                                                                                   table_mask, image_feat, image_mask,
                                                                                   image_ques_id, batch_text_len,
                                                                                   batch_image_len, input_token_type,
                                                                                   table_token_type, image_token_type)
        output_attentions = False
        output_hidden_states = True

        text_outputs = self.BART_encoder(input_ids=input_ids, attention_mask=input_mask,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        text_seq = text_outputs[0]

        image_mask_invert = torch.where(image_mask == 0,
                                        torch.full(image_mask.size(), fill_value=True).cuda(),
                                        torch.full(image_mask.size(), fill_value=False).cuda())
        image_outputs = self.OFA_encoder(input_ids=image_ques_id, attention_mask=image_mask_invert,
                                         token_type_ids=image_token_type, patch_images=image_feat,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        image_seq = image_outputs[0]

        seq_hidden, connect_span, connect_index, span_label, cell_span = self.get_table_connet(
            table_connect_num, flatten_connect_spans,
            flatten_connect_index, text_seq[:, 0],
            image_seq[:, 0], batch_text_len,
            batch_image_len, table_cell_num, span_label, flatten_cell_span)

        table_outputs, _, _ = self.BART_table_encoder(input_ids=table_text,
                                                      attention_mask=table_mask,
                                                      output_attentions=output_attentions,
                                                      output_hidden_states=output_hidden_states,
                                                      connect_span=connect_span,
                                                      connect_index=connect_index,
                                                      seq_hidden=seq_hidden, cell_span=cell_span,
                                                      gather_index=gather_index,
                                                      span_label=span_label,
                                                      is_eval_golden=True)
        table_seq = table_outputs[0]

        image_mask = torch.cat((torch.ones((image_mask.size(0), 16 * 16), dtype=torch.long).cuda(), image_mask), 1)
        encoder_hidden_states, encoder_attention_mask, encoder_pooled_states, pointer_masks, copy_masks, batch_ids, image_total_len = self.flatten_to_batch(
            text_seq, table_seq, image_seq, input_mask, table_mask, image_mask,
            batch_text_len, batch_image_len, golden_index, input_ids, table_text, image_ques_id)
        batch_size = len(encoder_hidden_states)
        outputs = torch.full((batch_size, self.max_gen_len), fill_value=self.tokenizer.pad_token_id,
                             dtype=torch.int64).cuda()
        gen_index = []
        total_correct = 0
        total_hidden_num = 0
        ans_ids = ans_ids.cuda()
        for i in range(batch_size):
            encoder_hidden = encoder_hidden_states[i].unsqueeze(0)
            encoder_mask = encoder_attention_mask[i].unsqueeze(0)
            pointer_mask = pointer_masks[i].unsqueeze(0)
            decoder_input = ans_ids[i]
            b_souce_index = torch.nonzero(decoder_input == self.b_sc.unsqueeze(0)).to(torch.int64)[0, -1]
            dec_input_id = decoder_input[:b_souce_index].unsqueeze(0)
            gpt_outputs = self.decoder(input_ids=dec_input_id,
                                       encoder_hidden_states=encoder_hidden,
                                       encoder_attention_mask=encoder_mask,
                                       use_cache=True,
                                       past_key_values=None)
            past_key_values = gpt_outputs.past_key_values
            dec_input_id = self.b_sc.unsqueeze(0)
            gpt_outputs = self.decoder(input_ids=dec_input_id,
                                       encoder_hidden_states=encoder_hidden,
                                       encoder_attention_mask=pointer_mask,
                                       use_cache=True,
                                       past_key_values=past_key_values, output_attentions=True)
            past_key_values = gpt_outputs.past_key_values
            gpt_out = gpt_outputs.last_hidden_state
            cls_logits = F.sigmoid(self.retreive_num_head(gpt_out))
            text_num = (batch_text_len[i, 0] // batch_text_len[i, 1]).item()
            text_len = input_ids.size(1)
            table_len = table_text.size(1)
            image_len = image_seq.size(1)
            pointer_hidden, source_index = self.get_pointer_hidden(encoder_hidden, gpt_outputs.cross_attentions,
                                                                   text_num, text_len, table_len, image_len)
            tmp_source_hidden = []
            tmp_source_index = []
            for index in range(self.max_source_len):
                if cls_logits.item() < 0.5:
                    break
                tmp_source_hidden.append(pointer_hidden.squeeze(0))
                if source_index not in tmp_source_index:
                    tmp_source_index.append(source_index)
                gpt_outputs = self.decoder(inputs_embeds=pointer_hidden,
                                           encoder_hidden_states=encoder_hidden,
                                           encoder_attention_mask=pointer_mask,
                                           use_cache=True,
                                           past_key_values=past_key_values, output_attentions=True)
                past_key_values = gpt_outputs.past_key_values
                gpt_out = gpt_outputs.last_hidden_state
                cls_logits = F.sigmoid(self.retreive_num_head(gpt_out))
                pointer_hidden, source_index = self.get_pointer_hidden(encoder_hidden, gpt_outputs.cross_attentions,
                                                                       text_num, text_len, table_len, image_len)

            sc_correct_num = len(list(set(tmp_source_index).intersection(set(golden_index[i].tolist()))))
            total_correct += sc_correct_num
            total_hidden_num += len(tmp_source_index)
            # source_hidden.append(tmp_source_hidden)
            gen_index.append(tmp_source_index)
            cur_unfinished = outputs.new(1).fill_(1)
            cur_len = 0
            tokens_to_add = self.e_sc
            copy_mask = torch.zeros_like(encoder_mask)
            for index in tmp_source_index:
                if index < text_num:
                    start = index * text_len
                    copy_mask[:, start:start + text_len] = encoder_mask[:, start:start + text_len]
                else:
                    start = text_num * text_len + table_len + (index - text_num - 1) * image_len
                    copy_mask[:, start:start + image_len] = encoder_mask[:, start:start + image_len]
            # 保证能够看到entity information
            start = text_num * text_len
            copy_mask[:, start:start + table_len] = encoder_mask[:, start:start + table_len]
            for index in range(self.max_gen_len - 1):
                dec_input_id = tokens_to_add.unsqueeze(0)
                gpt_outputs = self.decoder(input_ids=dec_input_id,
                                           encoder_hidden_states=encoder_hidden,
                                           encoder_attention_mask=copy_mask,
                                           use_cache=True,
                                           past_key_values=past_key_values)
                lm_logits = self.lm_head(gpt_outputs[0]) + self.final_logits_bias.cuda()

                past_key_values = gpt_outputs[1]
                gen_label = torch.argmax(lm_logits, dim=-1).squeeze(-1)
                tokens_to_add = gen_label * cur_unfinished + self.tokenizer.pad_token_id * (1 - cur_unfinished)
                outputs[i, index] = tokens_to_add
                cur_len += 1
                cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.tokenizer.eos_token_id).long())
                if cur_unfinished.max() == 0:
                    break
            if cur_len == self.max_gen_len:
                outputs[i, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.tokenizer.eos_token_id)
        return outputs, total_correct, total_hidden_num, gen_index

    def flatten_to_batch_test(self, text_seq, table_seq, image_seq, input_mask, table_mask, image_mask, batch_text_len,
                              batch_image_len, input_ids, table_text, image_ques_id):
        encoder_pooled_hidden_states = []
        encoder_hidden_states = []
        encoder_masks = []
        tmp_text_index = 0
        tmp_image_index = 0
        pointer_masks = []
        copy_masks = []
        batch_ids = []
        image_total_len = []
        for i in range(len(batch_text_len)):
            text_num = batch_text_len[i, 0] // batch_text_len[i, 1].item()
            tmp_hidden_states = text_seq[tmp_text_index:tmp_text_index + text_num]
            tmp_pooled_states = tmp_hidden_states[:, 0]
            tmp_masks_text = input_mask[tmp_text_index:tmp_text_index + text_num]
            text_ids = input_ids[tmp_text_index:tmp_text_index + text_num]
            pad_text_len = tmp_masks_text.size(1)
            tmp_pointer_mask_text = torch.tensor([[1] + [0] * (pad_text_len - 1)]).repeat(text_num, 1).cuda()
            tmp_text_index += text_num
            image_num = batch_image_len[i, 0].item()
            tmp_image_states = image_seq[tmp_image_index:tmp_image_index + image_num]
            tmp_image_pooled_hiddens = tmp_image_states[:, 0]
            tmp_image_masks = image_mask[tmp_image_index:tmp_image_index + image_num]
            pad_image_len = tmp_image_masks.size(1)
            tmp_image_ques_id = image_ques_id[tmp_image_index:tmp_image_index + image_num]
            tmp_pointer_mask_image = torch.tensor([[1] + [0] * (pad_image_len - 1)]).repeat(image_num, 1).cuda()
            tmp_image_index += image_num
            image_total_len.append(tmp_pointer_mask_image.reshape(-1).size(0))

            table_pooled = table_seq[i][0].unsqueeze(0)
            tmp_hidden_pooled_states = torch.cat(
                (tmp_pooled_states, table_pooled, tmp_image_pooled_hiddens), 0)
            tmp_hidden_states = torch.cat(
                (tmp_hidden_states.reshape(-1, tmp_hidden_states.size(-1)), table_seq[i],
                 tmp_image_states.reshape(-1, tmp_image_states.size(-1))), 0)
            tmp_masks = torch.cat((tmp_masks_text.reshape(-1), table_mask[i], tmp_image_masks.reshape(-1)), 0)

            table_len = table_mask[i].size(0)
            tmp_pointer_mask_table = torch.tensor([[1] + [0] * (table_len - 1)]).cuda()
            tmp_pointer_mask = torch.cat((tmp_pointer_mask_text.reshape(-1), tmp_pointer_mask_table.reshape(-1),
                                          tmp_pointer_mask_image.reshape(-1)), 0)

            image_ids = []
            image_pad_ids = torch.zeros(tmp_image_states.size(1) - tmp_image_ques_id.size(1), dtype=torch.int64).cuda()
            for idx in range(tmp_image_ques_id.size(0)):
                tmp_ids = torch.cat((tmp_image_ques_id[idx], image_pad_ids), dim=0)
                image_ids.append(tmp_ids)
            image_ids = torch.stack(image_ids)
            tmp_ids = torch.cat((text_ids.reshape(-1), table_text[i], image_ids.reshape(-1)), 0)
            encoder_pooled_hidden_states.append(tmp_hidden_pooled_states)
            encoder_hidden_states.append(tmp_hidden_states)
            encoder_masks.append(tmp_masks)
            pointer_masks.append(tmp_pointer_mask)
            batch_ids.append(tmp_ids)
        return encoder_hidden_states, encoder_masks, encoder_pooled_hidden_states, pointer_masks, batch_ids, image_total_len

    def test(self, input_ids, input_mask, image_feat, image_mask, image_ques_id, batch_text_len, batch_image_len,
             ans_ids, attn_mask, input_token_type, image_token_type, table_text, table_mask, table_token_type,
             table_connect_num, flatten_connect_spans, flatten_connect_index, table_cell_num,
             flatten_cell_span, gather_index, span_label, encoder_outputs=None):
        input_ids, input_mask, table_text, table_mask, image_feat, image_mask, image_ques_id, \
        input_token_type, table_token_type, image_token_type = self.get_input_list(input_ids, input_mask, table_text,
                                                                                   table_mask, image_feat, image_mask,
                                                                                   image_ques_id, batch_text_len,
                                                                                   batch_image_len, input_token_type,
                                                                                   table_token_type, image_token_type)
        output_attentions = False
        output_hidden_states = True

        text_outputs = self.BART_encoder(input_ids=input_ids, attention_mask=input_mask,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        text_seq = text_outputs[0]

        image_mask_invert = torch.where(image_mask == 0,
                                        torch.full(image_mask.size(), fill_value=True).cuda(),
                                        torch.full(image_mask.size(), fill_value=False).cuda())
        image_outputs = self.OFA_encoder(input_ids=image_ques_id, attention_mask=image_mask_invert,
                                         token_type_ids=image_token_type, patch_images=image_feat,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        image_seq = image_outputs[0]
        seq_hidden, connect_span, connect_index, span_label, cell_span = self.get_table_connet(
            table_connect_num, flatten_connect_spans,
            flatten_connect_index, text_seq[:, 0],
            image_seq[:, 0], batch_text_len,
            batch_image_len, table_cell_num, span_label, flatten_cell_span)

        table_outputs, has_connect, similarity = self.BART_table_encoder(input_ids=table_text,
                                                                         attention_mask=table_mask,
                                                                         output_attentions=output_attentions,
                                                                         output_hidden_states=output_hidden_states,
                                                                         connect_span=connect_span,
                                                                         connect_index=connect_index,
                                                                         seq_hidden=seq_hidden, cell_span=cell_span,
                                                                         gather_index=gather_index,
                                                                         span_label=span_label,
                                                                         is_train=False)
        table_seq = table_outputs[0]

        image_mask = torch.cat((torch.ones((image_mask.size(0), 16 * 16), dtype=torch.long).cuda(), image_mask), 1)
        encoder_hidden_states, encoder_attention_mask, encoder_pooled_states, pointer_masks, batch_ids, image_total_len = self.flatten_to_batch_test(
            text_seq, table_seq, image_seq, input_mask, table_mask, image_mask,
            batch_text_len, batch_image_len, input_ids, table_text, image_ques_id)
        batch_size = len(encoder_hidden_states)
        outputs = torch.full((batch_size, self.max_gen_len), fill_value=self.tokenizer.pad_token_id,
                             dtype=torch.int64).cuda()
        gen_index = []
        ans_ids = ans_ids.cuda()
        for i in range(batch_size):
            encoder_hidden = encoder_hidden_states[i].unsqueeze(0)
            encoder_mask = encoder_attention_mask[i].unsqueeze(0)
            pointer_mask = pointer_masks[i].unsqueeze(0)
            decoder_input = ans_ids[i]
            b_souce_index = torch.nonzero(decoder_input == self.b_sc.unsqueeze(0)).to(torch.int64)[0, -1]
            dec_input_id = decoder_input[:b_souce_index].unsqueeze(0)
            gpt_outputs = self.decoder(input_ids=dec_input_id,
                                       encoder_hidden_states=encoder_hidden,
                                       encoder_attention_mask=encoder_mask,
                                       use_cache=True,
                                       past_key_values=None)
            past_key_values = gpt_outputs.past_key_values
            dec_input_id = self.b_sc.unsqueeze(0)
            gpt_outputs = self.decoder(input_ids=dec_input_id,
                                       encoder_hidden_states=encoder_hidden,
                                       encoder_attention_mask=pointer_mask,
                                       use_cache=True,
                                       past_key_values=past_key_values, output_attentions=True)
            past_key_values = gpt_outputs.past_key_values
            gpt_out = gpt_outputs.last_hidden_state
            cls_logits = F.sigmoid(self.retreive_num_head(gpt_out))
            text_num = (batch_text_len[i, 0] // batch_text_len[i, 1]).item()
            text_len = input_ids.size(1)
            table_len = table_text.size(1)
            image_len = image_seq.size(1)
            pointer_hidden, source_index = self.get_pointer_hidden(encoder_hidden, gpt_outputs.cross_attentions,
                                                                   text_num, text_len, table_len, image_len)
            tmp_source_hidden = []
            tmp_source_index = []
            for index in range(self.max_source_len):
                if cls_logits.item() < 0.5:
                    break
                tmp_source_hidden.append(pointer_hidden.squeeze(0))
                if source_index not in tmp_source_index:
                    tmp_source_index.append(source_index)
                gpt_outputs = self.decoder(inputs_embeds=pointer_hidden,
                                           encoder_hidden_states=encoder_hidden,
                                           encoder_attention_mask=pointer_mask,
                                           use_cache=True,
                                           past_key_values=past_key_values, output_attentions=True)
                past_key_values = gpt_outputs.past_key_values
                gpt_out = gpt_outputs.last_hidden_state
                cls_logits = F.sigmoid(self.retreive_num_head(gpt_out))
                pointer_hidden, source_index = self.get_pointer_hidden(encoder_hidden, gpt_outputs.cross_attentions,
                                                                       text_num, text_len, table_len, image_len)

            gen_index.append(tmp_source_index)
            cur_unfinished = outputs.new(1).fill_(1)
            cur_len = 0
            tokens_to_add = self.e_sc
            copy_mask = torch.zeros_like(encoder_mask)
            for index in tmp_source_index:
                if index < text_num:
                    start = index * text_len
                    copy_mask[:, start:start + text_len] = encoder_mask[:, start:start + text_len]
                else:
                    start = text_num * text_len + table_len + (index - text_num - 1) * image_len
                    copy_mask[:, start:start + image_len] = encoder_mask[:, start:start + image_len]

            start = text_num * text_len
            copy_mask[:, start:start + table_len] = encoder_mask[:, start:start + table_len]
            for index in range(self.max_gen_len - 1):
                dec_input_id = tokens_to_add.unsqueeze(0)
                gpt_outputs = self.decoder(input_ids=dec_input_id,
                                           encoder_hidden_states=encoder_hidden,
                                           encoder_attention_mask=copy_mask,
                                           use_cache=True,
                                           past_key_values=past_key_values)
                lm_logits = self.lm_head(gpt_outputs[0]) + self.final_logits_bias.cuda()

                past_key_values = gpt_outputs[1]
                gen_label = torch.argmax(lm_logits, dim=-1).squeeze(-1)
                tokens_to_add = gen_label * cur_unfinished + self.tokenizer.pad_token_id * (1 - cur_unfinished)
                outputs[i, index] = tokens_to_add
                cur_len += 1
                cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.tokenizer.eos_token_id).long())
                if cur_unfinished.max() == 0:
                    break
            if cur_len == self.max_gen_len:
                outputs[i, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.tokenizer.eos_token_id)
        return outputs, gen_index

    def test_golden(self, input_ids, input_mask, image_feat, image_mask, image_ques_id, batch_text_len, batch_image_len,
                    ans_ids, attn_mask, input_token_type, image_token_type, table_text, table_mask, table_token_type,
                    table_connect_num, flatten_connect_spans, flatten_connect_index, table_cell_num,
                    flatten_cell_span, gather_index, span_label, encoder_outputs=None):
        input_ids, input_mask, table_text, table_mask, image_feat, image_mask, image_ques_id, \
        input_token_type, table_token_type, image_token_type = self.get_input_list(input_ids, input_mask, table_text,
                                                                                   table_mask, image_feat, image_mask,
                                                                                   image_ques_id, batch_text_len,
                                                                                   batch_image_len, input_token_type,
                                                                                   table_token_type, image_token_type)
        output_attentions = False
        output_hidden_states = True

        text_outputs = self.BART_encoder(input_ids=input_ids, attention_mask=input_mask,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        text_seq = text_outputs[0]

        image_mask_invert = torch.where(image_mask == 0,
                                        torch.full(image_mask.size(), fill_value=True).cuda(),
                                        torch.full(image_mask.size(), fill_value=False).cuda())
        image_outputs = self.OFA_encoder(input_ids=image_ques_id, attention_mask=image_mask_invert,
                                         token_type_ids=image_token_type, patch_images=image_feat,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        image_seq = image_outputs[0]
        seq_hidden, connect_span, connect_index, span_label, cell_span = self.get_table_connet(
            table_connect_num, flatten_connect_spans,
            flatten_connect_index, text_seq[:, 0],
            image_seq[:, 0], batch_text_len,
            batch_image_len, table_cell_num, span_label, flatten_cell_span)

        table_outputs, has_connect, similarity = self.BART_table_encoder(input_ids=table_text,
                                                                         attention_mask=table_mask,
                                                                         output_attentions=output_attentions,
                                                                         output_hidden_states=output_hidden_states,
                                                                         connect_span=connect_span,
                                                                         connect_index=connect_index,
                                                                         seq_hidden=seq_hidden, cell_span=cell_span,
                                                                         gather_index=gather_index,
                                                                         span_label=span_label,
                                                                         is_eval_golden=True)
        table_seq = table_outputs[0]

        image_mask = torch.cat((torch.ones((image_mask.size(0), 16 * 16), dtype=torch.long).cuda(), image_mask), 1)
        encoder_hidden_states, encoder_attention_mask, encoder_pooled_states, pointer_masks, batch_ids, image_total_len = self.flatten_to_batch_test(
            text_seq, table_seq, image_seq, input_mask, table_mask, image_mask,
            batch_text_len, batch_image_len, input_ids, table_text, image_ques_id)
        batch_size = len(encoder_hidden_states)
        outputs = torch.full((batch_size, self.max_gen_len), fill_value=self.tokenizer.pad_token_id,
                             dtype=torch.int64).cuda()
        gen_index = []
        ans_ids = ans_ids.cuda()
        for i in range(batch_size):
            encoder_hidden = encoder_hidden_states[i].unsqueeze(0)
            encoder_mask = encoder_attention_mask[i].unsqueeze(0)
            pointer_mask = pointer_masks[i].unsqueeze(0)
            decoder_input = ans_ids[i]
            b_souce_index = torch.nonzero(decoder_input == self.b_sc.unsqueeze(0)).to(torch.int64)[0, -1]
            dec_input_id = decoder_input[:b_souce_index].unsqueeze(0)
            gpt_outputs = self.decoder(input_ids=dec_input_id,
                                       encoder_hidden_states=encoder_hidden,
                                       encoder_attention_mask=encoder_mask,
                                       use_cache=True,
                                       past_key_values=None)
            past_key_values = gpt_outputs.past_key_values
            dec_input_id = self.b_sc.unsqueeze(0)
            gpt_outputs = self.decoder(input_ids=dec_input_id,
                                       encoder_hidden_states=encoder_hidden,
                                       encoder_attention_mask=pointer_mask,
                                       use_cache=True,
                                       past_key_values=past_key_values, output_attentions=True)
            past_key_values = gpt_outputs.past_key_values
            gpt_out = gpt_outputs.last_hidden_state
            cls_logits = F.sigmoid(self.retreive_num_head(gpt_out))
            text_num = (batch_text_len[i, 0] // batch_text_len[i, 1]).item()
            text_len = input_ids.size(1)
            table_len = table_text.size(1)
            image_len = image_seq.size(1)
            pointer_hidden, source_index = self.get_pointer_hidden(encoder_hidden, gpt_outputs.cross_attentions,
                                                                   text_num, text_len, table_len, image_len)
            tmp_source_hidden = []
            tmp_source_index = []
            for index in range(self.max_source_len):
                if cls_logits.item() < 0.5:
                    break
                tmp_source_hidden.append(pointer_hidden.squeeze(0))
                if source_index not in tmp_source_index:
                    tmp_source_index.append(source_index)
                gpt_outputs = self.decoder(inputs_embeds=pointer_hidden,
                                           encoder_hidden_states=encoder_hidden,
                                           encoder_attention_mask=pointer_mask,
                                           use_cache=True,
                                           past_key_values=past_key_values, output_attentions=True)
                past_key_values = gpt_outputs.past_key_values
                gpt_out = gpt_outputs.last_hidden_state
                cls_logits = F.sigmoid(self.retreive_num_head(gpt_out))
                pointer_hidden, source_index = self.get_pointer_hidden(encoder_hidden, gpt_outputs.cross_attentions,
                                                                       text_num, text_len, table_len, image_len)

            gen_index.append(tmp_source_index)
            cur_unfinished = outputs.new(1).fill_(1)
            cur_len = 0
            tokens_to_add = self.e_sc
            copy_mask = torch.zeros_like(encoder_mask)
            for index in tmp_source_index:
                if index < text_num:
                    start = index * text_len
                    copy_mask[:, start:start + text_len] = encoder_mask[:, start:start + text_len]
                else:
                    start = text_num * text_len + table_len + (index - text_num - 1) * image_len
                    copy_mask[:, start:start + image_len] = encoder_mask[:, start:start + image_len]

            start = text_num * text_len
            copy_mask[:, start:start + table_len] = encoder_mask[:, start:start + table_len]
            for index in range(self.max_gen_len - 1):
                dec_input_id = tokens_to_add.unsqueeze(0)
                gpt_outputs = self.decoder(input_ids=dec_input_id,
                                           encoder_hidden_states=encoder_hidden,
                                           encoder_attention_mask=copy_mask,
                                           use_cache=True,
                                           past_key_values=past_key_values)
                lm_logits = self.lm_head(gpt_outputs[0]) + self.final_logits_bias.cuda()

                past_key_values = gpt_outputs[1]
                gen_label = torch.argmax(lm_logits, dim=-1).squeeze(-1)
                tokens_to_add = gen_label * cur_unfinished + self.tokenizer.pad_token_id * (1 - cur_unfinished)
                outputs[i, index] = tokens_to_add
                cur_len += 1
                cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.tokenizer.eos_token_id).long())
                if cur_unfinished.max() == 0:
                    break
            if cur_len == self.max_gen_len:
                outputs[i, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.tokenizer.eos_token_id)
        return outputs, gen_index

    def test_beam(self, input_ids, input_mask, image_feat, image_mask, image_ques_id, batch_text_len, batch_image_len,
                  ans_ids, attn_mask, input_token_type, image_token_type, table_text, table_mask, table_token_type,
                  table_connect_num, flatten_connect_spans, flatten_connect_index, table_cell_num,
                  flatten_cell_span, gather_index, span_label, encoder_outputs=None):
        input_ids, input_mask, table_text, table_mask, image_feat, image_mask, image_ques_id, \
        input_token_type, table_token_type, image_token_type = self.get_input_list(input_ids, input_mask, table_text,
                                                                                   table_mask, image_feat, image_mask,
                                                                                   image_ques_id, batch_text_len,
                                                                                   batch_image_len, input_token_type,
                                                                                   table_token_type, image_token_type)
        output_attentions = False
        output_hidden_states = True

        text_outputs = self.BART_encoder(input_ids=input_ids, attention_mask=input_mask,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        text_seq = text_outputs[0]

        image_mask_invert = torch.where(image_mask == 0,
                                        torch.full(image_mask.size(), fill_value=True).cuda(),
                                        torch.full(image_mask.size(), fill_value=False).cuda())
        image_outputs = self.OFA_encoder(input_ids=image_ques_id, attention_mask=image_mask_invert,
                                         token_type_ids=image_token_type, patch_images=image_feat,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        image_seq = image_outputs[0]
        seq_hidden, connect_span, connect_index, span_label, cell_span = self.get_table_connet(
            table_connect_num, flatten_connect_spans,
            flatten_connect_index, text_seq[:, 0],
            image_seq[:, 0], batch_text_len,
            batch_image_len, table_cell_num, span_label, flatten_cell_span)

        table_outputs, has_connect, similarity = self.BART_table_encoder(input_ids=table_text,
                                                                         attention_mask=table_mask,
                                                                         output_attentions=output_attentions,
                                                                         output_hidden_states=output_hidden_states,
                                                                         connect_span=connect_span,
                                                                         connect_index=connect_index,
                                                                         seq_hidden=seq_hidden, cell_span=cell_span,
                                                                         gather_index=gather_index,
                                                                         span_label=span_label,
                                                                         is_train=False)
        table_seq = table_outputs[0]

        image_mask = torch.cat((torch.ones((image_mask.size(0), 16 * 16), dtype=torch.long).cuda(), image_mask), 1)
        encoder_hidden_states, encoder_attention_mask, encoder_pooled_states, pointer_masks, batch_ids, image_total_len = self.flatten_to_batch_test(
            text_seq, table_seq, image_seq, input_mask, table_mask, image_mask,
            batch_text_len, batch_image_len, input_ids, table_text, image_ques_id)
        batch_size = len(encoder_hidden_states)
        outputs = [''] * batch_size
        gen_index = []
        ans_ids = ans_ids.cuda()
        for i in range(batch_size):
            encoder_hidden = encoder_hidden_states[i].unsqueeze(0)
            encoder_mask = encoder_attention_mask[i].unsqueeze(0)
            pointer_mask = pointer_masks[i].unsqueeze(0)
            decoder_input = ans_ids[i]
            b_souce_index = torch.nonzero(decoder_input == self.b_sc.unsqueeze(0)).to(torch.int64)[0, -1]
            dec_input_id = decoder_input[:b_souce_index].unsqueeze(0)
            gpt_outputs = self.decoder(input_ids=dec_input_id,
                                       encoder_hidden_states=encoder_hidden,
                                       encoder_attention_mask=encoder_mask,
                                       use_cache=True,
                                       past_key_values=None)
            past_key_values = gpt_outputs.past_key_values
            dec_input_id = self.b_sc.unsqueeze(0)
            gpt_outputs = self.decoder(input_ids=dec_input_id,
                                       encoder_hidden_states=encoder_hidden,
                                       encoder_attention_mask=pointer_mask,
                                       use_cache=True,
                                       past_key_values=past_key_values, output_attentions=True)
            past_key_values = gpt_outputs.past_key_values
            gpt_out = gpt_outputs.last_hidden_state
            cls_logits = F.sigmoid(self.retreive_num_head(gpt_out))
            text_num = (batch_text_len[i, 0] // batch_text_len[i, 1]).item()
            text_len = input_ids.size(1)
            table_len = table_text.size(1)
            image_len = image_seq.size(1)
            pointer_hidden, source_index = self.get_pointer_hidden(encoder_hidden, gpt_outputs.cross_attentions,
                                                                   text_num, text_len, table_len, image_len)
            tmp_source_hidden = []
            tmp_source_index = []
            for index in range(self.max_source_len):
                if cls_logits.item() < 0.5:
                    break
                tmp_source_hidden.append(pointer_hidden.squeeze(0))
                if source_index not in tmp_source_index:
                    tmp_source_index.append(source_index)
                gpt_outputs = self.decoder(inputs_embeds=pointer_hidden,
                                           encoder_hidden_states=encoder_hidden,
                                           encoder_attention_mask=pointer_mask,
                                           use_cache=True,
                                           past_key_values=past_key_values, output_attentions=True)
                past_key_values = gpt_outputs.past_key_values
                gpt_out = gpt_outputs.last_hidden_state
                cls_logits = F.sigmoid(self.retreive_num_head(gpt_out))
                pointer_hidden, source_index = self.get_pointer_hidden(encoder_hidden, gpt_outputs.cross_attentions,
                                                                       text_num, text_len, table_len, image_len)

            gen_index.append(tmp_source_index)
            copy_mask = torch.zeros_like(encoder_mask)
            for index in tmp_source_index:
                if index < text_num:
                    start = index * text_len
                    copy_mask[:, start:start + text_len] = encoder_mask[:, start:start + text_len]
                else:
                    start = text_num * text_len + table_len + (index - text_num - 1) * image_len
                    copy_mask[:, start:start + image_len] = encoder_mask[:, start:start + image_len]
            start = text_num * text_len
            copy_mask[:, start:start + table_len] = encoder_mask[:, start:start + table_len]
            mask_value = torch.full((len(tmp_source_hidden),),
                                    fill_value=torch.tensor(self.tokenizer.mask_token_id)).cuda()
            prompt_decoder_input = torch.cat((decoder_input[:b_souce_index + 1], mask_value, self.e_sc))
            gs_index = [b_souce_index.item() + 1, b_souce_index.item() + 1 + len(tmp_source_hidden)]
            encoder_mask = encoder_mask.unsqueeze(1).repeat(1, prompt_decoder_input.size(0), 1)
            encoder_mask[0, gs_index[0] - 1:gs_index[1]] = pointer_mask.unsqueeze(0)
            encoder_mask[0, gs_index[1]:] = copy_mask.unsqueeze(0)
            beam_scorer = BeamSearchScorer(
                batch_size=1,
                num_beams=self.beam_size,
                max_length=self.max_gen_len + prompt_decoder_input.size(0),
                device=torch.device('cuda'),
                length_penalty=self.length_penalty)

            logits_processor = LogitsProcessorList([RepetitionPenaltyLogitsProcessor(penalty=0.9)])
            # instantiate logits processors
            logits_warper = LogitsProcessorList([TopKLogitsWarper(32)])

            beam_output = self.beam_sample(input_ids=prompt_decoder_input.unsqueeze(0), beam_scorer=beam_scorer,
                                           encoder_hidden_states=encoder_hidden,
                                           encoder_mask=encoder_mask,
                                           copy_mask=copy_mask,
                                           gs_index=gs_index,
                                           gs_hidden=torch.stack(tmp_source_hidden).squeeze(1),
                                           logits_processor=logits_processor,
                                           logits_warper=logits_warper,
                                           max_length=self.max_gen_len + prompt_decoder_input.size(0))
            outputs[i] = beam_output['sequences'][0, prompt_decoder_input.size(0):]

        return outputs, gen_index

    def test_acc_beam(self, input_ids, input_mask, image_feat, image_mask, image_ques_id, batch_text_len,
                      batch_image_len,
                      ans_ids, attn_mask, input_token_type, image_token_type, table_text, table_mask, table_token_type,
                      table_connect_num, flatten_connect_spans, flatten_connect_index, table_cell_num,
                      flatten_cell_span, gather_index, span_label, encoder_outputs=None):
        input_ids, input_mask, table_text, table_mask, image_feat, image_mask, image_ques_id, \
        input_token_type, table_token_type, image_token_type = self.get_input_list(input_ids, input_mask, table_text,
                                                                                   table_mask, image_feat, image_mask,
                                                                                   image_ques_id, batch_text_len,
                                                                                   batch_image_len, input_token_type,
                                                                                   table_token_type, image_token_type)
        output_attentions = False
        output_hidden_states = True

        text_outputs = self.BART_encoder(input_ids=input_ids, attention_mask=input_mask,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        text_seq = text_outputs[0]

        image_mask_invert = torch.where(image_mask == 0,
                                        torch.full(image_mask.size(), fill_value=True).cuda(),
                                        torch.full(image_mask.size(), fill_value=False).cuda())
        image_outputs = self.OFA_encoder(input_ids=image_ques_id, attention_mask=image_mask_invert,
                                         token_type_ids=image_token_type, patch_images=image_feat,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        image_seq = image_outputs[0]

        seq_hidden, connect_span, connect_index, span_label, cell_span = self.get_table_connet(
            table_connect_num, flatten_connect_spans,
            flatten_connect_index, text_seq[:, 0],
            image_seq[:, 0], batch_text_len,
            batch_image_len, table_cell_num, span_label, flatten_cell_span)

        table_outputs, has_connect, similarity = self.BART_table_encoder(input_ids=table_text,
                                                                         attention_mask=table_mask,
                                                                         output_attentions=output_attentions,
                                                                         output_hidden_states=output_hidden_states,
                                                                         connect_span=connect_span,
                                                                         connect_index=connect_index,
                                                                         seq_hidden=seq_hidden, cell_span=cell_span,
                                                                         gather_index=gather_index,
                                                                         span_label=span_label,
                                                                         is_train=False)
        table_seq = table_outputs[0]

        image_mask = torch.cat((torch.ones((image_mask.size(0), 16 * 16), dtype=torch.long).cuda(), image_mask), 1)
        encoder_hidden_states, encoder_attention_mask, encoder_pooled_states, pointer_masks, batch_ids, image_total_len = self.flatten_to_batch_test(
            text_seq, table_seq, image_seq, input_mask, table_mask, image_mask,
            batch_text_len, batch_image_len, input_ids, table_text, image_ques_id)
        batch_size = len(encoder_hidden_states)
        outputs = [''] * batch_size
        gen_index = []
        ans_ids = ans_ids.cuda()
        for i in range(batch_size):
            encoder_hidden = encoder_hidden_states[i].unsqueeze(0)
            encoder_mask = encoder_attention_mask[i].unsqueeze(0)
            pointer_mask = pointer_masks[i].unsqueeze(0)
            decoder_input = ans_ids[i]
            b_souce_index = torch.nonzero(decoder_input == self.b_sc.unsqueeze(0)).to(torch.int64)[0, -1]
            dec_input_id = decoder_input[:b_souce_index].unsqueeze(0)
            gpt_outputs = self.decoder(input_ids=dec_input_id,
                                       encoder_hidden_states=encoder_hidden,
                                       encoder_attention_mask=encoder_mask,
                                       use_cache=True,
                                       past_key_values=None)
            past_key_values = gpt_outputs.past_key_values
            dec_input_id = self.b_sc.unsqueeze(0)
            gpt_outputs = self.decoder(input_ids=dec_input_id,
                                       encoder_hidden_states=encoder_hidden,
                                       encoder_attention_mask=pointer_mask,
                                       use_cache=True,
                                       past_key_values=past_key_values, output_attentions=True)
            past_key_values = gpt_outputs.past_key_values
            gpt_out = gpt_outputs.last_hidden_state
            cls_logits = F.sigmoid(self.retreive_num_head(gpt_out))
            text_num = (batch_text_len[i, 0] // batch_text_len[i, 1]).item()
            text_len = input_ids.size(1)
            table_len = table_text.size(1)
            image_len = image_seq.size(1)
            pointer_hidden, source_index = self.get_pointer_hidden(encoder_hidden, gpt_outputs.cross_attentions,
                                                                   text_num, text_len, table_len, image_len)
            tmp_source_hidden = []
            tmp_source_index = []
            for index in range(self.max_source_len):
                if cls_logits.item() < 0.5:
                    break
                tmp_source_hidden.append(pointer_hidden.squeeze(0))
                if source_index not in tmp_source_index:
                    tmp_source_index.append(source_index)
                gpt_outputs = self.decoder(inputs_embeds=pointer_hidden,
                                           encoder_hidden_states=encoder_hidden,
                                           encoder_attention_mask=pointer_mask,
                                           use_cache=True,
                                           past_key_values=past_key_values, output_attentions=True)
                past_key_values = gpt_outputs.past_key_values
                gpt_out = gpt_outputs.last_hidden_state
                cls_logits = F.sigmoid(self.retreive_num_head(gpt_out))
                pointer_hidden, source_index = self.get_pointer_hidden(encoder_hidden, gpt_outputs.cross_attentions,
                                                                       text_num, text_len, table_len, image_len)

            gen_index.append(tmp_source_index)
            copy_mask = torch.zeros_like(encoder_mask)
            for index in tmp_source_index:
                if index < text_num:
                    start = index * text_len
                    copy_mask[:, start:start + text_len] = encoder_mask[:, start:start + text_len]
                else:
                    start = text_num * text_len + table_len + (index - text_num - 1) * image_len
                    copy_mask[:, start:start + image_len] = encoder_mask[:, start:start + image_len]
            start = text_num * text_len
            copy_mask[:, start:start + table_len] = encoder_mask[:, start:start + table_len]
            mask_value = torch.full((len(tmp_source_hidden),),
                                    fill_value=torch.tensor(self.tokenizer.mask_token_id)).cuda()
            prompt_decoder_input = torch.cat((decoder_input[:b_souce_index + 1], mask_value, self.e_sc))
            gs_index = [b_souce_index.item() + 1, b_souce_index.item() + 1 + len(tmp_source_hidden)]
            encoder_mask = encoder_mask.unsqueeze(1).repeat(1, prompt_decoder_input.size(0), 1)
            encoder_mask[0, gs_index[0] - 1:gs_index[1]] = pointer_mask.unsqueeze(0)
            encoder_mask[0, gs_index[1]:] = copy_mask.unsqueeze(0)
            add_score_ids = self.move_redundancy(prompt_decoder_input)
            beam_scorer = BeamSearchAccScorer(
                batch_size=1,
                num_beams=self.beam_size,
                max_length=self.max_gen_len + prompt_decoder_input.size(0),
                device=torch.device('cuda'),
                length_penalty=self.length_penalty, add_score_ids=add_score_ids, constrained=self.constrained)

            logits_processor = LogitsProcessorList([RepetitionPenaltyLogitsProcessor(penalty=0.9)])
            # instantiate logits processors
            logits_warper = LogitsProcessorList([TopKLogitsWarper(32)])
            beam_output = self.beam_sample(input_ids=prompt_decoder_input.unsqueeze(0), beam_scorer=beam_scorer,
                                           encoder_hidden_states=encoder_hidden,
                                           encoder_mask=encoder_mask,
                                           copy_mask=copy_mask,
                                           gs_index=gs_index,
                                           gs_hidden=torch.stack(tmp_source_hidden).squeeze(1),
                                           logits_processor=logits_processor,
                                           logits_warper=logits_warper,
                                           max_length=self.max_gen_len + prompt_decoder_input.size(0))
            outputs[i] = beam_output['sequences'][0, prompt_decoder_input.size(0):]

        return outputs, gen_index

    def move_redundancy(self, prompt_decoder_input):
        redundancy_list = ['a', 'is', 'am', 'are', 'the', 'an', 'it', 'has', 'very', 'and', 'his', 'her', 'there',
                           'she', 'they', 'what', 'this', 'that', 'these', 'those', 'do', 'does', 'why', 'for',
                           'with', 'how', 'when', 'where', 'woman', 'man', 'women', 'men', 'some', 'while',
                           'have', 'them', 'him', 'than', '[b_source]', '<mask>', '[e_source]', '<s>'] + list(
            string.punctuation)
        str_list = self.tokenizer.convert_ids_to_tokens(prompt_decoder_input.tolist())
        id_list = []
        for item in str_list:
            if item.replace('Ġ', '').lower() not in redundancy_list:
                id_list.append(item)
        id_list = self.tokenizer.convert_tokens_to_ids(id_list)
        return id_list

    def beam_sample(
            self,
            input_ids: torch.LongTensor,
            beam_scorer,
            encoder_hidden_states,
            encoder_mask,
            gs_index,
            gs_hidden,
            copy_mask,
            add_score_ids=None,
            logits_processor=None,
            stopping_criteria=None,
            logits_warper=None,
            max_length=None,
            pad_token_id=None,
            eos_token_id=None,
            output_attentions=None,
            output_hidden_states=None,
            output_scores=None,
            return_dict_in_generate=None,
            **model_kwargs,
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        max_length = max_length
        validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.tokenizer.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        # eos_token_id = eos_token_id if eos_token_id is not None else self.gpt_toker.convert_tokens_to_ids('<|e_exp|>')
        output_scores = True
        output_attentions = False
        output_hidden_states = (output_hidden_states)
        return_dict_in_generate = (return_dict_in_generate)

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size = encoder_hidden_states.size(0)
        num_beams = beam_scorer.num_beams

        input_ids = input_ids.unsqueeze(1).repeat(1, num_beams, 1).view(batch_size * num_beams, -1)
        batch_beam_size, cur_len = input_ids.shape

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))
        encoder_hidden_states_ex = encoder_hidden_states.unsqueeze(1).repeat(1, num_beams, 1, 1)
        encoder_hidden_states_ex = encoder_hidden_states_ex.reshape(
            (batch_size * num_beams, -1, encoder_hidden_states_ex.size(-1)))
        encoder_mask_ex = encoder_mask.unsqueeze(1).repeat(1, num_beams, 1, 1)
        encoder_mask_ex = encoder_mask_ex.reshape((batch_size * num_beams, encoder_mask.size(1), -1))
        copy_mask = copy_mask.unsqueeze(1).repeat(1, num_beams, 1)
        copy_mask = copy_mask.reshape((batch_size * num_beams, -1)).unsqueeze(1)
        gs_index_ex = []
        gs_hidden_ex = []
        for _ in range(batch_size * num_beams):
            gs_index_ex.append(gs_index)
            gs_hidden_ex.append(gs_hidden)
        while cur_len < max_length:
            gpt_outputs = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states_ex,
                                       encoder_attention_mask=encoder_mask_ex,
                                       gs_hidden=gs_hidden_ex, gs_index=gs_index_ex)
            encoder_mask_ex = torch.cat((encoder_mask_ex, copy_mask), dim=1)
            gpt_out = gpt_outputs.last_hidden_state[:, -1, :]
            # 只取最后一个作为当前步的输出
            next_token_logits = self.lm_head(gpt_out)
            # next_token_scores = torch.log(next_token_logits)
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (gpt_outputs.decoder_attentions,) if self.config.is_encoder_decoder else (
                            gpt_outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (gpt_outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (gpt_outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (gpt_outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = F.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if beam_scorer.is_done:
                break

            if stopping_criteria(input_ids, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        return sequence_outputs
