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
import math
import random
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    # logging,
    replace_return_docstrings,
)
from transformers.models.bart.configuration_bart import BartConfig
from .modeling_bart import BartPretrainedModel, BartLearnedPositionalEmbedding, BartEncoderLayer, BartDecoder
from .modeling_bart import BART_START_DOCSTRING, BART_INPUTS_DOCSTRING, BART_GENERATION_EXAMPLE, shift_tokens_right, \
    _expand_mask

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


class TableFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.retrieve_embed_proj = nn.Linear(hidden_dim, hidden_dim)
        self.retrieve_cls_proj = nn.Linear(hidden_dim, hidden_dim)
        self.connect_head = nn.Linear(hidden_dim * 2, 1)
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.xe_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        self.embed_dim = hidden_dim

    def forward(self, table_hidden: Optional[torch.Tensor] = None,
                connect_span: Optional[torch.Tensor] = None,
                connect_index: Optional[torch.Tensor] = None,
                cell_span: Optional[torch.Tensor] = None,
                gather_index=None, span_label=None,
                seq_hidden: Optional[torch.Tensor] = None):
        hidden_states_retrieve_proj = self.retrieve_embed_proj(table_hidden)
        connect_loss, similarity_loss = [], []
        for ba, offset in enumerate(cell_span):
            chunk = torch.zeros((offset.size(0) + 1, self.embed_dim),
                                dtype=hidden_states_retrieve_proj.dtype).cuda()
            chunk_hidden = hidden_states_retrieve_proj[ba, :]
            chunk = torch.index_add(chunk, 0, gather_index[ba].cuda(), chunk_hidden)
            chunk_len = [1] + [item[1].item() - item[0].item() for item in offset]
            chunk_len = torch.tensor(chunk_len).cuda()

            hidden_states_retrieve_proj_cell = chunk / chunk_len.unsqueeze(-1)
            # 第0位用于冗余编码：question/ROW/；/PAD等符号
            seq_hidden_retrieve_proj = self.retrieve_cls_proj(seq_hidden[ba])

            seq_hidden_cls = torch.cat(
                (seq_hidden_retrieve_proj, table_hidden[ba, :1, :].repeat(seq_hidden_retrieve_proj.size(0), 1)),
                dim=-1)
            has_connect_logits = self.connect_head(seq_hidden_cls).squeeze(-1)
            has_connect_lable = (span_label[ba].cuda() != 0).int()
            tmp_connect_loss = self.bce_criterion(has_connect_logits, has_connect_lable.to(torch.float))
            hidden_states_retrieve_proj_cell = hidden_states_retrieve_proj_cell.permute(1, 0)
            similarity_score = torch.matmul(seq_hidden_retrieve_proj, hidden_states_retrieve_proj_cell)
            tmp_similarity_loss = self.xe_criterion(similarity_score.reshape(-1, similarity_score.size(-1)),
                                                    span_label[ba].cuda().reshape(-1))
            connect_loss.append(tmp_connect_loss)
            similarity_loss.append(tmp_similarity_loss)
            for index in range(connect_span[ba].size(0)):
                span = connect_span[ba][index]
                tmp_connect_index = connect_index[ba][index]
                try:
                    for idx, item in enumerate(tmp_connect_index):
                        if item == -1:
                            break
                        table_hidden[ba, span[0].item():span[1].item()] += seq_hidden[ba][item]
                except:
                    table_hidden[ba, span[0].item():span[1].item()] += seq_hidden[ba][tmp_connect_index]
        connect_loss = torch.stack(connect_loss).mean()
        similarity_loss = torch.stack(similarity_loss).mean()
        return table_hidden, connect_loss, similarity_loss

    def evaluate(self, table_hidden: Optional[torch.Tensor] = None,
                 connect_span: Optional[torch.Tensor] = None,
                 connect_index: Optional[torch.Tensor] = None,
                 cell_span: Optional[torch.Tensor] = None,
                 gather_index=None, span_label=None,
                 seq_hidden: Optional[torch.Tensor] = None):
        hidden_states_retrieve_proj = self.retrieve_embed_proj(table_hidden)
        connect_list = []
        similarity_list = []
        for ba, offset in enumerate(cell_span):
            chunk = torch.zeros((offset.size(0) + 1, self.embed_dim),
                                dtype=hidden_states_retrieve_proj.dtype).cuda()
            chunk_hidden = hidden_states_retrieve_proj[ba, :]
            chunk = torch.index_add(chunk, 0, gather_index[ba].cuda(), chunk_hidden)
            chunk_len = [1] + [item[1] - item[0] for item in offset]
            chunk_len = torch.tensor(chunk_len).cuda()
            hidden_states_retrieve_proj_cell = chunk / chunk_len.unsqueeze(-1)
            seq_hidden_retrieve_proj = self.retrieve_cls_proj(seq_hidden[ba])
            seq_hidden_cls = torch.cat(
                (seq_hidden_retrieve_proj, table_hidden[ba, :1, :].repeat(seq_hidden_retrieve_proj.size(0), 1)), dim=-1)
            has_connect_logits = self.connect_head(seq_hidden_cls).squeeze(-1)
            hidden_states_retrieve_proj_cell = hidden_states_retrieve_proj_cell.permute(1, 0)
            similarity_score = torch.matmul(seq_hidden_retrieve_proj, hidden_states_retrieve_proj_cell)
            similarity_pred = torch.argmax(similarity_score, dim=-1)
            has_connect_logits = F.sigmoid(has_connect_logits)
            span_list = []
            for idx, item in enumerate(has_connect_logits):
                if item > 0.5:
                    cell_index = similarity_pred[idx] - 1
                    span = cell_span[ba][cell_index]
                    span_list.append(span.tolist())
                    if cell_index == -1:
                        continue
                    table_hidden[ba, span[0].item():span[1].item()] += seq_hidden[ba][idx]
            connect_list.append(torch.nonzero(has_connect_logits > 0.5).squeeze(-1).cpu().tolist())
            similarity_list.append(span_list)
        return table_hidden, connect_list, similarity_list


class BartFusionEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.fusion_layer_index = len(self.layers) // 2
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.table_fusion_layer = TableFusion(embed_dim)
        self.embed_dim = embed_dim
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            connect_span: Optional[torch.Tensor] = None,
            connect_index: Optional[torch.Tensor] = None,
            is_train=False, is_eval_golden=False,
            cell_span: Optional[torch.Tensor] = None,
            gather_index=None, span_label=None,
            seq_hidden: Optional[torch.Tensor] = None

    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if idx == self.fusion_layer_index:
                if self.training or is_eval_golden == True:
                    hidden_states, connect_loss, similarity_loss = self.table_fusion_layer(hidden_states,
                                                                                           connect_span=connect_span,
                                                                                           connect_index=connect_index,
                                                                                           cell_span=cell_span,
                                                                                           gather_index=gather_index,
                                                                                           span_label=span_label,
                                                                                           seq_hidden=seq_hidden)
                else:
                    hidden_states, has_connect, similarity = self.table_fusion_layer.evaluate(hidden_states,
                                                                                              connect_span=connect_span,
                                                                                              connect_index=connect_index,
                                                                                              cell_span=cell_span,
                                                                                              gather_index=gather_index,
                                                                                              span_label=span_label,
                                                                                              seq_hidden=seq_hidden)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if self.training or is_eval_golden == True:
            return BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
            ), connect_loss, similarity_loss
        else:
            return BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
            ), has_connect, similarity


class BartFusionModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartFusionEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            input_token_type: torch.LongTensor = None,
            connect_span: Optional[torch.Tensor] = None,
            connect_index: Optional[torch.Tensor] = None,
            is_train=False, is_eval_golden=False,
            cell_span: Optional[torch.Tensor] = None,
            gather_index=None, span_label=None,
            seq_hidden: Optional[torch.Tensor] = None
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict, connect_span=connect_span,
                connect_index=connect_index,
                cell_span=cell_span,
                gather_index=gather_index,
                span_label=span_label,
                is_train=True,
                seq_hidden=seq_hidden
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.", BART_START_DOCSTRING
)
class BartForTableFusion(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config)
        self.model = BartFusionModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        self.tokenizer = kwargs['tokenizer']
        self.e_sc = torch.tensor(self.tokenizer.convert_tokens_to_ids(["[e_source]"])).cuda()
        self.max_gen_len = 20

        # Initialize weights and apply final processing
        self.post_init()

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

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            ans_loss_mask: torch.LongTensor = None, input_token_type: torch.LongTensor = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            input_token_type=input_token_type,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        ans_labels = decoder_input_ids * ans_loss_mask
        ans_shift_labels = ans_labels[..., 1:].contiguous()
        shift_logits = lm_logits[..., :-1, :].contiguous()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), ans_shift_labels.view(-1))
        return gen_loss

    def evaluate(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, ans_loss_mask,
                 input_token_type
                 ):
        output_attentions = False
        input_ids = input_ids.cuda()
        input_mask = attention_mask.cuda()
        text_outputs = self.model.encoder(input_ids=input_ids, attention_mask=input_mask,
                                          output_attentions=output_attentions, output_hidden_states=True)
        text_seq = text_outputs[0]

        encoder_hidden_states = text_seq
        batch_size = encoder_hidden_states.size(0)
        outputs = torch.full((batch_size, self.max_gen_len), fill_value=self.tokenizer.pad_token_id,
                             dtype=torch.int64).cuda()
        ans_ids = decoder_input_ids.cuda()
        for i in range(batch_size):
            encoder_hidden = encoder_hidden_states[i].unsqueeze(0)
            encoder_mask = input_mask[i].unsqueeze(0)
            decoder_input = ans_ids[i]
            e_souce_index = torch.nonzero(decoder_input == self.e_sc.unsqueeze(0)).to(torch.int64)[0, -1]
            dec_input_id = decoder_input[:e_souce_index].unsqueeze(0)
            gpt_outputs = self.model.decoder(input_ids=dec_input_id,
                                             encoder_hidden_states=encoder_hidden,
                                             encoder_attention_mask=encoder_mask,
                                             use_cache=True,
                                             past_key_values=None)
            past_key_values = gpt_outputs.past_key_values
            cur_unfinished = outputs.new(1).fill_(1)
            cur_len = 0
            tokens_to_add = self.e_sc
            for index in range(self.max_gen_len - 1):
                dec_input_id = tokens_to_add.unsqueeze(0)
                gpt_outputs = self.model.decoder(input_ids=dec_input_id,
                                                 encoder_hidden_states=encoder_hidden,
                                                 encoder_attention_mask=encoder_mask,
                                                 use_cache=True,
                                                 past_key_values=past_key_values)
                lm_logits = self.lm_head(gpt_outputs[0]) + self.final_logits_bias
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

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
