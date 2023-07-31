""" PyTorch OFA model."""

import torch
from torch import nn
from torch.nn import functional as F
from .modeling_ofa_mask import OFAPreTrainedModel, OFAEncoder, OFADecoderLayer, LayerNorm, Embedding, \
    LayerDropModuleList, \
    OFA_INPUTS_DOCSTRING, OFA_START_DOCSTRING, make_token_bucket_position, make_image_bucket_position, \
    _make_causal_mask, \
    _expand_mask, shift_tokens_right, new_arange
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.file_utils import ModelOutput
from transformers.utils import logging
from .configuration_ofa import OFAConfig
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from torch.nn.utils.rnn import pad_sequence
import math
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "OFA-Sys/OFA-base"
_CONFIG_FOR_DOC = "OFAConfig"
_TOKENIZER_FOR_DOC = "OFATokenizer"

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

OFA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "OFA-Sys/OFA-tiny",
    "OFA-Sys/OFA-medium",
    "OFA-Sys/OFA-base",
    "OFA-Sys/OFA-large",
]


class OFADecoder_pointer(OFAPreTrainedModel):
    r"""
    OFA decoder consisting of layers of [`OFADecoderLayer`]

    Args:
        config: OFAConfig
        embed_tokens (`nn.Embedding`, *optional*): output embedding
    """

    def __init__(self, config: OFAConfig, embed_tokens: Optional[nn.Embedding] = None, output_projection=None):
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout)
        self.decoder_layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.disable_entangle = False
        self._future_mask = torch.empty(0)
        self.share_input_output_embed = config.share_decoder_input_output_embed
        self.num_attention_heads = config.decoder_attention_heads

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_dim = config.d_model
        self.output_embed_dim = config.d_model

        self.layers = nn.ModuleList([OFADecoderLayer(config) for _ in range(config.decoder_layers)])
        if config.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(self.embed_dim)
        else:
            self.layernorm_embedding = None

        self.window_size = config.code_image_size // 8

        self.embed_positions = Embedding(self.max_target_positions + 2, self.embed_dim)
        self.embed_image_positions = Embedding(config.image_bucket_size ** 2 + 1, self.embed_dim)
        self.pos_ln = LayerNorm(self.embed_dim)
        self.image_pos_ln = LayerNorm(self.embed_dim)
        self.pos_scaling = float(self.embed_dim / self.num_attention_heads * config.attn_scale_factor) ** -0.5
        self.self_pos_q_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.self_pos_k_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.cross_pos_q_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.cross_pos_k_linear = nn.Linear(self.embed_dim, self.embed_dim)

        if config.code_layernorm_embedding:
            self.code_layernorm_embedding = LayerNorm(self.embed_dim)
        else:
            self.code_layernorm_embedding = None

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        dpr = [x.item() for x in torch.linspace(0, config.decoder_drop_path_rate, config.decoder_layers)]
        self.layers.extend([OFADecoderLayer(config, drop_path_rate=dpr[i]) for i in range(config.decoder_layers)])
        self.num_layers = len(self.layers)

        if config.decoder_normalize_before:
            self.layer_norm = LayerNorm(self.embed_dim)
        else:
            self.layer_norm = None

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(config)

        self.token_bucket_size = config.token_bucket_size
        token_num_rel_dis = 2 * config.token_bucket_size - 1
        token_rp_bucket = make_token_bucket_position(config.token_bucket_size)
        self.token_rel_pos_table_list = nn.ModuleList(
            [
                Embedding(token_num_rel_dis, self.num_attention_heads, zero_init=True)
                for _ in range(config.decoder_layers)
            ]
        )

        self.image_bucket_size = config.image_bucket_size
        image_num_rel_dis = (2 * config.image_bucket_size - 1) * (2 * config.image_bucket_size - 1) + 3
        image_rp_bucket = make_image_bucket_position(config.image_bucket_size, image_num_rel_dis)
        image_position_idx = torch.arange(self.window_size).unsqueeze(0).expand(self.window_size, self.window_size) + \
                             torch.arange(self.window_size).unsqueeze(1) * config.image_bucket_size + 1
        image_position_idx = torch.cat([torch.tensor([0]), image_position_idx.view(-1)])
        image_position_idx = torch.cat([image_position_idx, torch.tensor([1024] * 768)])
        self.image_rel_pos_table_list = nn.ModuleList(
            [
                Embedding(image_num_rel_dis, self.num_attention_heads, zero_init=True)
                for _ in range(config.decoder_layers)
            ]
        )

        self.register_buffer("token_rp_bucket", token_rp_bucket)
        self.register_buffer("image_rp_bucket", image_rp_bucket)
        self.register_buffer("image_position_idx", image_position_idx)
        self.entangle_position_embedding = config.entangle_position_embedding

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def build_output_projection(self, config):
        if self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, config.vocab_size, bias=False
            )
            nn.init.normal_(self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5)

    def get_rel_pos_bias(self, x, idx):
        r"""
        Get the relative positional bias of the text, for attention.
        """

        seq_len = x.size(1)
        rp_bucket = self.token_rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.token_rel_pos_table_list[idx].weight)
        values = values.permute([2, 0, 1])
        return values.contiguous()

    def get_image_rel_pos_bias(self, x, idx):
        r"""
        Get the relative positional bias of the image, for attention.
        """

        seq_len = x.size(1)
        image_position_idx = self.image_position_idx[:seq_len]
        rp_bucket = self.image_rp_bucket[image_position_idx][:, image_position_idx]
        values = F.embedding(rp_bucket, self.image_rel_pos_table_list[idx].weight)
        values = values.permute(2, 0, 1)
        return values

    def get_pos_info(self, tgt_pos_embed, src_pos_embed=None, use_image=False):
        r"""
        Get the positional information.

        Args:
            tgt_pos_embed (`torch.FloatTensor` of shape `(bsz, tgt_len, embed_dim)`):
                the target-side positional embeddings.
            src_pos_embed (`torch.FloatTensor` of shape `(bsz, src_len, embed_dim)`, *optional*):
                the source-side positional embeddings.
            use_image (`bool`): whether to use image.

        Returns:
            abs_pos_bias (`torch.FloatTensor` of shape `(bsz, src_len, tgt_len, src_len)`):
                absolute positional bias for attention.
        """

        batch_size = tgt_pos_embed.size(0)
        tgt_len = tgt_pos_embed.size(1)
        tgt_pos_embed = self.image_pos_ln(tgt_pos_embed) if use_image else self.pos_ln(tgt_pos_embed)

        if src_pos_embed is not None:
            src_len = src_pos_embed.size(1)
            pos_q = self.cross_pos_q_linear(tgt_pos_embed).view(
                batch_size, tgt_len, self.num_attention_heads, -1
            ).transpose(1, 2) * self.pos_scaling
            pos_k = self.cross_pos_k_linear(src_pos_embed).view(
                batch_size, src_len, self.num_attention_heads, -1
            ).transpose(1, 2)
        else:
            src_len = tgt_pos_embed.size(1)
            pos_q = self.self_pos_q_linear(tgt_pos_embed).view(
                batch_size, tgt_len, self.num_attention_heads, -1
            ).transpose(1, 2) * self.pos_scaling
            pos_k = self.self_pos_k_linear(tgt_pos_embed).view(
                batch_size, src_len, self.num_attention_heads, -1
            ).transpose(1, 2)
        abs_pos_bias = torch.matmul(pos_q, pos_k.transpose(2, 3))

        return abs_pos_bias

    def get_input_embeddings(self):
        r"""
        Get the input embeddings
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        r"""
        Set the weights of the embeddings with the given tensor.
        """
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, dtype, past_key_values_length):
        r"""
        Create causal mask for unidirectional decoding.
        [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        """
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, dtype, past_key_values_length=past_key_values_length
            ).to(attention_mask.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return self.max_target_positions

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    def get_normalized_probs_scriptable(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def reorder_incremental_state_scripting(
            self,
            # incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            past_key_values: Optional[torch.Tensor],
            new_order: Tensor,
    ):
        """Main entry point for reordering the incremental state.

        Due to limitations in TorchScript, we call this function in
        :class:`fairseq.sequence_generator.SequenceGenerator` instead of
        calling :func:`reorder_incremental_state` directly.
        """
        input_buffer = past_key_values
        new_past_key_values = []
        if input_buffer is not None:
            for input_buffer_k in input_buffer:
                new_input_buffer_k = []
                for input in input_buffer_k:
                    if input is None:
                        input = None
                    else:
                        input = input.index_select(0, new_order)
                    new_input_buffer_k.append(input)
                new_past_key_values.append(new_input_buffer_k)
        return new_past_key_values

    def forward(
            self,
            input_ids: torch.Tensor = None,
            inputs_embeds: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            encoder_hidden_states: torch.Tensor = None,
            encoder_attention_mask: torch.Tensor = None,
            code_masks: Optional[torch.Tensor] = None,
            src_pos_embed: torch.Tensor = None,
            past_key_values: Optional[torch.Tensor] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            gs_hidden=None, gs_index=None

    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(bsz, seq_len)`): indices of the sequence in the vocabulary.
            attention_mask (`torch.Tensor` of shape `(bsz, seq_len)`): mask to avoid attention on padding tokens.
            encoder_hidden_states (`torch.FloatTensor` of shape `(bsz, seq_len, hidden)`): the last hidden state of the encoder.
            encoder_attention_mask (`torch.Tensor` of shape `(bsz, seq_len)`): the padding mask of the source side.
            code_masks (`torch.Tensor` of shape `(bsz, seq_len)`): masks only for code generation.
            src_pos_embed (`torch.FloatTensor` of shape `(bsz, seq_len, hidden)`): the positional embeddings of the source side.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(bsz, num_heads, tgt_len, head_size)`) and 2 additional tensors of
                shape `(bsz, num_heads, src_len, head_size)`.
            use_cache (`bool`): whether to use cache for faster inference.
            output_attentions (`bool`): whether to output attention weights.
            output_hidden_states (`bool`): whether to output hidden states.

        Returns:
            BaseModelOutputWithPastAndCrossAttentions or a plain tuple:
                last_hidden_state (`torch.FloatTensor` of shape `(bsz, seq_len, hidden)`): the last hidden states.
                past_key_values (`tuple(tuple(torch.FloatTensor)): past keys and values for faster inference.
                hidden_states (`tuple(torch.FloatTensor)`): hidden states of all layers.
                attentions (`tuple(torch.FloatTensor)): self attention weights of all layers.
                cross_attentions (`tuple(torch.FloatTensor)): cross attention weights of all layers.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if past_key_values is not None and len(past_key_values) > 0:
            size = past_key_values[0][0].size()
            bsz, tgt_len = size[0], size[-2] + 1
            token_position_idx = torch.arange(tgt_len, device=input_ids.device).expand([bsz, tgt_len]).contiguous()
        else:
            bsz, tgt_len = input_ids.shape
            token_position_idx = new_arange(input_ids)
        tgt_pos_embed = self.embed_positions(token_position_idx)
        if code_masks is not None and torch.any(code_masks):
            image_position_idx = self.image_position_idx[:input_ids.size(1)].unsqueeze(0).expand(bsz, tgt_len)
            tgt_pos_embed[code_masks] = self.embed_image_positions(image_position_idx)[code_masks]

        # self attn position bias
        self_abs_pos_bias = self.get_pos_info(tgt_pos_embed, use_image=False)
        if code_masks is not None and torch.any(code_masks):
            self_image_abs_pos_bias = self.get_pos_info(tgt_pos_embed, use_image=True)
            self_abs_pos_bias[code_masks] = self_image_abs_pos_bias[code_masks]
        # cross attn position bias
        cross_abs_pos_bias = self.get_pos_info(tgt_pos_embed, src_pos_embed=src_pos_embed)
        if code_masks is not None and torch.any(code_masks):
            cross_image_abs_pos_bias = self.get_pos_info(tgt_pos_embed, src_pos_embed=src_pos_embed, use_image=True)
            cross_abs_pos_bias[code_masks] = cross_image_abs_pos_bias[code_masks]
        cross_abs_pos_bias = cross_abs_pos_bias.reshape(-1, *cross_abs_pos_bias.size()[-2:])

        all_prev_output_tokens = input_ids.clone()
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids = input_ids[:, -1:]
            cross_abs_pos_bias = cross_abs_pos_bias[:, -1:, :]
            tgt_pos_embed = tgt_pos_embed[:, -1:, :]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(input_ids)

        if inputs_embeds is not None:
            x = inputs_embeds

        if gs_index is not None and gs_hidden is not None:
            for i in range(len(gs_hidden)):
                x[i, gs_index[i][0]:gs_index[i][1]] = gs_hidden[i]

        if self.entangle_position_embedding and not self.disable_entangle:
            x += tgt_pos_embed

        if self.layernorm_embedding is not None:
            if code_masks is None or not code_masks.any() or not self.code_layernorm_embedding:
                x = self.layernorm_embedding(x)
            elif code_masks is not None and code_masks.all():
                x = self.code_layernorm_embedding(x)
            else:
                x[~code_masks] = self.layernorm_embedding(x[~code_masks])
                x[code_masks] = self.code_layernorm_embedding(x[code_masks])

        hidden_states = self.dropout(x)

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None and len(
            past_key_values) > 0 else 0
        if input_ids is not None:
            shape, dtype = input_ids.shape, hidden_states.dtype
        else:
            shape, dtype = inputs_embeds.shape, hidden_states.dtype
            shape = shape[:2]
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, shape, dtype, past_key_values_length)
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, x.dtype, tgt_len=shape[-1])

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # decoder layers
        for idx, layer in enumerate(self.layers):
            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None and len(past_key_values) > 0 else None

            self_attn_bias = self_abs_pos_bias.clone()
            if code_masks is None or not code_masks.any():
                self_attn_bias += self.get_rel_pos_bias(all_prev_output_tokens, idx).unsqueeze(0)
            elif code_masks is not None and code_masks.all():
                self_attn_bias += self.get_image_rel_pos_bias(all_prev_output_tokens, idx).unsqueeze(0)
            else:
                self_attn_bias[~code_masks] += self.get_rel_pos_bias(all_prev_output_tokens, idx).unsqueeze(0)
                self_attn_bias[code_masks] += self.get_image_rel_pos_bias(all_prev_output_tokens, idx).unsqueeze(0)
            self_attn_bias = self_attn_bias.reshape(-1, *self_attn_bias.size()[-2:])
            if past_key_value is not None and len(past_key_values) > 0:
                self_attn_bias = self_attn_bias[:, -1:, :]

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                self_attn_bias=self_attn_bias,
                cross_attn_bias=cross_abs_pos_bias,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)

        if self.output_projection is not None:
            lm_logits = self.output_projection(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        ), lm_logits


@add_start_docstrings(
    "The bare OFA Model outputting raw hidden-states without any specific head on top.",
    OFA_START_DOCSTRING,
)
class OFAModel_MMQA(OFAPreTrainedModel):
    r"""
    The OFA model built with an encoder and a decoder only, without any classification head.

    Args:
        config (OFAConfig): OFA configuration.
    """

    def __init__(self, config: OFAConfig, **kwargs):
        super().__init__(config)
        self.disable_entangle = getattr(kwargs, 'disable_entangle', False)
        self.tokenizer = kwargs['tokenizer']

        self.padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        shared = nn.Embedding(vocab_size, config.d_model, self.padding_idx)
        self.retreive_num_head = nn.Linear(config.d_model, 1, bias=False)
        self.encoder = OFAEncoder(config, shared)
        self.decoder = OFADecoder_pointer(config, shared)
        self.dim = config.d_model
        # Initialize weights and apply final processing
        self.post_init()
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        self.sc_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        self.cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.b_sc = torch.tensor(self.tokenizer.encode("[b_source]", add_special_tokens=False)).cuda()
        self.e_sc = torch.tensor(self.tokenizer.encode("[e_source]", add_special_tokens=False)).cuda()
        self.max_gen_len = 20
        self.max_source_len = 5

    def get_input_embeddings(self):
        r"""
        Retrieve input embeddings.
        """
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        r"""
        Set values for input embeddings
        """
        shared = value
        self.encoder.embed_tokens = shared
        self.decoder.embed_tokens = shared

    def get_encoder(self):
        r"""
        Retrieve the encoder
        """
        return self.encoder

    def get_decoder(self):
        r"""
        Retrieve the decoder
        """
        return self.decoder

    @add_start_docstrings_to_model_forward(OFA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    def get_normalized_probs_scriptable(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        if hasattr(self, "decoder"):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            # syntactic sugar for simple models which don't have a decoder
            # (e.g., the classification tutorial)
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

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

    def get_input_list(self, input_ids, input_mask, tabel_text, tabel_mask, image_feat, image_mask, image_ques_id,
                       batch_text_len, batch_image_len, input_token_type, tabel_token_type, image_token_type):
        batch_size = input_ids.size(0)
        input_ids = input_ids.type(torch.int64).cuda()
        input_mask = input_mask.type(torch.int64).cuda()
        input_token_type = input_token_type.type(torch.int64).cuda()
        tabel_text = tabel_text.cuda()
        tabel_mask = tabel_mask.cuda()
        tabel_token_type = tabel_token_type.cuda()
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

        return input_ids_new, input_mask_new, tabel_text, tabel_mask, image_feat_new, image_mask_new, image_ques_id_new, \
               input_token_type_new, tabel_token_type, image_token_type_new

    def forward(self, input_ids, input_mask, tabel_text, tabel_mask, image_feat, image_mask, image_ques_id,
                batch_text_len, batch_image_len, ans_ids, attn_mask, ans_loss_mask, sc_loss_mask,
                golden_source_index, input_token_type, tabel_token_type, image_token_type, encoder_outputs=None,
                ):
        input_ids, input_mask, tabel_text, tabel_mask, image_feat, image_mask, image_ques_id, \
        input_token_type, tabel_token_type, image_token_type = self.get_input_list(input_ids, input_mask, tabel_text,
                                                                                   tabel_mask, image_feat, image_mask,
                                                                                   image_ques_id, batch_text_len,
                                                                                   batch_image_len, input_token_type,
                                                                                   tabel_token_type, image_token_type)

        output_attentions = False
        output_hidden_states = self.config.output_hidden_states
        use_cache = self.config.use_cache
        # input_mask = torch.where(input_mask == 0,
        #                          torch.full_like(input_mask, fill_value=1),
        #                          torch.zeros_like(input_mask))
        text_outputs = self.encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_token_type,
                                    output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        text_seq = text_outputs[0]
        text_pos_embed = text_outputs.position_embedding
        # tabel_mask = torch.where(tabel_mask == 0,
        #                          torch.full_like(tabel_mask, fill_value=1),
        #                          torch.zeros_like(tabel_mask))
        tabel_outputs = self.encoder(input_ids=tabel_text, attention_mask=tabel_mask, token_type_ids=tabel_token_type,
                                     output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        tabel_seq = tabel_outputs[0]
        tabel_pos_embed = tabel_outputs.position_embedding
        # image_mask = torch.where(image_mask == 0,
        #                          torch.full_like(image_mask, fill_value=1),
        #                          torch.zeros_like(image_mask))
        image_outputs = self.encoder(input_ids=image_ques_id, attention_mask=image_mask,
                                     token_type_ids=image_token_type, patch_images=image_feat,
                                     output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        image_seq = image_outputs[0]
        image_pos_embed = image_outputs.position_embedding

        image_mask = torch.cat((image_mask, torch.zeros((image_mask.size(0), 16 * 16), dtype=torch.long).cuda()), 1)
        encoder_hidden_states = torch.cat((text_seq.reshape(-1, self.dim), tabel_seq.reshape(-1, self.dim),
                                           image_seq.reshape(-1, self.dim))).unsqueeze(0)
        encoder_attention_mask = torch.cat(
            (input_mask.reshape(-1), tabel_mask.reshape(-1), image_mask.reshape(-1))).unsqueeze(0)
        text_pooled = text_seq[:, 0]
        tabel_pooled = tabel_seq[:, 0]
        image_pooled = image_seq[:, 0]
        encoder_pooled_states = torch.cat((text_pooled, tabel_pooled, image_pooled)).unsqueeze(0)
        # image_ids = []
        tmp_pointer_mask_text = torch.tensor([[0] + [1] * (text_seq.size(1) - 1)]).repeat(text_seq.size(0), 1).cuda()
        tmp_pointer_mask_image = torch.tensor([[0] + [1] * (image_seq.size(1) - 1)]).repeat(image_seq.size(0), 1).cuda()
        tmp_pointer_mask_tabel = torch.tensor([[0] + [1] * (tabel_seq.size(1) - 1)]).cuda()
        pointer_masks = torch.cat((tmp_pointer_mask_text.reshape(-1), tmp_pointer_mask_tabel.reshape(-1),
                                   tmp_pointer_mask_image.reshape(-1)), 0).unsqueeze(0)
        copy_masks = torch.ones_like(encoder_attention_mask).cuda()
        index_list = golden_source_index[0].cpu().tolist()
        for j in range(len(index_list)):
            if index_list[j] == -1:
                index_list = index_list[:j]
                break
        for index in index_list:
            if index < text_seq.size(0):
                start = index * text_seq.size(1)
                copy_masks[:, start:start + text_seq.size(1)] = encoder_attention_mask[:,
                                                                start:start + text_seq.size(1)]
            elif index == text_seq.size(0):
                start = text_seq.size(0) * text_seq.size(1)
                copy_masks[:, start:start + tabel_seq.size(1)] = encoder_attention_mask[:,
                                                                 start:start + tabel_seq.size(1)]
            else:
                start = text_seq.size(0) * text_seq.size(1) + tabel_seq.size(1) + (
                        index - text_seq.size(0) - 1) * image_seq.size(1)
                copy_masks[:, start:start + image_seq.size(1)] = encoder_attention_mask[:,
                                                                 start:start + image_seq.size(1)]
        del text_seq
        del text_outputs
        del tabel_seq
        del tabel_outputs
        del image_seq
        del image_outputs
        sc_loss_mask = sc_loss_mask.cuda()
        gs_hidden = self.find_golden_sc(encoder_pooled_states, golden_source_index)
        gs_index = self.find_ans_sc_index(sc_loss_mask)
        encoder_attention_mask = encoder_attention_mask.unsqueeze(0).repeat(1, ans_ids.size(1), 1)
        encoder_attention_mask[:, gs_index[0][0] - 1:gs_index[0][1]] = pointer_masks
        encoder_attention_mask[:, gs_index[0][1]:] = copy_masks

        ans_ids = ans_ids.cuda()
        attn_mask = attn_mask.cuda()
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1).to(torch.float32)
        encoder_attention_mask = torch.where(encoder_attention_mask == 1,
                                             torch.full_like(encoder_attention_mask,
                                                             torch.finfo(torch.float32).min),
                                             torch.zeros_like(encoder_attention_mask))
        src_pos_embed = torch.cat((text_pos_embed.view(-1, self.dim), tabel_pos_embed.view(-1, self.dim),
                                   image_pos_embed.view(-1, self.dim))).unsqueeze(0)
        attn_mask = torch.where(attn_mask == 0,
                                torch.full_like(attn_mask, fill_value=1),
                                torch.zeros_like(attn_mask))
        decoder_outputs, lm_logits = self.decoder(
            input_ids=ans_ids,
            attention_mask=attn_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            src_pos_embed=src_pos_embed,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            gs_hidden=gs_hidden, gs_index=gs_index)
        re_logits = self.retreive_num_head(decoder_outputs.last_hidden_state)
        cross_attention_weights = decoder_outputs.cross_attentions
        del decoder_outputs
        cross_attention_weights = torch.stack(cross_attention_weights).cuda().sum(dim=0)
        cross_attention_weights = cross_attention_weights.sum(dim=1)
        ans_loss_mask = ans_loss_mask.cuda()
        ans_labels = ans_ids * ans_loss_mask
        ans_shift_labels = ans_labels[..., 1:].contiguous()
        shift_logits = lm_logits[..., :-1, :].contiguous()
        # shift_logits = shift_logits.float()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), ans_shift_labels.view(-1))
        pointer_loss = []
        re_loss = []
        for i in range(ans_ids.size(0)):
            pointer_weight = cross_attention_weights[i, gs_index[i][0] - 1:gs_index[i][1] - 1]
            # Note -1
            cls_index = torch.nonzero(pointer_weight[-1]).squeeze(-1)
            # 同一样例label相同
            pointer_weight = pointer_weight[:, cls_index]

            label = golden_source_index[i].cuda()
            label = label[:pointer_weight.size(0)]
            # pointer_weight = pointer_weight.float()
            tmp_loss = self.sc_criterion(pointer_weight, label)
            pointer_loss.append(tmp_loss)
            re_logits = re_logits[i, gs_index[i][0] - 1:gs_index[i][1]]
            cls_len = re_logits.size(0)
            label = torch.cat(
                (torch.ones(cls_len - 1, dtype=torch.float32), torch.zeros(1, dtype=torch.float32))).cuda().to(
                dtype=re_logits.dtype)
            # re_logits = re_logits.float()
            tmp_loss = self.cls_criterion(re_logits.view(-1), label)
            re_loss.append(tmp_loss)
        pointer_loss = torch.stack(pointer_loss).mean()
        re_loss = torch.stack(re_loss).mean()
        return gen_loss, pointer_loss, re_loss

    def get_guidance(self, input_ids, input_mask, tabel_text, tabel_mask, image_feat, image_mask, image_ques_id,
                     batch_text_len, batch_image_len, ans_ids, attn_mask, ans_loss_mask, sc_loss_mask,
                     golden_source_index, input_token_type, tabel_token_type, image_token_type, encoder_outputs=None,
                     ):
        output_attentions = False
        output_hidden_states = self.config.output_hidden_states
        use_cache = self.config.use_cache
        text_outputs = self.encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_token_type,
                                    output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        text_seq = text_outputs[0]
        text_pos_embed = text_outputs.position_embedding
        tabel_outputs = self.encoder(input_ids=tabel_text, attention_mask=tabel_mask, token_type_ids=tabel_token_type,
                                     output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        tabel_seq = tabel_outputs[0]
        tabel_pos_embed = tabel_outputs.position_embedding
        image_outputs = self.encoder(input_ids=image_ques_id, attention_mask=image_mask,
                                     token_type_ids=image_token_type, patch_images=image_feat,
                                     output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        image_seq = image_outputs[0]
        image_pos_embed = image_outputs.position_embedding

        image_mask = torch.cat((image_mask, torch.zeros((image_mask.size(0), 16 * 16), dtype=torch.long).cuda()), 1)
        encoder_hidden_states = torch.cat((text_seq.reshape(-1, self.dim), tabel_seq.reshape(-1, self.dim),
                                           image_seq.reshape(-1, self.dim))).unsqueeze(0)
        encoder_attention_mask = torch.cat(
            (input_mask.reshape(-1), tabel_mask.reshape(-1), image_mask.reshape(-1))).unsqueeze(0)
        text_pooled = text_seq[:, 0]
        tabel_pooled = tabel_seq[:, 0]
        image_pooled = image_seq[:, 0]
        encoder_pooled_states = torch.cat((text_pooled, tabel_pooled, image_pooled)).unsqueeze(0)
        # image_ids = []
        tmp_pointer_mask_text = torch.tensor([[1] + [0] * (text_seq.size(1) - 1)]).repeat(text_seq.size(0), 1).cuda()
        tmp_pointer_mask_image = torch.tensor([[1] + [0] * (image_seq.size(1) - 1)]).repeat(image_seq.size(0), 1).cuda()
        tmp_pointer_mask_tabel = torch.tensor([[1] + [0] * (tabel_seq.size(1) - 1)]).cuda()
        pointer_masks = torch.cat((tmp_pointer_mask_text.reshape(-1), tmp_pointer_mask_tabel.reshape(-1),
                                   tmp_pointer_mask_image.reshape(-1)), 0).unsqueeze(0)
        copy_masks = torch.zeros_like(encoder_attention_mask).cuda()
        index_list = golden_source_index[0].cpu().tolist()
        for j in range(len(index_list)):
            if index_list[j] == -1:
                index_list = index_list[:j]
                break
        for index in index_list:
            if index < text_seq.size(0):
                start = index * text_seq.size(1)
                copy_masks[:, start:start + text_seq.size(1)] = encoder_attention_mask[:,
                                                                start:start + text_seq.size(1)]
            elif index == text_seq.size(0):
                start = text_seq.size(0) * text_seq.size(1)
                copy_masks[:, start:start + tabel_seq.size(1)] = encoder_attention_mask[:,
                                                                 start:start + tabel_seq.size(1)]
            else:
                start = text_seq.size(0) * text_seq.size(1) + tabel_seq.size(1) + (
                        index - text_seq.size(0) - 1) * image_seq.size(1)
                copy_masks[:, start:start + image_seq.size(1)] = encoder_attention_mask[:,
                                                                 start:start + image_seq.size(1)]
        del text_seq
        del text_outputs
        del tabel_seq
        del tabel_outputs
        del image_seq
        del image_outputs
        sc_loss_mask = sc_loss_mask.cuda()
        gs_hidden = self.find_golden_sc(encoder_pooled_states, golden_source_index)
        gs_index = self.find_ans_sc_index(sc_loss_mask)
        encoder_attention_mask = encoder_attention_mask.unsqueeze(0).repeat(1, ans_ids.size(1), 1)
        encoder_attention_mask[:, gs_index[0][0] - 1:gs_index[0][1]] = pointer_masks
        encoder_attention_mask[:, gs_index[0][1]:] = copy_masks

        ans_ids = ans_ids.cuda()
        attn_mask = attn_mask.cuda()
        src_pos_embed = torch.cat((text_pos_embed.view(-1, self.dim), tabel_pos_embed.view(-1, self.dim),
                                   image_pos_embed.view(-1, self.dim))).unsqueeze(0)

        decoder_outputs, lm_logits = self.decoder(
            input_ids=ans_ids,
            attention_mask=attn_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            src_pos_embed=src_pos_embed,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            gs_hidden=gs_hidden, gs_index=gs_index)
        ans_loss_mask = ans_loss_mask.cuda()
        ans_labels = ans_ids * ans_loss_mask
        ans_shift_labels = ans_labels[..., 1:].contiguous()
        shift_logits = lm_logits[..., :-1, :].contiguous()
        # shift_logits = shift_logits.float()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), ans_shift_labels.view(-1))

        return gen_loss, shift_logits

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

    def flatten_to_batch(self, text_seq, tabel_seq, image_seq, input_mask, tabel_mask, image_mask, text_pos_embed,
                         tabel_pos_embed, image_pos_embed, batch_text_len, batch_image_len, golden_source_index,
                         input_ids, tabel_text, image_ques_id):
        encoder_pooled_hidden_states = []
        encoder_hidden_states = []
        encoder_masks = []
        tmp_text_index = 0
        tmp_image_index = 0
        pointer_masks = []
        copy_masks = []
        batch_ids = []
        image_total_len = []
        src_pos_embed = []
        for i in range(len(batch_text_len)):
            text_num = batch_text_len[i, 0] // batch_text_len[i, 1].item()
            tmp_hidden_states = text_seq[tmp_text_index:tmp_text_index + text_num]
            tmp_pooled_states = tmp_hidden_states[:, 0]
            tmp_masks_text = input_mask[tmp_text_index:tmp_text_index + text_num]
            text_ids = input_ids[tmp_text_index:tmp_text_index + text_num]
            tmp_text_pos_embed = text_pos_embed[tmp_text_index:tmp_text_index + text_num]
            pad_text_len = tmp_masks_text.size(1)
            tmp_pointer_mask_text = torch.tensor([[0] + [1] * (pad_text_len - 1)]).repeat(text_num, 1).cuda()
            tmp_text_index += text_num

            image_num = batch_image_len[i, 0].item()
            tmp_image_states = image_seq[tmp_image_index:tmp_image_index + image_num]
            tmp_image_pooled_hiddens = tmp_image_states[:, 0]
            tmp_image_masks = image_mask[tmp_image_index:tmp_image_index + image_num]
            pad_image_len = tmp_image_masks.size(1)
            tmp_image_ques_id = image_ques_id[tmp_image_index:tmp_image_index + image_num]
            tmp_image_pos_embed = image_pos_embed[tmp_image_index:tmp_image_index + image_num]
            tmp_pointer_mask_image = torch.tensor([[0] + [1] * (pad_image_len - 1)]).repeat(image_num, 1).cuda()
            tmp_image_index += image_num
            image_total_len.append(tmp_pointer_mask_image.reshape(-1).size(0))

            tabel_pooled = tabel_seq[i][0].unsqueeze(0)
            tmp_hidden_pooled_states = torch.cat(
                (tmp_pooled_states, tabel_pooled, tmp_image_pooled_hiddens), 0)
            tmp_hidden_states = torch.cat(
                (tmp_hidden_states.reshape(-1, tmp_hidden_states.size(-1)), tabel_seq[i],
                 tmp_image_states.reshape(-1, tmp_image_states.size(-1))), 0)
            tmp_masks = torch.cat((tmp_masks_text.reshape(-1), tabel_mask[i], tmp_image_masks.reshape(-1)), 0)
            tmp_scr_pos_embed = torch.cat(
                (tmp_text_pos_embed.reshape(-1, tmp_hidden_states.size(-1)), tabel_pos_embed[i],
                 tmp_image_pos_embed.reshape(-1, tmp_hidden_states.size(-1))), 0)
            tabel_len = tabel_mask[i].size(0)
            tmp_pointer_mask_tabel = torch.tensor([[0] + [1] * (tabel_len - 1)]).cuda()
            tmp_pointer_mask = torch.cat((tmp_pointer_mask_text.reshape(-1), tmp_pointer_mask_tabel.reshape(-1),
                                          tmp_pointer_mask_image.reshape(-1)), 0)
            tmp_copy_mask = torch.ones_like(tmp_masks).cuda()
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
                elif index == text_num:
                    start = text_num * pad_text_len
                    tmp_copy_mask[start:start + tabel_len] = tmp_masks[start:start + tabel_len]
                else:
                    start = text_num * pad_text_len + tabel_len + (index - text_num - 1) * pad_image_len
                    tmp_copy_mask[start:start + pad_image_len] = tmp_masks[start:start + pad_image_len]
            image_ids = []
            image_pad_ids = torch.zeros(tmp_image_states.size(1) - tmp_image_ques_id.size(1), dtype=torch.int64).cuda()
            for idx in range(tmp_image_ques_id.size(0)):
                tmp_ids = torch.cat((tmp_image_ques_id[idx], image_pad_ids), dim=0)
                image_ids.append(tmp_ids)
            image_ids = torch.stack(image_ids)
            tmp_ids = torch.cat((text_ids.reshape(-1), tabel_text[i], image_ids.reshape(-1)), 0)
            encoder_pooled_hidden_states.append(tmp_hidden_pooled_states)
            encoder_hidden_states.append(tmp_hidden_states)
            encoder_masks.append(tmp_masks)
            pointer_masks.append(tmp_pointer_mask)
            copy_masks.append(tmp_copy_mask)
            batch_ids.append(tmp_ids)
            src_pos_embed.append(tmp_scr_pos_embed)
        return encoder_hidden_states, encoder_masks, encoder_pooled_hidden_states, pointer_masks, copy_masks, batch_ids, image_total_len, src_pos_embed

    def evaluate(self, input_ids, input_mask, tabel_text, tabel_mask, image_feat, image_mask, image_ques_id,
                 batch_text_len, batch_image_len, ans_ids, attn_mask, ans_loss_mask, sc_loss_mask,
                 golden_source_index, input_token_type, tabel_token_type, image_token_type, encoder_outputs=None
                 ):
        input_ids, input_mask, tabel_text, tabel_mask, image_feat, image_mask, image_ques_id, \
        input_token_type, tabel_token_type, image_token_type = self.get_input_list(input_ids, input_mask, tabel_text,
                                                                                   tabel_mask, image_feat, image_mask,
                                                                                   image_ques_id, batch_text_len,
                                                                                   batch_image_len, input_token_type,
                                                                                   tabel_token_type, image_token_type)
        output_attentions = False
        output_hidden_states = self.config.output_hidden_states
        # use_cache = self.config.use_cache
        input_mask = torch.where(input_mask == 0,
                                 torch.full_like(input_mask, fill_value=1),
                                 torch.zeros_like(input_mask))
        text_outputs = self.encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_token_type,
                                    output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        text_seq = text_outputs[0]
        text_pos_embed = text_outputs.position_embedding
        tabel_mask = torch.where(tabel_mask == 0,
                                 torch.full_like(tabel_mask, fill_value=1),
                                 torch.zeros_like(tabel_mask))
        tabel_outputs = self.encoder(input_ids=tabel_text, attention_mask=tabel_mask, token_type_ids=tabel_token_type,
                                     output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        tabel_seq = tabel_outputs[0]
        tabel_pos_embed = tabel_outputs.position_embedding
        image_mask = torch.where(image_mask == 0,
                                 torch.full_like(image_mask, fill_value=1),
                                 torch.zeros_like(image_mask))
        image_outputs = self.encoder(input_ids=image_ques_id, attention_mask=image_mask,
                                     token_type_ids=image_token_type, patch_images=image_feat,
                                     output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        image_seq = image_outputs[0]
        image_pos_embed = image_outputs.position_embedding

        image_mask = torch.cat((image_mask, torch.zeros((image_mask.size(0), 16 * 16), dtype=torch.long).cuda()), 1)
        encoder_hidden_states, encoder_attention_mask, encoder_pooled_states, pointer_masks, copy_masks, batch_ids, image_total_len, src_pos_embed = self.flatten_to_batch(
            text_seq, tabel_seq, image_seq, input_mask, tabel_mask, image_mask, text_pos_embed, tabel_pos_embed,
            image_pos_embed, batch_text_len, batch_image_len, golden_source_index, input_ids, tabel_text, image_ques_id)
        batch_size = len(encoder_hidden_states)
        outputs = torch.full((batch_size, self.max_gen_len), fill_value=self.tokenizer.pad_token_id,
                             dtype=torch.int64).cuda()
        gen_index = []
        total_correct = 0
        total_hidden_num = 0
        ans_ids = ans_ids.cuda()
        for i in range(batch_size):
            encoder_hidden = encoder_hidden_states[i].unsqueeze(0)
            encoder_mask = encoder_attention_mask[i].unsqueeze(0).unsqueeze(0).unsqueeze(1)
            encoder_mask = encoder_mask.to(torch.float32)
            encoder_mask = torch.where(encoder_mask == 1, torch.full_like(encoder_mask, torch.finfo(torch.float32).min),
                                       torch.zeros_like(encoder_mask))
            pointer_mask = pointer_masks[i].unsqueeze(0).unsqueeze(0).unsqueeze(1)
            pointer_mask = pointer_mask.to(torch.float32)
            pointer_mask = torch.where(pointer_mask == 1, torch.full_like(pointer_mask, torch.finfo(torch.float32).min),
                                       torch.zeros_like(pointer_mask))
            decoder_input = ans_ids[i]
            b_souce_index = torch.nonzero(decoder_input == self.b_sc.unsqueeze(0)).to(torch.int64)[0, -1]
            dec_input_id = decoder_input[:b_souce_index].unsqueeze(0)
            gpt_outputs, lm_logits = self.decoder(input_ids=dec_input_id,
                                                  attention_mask=torch.zeros_like(dec_input_id),
                                                  encoder_hidden_states=encoder_hidden,
                                                  src_pos_embed=src_pos_embed[i].unsqueeze(0),
                                                  encoder_attention_mask=encoder_mask.repeat(1, 1, dec_input_id.size(1),
                                                                                             1),
                                                  use_cache=True,
                                                  past_key_values=None)
            past_key_values = gpt_outputs.past_key_values
            dec_input_id = torch.cat((dec_input_id, self.b_sc.unsqueeze(0)), -1)
            gpt_outputs, lm_logits = self.decoder(input_ids=dec_input_id,
                                                  encoder_hidden_states=encoder_hidden,
                                                  encoder_attention_mask=pointer_mask,
                                                  src_pos_embed=src_pos_embed[i].unsqueeze(0), use_cache=True,
                                                  past_key_values=past_key_values, output_attentions=True)
            past_key_values = gpt_outputs.past_key_values
            gpt_out = gpt_outputs.last_hidden_state
            cls_logits = F.sigmoid(self.retreive_num_head(gpt_out))
            text_num = (batch_text_len[i, 0] // batch_text_len[i, 1]).item()
            text_len = input_ids.size(1)
            tabel_len = tabel_text.size(1)
            image_len = image_seq.size(1)
            pointer_hidden, source_index = self.get_pointer_hidden(encoder_hidden, gpt_outputs.cross_attentions,
                                                                   text_num, text_len, tabel_len, image_len)
            tmp_source_hidden = []
            tmp_source_index = []
            for index in range(self.max_source_len):
                if cls_logits.item() < 0.5:
                    break
                tmp_source_hidden.append(pointer_hidden.squeeze(0))
                if source_index not in tmp_source_index:
                    tmp_source_index.append(source_index)
                dec_input_id = torch.cat(
                    (dec_input_id, torch.tensor([self.tokenizer.pad_token_id]).unsqueeze(0).cuda()), -1)
                gpt_outputs, lm_logits = self.decoder(input_ids=dec_input_id, inputs_embeds=pointer_hidden,
                                                      encoder_hidden_states=encoder_hidden,
                                                      encoder_attention_mask=pointer_mask,
                                                      src_pos_embed=src_pos_embed[i].unsqueeze(0), use_cache=True,
                                                      past_key_values=past_key_values, output_attentions=True)
                past_key_values = gpt_outputs.past_key_values
                gpt_out = gpt_outputs.last_hidden_state
                cls_logits = F.sigmoid(self.retreive_num_head(gpt_out))
                pointer_hidden, source_index = self.get_pointer_hidden(encoder_hidden, gpt_outputs.cross_attentions,
                                                                       text_num, text_len, tabel_len, image_len)

            sc_correct_num = len(list(set(tmp_source_index).intersection(set(golden_source_index[i].tolist()))))
            total_correct += sc_correct_num
            total_hidden_num += len(tmp_source_index)
            # source_hidden.append(tmp_source_hidden)
            gen_index.append(tmp_source_index)
            cur_unfinished = outputs.new(1).fill_(1)
            cur_len = 0
            tokens_to_add = self.e_sc
            copy_mask = torch.full_like(encoder_mask, fill_value=torch.finfo(torch.float32).min)
            for index in tmp_source_index:
                if index < text_num:
                    start = index * text_len
                    copy_mask[:, :, :, start:start + text_len] = encoder_mask[:, :, :, start:start + text_len]
                elif index == text_num:
                    start = text_num * text_len
                    copy_mask[:, :, :, start:start + tabel_len] = encoder_mask[:, :, :, start:start + tabel_len]
                else:
                    start = text_num * text_len + tabel_len + (index - text_num - 1) * image_len
                    copy_mask[:, :, :, start:start + image_len] = encoder_mask[:, :, :, start:start + image_len]

            for index in range(self.max_gen_len - 1):
                dec_input_id = torch.cat((dec_input_id, tokens_to_add.unsqueeze(0)), -1)
                gpt_outputs, lm_logits = self.decoder(input_ids=dec_input_id,
                                                      encoder_hidden_states=encoder_hidden,
                                                      encoder_attention_mask=copy_mask,
                                                      src_pos_embed=src_pos_embed[i].unsqueeze(0),
                                                      use_cache=True,
                                                      past_key_values=past_key_values)
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

    def evaluate_golden(self, input_ids, input_mask, tabel_text, tabel_mask, image_feat, image_mask, image_ques_id,
                        batch_text_len, batch_image_len, ans_ids, attn_mask, ans_loss_mask, sc_loss_mask,
                        golden_source_index, input_token_type, tabel_token_type, image_token_type, encoder_outputs=None
                        ):
        input_ids, input_mask, tabel_text, tabel_mask, image_feat, image_mask, image_ques_id, \
        input_token_type, tabel_token_type, image_token_type = self.get_input_list(input_ids, input_mask, tabel_text,
                                                                                   tabel_mask, image_feat, image_mask,
                                                                                   image_ques_id, batch_text_len,
                                                                                   batch_image_len, input_token_type,
                                                                                   tabel_token_type, image_token_type)
        output_attentions = False
        output_hidden_states = self.config.output_hidden_states
        # use_cache = self.config.use_cache
        input_mask = torch.where(input_mask == 0,
                                 torch.full_like(input_mask, fill_value=1),
                                 torch.zeros_like(input_mask))
        text_outputs = self.encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_token_type,
                                    output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        text_seq = text_outputs[0]
        text_pos_embed = text_outputs.position_embedding
        tabel_mask = torch.where(tabel_mask == 0,
                                 torch.full_like(tabel_mask, fill_value=1),
                                 torch.zeros_like(tabel_mask))
        tabel_outputs = self.encoder(input_ids=tabel_text, attention_mask=tabel_mask, token_type_ids=tabel_token_type,
                                     output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        tabel_seq = tabel_outputs[0]
        tabel_pos_embed = tabel_outputs.position_embedding
        image_mask = torch.where(image_mask == 0,
                                 torch.full_like(image_mask, fill_value=1),
                                 torch.zeros_like(image_mask))
        image_outputs = self.encoder(input_ids=image_ques_id, attention_mask=image_mask,
                                     token_type_ids=image_token_type, patch_images=image_feat,
                                     output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        image_seq = image_outputs[0]
        image_pos_embed = image_outputs.position_embedding

        image_mask = torch.cat((image_mask, torch.zeros((image_mask.size(0), 16 * 16), dtype=torch.long).cuda()), 1)
        encoder_hidden_states, encoder_attention_mask, encoder_pooled_states, pointer_masks, copy_masks, batch_ids, image_total_len, src_pos_embed = self.flatten_to_batch(
            text_seq, tabel_seq, image_seq, input_mask, tabel_mask, image_mask, text_pos_embed, tabel_pos_embed,
            image_pos_embed, batch_text_len, batch_image_len, golden_source_index, input_ids, tabel_text, image_ques_id)
        batch_size = len(encoder_hidden_states)
        gs_hidden = self.find_golden_sc(encoder_pooled_states, golden_source_index)
        gs_index = self.find_ans_sc_index(sc_loss_mask)
        outputs = torch.full((batch_size, self.max_gen_len), fill_value=self.tokenizer.pad_token_id,
                             dtype=torch.int64).cuda()

        ans_ids = ans_ids.cuda()
        for i in range(batch_size):
            encoder_hidden = encoder_hidden_states[i].unsqueeze(0)
            encoder_mask = encoder_attention_mask[i].unsqueeze(0).unsqueeze(0).unsqueeze(1)
            encoder_mask = encoder_mask.to(torch.float32)
            encoder_mask = torch.where(encoder_mask == 1, torch.full_like(encoder_mask, torch.finfo(torch.float32).min),
                                       torch.zeros_like(encoder_mask))
            pointer_mask = pointer_masks[i].unsqueeze(0).unsqueeze(0).unsqueeze(1)
            pointer_mask = pointer_mask.to(torch.float32)
            pointer_mask = torch.where(pointer_mask == 1, torch.full_like(pointer_mask, torch.finfo(torch.float32).min),
                                       torch.zeros_like(pointer_mask))
            copy_mask = copy_masks[i].unsqueeze(0).unsqueeze(0).unsqueeze(1)
            copy_mask = copy_mask.to(torch.float32)
            copy_mask = torch.where(copy_mask == 1, torch.full_like(copy_mask, torch.finfo(torch.float32).min),
                                    torch.zeros_like(copy_mask))
            decoder_input = ans_ids[i]
            e_souce_index = torch.nonzero(decoder_input == self.e_sc.unsqueeze(0)).to(torch.int64)[0, -1]
            dec_input_id = decoder_input[:e_souce_index].unsqueeze(0)
            encoder_mask = encoder_mask.repeat(1, 1, dec_input_id.size(1), 1)
            encoder_mask[:, :, gs_index[i][0] - 1:gs_index[i][1]] = pointer_mask
            gpt_outputs, _ = self.decoder(
                input_ids=dec_input_id,
                attention_mask=torch.zeros_like(dec_input_id),
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=encoder_mask,
                use_cache=True,
                src_pos_embed=src_pos_embed[i].unsqueeze(0),
                output_attentions=True,
                output_hidden_states=output_hidden_states,
                gs_hidden=gs_hidden[i].unsqueeze(0), gs_index=[gs_index[i]])

            past_key_values = gpt_outputs.past_key_values
            tokens_to_add = self.e_sc
            cur_unfinished = outputs.new(1).fill_(1)
            cur_len = 0
            for index in range(self.max_gen_len - 1):
                dec_input_id = torch.cat((dec_input_id, tokens_to_add.unsqueeze(0)), -1)
                gpt_outputs, lm_logits = self.decoder(input_ids=dec_input_id,
                                                      encoder_hidden_states=encoder_hidden,
                                                      encoder_attention_mask=copy_mask,
                                                      src_pos_embed=src_pos_embed[i].unsqueeze(0),
                                                      use_cache=True,
                                                      past_key_values=past_key_values)
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
            decoder_input_ids=None,
            past=None,
            attention_mask=None,
            code_masks=None,
            use_cache=False,
            encoder_outputs=None,
            **kwargs
    ):
        # if attention_mask is None:
        attention_mask = decoder_input_ids.new_zeros(decoder_input_ids.shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,
            "patch_images": None,
            "patch_images_2": None,
            "patch_masks": None,
            "token_embeddings": None,
            "sample_patch_num": None,
            "attention_mask": attention_mask,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "code_masks": code_masks,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ):
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", "attention_mask"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        if encoder_kwargs.get("patch_masks") is None:
            encoder_kwargs["patch_masks"] = torch.tensor([True])

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)
        model_kwargs["attention_mask"] = None

        return model_kwargs

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    @staticmethod
    def _expand_inputs_for_generation(
            input_ids: torch.LongTensor,
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            attention_mask: Optional[torch.LongTensor] = None,
            encoder_outputs: Optional[ModelOutput] = None,
            **model_kwargs,
    ):
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            encoder_outputs["position_embedding"] = encoder_outputs.position_embedding.index_select(
                0, expanded_return_idx.to(encoder_outputs.position_embedding.device)
            )
            encoder_outputs["padding_mask"] = encoder_outputs.padding_mask.index_select(
                0, expanded_return_idx.to(encoder_outputs.padding_mask.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs
