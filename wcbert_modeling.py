import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.utils.checkpoint
from fastNLP import seq_len_to_mask
from torch import nn
import torch.nn.functional as F

from transformers.activations import gelu, gelu_new, ACT2FN
from transformers.configuration_bert import BertConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling

from function.radical_embedding_conv_snp import RadicalEmbeddingConvSnp
from module.crf import CRF
from module.bilstm import BiLSTM

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from transformers.modeling_bert import BertAttention, BertIntermediate, BertOutput, load_tf_weights_in_bert, BertModel

from wcbert_parser import get_argparse

BertLayerNorm = torch.nn.LayerNorm
args = get_argparse().parse_args()


class BertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type, boundary embeddings
    """

    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, boundary_ids=None, inputs_embeds=None):
        """
        here we add a boundary information
        boundary_ids: [batch_size, seq_length, boundary_size]
        boundary_mask: filter some boundary information
        """
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertCrossAttentionLayer(nn.Module):
    """
    融合偏旁部首信息和字符信息
    """

    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.output_dim = config.hidden_size
        self.num_heads = 1
        self.Q = nn.Linear(config.hidden_size, config.hidden_size)
        self.K = nn.Linear(config.hidden_size, config.hidden_size)
        self.V = nn.Linear(config.hidden_size, config.hidden_size)

        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.HP_dropout)

    def forward(
            self,
            character_output,
            radical_embeddings=None,
            radical_mask=None,
            head_mask=None,
            attention_mask=None
    ):
        batch_size_decoder, seq_len_decoder, _ = character_output.size()
        batch_size_encoder, seq_len_encoder, _ = radical_embeddings.size()  # torch.Size([6, 256, 768])

        query = (self.Q(character_output).view(batch_size_decoder, seq_len_decoder, self.num_heads, -1)
                 .transpose(1, 2))  # [batch_size_decoder, num_heads, seq_len_decoder, output_dim]
        key = (self.K(radical_embeddings).view(batch_size_encoder, seq_len_encoder, self.num_heads, -1)
               .transpose(1, 2))  # [batch_size_encoder, num_heads, seq_len_encoder, output_dim]
        value = (self.V(radical_embeddings).view(batch_size_encoder, seq_len_encoder, self.num_heads, -1)
                 .transpose(1, 2))  # [batch_size_encoder, num_heads, seq_len_encoder, output_dim]

        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / (
                self.output_dim ** 0.5)  # [batch_size_decoder, num_heads, seq_len_decoder, seq_len_encoder]
        if head_mask is not None:
            attention_weights = attention_weights.masked_fill(head_mask == 0, -1e9)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, value)  # [batch_size_decoder, num_heads, seq_len_decoder, output_dim]
        context = (context.transpose(1, 2).contiguous().view(batch_size_decoder, seq_len_decoder,
                                                             -1))  # [batch_size_decoder, seq_len_decoder, num_heads * output_dim]

        output = self.fc(context)  # [batch_size_decoder, seq_len_decoder, output_dim]
        return output


class BertLayer(nn.Module):
    # 1、Addition融合Cross-Attention和lexicon-Adapter的结果
    def __init__(self, config, feature_fusion_layer=False):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)

        # here we add an attention for matched word 为匹配的词添加注意力
        self.feature_fusion_layer = feature_fusion_layer
        if self.feature_fusion_layer:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.act = nn.Tanh()

            self.word_transform = nn.Linear(config.word_embed_dim, config.hidden_size)
            self.word_word_weight = nn.Linear(config.hidden_size, config.hidden_size)
            attn_W = torch.zeros(config.hidden_size, config.hidden_size)
            self.attn_W = nn.Parameter(attn_W)
            self.attn_W.data.normal_(mean=0.0, std=config.initializer_range)
            self.fuse_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

            self.linear = nn.Linear(config.hidden_size, config.hidden_size)

            self.dropout_rate = config.HP_dropout
            self.nb_heads = 2
            self.cross_attention_layer = BertCrossAttentionLayer(config)  # 添加偏旁部首信息

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_state,
            attention_mask=None,
            matched_word_embeddings=None,
            matched_word_mask=None,
            radical_embeddings=None,
            radical_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
    ):
        self_attention_outputs = self.attention(
            hidden_state,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]  # this is the contextual representation
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # decode need join attention from the outputs 解码需要从输出中加入attention
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            # 如果传入了`encoder_hidden_states` ，{self}必须通过设置`config.add_cross_attention = True` 来实例化cross-attention层。
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        # 字符向量character
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        character_output = layer_output

        if self.feature_fusion_layer:
            assert matched_word_mask is not None

            ## Lexicon-Adapter
            # transform 匹配词的矩阵
            word_outputs = self.word_transform(matched_word_embeddings)  # [N, L, W, D]
            word_outputs = self.act(word_outputs)
            word_outputs = self.word_word_weight(word_outputs)
            word_outputs = self.dropout(word_outputs)

            # attention_output = attention_output.unsqueeze(2) # [N, L, D] -> [N, L, 1, D]
            # Bi-linear Attention
            # 计算每个单词的相关系数α
            alpha = torch.matmul(layer_output.unsqueeze(2), self.attn_W)  # [N, L, 1, D]
            alpha = torch.matmul(alpha, torch.transpose(word_outputs, 2, 3))  # [N, L, 1, W]
            alpha = alpha.squeeze()  # [N, L, W]
            alpha = alpha + (1 - matched_word_mask.float()) * (-10000.0)
            alpha = torch.nn.Softmax(dim=-1)(alpha)  # [N, L, W]
            alpha = alpha.unsqueeze(-1)  # [N, L, W, 1]
            weighted_word_embedding = torch.sum(word_outputs * alpha, dim=2)  # [N, L, D] 所有词的加权总和
            layer_output = layer_output + weighted_word_embedding  # 将加权词典信息注入到字符向量中

            layer_output = self.dropout(layer_output)
            layer_output = self.fuse_layernorm(layer_output)
            lexicon_hidden_states = layer_output

            ## Cross-Attention
            attention_hidden_states = self.cross_attention_layer(
                character_output,
                radical_embeddings,
                radical_mask,
                head_mask,
                attention_mask)  # tensor

            ## 1、addition
            add_hidden_states = attention_hidden_states + lexicon_hidden_states

            layer_output = add_hidden_states
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.add_layers = config.add_layers
        total_layers = []
        for i in range(config.num_hidden_layers):
            if i in self.add_layers:
                total_layers.append(BertLayer(config, True))  # 在第1、2层之间加入特征融合模块
            else:
                total_layers.append(BertLayer(config, False))  # 不加入特征融合模块，使用原本的bert

        self.layer = nn.ModuleList(total_layers)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            matched_word_embeddings=None,
            matched_word_mask=None,
            radical_embeddings=None,
            radical_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        # print("Layer 0: \n")
        # print(hidden_states)
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    matched_word_embeddings,
                    matched_word_mask,
                    radical_embeddings,
                    radical_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    matched_word_embeddings,
                    matched_word_mask,
                    radical_embeddings,
                    radical_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]  # torch.Size([batch_size, 256, 768])
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        # false
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Parameter)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class WCBertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super(WCBertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            matched_word_embeddings=None,
            matched_word_mask=None,
            radical_embeddings=None,
            radical_mask=None,
            boundary_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

        batch_size: N 批大小
        seq_length: L 序列长度
        dim: D 维度
        word_num: W 词数量
        boundary_num: B 边界数量


        Args:
            input_ids: [N, L]
            attention_mask: [N, L]
            boundary_ids: [N, L, B]
            boundary_mask: [N, L, B]
            matched_word_embeddings: [B, L, W, D]
            matched_word_mask: [B, L, W]
        """
        #
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            boundary_ids=boundary_ids, inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            matched_word_embeddings=matched_word_embeddings,
            matched_word_mask=matched_word_mask,
            radical_embeddings=radical_embeddings,
            radical_mask=radical_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# 模型
class WCBertCRFForTokenClassification(BertPreTrainedModel):
    def __init__(self, config, pretrained_embeddings, num_labels, vocab):
        super().__init__(config)

        word_vocab_size = pretrained_embeddings.shape[0]
        embed_dim = pretrained_embeddings.shape[1]
        # self.datasets = datasets
        self.word_embeddings = nn.Embedding(word_vocab_size, embed_dim)
        self.radical_embeddings = RadicalEmbeddingConvSnp(vocab=vocab['lattice'], embed_size=50,
                                                          char_emb_size=30, filter_nums=[30], kernel_sizes=[3],
                                                          char_dropout=0.1, dropout=0.1, pool_method='max',
                                                          activation='relu', include_word_start_end=True,
                                                          min_char_freq=1)
        self.bert = WCBertModel(config)
        self.dropout = nn.Dropout(config.HP_dropout)
        self.num_labels = num_labels
        self.hidden2tag = nn.Linear(config.hidden_size, num_labels + 2)
        self.radical_proj = nn.Linear(self.radical_embeddings.embed_size, config.hidden_size)
        self.crf = CRF(num_labels, torch.cuda.is_available())

        self.init_weights()

        ## init the embedding
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        print("Load pretrained embedding from file.........")

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            matched_word_ids=None,
            matched_word_mask=None,
            boundary_ids=None,
            labels=None,
            flag="Train",
            lattice=None,
            seq_len=None,
            radical_mask=None
    ):
        #  匹配词嵌入
        matched_word_embeddings = self.word_embeddings(matched_word_ids)

        #  偏旁部首嵌入
        max_seq_len_and_lex_num = lattice.size(1)
        char_mask = seq_len_to_mask(seq_len, max_len=max_seq_len_and_lex_num).bool()
        char = lattice.masked_fill_(~char_mask, 0)
        char_mask = char_mask.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        radical_embeddings = self.radical_embeddings(char)
        radical_embeddings.masked_fill_(~char_mask.unsqueeze(-1), 0)
        radical_embeddings = self.radical_proj(radical_embeddings)  # torch.Size([16, 256, 768])

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            matched_word_embeddings=matched_word_embeddings,
            matched_word_mask=matched_word_mask,
            radical_embeddings=radical_embeddings,
            radical_mask=radical_mask,
            boundary_ids=boundary_ids,
        )

        # sequence_output = outputs[0]
        # sequence_output = self.dropout(sequence_output)
        # logits = self.hidden2tag(sequence_output)
        return outputs

        # if flag == 'Train':
        #     assert labels is not None
        #     loss = self.crf.neg_log_likelihood_loss(logits, attention_mask, labels)
        #     _, preds = self.crf._viterbi_decode(logits, attention_mask)
        #     return (loss, preds)
        # elif flag == 'Predict':
        #     _, preds = self.crf._viterbi_decode(logits, attention_mask)
        #     return (preds,)


class BertWordLSTMCRFForTokenClassification(BertPreTrainedModel):
    """
    model-level fusion baseline
    concat bert vector with attention weighted sum word embedding
    and then input to LSTM-CRF
    """

    def __init__(self, config, pretrained_embeddings, num_labels):
        super().__init__(config)

        word_vocab_size = pretrained_embeddings.shape[0]
        embed_dim = pretrained_embeddings.shape[1]
        self.word_embeddings = nn.Embedding(word_vocab_size, embed_dim)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.HP_dropout)

        self.act = nn.Tanh()
        self.word_transform = nn.Linear(config.word_embed_dim, config.hidden_size)
        self.word_word_weight = nn.Linear(config.hidden_size, config.hidden_size)
        self.bilstm = BiLSTM(config.hidden_size * 2, config.lstm_size, config.HP_dropout)

        attn_W = torch.zeros(config.hidden_size, config.hidden_size)
        self.attn_W = nn.Parameter(attn_W)
        self.attn_W.data.normal_(mean=0.0, std=config.initializer_range)

        self.num_labels = num_labels
        self.hidden2tag = nn.Linear(config.lstm_size * 2, num_labels + 2)
        self.crf = CRF(num_labels, torch.cuda.is_available())

        self.init_weights()

        ## init the embedding
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        print("Load pretrained embedding from file.........")

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            matched_word_ids=None,
            matched_word_mask=None,
            boundary_ids=None,
            labels=None,
            flag="Train"
    ):
        matched_word_embeddings = self.word_embeddings(matched_word_ids)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]

        matched_word_embeddings = self.word_transform(matched_word_embeddings)
        matched_word_embeddings = self.act(matched_word_embeddings)
        matched_word_embeddings = self.word_word_weight(matched_word_embeddings)
        matched_word_embeddings = self.dropout(matched_word_embeddings)

        alpha = torch.matmul(sequence_output.unsqueeze(2), self.attn_W)  # [N, L, 1, D]
        alpha = torch.matmul(alpha, torch.transpose(matched_word_embeddings, 2, 3))  # [N, L, 1, W]
        alpha = alpha.squeeze()  # [N, L, W]
        alpha = alpha + (1 - matched_word_mask.float()) * (-2 ** 31 + 1)
        alpha = torch.nn.Softmax(dim=-1)(alpha)  # [N, L, W]
        alpha = alpha.unsqueeze(-1)  # [N, L, W, 1]
        matched_word_embeddings = torch.sum(matched_word_embeddings * alpha, dim=2)  # [N, L, D]

        ## concat the embedding [B, L, N, D], [B, L, N]
        sequence_output = torch.cat((sequence_output, matched_word_embeddings), dim=-1)

        sequence_output = self.dropout(sequence_output)
        lstm_output = self.bilstm(sequence_output, attention_mask)
        logits = self.hidden2tag(lstm_output)

        if flag == 'Train':
            assert labels is not None
            loss = self.crf.neg_log_likelihood_loss(logits, attention_mask, labels)
            _, preds = self.crf._viterbi_decode(logits, attention_mask)
            return (loss, preds)
        elif flag == 'Predict':
            _, preds = self.crf._viterbi_decode(logits, attention_mask)
            return (preds,)
