from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastNLP import Vocabulary
from fastNLP.embeddings import TokenEmbedding, StaticEmbedding
from fastNLP.embeddings.utils import get_embeddings

char_info = dict()
radical_path = "data/dataset/NER/chaizi.txt"
with open(radical_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        char, info = line.split('\t', 1)
        char_info[char] = info.replace('\n', '').split('\t')


def char2radical(c):
    if c in char_info.keys():
        c_info = char_info[c]
        return list(c_info)
    return ['○']


def construct_radical_vocab_from_vocab(char_vocab: Vocabulary, min_freq: int = 1, include_word_start_end=True):
    r"""
    给定一个char的vocabulary生成radical的vocabulary.

    :param char_vocab: 从vocab
    :param include_word_start_end: 是否需要包含特殊的<bow>和<eos>
    :return:
    """
    # radical_vocab = []
    # for key, value in char_vocab.items():
    #     radical_vocab.append([char2radical(key), value])
    #     # radical_vocab[char2radical(key)] = value
    # if include_word_start_end:
    #     # radical_vocab[['<bow>', '<eow>']] = len(radical_vocab)
    #     radical_vocab.append([['<bow>', '<eow>'], len(radical_vocab)])
    # return radical_vocab

    radical_vocab = Vocabulary(min_freq=min_freq)
    for char, index in char_vocab:
        if not char_vocab._is_word_no_create_entry(char):
            radical_vocab.add_word_lst(char2radical(char))
    if include_word_start_end:
        radical_vocab.add_word_lst(['<bow>', '<eow>'])
    return radical_vocab


class RadicalEmbeddingConvSnp(TokenEmbedding):
    def __init__(self, vocab: Vocabulary, embed_size: int = 50, char_emb_size: int = 50, char_dropout: float = 0,
                 dropout: float = 0, filter_nums: List[int] = (40, 30, 20), kernel_sizes: List[int] = (5, 3, 1),
                 pool_method: str = 'max', activation='relu', min_char_freq: int = 2, pre_train_char_embed: str = None,
                 requires_grad: bool = True, include_word_start_end: bool = True):

        super(RadicalEmbeddingConvSnp, self).__init__(vocab, word_dropout=char_dropout, dropout=dropout)

        for kernel in kernel_sizes:
            assert kernel % 2 == 1, "Only odd kernel is allowed."

        assert pool_method in ('max', 'avg')
        self.pool_method = pool_method
        # activation function
        if isinstance(activation, str):
            if activation.lower() == 'relu':
                self.activation = F.relu
            elif activation.lower() == 'sigmoid':
                self.activation = F.sigmoid
            elif activation.lower() == 'tanh':
                self.activation = F.tanh
        elif activation is None:
            self.activation = lambda x: x
        elif callable(activation):
            self.activation = activation
        else:
            raise Exception(
                "Undefined activation function: choose from: [relu, tanh, sigmoid, or a callable function]")

        # logger.info("Start constructing character vocabulary.")
        # 建立radical的词表
        # self.radical_vocab = construct_radical_vocab_from_vocab(vocab, include_word_start_end=include_word_start_end)
        self.radical_vocab = construct_radical_vocab_from_vocab(vocab, min_freq=min_char_freq,
                                                                include_word_start_end=include_word_start_end)
        # exit()  # 查看radical表

        self.char_pad_index = self.radical_vocab.padding_idx

        # logger.info(f"In total, there are {len(self.radical_vocab)} distinct characters.")
        # 对vocab进行index
        max_radical_nums = max(map(lambda x: len(char2radical(x[0])), vocab))
        if include_word_start_end:
            max_radical_nums += 2
        self.register_buffer('chars_to_radicals_embedding',
                             torch.full((len(vocab), max_radical_nums), fill_value=self.char_pad_index,
                                        dtype=torch.long))
        self.register_buffer('word_lengths', torch.zeros(len(vocab)).long())
        for word, index in vocab:
            # if index!=vocab.padding_idx:  # 如果是pad的话，直接就为pad_value了。修改为不区分pad, 这样所有的<pad>也是同一个embed
            word = char2radical(word)
            if include_word_start_end:
                word = ['<bow>'] + word + ['<eow>']
            self.chars_to_radicals_embedding[index, :len(word)] = torch.LongTensor([self.radical_vocab.to_index(c) for c in word])
            self.word_lengths[index] = len(word)
        self.char_embedding = get_embeddings((len(self.radical_vocab), char_emb_size))

        self.convs = nn.ModuleList(
            [nn.Conv1d(self.char_embedding.embedding_dim, filter_nums[i], kernel_size=kernel_sizes[i], bias=True,
                       padding=kernel_sizes[i] // 2) for i in range(len(kernel_sizes))])
        self._embed_size = embed_size
        self.fc = nn.Linear(sum(filter_nums), embed_size)
        self.requires_grad = requires_grad

    def forward(self, words):
        r"""
        输入words的index后，生成对应的words的表示。

        :param words: [batch_size, max_len]
        :return: [batch_size, max_len, embed_size]
        """
        word = self.drop_word(words)
        word = word.to(torch.long)
        batch_size, max_len = word.size()
        chars = self.chars_to_radicals_embedding[word]  # batch_size x max_len x max_word_len
        word_lengths = self.word_lengths[word]  # batch_size x max_len
        max_word_len = word_lengths.max()
        chars = chars[:, :, :max_word_len]
        # 为1的地方为mask
        chars_masks = chars.eq(self.char_pad_index)  # batch_size x max_len x max_word_len 如果为0, 说明是padding的位置了
        chars = self.char_embedding(chars)  # batch_size x max_len x max_word_len x embed_size
        chars = self.dropout(chars)
        reshaped_chars = chars.reshape(batch_size * max_len, max_word_len, -1)
        reshaped_chars = reshaped_chars.transpose(1, 2)  # B' x E x M
        # 使用Conv-SNP 先非线性，再卷积
        conv_chars = self.activation(reshaped_chars)
        conv_chars = [conv(conv_chars).transpose(1, 2).reshape(batch_size, max_len, max_word_len, -1) for conv in
                      self.convs]
        conv_chars = torch.cat(conv_chars, dim=-1).contiguous()  # B x max_len x max_word_len x sum(filters)

        # conv_chars = [conv(reshaped_chars).transpose(1, 2).reshape(batch_size, max_len, max_word_len, -1)
        #               for conv in self.convs]
        # conv_chars = torch.cat(conv_chars, dim=-1).contiguous()  # B x max_len x max_word_len x sum(filters)
        # conv_chars = self.activation(conv_chars)

        if self.pool_method == 'max':
            conv_chars = conv_chars.masked_fill(chars_masks.unsqueeze(-1), float('-inf'))
            chars, _ = torch.max(conv_chars, dim=-2)  # batch_size x max_len x sum(filters)
        else:
            conv_chars = conv_chars.masked_fill(chars_masks.unsqueeze(-1), 0)
            chars = torch.sum(conv_chars, dim=-2) / chars_masks.eq(False).sum(dim=-1, keepdim=True).float()
        chars = self.fc(chars)

        return self.dropout(chars)
