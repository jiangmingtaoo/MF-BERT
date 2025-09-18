import copy
import os
import json
from functools import partial

import dgl
import scipy.sparse as sp

import numpy as np
from fastNLP import cache_results, Vocabulary, DataSet

from fastNLP.io import ConllLoader, JsonLoader
from tqdm import tqdm, trange
import torch
import pickle

from module.lexicon_tree import Trie
from module.sampler import SequentialDistributedSampler


def load_pretrain_embed(embedding_path, max_scan_num=1000000, add_seg_vocab=False):
    """
    从pretrained word embedding中读取前max_scan_num的词向量
    Args:
        embedding_path: 词向量路径
        max_scan_num: 最多读多少
    """
    ## 如果是使用add_seg_vocab, 则全局遍历
    if add_seg_vocab:
        max_scan_num = -1

    embed_dict = dict()
    embed_dim = -1
    with open(embedding_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if max_scan_num == -1:
            max_scan_num = len(lines)
        max_scan_num = min(max_scan_num, len(lines))
        line_iter = trange(max_scan_num)
        for idx in line_iter:
            line = lines[idx]
            line = line.strip()
            items = line.split()
            if len(items) == 2:
                embed_dim = int(items[1])
                continue
            elif len(items) == 201:
                token = items[0]
                embedd = np.empty([1, embed_dim])
                embedd[:] = items[1:]
                embed_dict[token] = embedd
            elif len(items) > 201:
                print("++++longer than 201+++++, line is: %s\n" % (line))
                token = items[0:-200]
                token = "".join(token)
                embedd = np.empty([1, embed_dim])
                embedd[:] = items[-200:]
                embed_dict[token] = embedd
            else:
                print("-------error word-------, line is: %s\n" % (line))

    return embed_dict, embed_dim


def build_pretrained_embedding_for_corpus(
        embedding_path,
        word_vocab,
        embed_dim=200,
        max_scan_num=1000000,
        saved_corpus_embedding_dir=None,
        add_seg_vocab=False
):
    """
    Args:
        embedding_path: 预训练的word embedding路径
        word_vocab: corpus的word vocab
        embed_dim: 维度
        max_scan_num: 最大浏览多大数量的词表
        saved_corpus_embedding_dir: 这个corpus对应的embedding保存路径
    """
    saved_corpus_embedding_file = os.path.join(saved_corpus_embedding_dir,
                                               'saved_word_embedding_{}.pkl'.format(max_scan_num))

    if os.path.exists(saved_corpus_embedding_file):
        with open(saved_corpus_embedding_file, 'rb') as f:
            pretrained_emb = pickle.load(f)
        return pretrained_emb, embed_dim

    embed_dict = dict()
    if embedding_path is not None:
        embed_dict, embed_dim = load_pretrain_embed(embedding_path, max_scan_num=max_scan_num,
                                                    add_seg_vocab=add_seg_vocab)

    scale = np.sqrt(3.0 / embed_dim)
    pretrained_emb = np.empty([word_vocab.item_size, embed_dim])

    matched = 0
    not_matched = 0

    for idx, word in enumerate(word_vocab.idx2item):
        if word in embed_dict:
            pretrained_emb[idx, :] = embed_dict[word]
            matched += 1
        else:
            pretrained_emb[idx, :] = np.random.uniform(-scale, scale, [1, embed_dim])
            not_matched += 1

    pretrained_size = len(embed_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, matched, not_matched, (not_matched + 0.) / word_vocab.item_size))

    with open(saved_corpus_embedding_file, 'wb') as f:
        pickle.dump(pretrained_emb, f, protocol=4)

    return pretrained_emb, embed_dim


def reverse_padded_sequence(inputs, lengths, batch_first=True):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """
    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)
    if len(lengths) != batch_size:
        raise ValueError("inputs is incompatible with lengths.")
    ind = [list(reversed(range(0, length))) + list(range(length, max_length))
           for length in lengths]
    ind = torch.LongTensor(ind).transpose(0, 1)
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = ind.expand_as(inputs)
    if inputs.is_cuda:
        ind = ind.cuda()
    reversed_inputs = torch.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)

    return reversed_inputs


def random_embedding(self, vocab_size, embedding_dim):
    pretrain_emb = np.empty([vocab_size, embedding_dim])
    scale = np.sqrt(3.0 / embedding_dim)
    for index in range(vocab_size):
        pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
    return pretrain_emb


def gather_indexes(sequence_tensor, positions):
    """
    gather specific tensor based on the positions
    Args:
        sequence_tensor: [B, L, D]
        positions: [B, P]
    """
    batch_size = sequence_tensor.size(0)
    seq_length = sequence_tensor.size(1)
    dim = sequence_tensor.size(2)

    whole_seq_length = torch.tensor([seq_length for _ in range(batch_size)], dtype=torch.long)
    whole_seq_length = whole_seq_length.to(sequence_tensor.device)

    flat_offsets = torch.cumsum(whole_seq_length, dim=-1)
    flat_offsets = flat_offsets - whole_seq_length  # [B]
    flat_offsets = flat_offsets.unsqueeze(-1)  # [B, 1]
    flat_positions = positions + flat_offsets  # [B, P]
    flat_positions = flat_positions.contiguous().view(-1)
    flat_sequence_tensor = sequence_tensor.contiguous().view(batch_size * seq_length, -1)  # [B * L, D]

    # output_tensor = flat_sequence_tensor[flat_positions]
    output_tensor = flat_sequence_tensor.index_select(0, flat_positions)
    output_tensor = output_tensor.contiguous().view(batch_size, -1)

    return output_tensor


def save_preds_for_seq_labelling(token_ids, tokenizer, true_labels, pred_labels, file):
    """
    save sequence labelling result into files
    Args:
        token_ids:
        tokenizer:
        true_labels:
        pred_labels:
        file:
    """
    error_num = 1
    with open(file, 'w', encoding='utf-8') as f:
        for w_ids, t_labels, p_labels in zip(token_ids, true_labels, pred_labels):
            tokens = tokenizer.convert_ids_to_tokens(w_ids)
            token_num = len(t_labels)
            tokens = tokens[1:token_num + 1]

            assert len(tokens) == len(t_labels), (len(tokens), len(t_labels))
            assert len(tokens) == len(p_labels), (len(tokens), len(p_labels))

            for w, t, p in zip(tokens, t_labels, p_labels):
                if t == p:
                    f.write("%s\t%s\t%s\n" % (w, t, p))
                else:
                    f.write("%s\t%s\t%s\t%d\n" % (w, t, p, error_num))
                    error_num += 1

            f.write("\n")


def get_bigrams(words):
    result = []
    for i, w in enumerate(words):
        if i != len(words) - 1:
            result.append(words[i] + words[i + 1])
        else:
            result.append(words[i] + '<end>')

    return result


@cache_results(_cache_fp='cache/ontonotes5ner', _refresh=False)
def load_ontonotes5ner(path, index_token=True, train_clip=False,
                       char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0):
    train_path = os.path.join(path, "train.bmes")
    dev_path = os.path.join(path, 'dev.bmes')
    test_path = os.path.join(path, "test.bmes")

    loader = ConllLoader(['chars', 'target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']

    datasets['train'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    print(datasets.keys())
    print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['dev'], datasets['test']])
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'], datasets['test']])
    label_vocab.from_dataset(datasets['train'], field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                 field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                   field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                  field_name='target', new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    vocabs['label'] = label_vocab

    # embeddings = {}
    # if char_embedding_path is not None:
    #     char_embedding = StaticEmbedding(char_vocab, char_embedding_path, word_dropout=0.01,
    #                                      min_freq=char_min_freq, only_train_min_freq=only_train_min_freq)
    #     embeddings['char'] = char_embedding
    #
    # if bigram_embedding_path is not None:
    #     bigram_embedding = StaticEmbedding(bigram_vocab, bigram_embedding_path, word_dropout=0.01,
    #                                        min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq)
    #     embeddings['bigram'] = bigram_embedding

    return datasets, vocabs


@cache_results(_cache_fp='cache/resume_ner', _refresh=False)
def load_resume_ner(path, index_token=True,
                    char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0):
    train_path = os.path.join(path, "train.char.bmes")
    dev_path = os.path.join(path, 'dev.char.bmes')
    test_path = os.path.join(path, "test.char.bmes")

    loader = ConllLoader(['chars', 'target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']

    datasets['train'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    print(datasets.keys())
    print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['dev'], datasets['test']])
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'], datasets['test']])
    label_vocab.from_dataset(datasets['train'], field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                 field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                   field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                  field_name='target', new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab

    # embeddings = {}
    # if char_embedding_path is not None:
    #     char_embedding = StaticEmbedding(char_vocab, char_embedding_path, word_dropout=0.01,
    #                                      min_freq=char_min_freq, only_train_min_freq=only_train_min_freq)
    #     embeddings['char'] = char_embedding
    #
    # if bigram_embedding_path is not None:
    #     bigram_embedding = StaticEmbedding(bigram_vocab, bigram_embedding_path, word_dropout=0.01,
    #                                        min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq)
    #     embeddings['bigram'] = bigram_embedding

    return datasets, vocabs


@cache_results(_cache_fp='cache/msraner1', _refresh=False)
def load_msra_ner_1(path, index_token=True, train_clip=False,
                    char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0):
    train_path = os.path.join(path, "train.char.bio")
    test_path = os.path.join(path, "test.char.bio")

    loader = ConllLoader(['chars', 'target'])
    train_bundle = loader.load(train_path)
    test_bundle = loader.load(test_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']

    datasets['train'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    print(datasets.keys())
    # print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['test']])
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['test']])
    label_vocab.from_dataset(datasets['train'], field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'], datasets['test'],
                                 field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'], datasets['test'],
                                   field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'], datasets['test'],
                                  field_name='target', new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    vocabs['label'] = label_vocab

    # embeddings = {}
    # if char_embedding_path is not None:
    #     char_embedding = StaticEmbedding(char_vocab, char_embedding_path, word_dropout=0.01,
    #                                      min_freq=char_min_freq, only_train_min_freq=only_train_min_freq)
    #     embeddings['char'] = char_embedding
    #
    # if bigram_embedding_path is not None:
    #     bigram_embedding = StaticEmbedding(bigram_vocab, bigram_embedding_path, word_dropout=0.01,
    #                                        min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq)
    #     embeddings['bigram'] = bigram_embedding

    return datasets, vocabs


@cache_results(_cache_fp='cache/weiboNER_uni+bi', _refresh=False)
def load_weibo_ner(path, index_token=True,
                   char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0, char_word_dropout=0.01, label='all'):
    loader = ConllLoader(['chars', 'target'])
    # 注释
    # bundle = loader.load(path)
    #
    # datasets = bundle.datasets
    #
    # print(datasets['train'][:5])
    # json_loader = JsonLoader()
    train_path = os.path.join(path, "train.char.bmes")
    dev_path = os.path.join(path, 'dev.char.bmes')
    test_path = os.path.join(path, "test.char.bmes")

    # train_path = os.path.join(path, "train.json")
    # dev_path = os.path.join(path, 'dev.json')
    # test_path = os.path.join(path, "test.json")

    paths = {'train': train_path, 'dev': dev_path, 'test': test_path}

    datasets = {}
    # for k, v in paths.items():
    #     json_data = json_loader.load(v)
    #     print('-------------------------json---------------------------')
    #     print(json_data.datasets)
    #     datasets[k] = json_data.datasets['train']
    for k, v in paths.items():
        bundle = loader.load(v)
        datasets[k] = bundle.datasets['train']
        # print('---------------------------k---------------------')
        # print(len(bundle.datasets['train']))

    for k, v in datasets.items():
        print('{}:{}'.format(k, len(v)))
    # print(*list(datasets.keys()))
    vocabs = {}
    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()

    for k, v in datasets.items():
        # ignore the word segmentation tag
        if label == 'ne':
            v.apply_field(lambda x: [w if len(w) > 1 and w.split('.')[1] == 'NAM' else 'O' for w in x], 'target',
                          'target')
        if label == 'nm':
            v.apply_field(lambda x: [w if len(w) > 1 and w.split('.')[1] == 'NOM' else 'O' for w in x], 'target',
                          'target')
        v.apply_field(lambda x: [w[0] for w in x], 'chars', 'chars')
        v.apply_field(get_bigrams, 'chars', 'bigrams')

    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['dev'], datasets['test']])
    label_vocab.from_dataset(datasets['train'], field_name='target')
    print('label_vocab:{}\n{}'.format(len(label_vocab), label_vocab.idx2word))

    for k, v in datasets.items():
        # v.set_pad_val('target',-100)
        v.add_seq_len('chars', new_field_name='seq_len')

    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab

    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'], datasets['test']])
    if index_token:
        char_vocab.index_dataset(*list(datasets.values()), field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(*list(datasets.values()), field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(*list(datasets.values()), field_name='target', new_field_name='target')

    # for k,v in datasets.items():
    #     v.set_input('chars','bigrams','seq_len','target')
    #     v.set_target('target','seq_len')

    vocabs['bigram'] = bigram_vocab

    # embeddings = {}
    #
    # if unigram_embedding_path is not None:
    #     unigram_embedding = StaticEmbedding(char_vocab, model_dir_or_name=unigram_embedding_path,
    #                                         word_dropout=char_word_dropout,
    #                                         min_freq=char_min_freq, only_train_min_freq=only_train_min_freq, )
    #     embeddings['char'] = unigram_embedding
    #
    # if bigram_embedding_path is not None:
    #     bigram_embedding = StaticEmbedding(bigram_vocab, model_dir_or_name=bigram_embedding_path,
    #                                        word_dropout=0.01,
    #                                        min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq)
    #     embeddings['bigram'] = bigram_embedding

    return datasets, vocabs


@cache_results(_cache_fp='cache/load_yangjie_rich_pretrain_word_list', _refresh=False)
def load_yangjie_rich_pretrain_word_list(embedding_path, drop_characters=True):
    f = open(embedding_path, 'r', encoding='utf-8')
    lines = f.readlines()
    w_list = []
    for line in lines:
        splited = line.strip().split(' ')
        w = splited[0]
        w_list.append(w)

    if drop_characters:
        w_list = list(filter(lambda x: len(x) != 1, w_list))

    return w_list


@cache_results(_cache_fp='need_to_defined_fp', _refresh=True)
def equip_chinese_ner_with_lexicon(datasets, vocabs, w_list, word_embedding_path=None,
                                   only_lexicon_in_train=False, word_char_mix_embedding_path=None,
                                   number_normalized=False,
                                   lattice_min_freq=1, only_train_min_freq=0):
    def normalize_char(inp):
        result = []
        for c in inp:
            if c.isdigit():
                result.append('0')
            else:
                result.append(c)

        return result

    def normalize_bigram(inp):
        result = []
        for bi in inp:
            tmp = bi
            if tmp[0].isdigit():
                tmp = '0' + tmp[:1]
            if tmp[1].isdigit():
                tmp = tmp[0] + '0'

            result.append(tmp)
        return result

    if number_normalized == 3:
        for k, v in datasets.items():
            v.apply_field(normalize_char, 'chars', 'chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                    no_create_entry_dataset=[datasets['dev'], datasets['test']])

        for k, v in datasets.items():
            v.apply_field(normalize_bigram, 'bigrams', 'bigrams')
        vocabs['bigram'] = Vocabulary()
        vocabs['bigram'].from_dataset(datasets['train'], field_name='bigrams',
                                      no_create_entry_dataset=[datasets['dev'], datasets['test']])

    if only_lexicon_in_train:
        print('已支持只加载在trian中出现过的词汇')

    def get_skip_path(chars, w_trie):
        sentence = ''.join(chars)
        result = w_trie.get_lexicon(sentence)
        # print(result)

        return result

    a = DataSet()
    w_trie = Trie()
    for w in w_list:
        w_trie.insert(w)

    if only_lexicon_in_train:
        lexicon_in_train = set()
        for s in datasets['train']['chars']:
            lexicon_in_s = w_trie.get_lexicon(s)
            for s, e, lexicon in lexicon_in_s:
                lexicon_in_train.add(''.join(lexicon))

        print('lexicon in train:{}'.format(len(lexicon_in_train)))
        print('i.e.: {}'.format(list(lexicon_in_train)[:10]))
        w_trie = Trie()
        for w in lexicon_in_train:
            w_trie.insert(w)

    for k, v in datasets.items():
        v.apply_field(partial(get_skip_path, w_trie=w_trie), 'chars', 'lexicons')
        v.apply_field(copy.copy, 'chars', 'raw_chars')
        v.add_seq_len('lexicons', 'lex_num')
        v.apply_field(lambda x: list(map(lambda y: y[0], x)), 'lexicons', 'lex_s')
        v.apply_field(lambda x: list(map(lambda y: y[1], x)), 'lexicons', 'lex_e')

    if number_normalized == 1:
        for k, v in datasets.items():
            v.apply_field(normalize_char, 'chars', 'chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                    no_create_entry_dataset=[datasets['dev'], datasets['test']])

    if number_normalized == 2:
        for k, v in datasets.items():
            v.apply_field(normalize_char, 'chars', 'chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                    no_create_entry_dataset=[datasets['dev'], datasets['test']])

        for k, v in datasets.items():
            v.apply_field(normalize_bigram, 'bigrams', 'bigrams')
        vocabs['bigram'] = Vocabulary()
        vocabs['bigram'].from_dataset(datasets['train'], field_name='bigrams',
                                      no_create_entry_dataset=[datasets['dev'], datasets['test']])

    def concat(ins):
        chars = ins['chars']
        lexicons = ins['lexicons']
        result = chars + list(map(lambda x: x[2], lexicons))
        return result

    def get_pos_s(ins):
        lex_s = ins['lex_s']
        seq_len = ins['seq_len']
        pos_s = list(range(seq_len)) + lex_s

        return pos_s

    def get_pos_e(ins):
        lex_e = ins['lex_e']
        seq_len = ins['seq_len']
        pos_e = list(range(seq_len)) + lex_e

        return pos_e

    for k, v in datasets.items():
        v.apply(concat, new_field_name='lattice')
        v.set_input('lattice')
        v.apply(get_pos_s, new_field_name='pos_s')
        v.apply(get_pos_e, new_field_name='pos_e')
        v.set_input('pos_s', 'pos_e')

    word_vocab = Vocabulary()
    word_vocab.add_word_lst(w_list)
    vocabs['word'] = word_vocab

    lattice_vocab = Vocabulary()
    lattice_vocab.from_dataset(datasets['train'], field_name='lattice',
                               no_create_entry_dataset=[v for k, v in datasets.items() if k != 'train'])
    vocabs['lattice'] = lattice_vocab

    # if word_embedding_path is not None:
    #     word_embedding = StaticEmbedding(word_vocab, word_embedding_path, word_dropout=0)
    #     embeddings['word'] = word_embedding
    #
    # if word_char_mix_embedding_path is not None:
    #     lattice_embedding = StaticEmbedding(lattice_vocab, word_char_mix_embedding_path, word_dropout=0.01,
    #                                         min_freq=lattice_min_freq, only_train_min_freq=only_train_min_freq)
    #     embeddings['lattice'] = lattice_embedding

    vocabs['char'].index_dataset(*(datasets.values()),
                                 field_name='chars', new_field_name='chars')
    vocabs['bigram'].index_dataset(*(datasets.values()),
                                   field_name='bigrams', new_field_name='bigrams')
    vocabs['label'].index_dataset(*(datasets.values()),
                                  field_name='target', new_field_name='target')
    vocabs['lattice'].index_dataset(*(datasets.values()),
                                    field_name='lattice', new_field_name='lattice')

    return datasets, vocabs



def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1， 1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解本文。
    参考：https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()
