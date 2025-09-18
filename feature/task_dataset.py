import os
import json
import random
import time

import numpy
import torch
import numpy as np
from torch.utils.data import Dataset
# from multiprocess import Pool
from function.preprocess import sent_to_matched_words_boundaries, get_pretrain_char_and_word
from function.utils import load_ontonotes5ner, load_resume_ner, load_weibo_ner, load_msra_ner_1, \
    load_yangjie_rich_pretrain_word_list, equip_chinese_ner_with_lexicon
from wcbert_parser import get_argparse

random.seed(106524)
args = get_argparse().parse_args()


class TaskDataset(Dataset):
    def __init__(self, file, params, do_shuffle=False):
        """
        Args:
            file: data file
            params: 1.vocab.txt, tokenizer
                    2.word_vocab
                    3.label_vocab
                    4.max_word_num
                    5.max_scan_num
                    6.max_seq_length
        """
        self.max_word_num = params['max_word_num']
        self.tokenizer = params['tokenizer']
        self.label_vocab = params['label_vocab']
        self.word_vocab = params['word_vocab']
        self.lexicon_tree = params['lexicon_tree']
        self.max_scan_num = params['max_scan_num']
        self.max_seq_length = params['max_seq_length']
        self.default_label = params['default_label']
        self.do_shuffle = do_shuffle

        self.file = file
        file_items = file.split("/")
        item = file_items[-1].split("\\")
        file_items[-1] = item[0]
        file_items.append(item[1])
        data_dir = "/".join(file_items[:-1])

        file_name = "saved_maxword_{}_maxseq_{}_".format(self.max_word_num, self.max_seq_length) + \
                    file_items[-1].split('.')[0] + "_{}.npz".format(self.max_scan_num)
        saved_np_file = os.path.join(data_dir, file_name)
        self.np_file = saved_np_file

        self.init_np_dataset()

    def init_np_dataset(self):
        """
        generate np file, accumulate the read speed.
        we need
            2. tokenizer
            3. word_vocab
            4. label vocab
            5. max_scan_num
            6, max_word_num
        """
        print_flag = True
        if os.path.exists(self.np_file):
            with np.load(self.np_file) as dataset:
                self.input_ids = dataset["input_ids"]
                self.segment_ids = dataset["segment_ids"]
                self.attention_mask = dataset["attention_mask"]
                self.input_matched_word_ids = dataset["input_matched_word_ids"]
                self.input_matched_word_mask = dataset["input_matched_word_mask"]
                self.input_boundary_ids = dataset["input_boundary_ids"]
                self.labels = dataset["labels"]
                # self.all_text = dataset["all_text"]
            print("核对%s中id和词是否匹配: " % (self.file))
            print(self.input_ids[0][:10])
            print(self.tokenizer.convert_ids_to_tokens(self.input_ids[0][:10]))
            for idx in range(10):
                print(self.input_matched_word_ids[0][idx])
                print(self.word_vocab.convert_ids_to_items(self.input_matched_word_ids[0][idx]))

        else:
            all_input_ids = []
            all_segment_ids = []
            all_attention_mask = []
            all_input_matched_word_ids = []
            all_input_matched_word_mask = []
            all_input_boundary_ids = []
            all_labels = []

            with open(self.file, 'r', encoding='utf-8') as f:
                f = f.read()
                # print(f)
                sample = json.loads(f)
                # print(sample)
                for s in sample:
                    # s = s.strip()
                    if s:
                        text = s['text']
                        labels = s['labels']
                        label = []
                        for lab in labels:
                            # label.append([lab[2]+1, lab[3], lab[1]])
                            label.append([lab[2], lab[3], lab[1]])
                        tokens = [i for i in text]
                        if len(tokens) > self.max_seq_length - 2:
                        # if len(tokens) > self.max_seq_length:
                            tokens = tokens[:self.max_seq_length - 2]
                            # tokens = tokens[:self.max_seq_length]
                            # label = label[:self.max_seq_length - 2]
                        # text.insert(0, '[CLS]')
                        # label.insert(0, self.default_label)
                        # text.append('[SEP]')
                        tokens = ['[CLS]'] + tokens + ['[SEP]']
                        # label.append(self.default_label)
                        # all_text.append(tokens)

                        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                        input_ids = np.zeros(self.max_seq_length, dtype=np.int)
                        segment_ids = np.ones(self.max_seq_length, dtype=np.int)
                        attention_mask = np.zeros(self.max_seq_length, dtype=np.int)
                        matched_word_ids = np.zeros((self.max_seq_length, self.max_word_num), dtype=np.int)
                        matched_word_mask = np.zeros((self.max_seq_length, self.max_word_num), dtype=np.int)
                        boundary_ids = np.zeros(self.max_seq_length, dtype=np.int)
                        np_label = np.zeros((len(self.label_vocab.item2idx), self.max_seq_length, self.max_seq_length), dtype=np.float16)
                        for start, end, label in label:
                            # 排除SEP及之后的
                            if end >= self.max_seq_length - 1:
                                continue
                            label_id = self.label_vocab.item2idx[label]
                            np_label[label_id, start, end] = 1

                        # token_ids, segment_ids, attention_mask
                        input_ids[:len(token_ids)] = token_ids
                        segment_ids[:len(token_ids)] = 0
                        attention_mask[:len(token_ids)] = 1

                        # matched word, boubdary
                        matched_words, sent_boundaries = \
                            sent_to_matched_words_boundaries(tokens, self.lexicon_tree, self.max_word_num)
                        sent_length = len(tokens)
                        boundary_ids[:len(sent_boundaries)] = sent_boundaries
                        for idy in range(sent_length):
                            now_words = matched_words[idy]
                            now_word_ids = self.word_vocab.convert_items_to_ids(now_words)
                            matched_word_ids[idy][:len(now_word_ids)] = now_word_ids
                            matched_word_mask[idy][:len(now_word_ids)] = 1

                        if print_flag:
                            print("核对%s中id和词是否匹配: " % (self.file))
                            print(input_ids[:10])
                            print(self.tokenizer.convert_ids_to_tokens(input_ids[:10]))
                            for idx in range(10):
                                print(matched_word_ids[idx])
                                print(self.word_vocab.convert_ids_to_items(matched_word_ids[idx]))

                            print(matched_words)
                            print(matched_words[:10])
                            print(matched_word_ids[:10])
                            print_flag = False

                        all_input_ids.append(input_ids)
                        all_segment_ids.append(segment_ids)
                        all_attention_mask.append(attention_mask)
                        all_input_matched_word_ids.append(matched_word_ids)
                        all_input_matched_word_mask.append(matched_word_mask)
                        all_input_boundary_ids.append(boundary_ids)
                        all_labels.append(np_label)

            assert len(all_input_ids) == len(all_segment_ids), (len(all_input_ids), len(all_segment_ids))
            assert len(all_input_ids) == len(all_attention_mask), (len(all_input_ids), len(all_attention_mask))
            assert len(all_input_ids) == len(all_input_matched_word_ids), (
                len(all_input_ids), len(all_input_matched_word_ids))
            assert len(all_input_ids) == len(all_input_matched_word_mask), (
                len(all_input_ids), len(all_input_matched_word_mask))
            assert len(all_input_ids) == len(all_input_boundary_ids), (len(all_input_ids), len(all_input_boundary_ids))
            assert len(all_input_ids) == len(all_labels), (len(all_input_ids), len(all_labels))

            all_input_ids = np.array(all_input_ids)
            all_segment_ids = np.array(all_segment_ids)
            all_attention_mask = np.array(all_attention_mask)
            all_input_matched_word_ids = np.array(all_input_matched_word_ids)
            all_input_matched_word_mask = np.array(all_input_matched_word_mask)
            all_input_boundary_ids = np.array(all_input_boundary_ids)
            all_labels = np.array(all_labels)
            np.savez(
                self.np_file, input_ids=all_input_ids, segment_ids=all_segment_ids, attention_mask=all_attention_mask,
                input_matched_word_ids=all_input_matched_word_ids, input_matched_word_mask=all_input_matched_word_mask,
                input_boundary_ids=all_input_boundary_ids, labels=all_labels
            )

            self.input_ids = all_input_ids
            self.segment_ids = all_segment_ids
            self.attention_mask = all_attention_mask
            self.input_matched_word_ids = all_input_matched_word_ids
            self.input_matched_word_mask = all_input_matched_word_mask
            self.input_boundary_ids = all_input_boundary_ids
            self.labels = all_labels
            # self.all_text = all_text

        self.total_size = self.input_ids.shape[0]
        self.indexes = list(range(self.total_size))
        if self.do_shuffle:
            random.shuffle(self.indexes)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        index = self.indexes[index]
        return (
            torch.tensor(self.input_ids[index]),
            torch.tensor(self.segment_ids[index]),
            torch.tensor(self.attention_mask[index]),
            torch.tensor(self.input_matched_word_ids[index]),
            torch.tensor(self.input_matched_word_mask[index]),
            torch.tensor(self.input_boundary_ids[index]),
            torch.tensor(self.labels[index])
        )


def radical_task_dataset():
    # path
    # pretrain_unigram_path = 'data/dataset/NER/gigaword_chn.all.a2b.uni.ite50.vec'
    # pretrain_bigram_path = 'data/dataset/NER/gigaword_chn.all.a2b.bi.ite50.vec'
    pretrain_word_path = 'data/dataset/NER/ctb.50d.vec'
    # this path is for the output of preprocessing
    pretrain_char_and_word_path = 'data/dataset/NER/word_char_mix.txt'

    raw_dataset_cache_name = os.path.join('cache', args.dataset +
                                          '_trainClip_{}'.format(args.train_clip)
                                          + 'bgminfreq_{}'.format(args.bigram_min_freq)
                                          + 'char_min_freq_{}'.format(args.char_min_freq)
                                          + 'word_min_freq_{}'.format(args.word_min_freq)
                                          + 'only_train_min_freq{}'.format(args.only_train_min_freq)
                                          + 'number_norm{}'.format(args.number_normalized)
                                          + 'load_dataset_seed{}'.format(100)
                                          )

    # 加载数据集、vocabulary、embedding
    if args.dataset == 'note5':
        raw_datasets, raw_vocabs = load_ontonotes5ner(args.data_dir, _refresh=False, index_token=False,
                                                                      train_clip=args.train_clip,
                                                                      _cache_fp=raw_dataset_cache_name,
                                                                      char_min_freq=args.char_min_freq,
                                                                      bigram_min_freq=args.bigram_min_freq,
                                                                      only_train_min_freq=args.only_train_min_freq
                                                                      )
        print('----------------------')
        # print(raw_datasets['train'])
        with open('data/dataset/NER/note5/text.txt', 'w', encoding='utf-8') as f:
            for l in raw_datasets['train']['chars'].content:
                for word in l:
                    f.write(word)
                f.write('\n')
        print('----------------------')
    elif args.dataset == 'resume':
        raw_datasets, raw_vocabs = load_resume_ner(args.data_dir, _refresh=False, index_token=False,
                                                                   _cache_fp=raw_dataset_cache_name,
                                                                   char_min_freq=args.char_min_freq,
                                                                   bigram_min_freq=args.bigram_min_freq,
                                                                   only_train_min_freq=args.only_train_min_freq
                                                                   )
    elif args.dataset == 'weibo':
        if args.label == 'ne':
            raw_dataset_cache_name = 'ne' + raw_dataset_cache_name
        elif args.label == 'nm':
            raw_dataset_cache_name = 'nm' + raw_dataset_cache_name
        raw_datasets, raw_vocabs = load_weibo_ner(args.data_dir, _refresh=False, index_token=False,
                                                                  _cache_fp=raw_dataset_cache_name,
                                                                  char_min_freq=args.char_min_freq,
                                                                  bigram_min_freq=args.bigram_min_freq,
                                                                  only_train_min_freq=args.only_train_min_freq
                                                                  )
    elif args.dataset == 'msra':
        raw_datasets, raw_vocabs = load_msra_ner_1(args.data_dir, _refresh=False, index_token=False,
                                                                   train_clip=args.train_clip,
                                                                   _cache_fp=raw_dataset_cache_name,
                                                                   char_min_freq=args.char_min_freq,
                                                                   bigram_min_freq=args.bigram_min_freq,
                                                                   only_train_min_freq=args.only_train_min_freq
                                                                   )

    cache_name = os.path.join('cache', (args.dataset + '_lattice_only_train_{}' +
                                        '_trainClip_{}' + '_norm_num_{}'
                                        + 'char_min_freq_{}' + 'bigram_min_freq_{}' + 'word_min_freq_{}' + 'only_train_min_freq_{}'
                                        + 'number_norm_{}' + 'lexicon_{}' + 'load_dataset_seed_{}')
                              .format(args.only_lexicon_in_train,
                                      args.train_clip, args.number_normalized, args.char_min_freq,
                                      args.bigram_min_freq, args.word_min_freq, args.only_train_min_freq,
                                      args.number_normalized, args.lexicon_name, 100))
    if args.dataset == 'weibo':
        if args.label == 'ne':
            cache_name = 'ne' + cache_name
        elif args.label == 'nm':
            cache_name = 'nm' + cache_name

    # 加载预训练数据中的中文字符
    w_list = load_yangjie_rich_pretrain_word_list(pretrain_word_path,
                                                  _refresh=False, _cache_fp='cache/{}'.format(args.lexicon_name))
    #  预处理
    # get_pretrain_char_and_word(pretrain_word_path, pretrain_unigram_path, pretrain_char_and_word_path)

    datasets, vocabs = equip_chinese_ner_with_lexicon(raw_datasets, raw_vocabs,
                                                                  w_list, pretrain_word_path,
                                                                  _refresh=False, _cache_fp=cache_name,
                                                                  only_lexicon_in_train=args.only_lexicon_in_train,
                                                                  word_char_mix_embedding_path=pretrain_char_and_word_path,
                                                                  number_normalized=args.number_normalized,
                                                                  lattice_min_freq=args.lattice_min_freq,
                                                                  only_train_min_freq=args.only_train_min_freq)

    for k, v in datasets.items():
        if args.lattice:
            v.set_input('lattice', 'bigrams', 'seq_len', 'target')
            v.set_input('lex_num', 'pos_s', 'pos_e')
            v.set_target('target', 'seq_len')
        else:
            v.set_input('chars', 'bigrams', 'seq_len', 'target')
            v.set_target('target', 'seq_len')

    return datasets, vocabs
