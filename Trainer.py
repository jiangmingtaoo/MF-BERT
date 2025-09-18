import logging
import os
import random
import time
from collections import defaultdict
from itertools import zip_longest

from apex.amp import scaler
from packaging import version

import torch
import numpy as np
import torch.distributed as dist
from thop import profile, clever_format
from torch.cuda.amp import autocast
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler
from tqdm.auto import tqdm, trange

from transformers.configuration_bert import BertConfig
from transformers.tokenization_bert import BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup, AdamW, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup

from module.GlobalPointer import GlobalPointer
from wcbert_parser import get_argparse
from wcbert_modeling import WCBertCRFForTokenClassification, BertWordLSTMCRFForTokenClassification
from module.sampler import SequentialDistributedSampler
from feature.task_dataset import TaskDataset, radical_task_dataset
from feature.vocab import ItemVocabFile, ItemVocabArray
from function.metrics import seq_f1_with_mask
from function.preprocess import build_lexicon_tree_from_vocabs, get_corpus_matched_word_from_lexicon_tree
from function.utils import build_pretrained_embedding_for_corpus, save_preds_for_seq_labelling, \
    multilabel_categorical_crossentropy

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        print("No Tensorboard Found!!!")

### to enable fp16 training, note pytorch >= 1.16.0 #########
from apex import amp

_use_apex = True
_use_native_amp = False

###### for multi-gpu DistributedDataParallel training  #########
os.environ['NCCL_DEBUG'] = 'INFO'  # print more detailed NCCL log information
os.environ['NCCL_IB_DISABLE'] = '1'  # force IP sockets usage
# os.environ['RANK'] = '0'
# os.environ['MASTER_PORT'] = '5678'

# set logger, print to console and write to file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()  # 输出到控制台的handler
chlr.setFormatter(formatter)
logfile = './data/dataset/NER/weibo/log/log_file_{}.txt'.format(time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time())))
fh = logging.FileHandler(logfile)
fh.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fh)

PREFIX_CHECKPOINT_DIR = "checkpoint"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloader(dataset, args, mode='train'):
    """
    generator datasetloader for training.
    Note that: for training, we need random sampler, same to shuffle
               for eval or predict, we need sequence sampler, same to no shuffle
    Args:
        dataset:
        args:
        mode: train or non-train
    """
    # print("Dataset length: ", len(dataset))
    if args.local_rank != -1:
        if mode == 'train':
            sampler = SequentialDistributedSampler(dataset, do_shuffle=True)
        else:
            sampler = SequentialDistributedSampler(dataset)
    else:
        if mode == 'train':
            sampler = SequentialSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
    if mode == 'train':
        batch_size = args.per_gpu_train_batch_size
    else:
        batch_size = args.per_gpu_eval_batch_size

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )

    return data_loader


def get_optimizer(model, args, num_training_steps):
    """
    Setup the optimizer and the learning rate scheduler

    we provide a reasonable default that works well
    If you want to use something else, you can pass a tuple in the Trainer's init,
    or override this method in a subclass.
    """
    no_bigger = ["word_embedding", "attn_w", "word_transform", "word_word_weight", "hidden2tag",
                 "lstm", "crf"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_bigger)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_bigger)],
            "lr": 0.0001
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    return optimizer, scheduler


def print_log(logs, epoch, global_step, eval_type, tb_writer, iterator=None):
    if epoch is not None:
        logs['epoch'] = epoch
    if global_step is None:
        global_step = 0
    if eval_type in ["Dev", "Test"]:
        print("#############  %s's result  #############" % (eval_type))
    if tb_writer:
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                tb_writer.add_scalar(k, v, global_step)
            else:
                logger.warning(
                    "Trainer is attempting to log a value of "
                    '"%s" of type %s for key "%s" as a scalar. '
                    "This invocation of Tensorboard's writer.add_scalar() "
                    "is incorrect so we dropped this attribute.",
                    v,
                    type(v),
                    k,
                )
        tb_writer.flush()

    output = {**logs, **{"step": global_step}}
    if iterator is not None:
        iterator.write(output)
    else:
        logger.info(output)


def loss_fun(y_true, y_pred):
    """
    loss func
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size, seq_len = y_pred.shape[:3]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss


def train(model, args, train_dataset, dev_dataset, test_dataset, label_vocab, tb_writer, datasets, device,
               model_path=None):
    ## 1.prepare data

    train_dataloader = get_dataloader(train_dataset, args, mode='train')

    data = datasets['train']['lattice']
    max_len = args.max_seq_length
    radical_mask = np.ones((len(data), max_len), dtype=np.int32)
    result = np.zeros((len(data), max_len), dtype=np.int32)
    for i, row in enumerate(data):
        if len(row) > max_len:
            row = row[:max_len - 1]
        result[i][:len(row)] = row
        radical_mask[i][:len(row)] = 0
    radical_train_dataloader = get_dataloader(result, args, mode='train')
    radical_mask_dataloader = get_dataloader(radical_mask, args, mode='train')

    seq_length = torch.Tensor(datasets['train']['seq_len'])
    seq_length = get_dataloader(seq_length, args, mode='train')

    if args.max_steps > 0:
        t_total = args.max_steps
        num_train_epochs = (
                args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1)
    else:
        t_total = int(len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs)
        num_train_epochs = args.num_train_epochs

    ## 2.optimizer and model
    optimizer, scheduler = get_optimizer(model, args, t_total)

    if args.fp16 and _use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Check if saved optimizer or scheduler states exist
    if (model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))):
        optimizer.load_state_dict(
            torch.load(os.path.join(model_path, "optimizer.pt"), map_location=args.device)
        )
        scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

    if args.local_rank != -1:
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    ## 3.begin train
    total_train_batch_size = args.per_gpu_train_batch_size * args.gradient_accumulation_steps
    if args.local_rank == 0 or args.local_rank == -1:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader.dataset))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epoch = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    if model_path is not None:  # load checkpoint and continue training
        try:
            global_step = int(model_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
            )
            model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            global_step = 0
            logger.info("  Starting fine-tuning.")

    total_loss = 0.0
    logging_loss = 0.0
    # for batch_ind, batch_data in pbar:
    train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch")
    for epoch in train_iterator:
        if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        radical_epoch_iterator = tqdm(radical_train_dataloader, desc="radical_Iteration")
        for (step, batch), (radical_step, radical_batch), seq_len, (mask_step, radical_level_mask) in zip_longest(
                enumerate(epoch_iterator),
                enumerate(radical_epoch_iterator),
                seq_length, enumerate(radical_mask_dataloader)):
            if steps_trained_in_current_epoch > 0:
                # Skip past any already trained steps if resuming training
                steps_trained_in_current_epoch -= 1
                continue
            model.train()

            batch_data = (batch[0], batch[2], batch[1], batch[3], batch[4], batch[5], batch[6])
            new_batch = batch_data
            batch = tuple(t.to(args.device) for t in new_batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                      "matched_word_ids": batch[3], "matched_word_mask": batch[4], "boundary_ids": batch[5],
                      "labels": batch[6], "flag": "Train", "lattice": radical_batch, "seq_len": seq_len,
                      "radical_mask": radical_level_mask}
            batch_data = None
            new_batch = None

            if args.fp16 and _use_native_amp:
                with autocast():
                    logits = model(**inputs)
                    loss = loss_fun(batch[6], logits)
            else:
                logits = model(**inputs)
                loss = loss_fun(batch[6], logits)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16 and _use_native_amp:
                scaler.scale(loss).backward()
            elif args.fp16 and _use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            total_loss += loss.item()

            ## update gradient
            if (step + 1) % args.gradient_accumulation_steps == 0 or ((step + 1) == len(epoch_iterator)):
                if args.fp16 and _use_native_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                elif args.fp16 and _use_apex:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if args.fp16 and _use_native_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                model.zero_grad()
                global_step += 1

                ## logger and evaluate
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # logs = {"loss": avg_loss, "learning_rate": optimizer.param_groups[0]['lr']}

                    logs = {"loss": (total_loss - logging_loss) / args.logging_steps, "learning_rate": (
                        scheduler.get_last_lr()[0]
                        if version.parse(torch.__version__) >= version.parse("1.4")
                        else scheduler.get_lr()[0]
                    )}
                    # backward compatibility for pytorch schedulers
                    logging_loss = total_loss
                    if args.local_rank == 0 or args.local_rank == -1:
                        print_log(logs, epoch, global_step, "", tb_writer)

                ## save checkpoint
                if False and args.save_steps > 0 and global_step % args.save_steps == 0 and \
                        (args.local_rank == 0 or args.local_rank == -1):
                    # output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
                    # os.makedirs(output_dir, exist_ok=True)
                    #
                    # torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                    if False and args.evaluate_during_training:
                        # for dev
                        if "msra" in args.data_dir:
                            final_metrics = evaluate(model, args, dev_dataset, datasets['test'], label_vocab,
                                                          global_step, device, description="Dev", write_file=True)
                        else:
                            final_metrics = evaluate(model, args, dev_dataset, datasets['dev'], label_vocab,
                                                          global_step, device, description="Dev", write_file=True)
                        if args.local_rank == 0 or args.local_rank == -1:
                            print_log(final_metrics, epoch, global_step, "Dev", tb_writer)

                        # for test
                        final_metrics = evaluate(
                            model, args, test_dataset, datasets['test'], label_vocab, global_step, device,
                            description="Test")
                        if args.local_rank == 0 or args.local_rank == -1:
                            print_log(final_metrics, epoch, global_step, "Test", tb_writer)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        # save after each epoch
        # output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
        # os.makedirs(output_dir, exist_ok=True)
        #
        # torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        # evaluate after each epoch
        if args.evaluate_during_training:
            # for dev
            if "msra" in args.data_dir:
                final_metrics = evaluate(model, args, dev_dataset, datasets['test'], label_vocab, global_step,
                                              device, description="Dev", write_file=True)
            else:
                final_metrics = evaluate(model, args, dev_dataset, datasets['dev'], label_vocab, global_step,
                                              device, description="Dev", write_file=True)
            if args.local_rank == 0 or args.local_rank == -1:
                print_log(final_metrics, epoch, global_step, "Dev", tb_writer)

            # for test
            final_metrics = evaluate(model, args, test_dataset, datasets['test'], label_vocab, global_step, device,
                                          description="Test", write_file=True)
            if args.local_rank == 0 or args.local_rank == -1:
                print_log(final_metrics, epoch, global_step, "Test", tb_writer)

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    # save the last one
    # output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
    # os.makedirs(output_dir, exist_ok=True)
    # torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model .bin"))
    #
    # # model.save_pretrained(os.path.join(output_dir, "pytorch-model.bin"))
    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    print("global_step: ", global_step)
    logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    return global_step, total_loss / global_step


def evaluate(model, args, dataset, radical_dataset, label_vocab, global_step, device, description="dev",
                  write_file=False):
    dataloader = get_dataloader(dataset, args, mode='dev')

    data = radical_dataset['lattice']
    max_len = args.max_seq_length
    result = np.zeros((len(data), max_len))
    radical_mask = np.ones((len(data), max_len), dtype=np.int32)
    for i, row in enumerate(data):
        if len(row) > max_len:
            row = row[:max_len - 1]
        result[i][:len(row)] = row
        radical_mask[i][:len(row)] = 0
    radical_dev_dataloader = get_dataloader(result, args, mode='dev')
    radical_mask_dataloader = get_dataloader(radical_mask, args, mode='dev')

    seq_length = torch.Tensor(radical_dataset['seq_len'])
    seq_length = get_dataloader(seq_length, args, mode='dev')

    if (not args.do_train) and (not args.no_cuda) and args.local_rank != -1:
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    batch_size = dataloader.batch_size
    if args.local_rank == 0 or args.local_rank == -1:
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", len(dataloader.dataset))
        logger.info("  Batch size = %d", batch_size)

    model.eval()

    all_label_ids = None
    all_predict_ids = None
    all_attention_mask = None

    for batch, radical_batch, seq_len, radical_level_mask in zip_longest(tqdm(dataloader, desc=description),
                                                                         tqdm(radical_dev_dataloader,
                                                                              desc="radical_" + description),
                                                                         seq_length, radical_mask_dataloader):

        batch_data = (batch[0], batch[2], batch[1], batch[3], batch[4], batch[5], batch[6])
        new_batch = batch_data
        batch = tuple(t.to(args.device) for t in new_batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                  "matched_word_ids": batch[3], "matched_word_mask": batch[4],
                  "boundary_ids": batch[5], "labels": batch[6], "flag": "Predict", "lattice": radical_batch,
                  "seq_len": seq_len}
        with torch.no_grad():
            batch_logits = model(**inputs)

        y_pred = batch_logits.cpu().numpy()
        y_true = batch[6].cpu().numpy()

        all_pred_labels = np.full((len(y_pred), args.max_seq_length), len(label_vocab.item2idx)-1)
        all_true_labels = np.full((len(y_true), args.max_seq_length), len(label_vocab.item2idx)-1)

        for b, l, start, end in zip(*np.where(y_pred > 0)):
            label = np.full(end-start, l)
            all_pred_labels[b, start:end] = label

        for b, l, start, end in zip(*np.where(y_true > 0)):
            label = np.full(end-start, l)
            all_true_labels[b, start:end] = label

        attention_mask = batch[1].detach().cpu().numpy()
        if all_label_ids is None:
            all_label_ids = all_true_labels
            all_predict_ids = all_pred_labels
            all_attention_mask = attention_mask
        else:
            all_label_ids = np.append(all_label_ids, all_true_labels, axis=0)
            all_predict_ids = np.append(all_predict_ids, all_pred_labels, axis=0)
            all_attention_mask = np.append(all_attention_mask, attention_mask, axis=0)

    acc, p, r, f1, all_true_labels, all_pred_labels = seq_f1_with_mask(
        all_label_ids, all_predict_ids, all_attention_mask, label_vocab)

    final_metrics = {'acc': acc, 'p': p, 'r': r, 'f1': f1}

    return final_metrics


def main():
    args = get_argparse().parse_args()
    args.no_cuda = not torch.cuda.is_available()

    ########### for multi-gpu training ##############
    if torch.cuda.is_available() and args.local_rank != -1:
        args.n_gpu = 1
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='gloo', init_method='env://')
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    #################################################

    args.device = device
    logger.info(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    logger.info("Training/evaluation parameters %s", args)
    tb_writer = SummaryWriter(log_dir=args.logging_dir)
    set_seed(args.seed)

    ## 1.prepare data
    # a. lexicon tree
    lexicon_tree = build_lexicon_tree_from_vocabs([args.word_vocab_file], scan_nums=[args.max_scan_num])
    embed_lexicon_tree = lexicon_tree

    # b. word vocab, label vocab
    train_data_file = os.path.join(args.data_dir, "train.json")
    train_mid_data_file = os.path.join(args.mid_data_dir, "train.json")
    # if only has test_set no dev_set, such as msra NER
    if "msra" in args.data_dir:
        dev_data_file = os.path.join(args.data_dir, "test.json")
        dev_mid_data_file = os.path.join(args.mid_data_dir, "test.json")
    else:
        dev_data_file = os.path.join(args.data_dir, "dev.json")
        dev_mid_data_file = os.path.join(args.mid_data_dir, "dev.json")
    test_data_file = os.path.join(args.data_dir, "test.json")
    test_mid_data_file = os.path.join(args.mid_data_dir, "test.json")
    data_files = [train_data_file, dev_data_file, test_data_file]
    matched_words = get_corpus_matched_word_from_lexicon_tree(data_files, embed_lexicon_tree)
    word_vocab = ItemVocabArray(items_array=matched_words, is_word=True, has_default=False, unk_num=5)  # 匹配词的vocabulary
    label_vocab = ItemVocabFile(files=[args.mid_label_file], is_word=False)
    tokenizer = BertTokenizer.from_pretrained(args.vocab_file)

    with open("word_vocab.txt", "w", encoding="utf-8") as f:
        for idx, word in enumerate(word_vocab.idx2item):
            f.write("%d\t%s\n" % (idx, word))

    # c. prepare embedding
    pretrained_word_embedding, embed_dim = build_pretrained_embedding_for_corpus(
        embedding_path=args.word_embedding,
        word_vocab=word_vocab,
        embed_dim=args.word_embed_dim,
        max_scan_num=args.max_scan_num,
        saved_corpus_embedding_dir=args.saved_embedding_dir,
    )

    datasets, vocabs = radical_task_dataset()

    # d. define model
    config = BertConfig.from_pretrained(args.config_name)
    if args.model_type == "WCBertCRF_Token":
        model = WCBertCRFForTokenClassification.from_pretrained(
            args.model_name_or_path, config=config,
            pretrained_embeddings=pretrained_word_embedding,
            num_labels=label_vocab.get_item_size(), vocab=vocabs)
        encoder = model
        model = GlobalPointer(encoder, label_vocab.get_item_size(), config.hidden_size)
    elif args.model_type == "BertWordLSTMCRF_Token":
        model = BertWordLSTMCRFForTokenClassification.from_pretrained(
            args.model_name_or_path, config=config,
            pretrained_embeddings=pretrained_word_embedding,
            num_labels=label_vocab.get_item_size(), vocab=vocabs
        )
        encoder = model
        model = GlobalPointer(encoder, label_vocab.get_item_size(), config.hidden_size)

    if not args.no_cuda:
        model = model.cuda()
    args.label_size = label_vocab.get_item_size()
    dataset_params = {
        'tokenizer': tokenizer,
        'word_vocab': word_vocab,
        'label_vocab': label_vocab,
        'lexicon_tree': lexicon_tree,
        'max_seq_length': args.max_seq_length,
        'max_scan_num': args.max_scan_num,
        'max_word_num': args.max_word_num,
        'default_label': args.default_label,
    }

    model_name_or_path = args.model_name_or_path

    if args.do_train:
        args.model_name_or_path = None
        train_dataset = TaskDataset(train_mid_data_file, params=dataset_params, do_shuffle=args.do_shuffle)
        dev_dataset = TaskDataset(dev_mid_data_file, params=dataset_params, do_shuffle=False)
        test_dataset = TaskDataset(test_mid_data_file, params=dataset_params, do_shuffle=False)

        train(model, args, train_dataset, dev_dataset, test_dataset, label_vocab, tb_writer, datasets=datasets,
                   device=device)

    if args.do_eval:
        logger.info("*** Dev Evaluate ***")
        dev_dataset = TaskDataset(dev_mid_data_file, params=dataset_params, do_shuffle=False)
        global_steps = model_name_or_path.split("/")[-2].split("-")[-1]
        if "msra" in args.data_dir:
            eval_output = evaluate(model, args, dev_dataset, datasets['test'], label_vocab, global_steps, device,
                                        "dev", write_file=True)
        else:
            eval_output = evaluate(model, args, dev_dataset, datasets['dev'], label_vocab, global_steps, device,
                                        "dev", write_file=True)
        eval_output["global_steps"] = global_steps
        print("Dev Result: acc: %.4f, p: %.4f, r: %.4f, f1: %.4f\n" %
              (eval_output['acc'], eval_output['p'], eval_output['r'], eval_output['f1']))
        logger.info("Dev Result: acc: %.4f, p: %.4f, r: %.4f, f1: %.4f\n" %
                    (eval_output['acc'], eval_output['p'], eval_output['r'], eval_output['f1']))

    if args.do_predict:
        logger.info("*** Test Evaluate ***")
        test_dataset = TaskDataset(test_mid_data_file, params=dataset_params, do_shuffle=False)
        global_steps = model_name_or_path.split("/")[-2].split("-")[-1]
        eval_output = evaluate(model, args, test_dataset, datasets['test'], label_vocab, global_steps, device,
                                    "test",
                                    write_file=True)
        eval_output["global_steps"] = global_steps
        print("Test Result: acc: %.4f, p: %.4f, r: %.4f, f1: %.4f\n" %
              (eval_output['acc'], eval_output['p'], eval_output['r'], eval_output['f1']))
        logger.info("Test Result: acc: %.4f, p: %.4f, r: %.4f, f1: %.4f\n" %
                    (eval_output['acc'], eval_output['p'], eval_output['r'], eval_output['f1']))


if __name__ == "__main__":
    main()
