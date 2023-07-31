# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for text question-answering on MultimodalQA (DistilBERT, Bert, XLM, XLNet)."""
import sys

sys.path.append('../')
import argparse
import logging
import os
from torch.cuda.amp import autocast, GradScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from ofa import OFATokenizer

from OFA.datasets.generative_table_connect_dataset_webqa import generative_webqa_dataset, generative_webqa_dataset_test
from OFA.ofa.modeling_bart import BartForConditionalGeneration
from OFA.ofa.modeling_Fusion_bart import BartForTableFusion
from OFA.ofa.modeling_bart_OFA_re_source2cell_webqa import OFABart_AnsAcc
import errno

from torch.utils.tensorboard import SummaryWriter
from ofa.modeling_mmqa import OFAModel_MMQA
from utils.webqa_eval import webqa_metrics_approx

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
import string
from OFA.utils.BARTScore.bart_score import BARTScorer

TABLE = str.maketrans(dict.fromkeys(string.punctuation))


def normalize_text_for_bart(x):  # Light text normalization for WebQA eval: white space fix + punctuation removal
    return " ".join(x.translate(TABLE).split())


def compute_bartscore_ParaBank(c, a, switch=False):
    bart_scorer_ParaBank = BARTScorer(device='cuda', checkpoint='./bart-large-cnn')
    bart_scorer_ParaBank.load(
        path='./MMQA/OFA/utils/BARTScore/bart_score.pth')  # Please change the path to bart.pth
    c_removepunc = [normalize_text_for_bart(x) for x in c]
    a_removepunc = [normalize_text_for_bart(x) for x in a]
    if switch:
        score = np.exp(bart_scorer_ParaBank.score(c_removepunc, a_removepunc))
    else:
        score = np.exp(bart_scorer_ParaBank.score(a_removepunc, c_removepunc))
    return score


def mkdir(path):
    # if it is the current folder, skip.
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_latest_checkpoint(model, tokenizer, args, optimizer, scheduler, num_trial=10):
    checkpoint_dir = os.path.join(os.path.join(args.prefix_path, args.output_dir), 'checkpoint-latest')
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            checkpoint_dir_model = os.path.join(checkpoint_dir, "model.pth")
            torch.save(model_to_save.state_dict(), checkpoint_dir_model)
            torch.save(args, os.path.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            checkpoint_dir_op = os.path.join(checkpoint_dir, "optimizer.pth")
            torch.save(optimizer.state_dict(), checkpoint_dir_op)
            checkpoint_dir_sc = os.path.join(checkpoint_dir, "scheduler.pth")
            torch.save(scheduler.state_dict(), checkpoint_dir_sc)
            print("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir


def save_checkpoint(model, tokenizer, args, epoch, iteration, optimizer, scheduler, num_trial=10, QA_scores=0.0,
                    bart_scores=0.0,
                    precision=0.0, recall=0.0, ACC=0.0, span_pre=0.0, span_re=0.0, connect_re=0.0, connect_sc=0.0,
                    res_str=None):
    checkpoint_dir = os.path.join(os.path.join(args.prefix_path, args.output_dir),
                                  'checkpoint-{}-{}-QA-{:.4f}-ACC-{:.4f}-BART-{:.4f}-PRE-{:.4f}-REC-{:.4f}'.format(
                                      epoch,
                                      iteration, QA_scores, ACC,
                                      bart_scores,
                                      precision,
                                      recall))

    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            checkpoint_dir_model = os.path.join(checkpoint_dir, "model.pth")
            torch.save(model_to_save.state_dict(), checkpoint_dir_model)
            torch.save(args, os.path.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            checkpoint_dir_op = os.path.join(checkpoint_dir, "optimizer.pth")
            torch.save(optimizer.state_dict(), checkpoint_dir_op)
            checkpoint_dir_sc = os.path.join(checkpoint_dir, "scheduler.pth")
            torch.save(scheduler.state_dict(), checkpoint_dir_sc)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    if res_str is not None:
        file_name = 'MMQA-QA-{:.4f}-ACC-{:.4f}-BART-{:.4f}-span_pre-{:.4f}-span_re-{:.4f}-connect_sc-{:.4f}-connect_re-{:.4f}.json'.format(
            QA_scores, ACC, bart_scores, span_pre, span_re, connect_sc, connect_re)
        with open(os.path.join(checkpoint_dir, file_name), 'w') as json_file:
            json_file.write(res_str)
        print('写入完成')
    return checkpoint_dir


def build_dataloader(dataset, is_train, opts):
    if is_train:
        dataloader = DataLoader(dataset, drop_last=True, batch_size=opts.per_gpu_train_batch_size * opts.n_gpu,
                                num_workers=opts.num_workers,
                                shuffle=True, collate_fn=dataset.mmqa_collate)
    else:
        dataloader = DataLoader(dataset, batch_size=opts.per_gpu_eval_batch_size,
                                num_workers=opts.num_workers, shuffle=False, collate_fn=dataset.mmqa_collate)
    return dataloader


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[
                args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    train_dataset = generative_webqa_dataset(tokenizer, args.example_file_train,
                                             args.text_dir,
                                             args.img_dir, args.table_path, args.connect_path, args.span_path,
                                             args.prefix_path, args.doc_stride, args.max_img_length,
                                             args.max_query_length,
                                             args.max_seq_length, is_debug=args.is_debug)
    train_dataloader = build_dataloader(train_dataset, True, args)

    val_dataset = generative_webqa_dataset(tokenizer, args.example_file_eval,
                                           args.text_dir,
                                           args.img_dir, args.table_path, args.connect_path, args.span_path,
                                           args.prefix_path,
                                           args.doc_stride, args.max_img_length,
                                           args.max_query_length,
                                           args.max_seq_length, is_debug=args.is_debug, is_val=True)
    val_dataloader = build_dataloader(val_dataset,
                                      False, args)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
                len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader) * args.per_gpu_train_batch_size * args.n_gpu)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = args.global_step
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    if global_step > 0:
        model = model.module if hasattr(model, 'module') else model
        model_file = os.path.join(args.prefix_path, args.eval_model_name_or_path, 'model.pth')
        model.load_state_dict(torch.load(model_file))
        optimizer.load_state_dict(
            torch.load(os.path.join(args.prefix_path, args.eval_model_name_or_path, 'optimizer.pth')))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()  # an optimizer.cuda() method for this operation would be nice
        scheduler.load_state_dict(
            torch.load(os.path.join(args.prefix_path, args.eval_model_name_or_path, 'scheduler.pth')))
        logger.info("  Resume from %s", args.eval_model_name_or_path)
        model = torch.nn.DataParallel(model)
        epochs_trained = global_step // (len(train_dataloader) //
                                         args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps)

    tr_loss, logging_loss = 0.0, 0.0
    tr_gen_loss, logging_gen_loss = 0.0, 0.0
    tr_pointer_loss, logging_pointer_loss = 0.0, 0.0
    tr_re_loss, logging_re_loss = 0.0, 0.0
    tr_connect_loss, logging_connect_loss = 0.0, 0.0
    tr_similarity_loss, logging_similarity_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    model.train()
    scaler = GradScaler()

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs = {'input_ids': batch['input_ids'], 'input_mask': batch['input_mask'],
                      'input_token_type': batch['input_token_type'], 'batch_text_len': batch['batch_text_len'],
                      'table_text': batch['table_text'], 'table_mask': batch['table_mask'],
                      'table_token_type': batch['table_token_type'],
                      'image_token_type': batch['image_token_type'], 'image_feat': batch['image_feat'],
                      'image_mask': batch['image_mask'], 'batch_image_len': batch['batch_image_len'],
                      'image_ques_id': batch['image_ques_id'],
                      'ans_ids': batch['ans_ids'], 'attn_mask': batch['attn_mask'],
                      'ans_loss_mask': batch['ans_loss_mask'], 'sc_loss_mask': batch['sc_loss_mask'],
                      'golden_index': batch['golden_index'],
                      'table_connect_num': batch['table_connect_num'],
                      'flatten_connect_spans': batch['flatten_connect_spans'],
                      'flatten_connect_index': batch['flatten_connect_index'],
                      'table_cell_num': batch['table_cell_num'], 'flatten_cell_span': batch['flatten_cell_span'],
                      'gather_index': batch['gather_index'], 'span_label': batch['flatten_span_label']
                      }
            with autocast():
                gen_loss, pointer_loss, re_loss, connect_loss, similarity_loss = model(**inputs)

            # model outputs are always tuple in transformers (see doc)
            if args.n_gpu > 1:
                gen_loss = gen_loss.mean()
                pointer_loss = pointer_loss.mean()
                re_loss = re_loss.mean()
                connect_loss = connect_loss.mean()
                similarity_loss = similarity_loss.mean()

            loss = gen_loss + pointer_loss + re_loss + connect_loss + similarity_loss
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                gen_loss = gen_loss / args.gradient_accumulation_steps
                pointer_loss = pointer_loss / args.gradient_accumulation_steps
                re_loss = re_loss / args.gradient_accumulation_steps
                connect_loss = connect_loss / args.gradient_accumulation_steps
                similarity_loss = similarity_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                scaler.scale(loss).backward()
                # loss.backward()

            tr_loss += loss.item()
            tr_gen_loss += gen_loss.item()
            tr_pointer_loss += pointer_loss.item()
            tr_re_loss += re_loss.item()
            tr_connect_loss += connect_loss.item()
            tr_similarity_loss += similarity_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar(
                        "lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    print("global steps {}".format(global_step))
                    print("current loss: {}".format(
                        (tr_loss - logging_loss) / args.logging_steps))
                    logging_loss = tr_loss
                    print("current gen loss: {}".format(
                        (tr_gen_loss - logging_gen_loss) / args.logging_steps))
                    print("current pointer loss: {}".format(
                        (tr_pointer_loss - logging_pointer_loss) / args.logging_steps))
                    print("current re loss: {}".format(
                        (tr_re_loss - logging_re_loss) / args.logging_steps))
                    print("current source2cell bce loss: {}".format(
                        (tr_connect_loss - logging_connect_loss) / args.logging_steps))
                    print("current source2cell xe loss: {}".format(
                        (tr_similarity_loss - logging_similarity_loss) / args.logging_steps))
                    logging_gen_loss = tr_gen_loss
                    logging_pointer_loss = tr_pointer_loss
                    logging_re_loss = tr_re_loss
                    logging_similarity_loss = tr_similarity_loss
                    logging_connect_loss = tr_connect_loss
                # Save model checkpoint
                if epoch >= args.epoch_begin and global_step % args.valid_steps == 0:
                    QA_scores, bart_scores, precision, recall, ACC, span_pre, span_re, connect_re, connect_sc, result_str = evaluate(
                        args, model, val_dataloader, tokenizer)

                    checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, global_step, optimizer,
                                                     scheduler, QA_scores=QA_scores, bart_scores=bart_scores,
                                                     precision=precision,
                                                     recall=recall, ACC=ACC, span_pre=span_pre, span_re=span_re,
                                                     connect_re=connect_re,
                                                     connect_sc=connect_sc, res_str=result_str)
                    model.train()
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_dir = save_latest_checkpoint(model, tokenizer, args, optimizer, scheduler)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataloader, tokenizer, prefix=""):
    if not os.path.exists(os.path.join(args.prefix_path, args.output_dir)) and args.local_rank in [-1, 0]:
        os.makedirs(os.path.join(args.prefix_path, args.output_dir))

    args.eval_batch_size = args.per_gpu_eval_batch_size
    model = model.module if hasattr(model, 'module') else model

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader) * args.eval_batch_size)
    logger.info("  Batch size = %d", args.eval_batch_size)

    total_correct = 0
    total_re_num = 0
    total_sc_num = 0
    n_examples = 0
    # em_scores = 0
    # f1_scores = 0
    acc_scores = []

    total_connect = 0
    total_connect_re_num = 0
    total_connect_sc_num = 0
    span_correct = 0
    result_dict = {}
    e_sc = model.tokenizer.encode("[e_source]", add_special_tokens=False)[0]
    golden_res_list = []
    gen_res_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            inputs = {'input_ids': batch['input_ids'], 'input_mask': batch['input_mask'],
                      'input_token_type': batch['input_token_type'], 'batch_text_len': batch['batch_text_len'],
                      'table_text': batch['table_text'], 'table_mask': batch['table_mask'],
                      'table_token_type': batch['table_token_type'],
                      'image_token_type': batch['image_token_type'], 'image_feat': batch['image_feat'],
                      'image_mask': batch['image_mask'], 'batch_image_len': batch['batch_image_len'],
                      'image_ques_id': batch['image_ques_id'],
                      'ans_ids': batch['ans_ids'], 'attn_mask': batch['attn_mask'],
                      'ans_loss_mask': batch['ans_loss_mask'], 'sc_loss_mask': batch['sc_loss_mask'],
                      'golden_index': batch['golden_index'],
                      'table_connect_num': batch['table_connect_num'],
                      'flatten_connect_spans': batch['flatten_connect_spans'],
                      'flatten_connect_index': batch['flatten_connect_index'],
                      'table_cell_num': batch['table_cell_num'], 'flatten_cell_span': batch['flatten_cell_span'],
                      'gather_index': batch['gather_index'], 'span_label': batch['flatten_span_label']
                      }
            outputs, re_correct, total_hidden_num, gen_index_list, has_connect, similarity_list = model.evaluate(
                **inputs)
            qids = batch['ques_id']
            total_correct += re_correct
            total_re_num += total_hidden_num
            qus_answer_list = batch['ans_ids']
            answer_list = batch['answer_list']
            Qcate_list = batch['Qcate']
            span_lable_list = batch['span_lable']
            flatten_cell_span = batch['flatten_cell_span']
            input_text_keys = batch['input_text_keys']
            image_keys = batch['images_keys']
            for qid, gen_ans, qus_answer, golden_answers, Qcate, golden_index, gen_index, pred_connect, \
                pred_similarity, span_lable, cell_span, table_text, \
                input_text_key, image_key in zip(qids, outputs, qus_answer_list, answer_list,
                                                 Qcate_list, batch['golden_index'],
                                                 gen_index_list, has_connect,
                                                 similarity_list, span_lable_list,
                                                 flatten_cell_span, batch['table_text'],
                                                 input_text_keys, image_keys):
                n_examples += 1
                golden_index = golden_index.tolist()
                golden_keys = []
                text_num = len(input_text_key)
                for j in range(len(golden_index)):
                    if golden_index[j] == -1:
                        golden_index = golden_index[:j]
                        break
                    else:
                        if golden_index[j] < text_num:
                            golden_keys.append(input_text_key[golden_index[j]])
                        else:
                            golden_keys.append(image_key[golden_index[j] - text_num - 1])
                total_sc_num += len(golden_index)
                gen_keys = []
                for index in gen_index:
                    if index < text_num:
                        gen_keys.append(input_text_key[index])
                    else:
                        gen_keys.append(image_key[index - text_num - 1])
                qus_answer = qus_answer.tolist()
                b_sc_index = qus_answer.index(e_sc)

                ques_str = model.tokenizer.decode(qus_answer[:b_sc_index], skip_special_tokens=True).lower()
                golden_ans = golden_answers[0].lower()
                gen_ans_str = model.tokenizer.decode(gen_ans, skip_special_tokens=True).lower()
                res_dict = webqa_metrics_approx(gen_ans_str, golden_ans, Qcate)
                accuracy = res_dict['acc_approx']
                acc_scores.append(accuracy)
                golden_res_list.append(golden_ans)
                gen_res_list.append(gen_ans_str)

                tmp_has_connect = torch.nonzero(span_lable).squeeze(-1).tolist()
                total_connect_re_num += len(tmp_has_connect)
                total_connect_sc_num += len(pred_connect)
                inter_list = list(set(pred_connect).intersection(set(tmp_has_connect)))
                total_connect += len(inter_list)
                cell_span = cell_span.reshape(-1, 2)
                # golden_span = cell_span[span_lable[inter_list] - 1]
                for idx, item in enumerate(pred_connect):
                    if item in inter_list:
                        if pred_similarity[idx] == cell_span[span_lable[item] - 1].tolist():
                            span_correct += 1
                golden_source_to_span = {}
                for idx, item in enumerate(span_lable):
                    if item != 0:
                        tmp_span = cell_span[item - 1]
                        span_str = model.tokenizer.decode(table_text[tmp_span[0]:tmp_span[1]])
                        golden_source_to_span[idx] = span_str
                pred_source_to_span = {}
                for idx, item in enumerate(pred_connect):
                    tmp_span = pred_similarity[idx]
                    span_str = model.tokenizer.decode(table_text[tmp_span[0]:tmp_span[1]])
                    pred_source_to_span[item] = span_str
                result_dict[qid] = {'question': ques_str, 'golden_ans': golden_ans, 'gen_ans': gen_ans_str,
                                    'gen_id': gen_keys, 'golden_id': golden_keys,
                                    'golden_source_to_span': golden_source_to_span,
                                    'pred_source_to_span': pred_source_to_span, }

    normalizer = compute_bartscore_ParaBank(golden_res_list, golden_res_list)
    BARTscore = compute_bartscore_ParaBank(gen_res_list, golden_res_list) / np.array(normalizer)
    bart_scores = np.where(BARTscore > 1, 1, BARTscore)
    acc_scores = np.array(acc_scores)
    QA_scores = bart_scores * acc_scores
    QA_scores = QA_scores.mean()
    acc_scores = acc_scores.mean()
    bart_scores = bart_scores.mean()
    result_str = json.dumps(result_dict, indent=4)
    # span_ac = span_correct / total_connect
    span_pre = span_correct / total_connect_sc_num
    span_re = span_correct / total_connect_re_num
    connect_re = total_connect / total_connect_re_num
    connect_sc = total_connect / total_connect_sc_num
    return QA_scores, bart_scores, total_correct / total_re_num, total_correct / total_sc_num, acc_scores, span_pre, span_re, connect_re, connect_sc, result_str


def test(args, model, eval_dataloader, tokenizer, prefix=""):
    if not os.path.exists(os.path.join(args.prefix_path, args.output_dir)) and args.local_rank in [-1, 0]:
        os.makedirs(os.path.join(args.prefix_path, args.output_dir))
    args.eval_batch_size = args.per_gpu_eval_batch_size
    model = model.module if hasattr(model, 'module') else model
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader) * args.eval_batch_size)
    logger.info("  Batch size = %d", args.eval_batch_size)
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    exclude = list(string.punctuation)
    n_examples = 0
    result_dict = {}
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            inputs = {'input_ids': batch['input_ids'], 'input_mask': batch['input_mask'],
                      'input_token_type': batch['input_token_type'], 'batch_text_len': batch['batch_text_len'],
                      'image_token_type': batch['image_token_type'], 'image_feat': batch['image_feat'],
                      'image_mask': batch['image_mask'], 'batch_image_len': batch['batch_image_len'],
                      'image_ques_id': batch['image_ques_id'],
                      'ans_ids': batch['ans_ids'], 'attn_mask': batch['attn_mask'],
                      'table_text': batch['table_text'], 'table_mask': batch['table_mask'],
                      'table_token_type': batch['table_token_type'],
                      'table_connect_num': batch['table_connect_num'],
                      'flatten_connect_spans': batch['flatten_connect_spans'],
                      'flatten_connect_index': batch['flatten_connect_index'],
                      'table_cell_num': batch['table_cell_num'], 'flatten_cell_span': batch['flatten_cell_span'],
                      'gather_index': batch['gather_index'], 'span_label': batch['flatten_span_label']
                      }

            outputs, gen_index_list = model.test(**inputs)
        qids = batch['ques_id']
        qus_answer_list = batch['ans_ids']
        guide_keys_list = batch['guide_keys']
        for qid, gen_ans, qus_answer, guide_keys, gen_index in zip(qids, outputs,
                                                                   qus_answer_list,
                                                                   guide_keys_list,
                                                                   gen_index_list):
            n_examples += 1
            gen_ans_str = model.tokenizer.decode(gen_ans, skip_special_tokens=True)
            answer = ''
            for index, char in enumerate(gen_ans_str):
                if char in exclude:
                    if gen_ans_str[index - 1] != ' ':
                        answer += ' '
                    answer += char
                    if index + 1 < len(gen_ans_str):
                        if gen_ans_str[index + 1] != ' ':
                            answer += ' '
                else:
                    answer += char
            sources = []
            for item in gen_index:
                sources.append(guide_keys[item])
            result_dict[qid] = {'answer': answer, 'sources': sources}
    result_str = json.dumps(result_dict, indent=4)
    return result_str


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--prefix_path",
                        default="./MMQA/",
                        type=str,
                        help="Prefix to project path", )
    parser.add_argument("--example_file_train",
                        default="dataset/webqa/WebQA_train_val.json",
                        type=str, help="Path to WebQA train set ", )
    parser.add_argument("--example_file_eval",
                        default="dataset/webqa/WebQA_train_val.json",
                        type=str, help="Path to MultimodalQA eval set ", )
    parser.add_argument("--example_file_test",
                        default="dataset/webqa/WebQA_test.json",
                        type=str, help="Path to MultimodalQA test set ")
    parser.add_argument("--text_dir",
                        default="dataset/webqa/WebQA_texts_ofa_feats.pkl",
                        type=str, help="Path to WebQA texts feat")
    parser.add_argument("--test_text_dir",
                        default="dataset/webqa/WebQA_texts_ofa_feats_test.pkl",
                        type=str, help="Path to WebQA texts feat on test set")
    parser.add_argument("--img_dir",
                        default="dataset/webqa/images_ofa_feats",
                        type=str, help="Path to WebQA img feats directory")
    parser.add_argument("--test_img_dir",
                        default="dataset/webqa/images_ofa_feats",
                        type=str, help="Path to WebQA img feats directory")
    parser.add_argument("--img_index",
                        default="dataset/webqa/imgs.lineidx",
                        type=str, help="Path to WebQA img index")
    parser.add_argument("--img_tsv",
                        default="dataset/webqa/imgs.tsv",
                        type=str, help="Path to WebQA img tsv")
    parser.add_argument("--connect_path",
                        default="dataset/webqa/WebQA_opennre_tables_connect_rel.pkl",
                        type=str, help="Path to file that contains alignment between tables and sources on train set")
    parser.add_argument("--connect_path_test",
                        default="dataset/webqa/WebQA_opennre_tables_connect_rel_test_full.pkl",
                        type=str, help="Path to file that contains alignment between tables and sources on test set")
    parser.add_argument("--span_path",
                        default="dataset/webqa/WebQA_opennre_tables_spans_rel.pkl",
                        type=str, help="Path to file that contains spans of head entities on tables on train set")
    parser.add_argument("--span_path_test",
                        default="dataset/webqa/WebQA_opennre_tables_spans_rel_test_full.pkl",
                        type=str, help="Path to file that contains spans of head entities on tables on test set")
    parser.add_argument("--table_path",
                        default="dataset/webqa/WebQA_opennre_tables_ofa_feats_rel.pkl",
                        type=str, help="Path to file that contains extracted KGs feats on train set")
    parser.add_argument("--table_path_test",
                        default="dataset/webqa/WebQA_opennre_tables_ofa_feats_rel_test_full.pkl",
                        type=str, help="Path to file that contains extracted KGs feats on test set")
    parser.add_argument(
        "--bart_model_name_or_path",
        default='./BART',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--model_name_or_path",
        default='./OFA',
        type=str,
        help="Path to pretrained OFA",
    )
    parser.add_argument(
        "--ofa_squad_model_name_or_path",
        default='OFA/outputs/OFA_squad/checkpoint-5-7600-em-66.9703-f1-78.1277-pre-100.0000-re-100.0000',
        type=str,
        help="Path to pretrained OFA on SQUAD 2.0 ",
    )
    parser.add_argument(
        "--bart_squad_model_name_or_path",
        default='OFA/outputs/Bart_squad/checkpoint-6-9200-em-78.1377-f1-88.0597-pre-100.0000-re-100.0000',
        type=str,
        help="Path to pretrained BART on SQUAD 2.0",
    )
    parser.add_argument(
        "--eval_model_name_or_path",
        default='OFA/outputs/webqa/OFA_Bart_re_source2cell_AnsAcc_rel_opennre/checkpoint-latest',
        type=str,
        help="Path to SKURG model that need to be evaluated",
    )
    parser.add_argument(
        "--output_dir",
        default='OFA/outputs/webqa/OFA_Bart_re_source2cell_AnsAcc_rel_opennre',
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=768,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--max_img_length",
        default=200,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=300,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=256,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
             "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_beam", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_golden_source2entity", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true",
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_test_beam", action="store_true",
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_test_acc_beam", action="store_true",
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_test_golden", action="store_true",
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_golden_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_golden_source", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=2,
                        type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=3e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
             "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
             "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int,
                        default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=800,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--epoch_begin", type=int, default=3,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--valid_steps", type=int, default=800,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="",
                        help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="",
                        help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1,
                        help="multiple threads for converting example to features")

    parser.add_argument(
        "--insert_title",
        action="store_true",
        help="insert title information for QA model",
    )
    parser.add_argument(
        "--convert_q", action="store_true")
    parser.add_argument("--is_debug", action='store_true', help="Whether to run training.")
    parser.add_argument(
        "--global_step",
        type=int,
        default=0,
        help="Resume global step.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Num workers.",
    )
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=2,
        help="Resume global step.",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.0,
        help="Resume global step.",
    )
    parser.add_argument(
        "--constrained",
        type=float,
        default=1.0,
        help="Resume global step.",
    )
    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    tokenizer = OFATokenizer.from_pretrained(args.model_name_or_path)
    # bart_tokenizer=BartTokenizer.from_pretrained(args.bart_model_name_or_path)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<title>', '</title>', 'ROW', '[b_ans]', '[e_ans]', '[b_source]',
                                       '[e_source]']})
    OFA_model = OFAModel_MMQA.from_pretrained(args.model_name_or_path, tokenizer=tokenizer)
    OFA_model.resize_token_embeddings(len(tokenizer))
    # 加载squad预训练
    model_file = os.path.join(os.path.join(args.prefix_path, args.ofa_squad_model_name_or_path), 'model.pth')
    pretrained_dict = torch.load(model_file)
    model_dict = OFA_model.state_dict()
    model_dict.update(pretrained_dict)
    OFA_model.load_state_dict(model_dict)
    logger.info("load squad pretrained OFA from %s", args.ofa_squad_model_name_or_path)

    Bart_model = BartForConditionalGeneration.from_pretrained(args.bart_model_name_or_path, tokenizer=tokenizer)
    Bart_model.resize_token_embeddings(len(tokenizer))
    # 加载squad预训练
    model_file = os.path.join(os.path.join(args.prefix_path, args.bart_squad_model_name_or_path), 'model.pth')
    pretrained_dict = torch.load(model_file)
    model_dict = Bart_model.state_dict()
    model_dict.update(pretrained_dict)
    Bart_model.load_state_dict(model_dict)
    logger.info("load squad pretrained Bart from %s", args.bart_squad_model_name_or_path)

    Bart_table_model = BartForTableFusion.from_pretrained(args.bart_model_name_or_path, tokenizer=tokenizer)
    Bart_table_model.resize_token_embeddings(len(tokenizer))
    # 加载squad预训练
    model_file = os.path.join(os.path.join(args.prefix_path, args.bart_squad_model_name_or_path), 'model.pth')
    pretrained_dict = torch.load(model_file)
    model_dict = Bart_table_model.state_dict()
    model_dict.update(pretrained_dict)
    Bart_table_model.load_state_dict(model_dict)
    logger.info("load squad pretrained Bart from %s", args.bart_squad_model_name_or_path)

    model = OFABart_AnsAcc(OFA_model, Bart_model, Bart_table_model, tokenizer, args.length_penalty, args.constrained)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if args.do_test and args.local_rank in [-1, 0]:
        model_file = os.path.join(args.prefix_path, args.eval_model_name_or_path, 'model.pth')
        model.load_state_dict(torch.load(model_file))
        val_dataset = generative_webqa_dataset_test(tokenizer, args.example_file_test,
                                                    args.test_text_dir,
                                                    args.img_dir, args.table_path_test, args.connect_path_test,
                                                    args.span_path_test,
                                                    args.prefix_path, args.doc_stride,
                                                    args.max_query_length,
                                                    args.max_seq_length, is_debug=args.is_debug, is_val=True)
        model.to(args.device)
        test_dataloader = build_dataloader(val_dataset, False, args)

        result_str = test(args, model, test_dataloader, tokenizer)
        file_name = 'WebQA_bart_ofa_re_source2cell.json'
        with open(os.path.join(args.prefix_path, args.eval_model_name_or_path, file_name), 'w') as json_file:
            json_file.write(result_str)
        print('写入完成 ', file_name)

    if args.do_eval:
        model_file = os.path.join(args.prefix_path, args.eval_model_name_or_path, 'model.pth')
        model.load_state_dict(torch.load(model_file))
        val_dataset = generative_webqa_dataset(tokenizer, args.example_file_eval,
                                               args.text_dir,
                                               args.img_dir, args.table_path, args.connect_path, args.span_path,
                                               args.prefix_path,
                                               args.doc_stride, args.max_img_length,
                                               args.max_query_length,
                                               args.max_seq_length, is_debug=args.is_debug, is_val=True)
        model.to(args.device)
        val_dataloader = build_dataloader(val_dataset, False, args)

        QA_scores, bart_scores, precision, recall, ACC, span_pre, span_re, connect_re, connect_sc, result_str = evaluate(
            args, model, val_dataloader, tokenizer)
        file_name = 'MMQA-ACC-{:.4f}-BART-{:.4f}-span_pre-{:.4f}-span_re-{:.4f}-connect_sc-{:.4f}-connect_re-{:.4f}.json'.format(
            ACC, bart_scores, span_pre, span_re, connect_sc, connect_re)
        with open(os.path.join(args.prefix_path, args.eval_model_name_or_path, file_name), 'w') as json_file:
            json_file.write(result_str)
        print('写入完成 ', file_name)


if __name__ == "__main__":
    main()
