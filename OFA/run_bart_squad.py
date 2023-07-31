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
import glob
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
from ofa import OFATokenizer, OFAConfig

from OFA.datasets.generative_dataset_squad import generative_mmqa_dataset
from OFA.ofa.modeling_squad_bart import BartForConditionalGeneration
import errno
from utils.evaluate import evaluate_predictions_example
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


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
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-latest')
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


def save_checkpoint(model, tokenizer, args, epoch, iteration, optimizer, scheduler, num_trial=10, em=0.0, f1=0.0,
                    precision=0.0, recall=0.0, res_str=None):
    checkpoint_dir = os.path.join(args.output_dir,
                                  'checkpoint-{}-{}-em-{:.4f}-f1-{:.4f}-pre-{:.4f}-re-{:.4f}'.format(epoch, iteration,
                                                                                                     em,
                                                                                                     f1, precision,
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
        file_name = 'MMQA-em-{:.4f}-f1-{:.4f}.json'.format(em, f1)
        with open(os.path.join(checkpoint_dir, file_name), 'w') as json_file:
            json_file.write(res_str)
        print('Write to', file_name)
    return checkpoint_dir


def build_dataloader(dataset, is_train, opts):
    if is_train:
        dataloader = DataLoader(dataset, drop_last=True, batch_size=opts.per_gpu_train_batch_size * opts.n_gpu,
                                num_workers=opts.num_workers,
                                shuffle=True, collate_fn=dataset.mmqa_collate)
    else:
        dataloader = DataLoader(dataset, batch_size=opts.per_gpu_eval_batch_size,
                                num_workers=0, shuffle=False, collate_fn=dataset.mmqa_collate)
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

    train_dataset = generative_mmqa_dataset(tokenizer, args.example_file_train,
                                            args.doc_stride,
                                            args.max_query_length,
                                            args.max_seq_length)
    train_dataloader = build_dataloader(train_dataset, True, args)

    val_dataset = generative_mmqa_dataset(tokenizer, args.example_file_eval,
                                          args.doc_stride, args.max_query_length,
                                          args.max_seq_length, is_test=True)
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
        # resume from args.eval_model_name_or_path
        model = model.module if hasattr(model, 'module') else model
        model_file = os.path.join(args.eval_model_name_or_path, 'model.pth')
        model.load_state_dict(torch.load(model_file))
        optimizer.load_state_dict(torch.load(os.path.join(args.eval_model_name_or_path, 'optimizer.pth')))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()  # an optimizer.cuda() method for this operation would be nice
        scheduler.load_state_dict(torch.load(os.path.join(args.eval_model_name_or_path, 'scheduler.pth')))
        logger.info("  Resume from %s", args.eval_model_name_or_path)
        model = torch.nn.DataParallel(model)
        epochs_trained = global_step // (len(train_dataloader) //
                                         args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps)

    tr_loss, logging_loss = 0.0, 0.0
    tr_gen_loss, logging_gen_loss = 0.0, 0.0

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

            inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['input_mask'],
                      'input_token_type': batch['input_token_type'],
                      'decoder_input_ids': batch['ans_ids'], 'decoder_attention_mask': batch['attn_mask'],
                      'ans_loss_mask': batch['ans_loss_mask']
                      }
            with autocast():
                gen_loss = model(**inputs)

            if args.n_gpu > 1:
                gen_loss = gen_loss.mean()

            loss = gen_loss
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                gen_loss = gen_loss / args.gradient_accumulation_steps


            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                scaler.scale(loss).backward()
                # loss.backward()

            tr_loss += loss.item()
            tr_gen_loss += gen_loss.item()
            # tr_pointer_loss += pointer_loss.item()
            # tr_re_loss += re_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
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
                    logging_gen_loss = tr_gen_loss
                # Save model checkpoint
                if epoch >= args.epoch_begin and global_step % args.valid_steps == 0:
                    em, f1, result_str = evaluate(args, model, val_dataloader, tokenizer)

                    checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, global_step, optimizer,
                                                     scheduler, em=em, f1=f1, precision=100.0, recall=100.0,
                                                     res_str=result_str)
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
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    model = model.module if hasattr(model, 'module') else model

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader) * args.eval_batch_size)
    logger.info("  Batch size = %d", args.eval_batch_size)

    n_examples = 0
    em_scores = 0
    f1_scores = 0
    result_dict = {}

    e_sc = model.tokenizer.convert_tokens_to_ids("[e_source]")

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['input_mask'],
                      'input_token_type': batch['input_token_type'],
                      'decoder_input_ids': batch['ans_ids'], 'decoder_attention_mask': batch['attn_mask'],
                      'ans_loss_mask': batch['ans_loss_mask']
                      }
            outputs = model.evaluate(**inputs)
            qids = batch['ques_id']
            qus_answer_list = batch['ans_ids']
            answer_list = batch['answer_list']
            for qid, gen_ans, qus_answer, golden_answers in zip(qids, outputs, qus_answer_list, answer_list):
                n_examples += 1
                qus_answer = qus_answer.tolist()
                e_sc_index = qus_answer.index(e_sc)

                ques_str = model.tokenizer.decode(qus_answer[:e_sc_index], skip_special_tokens=True).lower()
                golden_ans = [str(item['text']).lower() for item in golden_answers]
                gen_ans_str = model.tokenizer.decode(gen_ans, skip_special_tokens=True).lower()
                eval_scores = evaluate_predictions_example(gen_ans_str, golden_ans)
                em_scores += eval_scores['list_em'] * 100
                f1_scores += eval_scores['list_f1'] * 100
                result_dict[qid] = {'question': ques_str, 'golden_ans': golden_ans, 'gen_ans': gen_ans_str}

    result_str = json.dumps(result_dict, indent=4)
    return em_scores / n_examples, f1_scores / n_examples, result_str


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--example_file_train",
                        default="./MMQA/dataset/train-v2.0.json",
                        type=str)
    parser.add_argument("--example_file_eval",
                        default="./MMQA/dataset/dev-v2.0.json",
                        type=str)
    parser.add_argument(
        "--bart_model_name_or_path",
        default='/./bart-base',
        type=str,
        help="Path to bart-base",
    )
    parser.add_argument(
        "--model_name_or_path",
        default='/./OFA',
        type=str,
        help="Path to ofa-base",
    )
    parser.add_argument(
        "--eval_model_name_or_path",
        default='./MMQA/OFA/outputs/Bart_squad_base/checkpoint-latest',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default='./MMQA/OFA/outputs/Bart_squad_base',
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
        default=1024,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=768,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
             "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8,
                        type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=16, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=3e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=8.0, type=float, help="Total number of training epochs to perform."
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
    parser.add_argument("--save_steps", type=int, default=400,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--epoch_begin", type=int, default=3,
                        help="Start evaluation after X epochs.")
    parser.add_argument("--valid_steps", type=int, default=400,
                        help="Evaluate model every X updates steps.")
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
    parser.add_argument(
        "--global_step",
        type=int,
        default=0,
        help="Resume global step.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Resume global step.",
    )
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
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
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<title>', '</title>', 'ROW', '[b_ans]', '[e_ans]', '[b_source]',
                                       '[e_source]']})
    model = BartForConditionalGeneration.from_pretrained(args.bart_model_name_or_path, tokenizer=tokenizer)
    model.resize_token_embeddings(len(tokenizer))
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
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        model_file = os.path.join(args.eval_model_name_or_path, 'model.pth')
        model.load_state_dict(torch.load(model_file))
        val_dataset = generative_mmqa_dataset(tokenizer, args.example_file_eval,
                                              args.doc_stride, args.max_query_length,
                                              args.max_seq_length,  is_test=True)
        model.to(args.device)
        test_dataloader = build_dataloader(val_dataset, False, args)

        em, f1, result_str = evaluate(args, model, test_dataloader, tokenizer)
        file_name = 'MMQA-em-{:.4f}-f1-{:.4f}.json'.format(em, f1)
        with open(os.path.join(args.eval_model_name_or_path, file_name), 'w') as json_file:
            json_file.write(result_str)
        print('Write to ', file_name)

    return results


if __name__ == "__main__":
    main()
