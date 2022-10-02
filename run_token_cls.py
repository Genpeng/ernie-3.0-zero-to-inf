# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import functools
import json
import os
import sys
import time

import paddle
from paddle.io import DataLoader
from paddlenlp.data import DataCollatorForTokenClassification
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.transformers import AutoModelForTokenClassification, AutoTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # region Data Arguments
    parser.add_argument(
        "--task_name",
        default="msra_ner",
        type=str,
        choices=["msra_ner", "xhs_ner_v1"],
        help="The named entity recognition datasets."
    )
    parser.add_argument(
        "--train_file_path",
        default="data/train.tsv",
        type=str,
        help="The file path of training dataset."
    )
    parser.add_argument(
        "--eval_file_path",
        default="data/eval.tsv",
        type=str,
        help="The file path of validation dataset."
    )
    parser.add_argument(
        "--test_file_path",
        default="data/test.tsv",
        type=str,
        help="The file path of test dataset."
    )
    parser.add_argument(
        "--label_map_file_path",
        default="data/label_map.json",
        type=str,
        help="The file used to store label mapping relation (label -> label_id)."
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. "
             "Sequences longer than this will be truncated, sequences shorter will be padded."
    )
    # endregion

    # region Model Arguments
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True
    )
    # endregion

    # region Paddle Training Arguments
    parser.add_argument(
        "--output_dir",
        default="ernie-ckpt",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument("--do_train", action='store_true', help="Whether to train.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to validate.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to predict.")
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. "
             "Override num_train_epochs."
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X updates steps."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for initialization"
    )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu."
    )
    # endregion

    args = parser.parse_args()
    return args


def tokenize_and_align_labels_v1(examples, tokenizer, label_map, max_seq_length=128):
    no_entity_id = label_map["O"]
    examples_encoded = tokenizer(
        examples['tokens'],
        max_seq_len=max_seq_length,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
        return_length=True
    )

    batch_labels = []
    for i, labels in enumerate(examples['labels']):
        labels = [label_map.get(label, no_entity_id) for label in labels]
        input_ids_len = len(examples_encoded['input_ids'][i])
        if input_ids_len - 2 < len(labels):
            labels = labels[:input_ids_len -2]
        labels = [no_entity_id] + labels + [no_entity_id]
        labels += [no_entity_id] * (input_ids_len - len(labels))
        batch_labels.append(labels)

    examples_encoded["labels"] = batch_labels
    return examples_encoded


def tokenize_and_align_labels_v2(example, tokenizer, label_map, max_seq_length=128):
    no_entity_id = label_map["O"]
    tokens = example["tokens"]  # list of tokens
    labels = [label_map.get(label, no_entity_id) for label in example["labels"]]  # list of label ids

    tokens_encoded = tokenizer(tokens, return_length=True, is_split_into_words=True, max_seq_len=max_seq_length)

    input_ids_len = len(tokens_encoded["input_ids"])  # input_ids_len = max_seq_len
    # 如果 input_ids_len - 2 < len(labels)，说明输入的 tokens 的长度超过 max_seq_len，被截断了
    if input_ids_len - 2 < len(labels):
        labels = labels[:input_ids_len - 2]
    tokens_encoded["labels"] = [no_entity_id] + labels + [no_entity_id]
    tokens_encoded["labels"] += [no_entity_id] * (input_ids_len - len(tokens_encoded["labels"]))
    return tokens_encoded


@paddle.no_grad()
def evaluate(data_loader, model, loss_op, metric_op):
    model.eval()
    metric_op.reset()

    avg_loss, precision, recall, f1_score = 0, 0, 0, 0
    for batch in data_loader:
        logits = model(batch['input_ids'], batch['token_type_ids'])
        loss = loss_op(logits, batch['labels'])  # (batch_size, )
        avg_loss = paddle.mean(loss)
        preds = logits.argmax(axis=2)  # (batch_size, seq_len)
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric_op.compute(
            batch['seq_len'], preds, batch['labels']
        )
        metric_op.update(
            num_infer_chunks.numpy(), num_label_chunks.numpy(), num_correct_chunks.numpy()
        )
        precision, recall, f1_score = metric_op.accumulate()
    print(
        "[EVAL] loss: %.6f, precision: %.6f, recall: %.6f, f1: %.6f"
        % (avg_loss, precision, recall, f1_score)
    )

    model.train()
    return avg_loss, precision, recall, f1_score


def print_arguments(args):
    """print arguments"""
    print('=============== Configuration Arguments ===============')
    for arg, value in sorted(vars(args).items(), key=lambda x: x[0]):
        print('%s: %s' % (arg, value))
    print('=======================================================')


def do_train(args):
    start_time = time.time()
    task_name = args.task_name

    print(f"start execute task '{task_name}'...")

    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    train_file_path = args.train_file_path
    eval_file_path = args.eval_file_path
    label_map_file_path = args.label_map_file_path
    if not os.path.exists(train_file_path) or not os.path.isfile(train_file_path):
        sys.exit(f"{label_map_file_path} dose not exists or is not a file.")
    if not os.path.exists(eval_file_path) or not os.path.isfile(eval_file_path):
        sys.exit(f"{label_map_file_path} dose not exists or is not a file.")
    if not os.path.exists(label_map_file_path) or not os.path.isfile(label_map_file_path):
        sys.exit(f"{label_map_file_path} dose not exists or is not a file.")

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # region Load training and validation set

    def _read_token_cls_data(data_path):
        with open(data_path, "r") as fin:
            for line in fin:
                line = line.rstrip()
                tokens_str, labels_str = line.split("\t")
                tokens = tokens_str.split("\002")
                labels = labels_str.split("\002")
                yield {"tokens": tokens, "labels": labels}

    train_dataset = load_dataset(_read_token_cls_data, data_path=train_file_path, lazy=False)
    eval_dataset = load_dataset(_read_token_cls_data, data_path=eval_file_path, lazy=False)

    with open(label_map_file_path, "r") as fin:
        label_map = json.load(fin)
    num_classes = len(label_map)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    transform_func = functools.partial(
        tokenize_and_align_labels_v2,
        tokenizer=tokenizer,
        label_map=label_map,
        max_seq_length=args.max_seq_length
    )
    train_dataset = train_dataset.map(transform_func)
    eval_dataset = eval_dataset.map(transform_func)

    collate_fn = DataCollatorForTokenClassification(tokenizer=tokenizer)

    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=collate_fn,
        return_list=True
    )

    eval_batch_sampler = paddle.io.BatchSampler(
        eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    eval_data_loader = DataLoader(
        dataset=eval_dataset,
        batch_sampler=eval_batch_sampler,
        collate_fn=collate_fn,
        return_list=True
    )

    train_steps_per_epoch = len(train_data_loader)
    eval_steps_per_epoch = len(eval_data_loader)
    print(f"train_steps_per_epoch: {train_steps_per_epoch}, eval_steps_per_epoch: {eval_steps_per_epoch}")

    # endregion

    # region Define model, loss, metric and optimizer

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path, num_classes=num_classes
    )
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = args.max_steps if args.max_steps > 0 else train_steps_per_epoch * args.num_train_epochs
    lr_scheduler = LinearDecayWithWarmup(
        args.learning_rate, num_training_steps, args.warmup_steps
    )
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        param.name for name, param in model.named_parameters()
        if not any(_name in name for _name in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params
    )

    loss_obj = paddle.nn.loss.CrossEntropyLoss()
    metric_obj = ChunkEvaluator(label_list=list(label_map.keys()))

    # endregion

    # region Training & Validation

    global_step = 0
    best_step = 0
    best_f1_loss, best_f1_precision, best_f1_recall, best_f1 = 0.0, 0.0, 0.0, 0.0
    tic_train = time.time()
    for epoch in range(1, args.num_train_epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            global_step += 1

            logits = model(batch['input_ids'], batch['token_type_ids'])
            loss = loss_obj(logits, batch['labels'])  # (batch_size, )
            avg_loss = paddle.mean(loss)

            if global_step % args.logging_steps == 0:
                print(
                    "[TRAIN] global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_step, epoch, step, avg_loss, args.logging_steps / (time.time() - tic_train))
                )
                tic_train = time.time()

            avg_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.save_steps == 0 or global_step == num_training_steps:
                if paddle.distributed.get_rank() == 0:
                    _loss, _precision, _recall, _f1 = evaluate(eval_data_loader, model, loss_obj, metric_obj)
                    if _f1 > best_f1:
                        best_step = global_step
                        best_f1_loss, best_f1_precision, best_f1_recall, best_f1 = (
                            _loss, _precision, _recall, _f1
                        )
                        # Need better way to get inner model of DataParallel
                        model_to_save = model._layers if isinstance(model, paddle.DataParallel) else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

            if global_step >= num_training_steps:
                break
                # end_time = time.time()
                # print(
                #     "best_step: %d, loss: %.6f, precision: %.4f, recall: %.4f, best_f1: %.4f"
                #     % (best_step, best_f1_loss, best_f1_precision, best_f1_recall, best_f1)
                # )
                # print(f"finish job '{task_name}', time: {end_time-start_time}")
                # return

    # 设置的 max_step 过大（大于 num_train_epochs * train_steps_per_epoch）
    end_time = time.time()
    print(
        "best_step: %d, loss: %.6f, precision: %.4f, recall: %.4f, best_f1: %.4f"
        % (best_step, best_f1_loss, best_f1_precision, best_f1_recall, best_f1)
    )
    print(f"finish job '{task_name}', time: {end_time - start_time}")

    # endregion


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    if args.do_train:
        do_train(args)
