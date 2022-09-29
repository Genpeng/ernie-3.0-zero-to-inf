#!/usr/bin/env bash

pip install paddlenlp==2.4.0

TASK_NAME="xhs_ner_v1"
TRAIN_FILE_PATH="datasets/xhs_ner_v1/train.tsv"
EVAL_FILE_PATH="datasets/xhs_ner_v1/test.tsv"
LABEL_MAP_FILE_PATH="datasets/xhs_ner_v1/label_map.json"
MODEL_NAME="ernie-3.0-medium-zh"
OUTPUT_DIR="ernie-ckpt/ernie-3.0-medium-zh-xhs-ner-v1"
python run_token_cls.py \
    --task_name $TASK_NAME \
    --train_file_path $TRAIN_FILE_PATH \
    --eval_file_path $EVAL_FILE_PATH \
    --label_map_file_path $LABEL_MAP_FILE_PATH \
    --max_seq_length 128 \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --do_train \
    --num_train_epochs 20 \
    --batch_size 128 \
    --learning_rate 5e-5 \
    --logging_steps 10 \
    --save_steps 1000 \
    --device "gpu"