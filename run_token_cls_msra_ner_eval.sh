#!/usr/bin/env bash

pip install paddlenlp==2.4.0

TASK_NAME="msra_ner"
EVAL_FILE_PATH="datasets/msra_ner/test.tsv"
LABEL_MAP_FILE_PATH="datasets/msra_ner/label_map.json"
MODEL_NAME="ernie-3.0-medium-zh"
INIT_CHECKPOINT_PATH="ernie-ckpt/ernie-3.0-medium-zh-msra-ner"
python run_token_cls.py \
    --task_name $TASK_NAME \
    --eval_file_path $EVAL_FILE_PATH \
    --label_map_file_path $LABEL_MAP_FILE_PATH \
    --max_seq_length 128 \
    --model_name_or_path $MODEL_NAME \
    --init_checkpoint_path $INIT_CHECKPOINT_PATH \
    --do_eval \
    --batch_size 256 \
    --device gpu