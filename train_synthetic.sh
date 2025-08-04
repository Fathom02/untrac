#!/bin/sh

if [ ! -d data ]; then
  mkdir data
fi

python preprocess_synthetic.py

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_name_or_path t5-small \
    --optim adafactor \
    --per_device_train_batch_size 2 \
    --train_dir data/synthetic_train00_dataset \
    --eval_dir data/synthetic_eval00_dataset \
    --eval_steps 128 \
    --logging_steps 128 \
    --save_strategy no \
    --num_train_epochs 5 \
    --output_dir model/t5_00 \