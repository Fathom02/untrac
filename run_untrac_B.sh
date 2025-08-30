#!/bin/bash
DATASETS=(dataset1 dataset2 dataset3 dataset4)
CKPT="/home/lei/untrac/model/t5_00"

for ds in "${DATASETS[@]}"; do
  echo "=== Unlearning $ds ==="
  CUDA_VISIBLE_DEVICES=0 python main.py \
    --unlearn \
    --optim adafactor \
    --max_grad_norm 0 \
    --fp16 \
    --per_device_train_batch_size 2 \
    --model_name_or_path "$CKPT" \
    --train_dir data/synthetic_train01_dataset \
    --dataset_names "$ds" \
    --eval_dir data/synthetic_eval01_dataset \
    --eval_steps 16 \
    --logging_steps 16 \
    --save_strategy no \
    --num_train_epochs 1 \
    --output_dir unlearn_outputs/$ds \
    --overwrite_output_dir
done
