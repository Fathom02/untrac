#!/usr/bin/env bash


CKPT="./model/t5_00"                     
TEST_DIR="data/synthetic_eval01_dataset" 
TEST_DS_NAMES=('dataset0')
TRAIN_SPLITS=(                           
  synthetic_train01_dataset
  synthetic_train10_dataset
)
OUT_ROOT="untrac_inv_outputs"            
mkdir -p "$OUT_ROOT"

for TRAIN_DIR in "${TRAIN_SPLITS[@]}"; do
  DS_NAME=$(basename "$TRAIN_DIR")
  echo "=== evaluating impact on $DS_NAME ==="
  python main.py \
    --unlearn \
    --fp16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 128 \
    --max_grad_norm 0 \
    --model_name_or_path "$CKPT" \
    --train_dir "$TEST_DIR" \
    --dataset_names "${TEST_DS_NAMES[@]}" \
    --eval_dir  "data/$TRAIN_DIR" \
    --each_eval_samples 256 \
    --eval_steps 5 \
    --logging_steps 5 \
    --save_strategy no \
    --max_steps 50 \
    --output_dir "$OUT_ROOT/$DS_NAME" \
    --overwrite_output_dir
done
