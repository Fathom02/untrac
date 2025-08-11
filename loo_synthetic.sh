#!/usr/bin/env bash
set -euo pipefail


#   bash loo_synthetic.sh            
#   bash loo_synthetic.sh 10         
SPLIT="${1:-00}"


CKPT="./model/t5_00"                              
TRAIN_DIR="data/synthetic_train${SPLIT}_dataset"  
EVAL_DIR="data/synthetic_eval${SPLIT}_dataset"    
BATCH=2
EPOCHS=1
EVAL_STEPS=128
LOG_STEPS=$EVAL_STEPS
OUT_ROOT="model/t5_00_loo_${SPLIT}"
mkdir -p "$OUT_ROOT"

mapfile -t DATASET_NAMES < <(python - <<'PY' "$TRAIN_DIR"
import sys
from datasets import load_from_disk
ds = load_from_disk(sys.argv[1])
print(*sorted(set(ds["dataset"])), sep="\n")
PY
)

echo "Found datasets in ${TRAIN_DIR}: ${DATASET_NAMES[*]}"

for name in "${DATASET_NAMES[@]}"; do
  echo "=== LOO: leaving out ${name} (split ${SPLIT}) ==="
  CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_name_or_path "$CKPT" \
    --optim adafactor \
    --fp16 \
    --per_device_train_batch_size "$BATCH" \
    --train_dir "$TRAIN_DIR" \
    --loo_dataset_names "$name" \
    --eval_dir "$EVAL_DIR" \
    --eval_steps "$EVAL_STEPS" \
    --logging_steps "$LOG_STEPS" \
    --save_strategy no \
    --num_train_epochs "$EPOCHS" \
    --output_dir "$OUT_ROOT/$name" \
    --overwrite_output_dir
done

echo "Done. Results under: $OUT_ROOT"
