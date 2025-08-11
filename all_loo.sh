CUDA_VISIBLE_DEVICES=0 python main.py \
  --model_name_or_path ./model/t5_00 \
  --optim adafactor --fp16 \
  --per_device_train_batch_size 2 \
  --train_dir data/synthetic_train00_dataset \
  --eval_dir  data/synthetic_eval00_dataset \
  --eval_steps 128 --logging_steps 128 \
  --save_strategy no --num_train_epochs 1 \
  --output_dir model/t5_00_loo_00/all \
  --overwrite_output_dir
