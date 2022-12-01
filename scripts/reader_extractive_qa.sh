model_name_or_path=bert-base-uncased
dataset_name=squad
output_dir=models
batch_size=16
num_train_epochs=10
max_seq_length=384
save_steps=10000

CUDA_VISIBLE_DEVICES=7 python3 baselines/reader/train.py \
  --model_name_or_path /home/oogundep/african_qa/models/bert-base-uncased/checkpoint-55000 \
  --dataset_name $dataset_name \
  --do_train \
  --do_eval \
  --per_device_train_batch_size $batch_size  \
  --per_device_eval_batch_size $batch_size  \
  --learning_rate 3e-5 \
  --num_train_epochs $num_train_epochs \
  --max_seq_length $max_seq_length \
  --doc_stride 128 \
  --output_dir $output_dir/$model_name_or_path \
  --save_steps $save_steps