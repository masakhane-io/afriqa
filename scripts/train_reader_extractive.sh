model_name_or_path=ToluClassics/extractive_reader_nq
# dataset_name=Tevatron/wikipedia-nq+
dataset_name=squad_v2
output_dir=models
batch_size=64
num_train_epochs=5
max_seq_length=384
save_steps=10000

CUDA_VISIBLE_DEVICES=4 python3 baselines/reader/train_extractive.py \
  --model_name_or_path $model_name_or_path \
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
  --save_steps $save_steps \
  --overwrite_output_dir \
  --push_to_hub \
  --push_to_hub_model_id=extractive_reader_nq_squad_v2 \
  --weight_decay 0.01 \
  --eval_steps 1000 \
  --logging_steps 1000 \
  --evaluation_strategy="steps" \
  --version_2_with_negative
