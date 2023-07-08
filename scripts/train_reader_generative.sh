model_name_or_path=$1
dataset_name=squad_v2
output_dir=models
batch_size= 32
num_train_epochs=5
max_seq_length=384
save_steps=10000

CUDA_VISIBLE_DEVICES=4 python3 baselines/reader/train_seq_2_seq.py \
  --model_name_or_path $model_name_or_path \
  --dataset_name $dataset_name \
  --do_train \
  --do_eval \
  --learning_rate 3e-5 \
  --num_train_epochs $num_train_epochs \
  --max_seq_length $max_seq_length \
  --doc_stride 128 \
  --output_dir $output_dir/$model_name_or_path \
  --save_steps $save_steps \
  --overwrite_output_dir \
  --context_column context \
  --predict_with_generate True \
  --question_column question \
  --answer_column answers \
  --weight_decay 0.01 \
  --eval_steps 1000 \
  --logging_steps 1000 \
  --evaluation_strategy="steps" \
  --version_2_with_negative
