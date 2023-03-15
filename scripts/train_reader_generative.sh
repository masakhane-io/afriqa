export HF_TOKEN=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
model_name_or_path=Atnafu/mt5-base-squad2-fin
# dataset_name=Tevatron/wikipedia-nq+
dataset_name=squad_v2
output_dir=models
batch_size=8
num_train_epochs=5
max_seq_length=384
save_steps=10000

CUDA_VISIBLE_DEVICES=0 python3 baselines/reader/train_seq_2_seq.py \
  --model_name_or_path $model_name_or_path \
  --dataset_name $dataset_name \
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
  --push_to_hub_token $HF_TOKEN \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --push_to_hub_model_id=extractive_reader_nq_squad_v2 \
  --weight_decay 0.01 \
  --eval_steps 1000 \
  --logging_steps 1000 \
  --metric_for_best_model="eval_f1,eval_HasAns_f1,eval_NoAns_f1,exact" \
  --greater_is_better=True \
  --evaluation_strategy="steps" \
  --version_2_with_negative
