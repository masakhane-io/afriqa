# model_name_or_path=Davlan/afro-xlmr-base
# dataset_name=squad_v2
# output_dir=models
# batch_size=64
# num_train_epochs=5
# max_seq_length=384
# save_steps=10000

# CUDA_VISIBLE_DEVICES=1 python3 baselines/reader/train_extractive.py \
#   --model_name_or_path $model_name_or_path \
#   --dataset_name $dataset_name \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size $batch_size  \
#   --per_device_eval_batch_size $batch_size  \
#   --learning_rate 3e-5 \
#   --num_train_epochs $num_train_epochs \
#   --max_seq_length $max_seq_length \
#   --doc_stride 128 \
#   --output_dir $output_dir/$model_name_or_path \
#   --save_steps $save_steps \
#   --overwrite_output_dir \
#   --weight_decay 0.01 \
#   --eval_steps 1000 \
#   --logging_steps 1000 \
#   --evaluation_strategy="steps" \
#   --version_2_with_negative \
#   --question_column question \
#   --context_column context \
#   --answer_column answers \
#   --push_to_hub \
#   --push_to_hub_model_id=extractive_reader_afroxlmr_squad_v2 \


translation_type=$1
data_file_path=$2

for lang in kin
do
    for split in test
    do
        model_name_or_path=ToluClassics/extractive_reader_afroxlmr_squad_v2
        validation_file=${data_file_path}/${split}.${lang}.${translation_type}.json
        output_dir=models
        batch_size=16
        max_seq_length=384

        # CUDA_VISIBLE_DEVICES=4 python3 baselines/reader/train_extractive.py \
        #     --model_name_or_path $model_name_or_path \
        #     --validation_file $validation_file \
        #     --do_eval \
        #     --per_device_eval_batch_size $batch_size  \
        #     --learning_rate 3e-5 \
        #     --max_seq_length $max_seq_length \
        #     --doc_stride 128 \
        #     --output_dir $output_dir/$model_name_or_path/multilingual_translation/$lang \
        #     --overwrite_output_dir \
        #     --eval_steps 1000 \
        #     --logging_steps 1000 \
        #     --evaluation_strategy="steps" \
        #     --version_2_with_negative \
        #     --question_column question_lang \
        #     --context_column context \
            # --answer_column answer_pivot 

        # CUDA_VISIBLE_DEVICES=4 python3 baselines/reader/train_extractive.py \
        #     --model_name_or_path $model_name_or_path \
        #     --validation_file $validation_file \
        #     --do_eval \
        #     --per_device_eval_batch_size $batch_size  \
        #     --learning_rate 3e-5 \
        #     --max_seq_length $max_seq_length \
        #     --doc_stride 128 \
        #     --output_dir $output_dir/$model_name_or_path/$translation_type/$lang \
        #     --overwrite_output_dir \
        #     --eval_steps 1000 \
        #     --logging_steps 1000 \
        #     --evaluation_strategy="steps" \
        #     --version_2_with_negative \
        #     --question_column question_translated \
        #     --context_column context \
        #     --answer_column answer_pivot 
    done
done
