translation_type=$1
data_file_path=$2
model_name_or_path=$3


# #==================================================================================================#
# # Generative QA inference
# #==================================================================================================#

for lang in bem yor zul hau ibo kin twi swa fon
do
    for split in test
    do

    model_name_or_path=$model_name_or_path
    # validation_file=${data_file_path}/${split}.${lang}.${translation_type}.json
    validation_file=data/gold_passages/${lang}/gold_span_passages.afriqa.${lang}.en.${split}.json
    output_dir=models
    batch_size=8
    num_train_epochs=10
    max_seq_length=384
    save_steps=10000

    CUDA_VISIBLE_DEVICES=7 python baselines/reader/train_seq_2_seq.py \
        --model_name_or_path $model_name_or_path  \
        --validation_file $validation_file \
        --context_column context \
        --question_column question_translated \
        --answer_column answer_pivot \
        --do_eval \
        --per_device_train_batch_size $batch_size \
        --per_device_eval_batch_size $batch_size \
        --learning_rate 3e-5 \
        --num_train_epochs $num_train_epochs \
        --max_seq_length $max_seq_length \
        --doc_stride 128 \
        --output_dir $output_dir/$model_name_or_path/$translation_type/$lang  \
        --save_steps $save_steps \
        --overwrite_output_dir \
        --predict_with_generate True \
        --version_2_with_negative 
    done
done

# #==================================================================================================#
# # Multingual Generative QA using a finetuned mt5-base using the in-language queries
# #==================================================================================================#


for lang in bem yor zul hau ibo kin twi swa fon
do
    for split in test
    do

    validation_file=data/gold_passages/${lang}/gold_span_passages.afriqa.${lang}.en.${split}.json
    output_dir=models
    batch_size=8
    num_train_epochs=10
    max_seq_length=384
    save_steps=10000

    CUDA_VISIBLE_DEVICES=7 python baselines/reader/train_seq_2_seq.py \
        --model_name_or_path $model_name_or_path  \
        --validation_file $validation_file \
        --context_column context \
        --question_column question_lang \
        --answer_column answer_pivot \
        --do_eval \
        --per_device_train_batch_size $batch_size \
        --per_device_eval_batch_size $batch_size \
        --learning_rate 3e-5 \
        --num_train_epochs $num_train_epochs \
        --max_seq_length $max_seq_length \
        --doc_stride 128 \
        --output_dir $output_dir/$model_name_or_path/multilingual_translation/$lang \
        --save_steps $save_steps \
        --overwrite_output_dir \
        --predict_with_generate True \
        --version_2_with_negative 
    done
done