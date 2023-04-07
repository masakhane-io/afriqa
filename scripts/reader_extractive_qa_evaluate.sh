# translation_dir=$1
# translation_type=$2


# for lang in ibo hau zul fon
# do
#     for split in test
#     do  
#         echo "================================================="
#         echo "Language: ${lang}"
#         echo "Split: ${split}"
#         echo "================================================="
#         python3 scripts/generate_translation_gold_span_file.py --input_gold_data_split queries/json_lines_gold/data_split_gold_passages_${lang}_${split}.json \
#             --input_translation_directory ${translation_dir} \
#             --output_directory queries/eval_gold_span \
#             --lang ${lang} \
#             --split ${split} \
#             --translation ${translation_type}
#     done
# done



translation_type=$1
data_file_path=$2

# for lang in swa yor bem wol;
# do
#     for split in test
#     do
#         if [ $lang = "wol" ]; then
#             model_name_or_path=ToluClassics/extractive_reader_afroxlmr_fquad
#         else
#             model_name_or_path=ToluClassics/extractive_reader_afroxlmr_squad_v2
#         fi
        
#         validation_file=${data_file_path}/${split}.${lang}.${translation_type}.json
#         output_dir=models
#         batch_size=16
#         max_seq_length=384

#         CUDA_VISIBLE_DEVICES=4 python3 baselines/reader/train_extractive.py \
#             --model_name_or_path $model_name_or_path \
#             --validation_file $validation_file \
#             --do_eval \
#             --per_device_eval_batch_size $batch_size  \
#             --learning_rate 3e-5 \
#             --max_seq_length $max_seq_length \
#             --doc_stride 128 \
#             --output_dir $output_dir/$model_name_or_path/multilingual_translation/$lang \
#             --overwrite_output_dir \
#             --eval_steps 1000 \
#             --logging_steps 1000 \
#             --evaluation_strategy="steps" \
#             --version_2_with_negative \
#             --question_column question_lang \
#             --context_column context \
#             --answer_column answer_pivot 
#     done
# done

for lang in swa yor bem wol;
do
    for split in test
    do
        if [ $lang = "wol" ]; then
            model_name_or_path=ToluClassics/extractive_reader_afroxlmr_fquad
        else
            model_name_or_path=ToluClassics/extractive_reader_afroxlmr_squad_v2
        fi
        validation_file=${data_file_path}/${split}.${lang}.${translation_type}.json
        output_dir=models
        batch_size=16
        max_seq_length=384

        CUDA_VISIBLE_DEVICES=4 python3 baselines/reader/train_extractive.py \
            --model_name_or_path $model_name_or_path \
            --validation_file $validation_file \
            --do_eval \
            --per_device_eval_batch_size $batch_size  \
            --learning_rate 3e-5 \
            --max_seq_length $max_seq_length \
            --doc_stride 128 \
            --output_dir $output_dir/$model_name_or_path/$translation_type/$lang \
            --overwrite_output_dir \
            --eval_steps 1000 \
            --logging_steps 1000 \
            --evaluation_strategy="steps" \
            --version_2_with_negative \
            --question_column question_translated \
            --context_column context \
            --answer_column answer_pivot 
    done
done

# translation_type=$1

# echo "================================================="
# echo "Translation Type: ${translation_type}"
# echo "================================================="

# for lang in swa zul hau ibo bem twi kin yor
# do
#     for run_type in mdpr-hybrid
#     do  
#         echo "================================================="
#         echo "Language: ${lang}"
#         echo "Run Type: ${run_type}"
#         echo "================================================="
#         python3 /home/oogundep/african_qa/baselines/reader/evaluate_reader.py --retrieval-file /home/oogundep/african_qa/runs/run.xqa.$lang.test.en.${translation_type}.$run_type.json \
#             --reader dpr --retriever score --settings dpr \
#             --output-file /home/oogundep/african_qa/runs/reader/dpr-reader-single-nq-base/reader.xqa.$lang.$run_type.${translation_type}.json  \
#             --topk-em 10 20 30 40 50 60 70 80 90 100 \
#             --model-name facebook/dpr-reader-single-nq-base \
#             --tokenizer-name facebook/dpr-reader-single-nq-base   
#         echo "================================================="
#         echo "===> Done with ${lang} ${run_type}======"
#     done
# done


# for lang in fon wol
# do
#     for run_type in bm25 mdpr
#     do  
#         echo "================================================="
#         echo "Language: ${lang}"
#         echo "Run Type: ${run_type}"
#         echo "================================================="
#         python3 /home/oogundep/african_qa/baselines/reader/evaluate_reader.py --retrieval-file /home/oogundep/african_qa/runs/run.xqa.$lang.test.fr.${translation_type}.$run_type.json \
#             --reader dpr --retriever score --settings dpr \
#             --output-file /home/oogundep/african_qa/runs/reader/dpr-reader-single-nq-base/reader.xqa.$lang.$run_type.${translation_type}.json  \
#             --topk-em 10 20 30 40 50 60 70 80 90 100 \
#             --model-name /home/oogundep/african_qa/models/checkpoint_new_hgf  \
#             --tokenizer-name bert-base-multilingual-uncased
#         echo "================================================="
#         echo "===> Done with ${lang} ${run_type}======"
#     done
# done



# for lang in hau
# do
#     for run_type in bm25
#     do  
#         echo "================================================="
#         echo "Language: ${lang}"
#         echo "Run Type: ${run_type}"
#         echo "================================================="
#         CUDA_VISIBLE_DEVICES=1 python3 /home/oogundep/african_qa/baselines/reader/evaluate_reader.py --retrieval-file /home/oogundep/african_qa/runs/run.xqa.$lang.test.en.${translation_type}.$run_type.json \
#             --reader dpr --retriever score --settings dpr \
#             --output-file /home/oogundep/african_qa/runs/reader/dpr-reader-single-nq-base/reader.xqa.$lang.$run_type.${translation_type}.json  \
#             --topk-em 10 50 100 \
#             --model-name /home/oogundep/african_qa/models/dpr_mult_25.pth \
#             --tokenizer-name bert-base-multilingual-uncased \
#             --device cuda:0
#         echo "================================================="
#         echo "===> Done with ${lang} ${run_type}======"
#     done
# done


# hau ibo bem kin twi zul
# # model_name_or_path=bert-base-uncased
# # dataset_name=Tevatron/wikipedia-nq
# # output_dir=models
# # batch_size=64
# # num_train_epochs=10
# # max_seq_length=384
# # save_steps=10000

# # CUDA_VISIBLE_DEVICES=7 python3 baselines/reader/train_extractive.py \
# #   --model_name_or_path $model_name_or_path \
# #   --dataset_name $dataset_name \
# #   --do_train \
# #   --do_eval \
# #   --per_device_train_batch_size $batch_size  \
# #   --per_device_eval_batch_size $batch_size  \
# #   --learning_rate 3e-5 \
# #   --num_train_epochs $num_train_epochs \
# #   --max_seq_length $max_seq_length \
# #   --doc_stride 128 \
# #   --output_dir $output_dir/$model_name_or_path \
# #   --save_steps $save_steps


# # python3 /home/oogundep/african_qa/baselines/reader/convert_dpr_original_checkpoint_to_pytorch.py --type reader --src /home/oogundep/african_qa/models/dpr_extractive_reader_multilingual.2.3304 --dest /home/oogundep/african_qa/models/hgf
# # python3 /home/oogundep/african_qa/baselines/reader/evaluate_reader.py --retrieval-file /home/oogundep/african_qa/runs/run.xqa.wol.test.fr.human_translation.mdpr-hybrid.json --reader dpr --retriever score --settings dpr --output-file /home/oogundep/african_qa/runs/reader/out.json --model-name /home/oogundep/african_qa/models/hgf --tokenizer-name bert-base-multilingual-uncased --topk-em 10 20 50 100