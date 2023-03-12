translation_type=$1

echo "================================================="
echo "Translation Type: ${translation_type}"
echo "================================================="

# for lang in wol fon
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
#             --topk-em 10 50 100 \
#             --model-name facebook/dpr-reader-single-nq-base \
#             --tokenizer-name facebook/dpr-reader-single-nq-base
#         echo "================================================="
#         echo "===> Done with ${lang} ${run_type}======"
#     done
# done

for lang in hau
do
    for run_type in bm25
    do  
        echo "================================================="
        echo "Language: ${lang}"
        echo "Run Type: ${run_type}"
        echo "================================================="
        CUDA_VISIBLE_DEVICES=1 python3 /home/oogundep/african_qa/baselines/reader/evaluate_reader.py --retrieval-file /home/oogundep/african_qa/runs/run.xqa.$lang.test.en.${translation_type}.$run_type.json \
            --reader dpr --retriever score --settings dpr \
            --output-file /home/oogundep/african_qa/runs/reader/dpr-reader-single-nq-base/reader.xqa.$lang.$run_type.${translation_type}.json  \
            --topk-em 10 50 100 \
            --model-name /home/oogundep/african_qa/models/dpr_mult_25.pth \
            --tokenizer-name bert-base-multilingual-uncased \
            --device cuda:0
        echo "================================================="
        echo "===> Done with ${lang} ${run_type}======"
    done
done


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