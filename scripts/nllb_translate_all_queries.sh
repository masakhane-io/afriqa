declare -A src_lang_to_full=(["ibo"]="igbo" ["hau"]="hausa" ["kin"]="kinyarwanda" ["zul"]="zulu" ["wol"]="wolof" ["twi"]="twi" ["bem"]="bemba" ["fon"]="fon" ["yor"]="yoruba" ["swa"]="swahili" )
declare -A src_lang_to_pivot=(["ibo"]="english" ["hau"]="english" ["fon"]="french" ["yor"]="english" ["swa"]="english" ["kin"]="english" ["zul"]="english" ["wol"]="french" ["twi"]="english" ["bem"]="english")

for lang in yor swa;
do
    CUDA_VISIBLE_DEVICES=4 python3 translation_script/nllb_translate.py --queries_file queries/official_topics/$lang/test.$lang.tsv \
        --lang ${src_lang_to_full[$lang]} --output_file queries/nllb_topics_new/test.$lang.tsv

    python3  translation_script/translate_queries_gmt.py --questions_file_path queries/official_topics/$lang/test.$lang.tsv --source ${src_lang_to_full[$lang]} --pivot ${src_lang_to_pivot[$lang]} \
        --output_file_path queries/gmt_topics_new/test.$lang.tsv
done


# CUDA_VISIBLE_DEVICES=0 python3 -m pyserini.encode   input --corpus collections/a --fields text --shard-id 0 --shard-num 1 output  --embeddings /store2/scratch/oogundep/indexes/enwiki-20220501-index-mdpr/a encoder --encoder castorini/mdpr-tied-pft-msmarco --fields text --batch 128 --encoder-class 'auto'  --fp16