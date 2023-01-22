declare -A src_lang_to_full=(["ibo"]="igbo" ["hau"]="hausa" ["kin"]="kinyarwanda" ["zul"]="zulu" ["wol"]="wolof" ["twi"]="twi" ["bem"]="bemba")

for lang in kin
do
    CUDA_VISIBLE_DEVICES=0 python3 translation_script/nllb_translate.py --queries_file queries/raw_topics/test.$lang.csv \
        --lang ${src_lang_to_full[$lang]} --output_file queries/nllb_topics/test.$lang.tsv 
done


# CUDA_VISIBLE_DEVICES=0 python3 -m pyserini.encode   input --corpus collections/a --fields text --shard-id 0 --shard-num 1 output  --embeddings /store2/scratch/oogundep/indexes/enwiki-20220501-index-mdpr/a encoder --encoder castorini/mdpr-tied-pft-msmarco --fields text --batch 128 --encoder-class 'auto'  --fp16