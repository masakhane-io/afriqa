split=$1
translation=$2
queries_dir=$3
device=$4
index_path=indexes
collection_path=collections
retrieval_model=castorini/mdpr-tied-pft-msmarco
index_link=https://huggingface.co/datasets/masakhane/afriqa-prebuilt-sparse-indexes/blob/main/

declare -A src_lang_to_pivot=(["ibo"]="en" ["hau"]="en" ["fon"]="fr" ["yor"]="en" ["swa"]="en" ["kin"]="en" ["zul"]="en" ["wol"]="fr" ["twi"]="en" ["bem"]="en")
declare -A pivot_lang_to_index=(["fr"]="frwiki-20220420-index-mdpr" ["en"]="enwiki-20220501-index-mdpr")
declare -A pivot_lang_to_bm25_index=(["fr"]="frwiki-20220420-index" ["en"]="enwiki-20220501-index")

export CUDA_VISIBLE_DEVICES=$device


for lang in fr en; do
    if [ $lang = "en" ]; then
        date="20220501"
    else
        date="20220420"
    fi

    wiki_index=${index_path}/${lang}wiki-${date}-index
    echo ${wiki_index}
    if [ ! -d ${wiki_index} ]; then
        echo "Downloading prebuilt BM25 indexes from huggingface"

        wget ${index_link}/${lang}wiki-${date}-index.tar.gz -P indexes/
    fi

done


#==================================================================================================#
# Hybrid Retrieval using MDPR and BM25 for French and English
#==================================================================================================#

for source_lang in bem hau ibo kin swa twi yor zul fon wol;
do
    echo "================================================="
    echo "[INFO] The Pivot language for ${source_lang} is ${src_lang_to_pivot[$source_lang]}"
    echo "[INFO] Searching Index: ${pivot_lang_to_index[${src_lang_to_pivot[$source_lang]}]}"

    trec_run_file=runs/run.xqa.${source_lang}.${split}.${src_lang_to_pivot[$source_lang]}.$translation.mdpr-hybrid.trec
    json_run_file=runs/run.xqa.${source_lang}.${split}.${src_lang_to_pivot[$source_lang]}.$translation.mdpr-hybrid.json
    queries=$queries_dir/queries.xqa.${source_lang}.${split}.${src_lang_to_pivot[$source_lang]}.$translation.txt

    python baselines/retriever/dense/pyserini/search_hybrid.py \
        dense --index ${index_path}/frwiki-20220420-index-mdpr \
        --encoder ${retrieval_model} --encoder-class auto \
        sparse --index ${index_path}/${pivot_lang_to_bm25_index[${src_lang_to_pivot[$source_lang]}]} \
        fusion --alpha 0.95 \
        run --topics ${queries} \
        --output ${trec_run_file} \
        --threads 32 \
        --batch-size 128 \
        --hits 100

    # Convert TREC Run File to Readable JSON format
    echo "[INFO] Converting TREC Run File to Readable JSON format"
    python3 baselines/retriever/BM25/pyserini/convert_trec_run_to_dpr_retrieval_run.py \
        --topics ${queries} \
        --index ${index_path}/${pivot_lang_to_bm25_index[${src_lang_to_pivot[$source_lang]}]} \
        --input ${trec_run_file} \
        --output ${json_run_file} \
        --store-raw

    # Evaluate the retriever
    echo "[INFO] Multilingual Dense Passage Retrieval Evaluation Results"
    python -m pyserini.eval.evaluate_dpr_retrieval --topk 10 20 100 --retrieval ${json_run_file}

    echo "================================================="
done