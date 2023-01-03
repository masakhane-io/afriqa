source_lang=$1
translation=$2
index_path=indexes
collection_path=collections
index_link=https://huggingface.co/datasets/ToluClassics/masakhane-xqa-prebuilt-sparse-indexes/resolve/main

declare -A src_lang_to_pivot=(["ibo"]="en" ["hau"]="en" ["fon"]="fr" ["yor"]="en" ["swa"]="en" ["kin"]="en" ["zul"]="en" ["wol"]="fr" ["twi"]="en" ["bem"]="en")
declare -A pivot_lang_to_index=(["fr"]="frwiki-20220420-index" ["en"]="enwiki-20220501-index")

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

        wget ${index_link}/enwiki-20220501-index.tar.gz -P indexes/
    fi

done

echo "================================================="
echo "[INFO] The Pivot language for ${source_lang} is ${src_lang_to_pivot[$source_lang]}"
echo "[INFO] Searching Index: ${pivot_lang_to_index["en"]}"

trec_run_file=runs/run.xqa.${source_lang}.${src_lang_to_pivot[$source_lang]}.$translation.bm25.trec
json_run_file=runs/run.xqa.${source_lang}.${src_lang_to_pivot[$source_lang]}.$translation.bm25.json
queries=queries/queries.xqa.${source_lang}.${src_lang_to_pivot[$source_lang]}.txt

# Search index and generate a TREC format run file
python3 baselines/retriever/BM25/pyserini/search.py \
    --index indexes/${pivot_lang_to_index["en"]} \
    --topics ${queries} \
    --output ${trec_run_file}

# Convert TREC Run File to Readable JSON format
echo "[INFO] Converting TREC Run File to Readable JSON format"
python3 baselines/retriever/BM25/pyserini/convert_trec_run_to_dpr_retrieval_run.py \
    --topics ${queries} \
    --index indexes/${pivot_lang_to_index["en"]} \
    --input ${trec_run_file} \
    --output ${json_run_file} \
    --store-raw

# Evaluate the retriever
echo "[INFO] BM25 Evaluation Results"
python -m pyserini.eval.evaluate_dpr_retrieval --topk 10 20 50 100 --retrieval ${json_run_file}

echo "================================================="
