translation_type=$1
query_dir=$2


for lang in fon;
do
    for retriever in mdpr;
    do
        echo "================================================="
        echo "Language: ${lang}"
        echo "Retriever: ${retriever}"
        echo queries/official_topics/${lang}/test.${lang}.tsv
        echo "================================================="

        python3 translation_script/full_trip_scores.py --reader_file runs/reader/dpr-reader-single-nq-base/reader.xqa.${lang}.${retriever}.${translation_type}.json \
            --query_file ${query_dir}/test.${lang}.tsv \
            --output_directory runs/reader/full_trip \
            --lang ${lang} \
            --translation_type ${translation_type} \
            --split test \
            --retriever ${retriever} 
    done
done