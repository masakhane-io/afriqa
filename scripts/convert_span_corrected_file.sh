translation_type=$1
translation_dir=$2
output_dir=queries/processed_topics_new

for lang in "kin" "ibo" "hau" "zul" "wol" "twi" "bem" "fon" "yor" "swa"; do
    for split in "test"; do
        python3 baselines/retriever/BM25/pyserini/convert_tsv_to_query_file.py \
            --input_annotation_file $translation_dir/$split.$lang.tsv \
            --lang $lang \
            --output_dir $output_dir \
            --translation_type $translation_type \
            --output_file_extension txt --split $split \
            --translation_row_index -1
    done
done
