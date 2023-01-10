lang=$1
translation_type=$2
output_dir=queries/processed_topics

for split in "dev" "train" "test"; do
    python3 baselines/retriever/BM25/pyserini/convert_tsv_to_query_file.py \
        --input_annotation_file /home/oogundep/african_qa/queries/raw_topics/$split.$lang.tsv \
        --lang $lang \
        --output_dir $output_dir \
        --translation_type $translation_type \
        --output_file_extension txt --split $split
done
