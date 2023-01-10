CUDA_VISIBLE_DEVICES=0 python -m pyserini.encode \
    input --corpus /home/oogundep/african_qa/collections/enwiki-20220501-jsonl \
    --fields text \
    --shard-id 0 \
    --shard-num 1 \
    output --embeddings indexes/enwiki-20220501-index-mdpr \
    --to-faiss \
    encoder --encoder castorini/mdpr-tied-pft-msmarco \
    --fields text \
    --batch 64 \
    --fp16
