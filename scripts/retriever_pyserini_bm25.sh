collection_path=collections
index_path=indexes

for lang in fr en; do
    if [ $lang = "en" ]; then
        date="20220501"
    else
        date="20220420"
    fi

    python -m pyserini.index.lucene --collection MrTyDiCollection \
        --input ${collection_path}/${lang}wiki-${date}-jsonl \
        --language ${lang} \
        --index ${index_path}/${lang}wiki-${date}-index \
        --generator DefaultLuceneDocumentGenerator \
        --threads 20 \
        --storePositions --storeRaw --optimize
done

# from pyserini.index.lucene import IndexReader
# index_reader = IndexReader.from_prebuilt_index('robust04')
# index_reader.stats()

python3 baselines/retriever/BM25/pyserini/search.py
--index indexes/enwiki-20220501-index
--topics /home/oogundep/african_qa/queries/queries.xqa.hausa.english.txt
--output runs/run.xqa.hausa.english.bm25.trec

python3 baselines/retriever/BM25/pyserini/convert_trec_run_to_dpr_retrieval_run.py
--topics /home/oogundep/african_qa/queries/queries.xqa.hausa.english.txt
--index indexes/enwiki-20220501-index
--input runs/run.xqa.hausa.english.bm25.trec
--output runs/run.xqa.hausa.english.bm25.json

python -m pyserini.eval.evaluate_dpr_retrieval
--topk 100 --retrieval runs/run.xqa.hausa.english.bm25.json
