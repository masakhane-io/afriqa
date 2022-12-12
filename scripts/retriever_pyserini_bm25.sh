collection_path=collections
index_path=indexes

for lang in fr; do
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
