# Retriever Baselines (BM25) on Pyserini

This page contains information on how to run BM25 retrieval baselines on {{Data}} using the [Pyserini](https://github.com/castorini/pyserini) Information Retrieval toolkit. Pyserini is built on the popular Lucene search library.

## Installation

- To install via PyPI

    ```bash
    pip install pyserini
    ```
- For a more detailed development installation, consult [their documentation](https://github.com/castorini/pyserini/blob/master/docs/installation.md)

## Indexing the Data 

To build an [inverted index](https://nlp.stanford.edu/IR-book/html/htmledition/a-first-take-at-building-an-inverted-index-1.html) to store the data and search the data effectively using the Pyserini toolkit. Follow the steps below:

- Format the data into a [jsonlines](https://jsonlines.org/) file such that each line of the `json or jsonl` file is a valid json e.g
    
    ```bash
    {"id": "1", "contents": "..."}
    {"id": "1", "contents": "..."}
    ```

    the data can also be in some other formats e.g. Folder with each [JSON in their own file like this](https://github.com/castorini/pyserini/blob/master/tests/resources/sample_collection_json).
    For additional information consult this [documentation](https://github.com/castorini/pyserini#how-do-i-index-and-search-my-own-documents)

- Run the following command :
    
    ```bash
    language=""
    collection=""
    index_name=""

    python -m pyserini.index.lucene \
        --collection JsonCollection \
        --input ${collection} \
        --language ${language} \
        --index indexes/${index_name} \
        --generator DefaultLuceneDocumentGenerator \
        --threads 4 \
        --storePositions --storeDocvectors --storeRaw
    ```

## Simple Searcher

After indexing is done, we can search the data using the following code snippet.

```python
import json
from pyserini.search.lucene import LuceneSearcher

num_hits=10
index_name=""
query="how many countries are in Africa"

# search the index
searcher = LuceneSearcher(f'indexes/{index_name}')
searcher.set_language(language)
hits = searcher.search(query, k=num_hits)

for i in range(len(hits)):
    print(f'{i+1:2} {hits[i].docid:4} {hits[i].score:.5f}')
    print(f"\tcontents: {json.loads(hits[i].raw)['contents']}")
    print("==============================================")
```

you should get something like this::

```bash
 1 doc2 0.25620
   contents: ...
 ===============================
 2 doc3 0.23140
   contents: ...
```

## Batch Retrieval Run

## Evaluate