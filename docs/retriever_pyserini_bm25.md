# Retriever Baselines (BM25) on Pyserini 🦆

This page contains information on how to run BM25 retrieval baselines on {{Data}} using the [Pyserini](https://github.com/castorini/pyserini) Information Retrieval toolkit. Pyserini is built on the popular Lucene search library.

## Installation

- To install via PyPI

    ```bash
    pip install pyserini
    ```
- For a more detailed development installation, consult [their documentation](https://github.com/castorini/pyserini/blob/master/docs/installation.md)

## Indexing the Data (Skip this section if you're using prebuilt indexes)

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
    language="en"
    collection="collection/enwiki-20220501-jsonl"
    index_name="indexes/enwiki-20220501-index"

    python -m pyserini.index.lucene \
        --collection MrTyDiCollection \
        --input ${collection} \
        --language ${language} \
        --index indexes/${index_name} \
        --generator DefaultLuceneDocumentGenerator \
        --threads 20 \
        --storePositions --optimize --storeRaw
    ```

## Simple Searcher (This is for sanity checking the index if you want to 😊)

After indexing is done, we can search the data using the following code snippet.

```python
import json
from pyserini.search.lucene import LuceneSearcher

num_hits=10
language="en"
index_name="indexes/enwiki-20220501-index"
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

# Batch Retrieval Run and Evaluation ⌛️

To Evaluate BM25 retriever accuracy over a set of question pairs for a particular language, you can run this [script](scripts/retriever_pyserini_bm25.sh) as shown below by passing the `iso-3 language code` and `translation` type as arguments;

```bash
bash scripts/retriever_pyserini_bm25.sh ibo human_translation
```

What this script does is:

- Download the prebuilt BM25 indexes from [huggingface](https://huggingface.co/datasets/ToluClassics/masakhane-xqa-prebuilt-sparse-indexes)
- Run retrieval on the indexes using the provided queries and generate a trec run file; a TREC run file is a txt that shows retrieval runs in the format `query_id Q0 document_id rank relevance_score run_tag` e.g

    ```txt
    0 Q0 10823634 1 11.780300 Anserini
    0 Q0 8289940 2 11.101400 Anserini
    ```
- Convert the TREC run file into a DPR style json file e.g

    ```json
    {
        "0": {
            "question": "What is a pharoah in Egypt?",
            "answers": [
                "monarchs"
            ],
            "contexts": [
                {
                    "docid": "10823634",
                    "score": "11.780300",
                    "text": "continue their task. A herald announces the procession of Pharaoh Ramesses II, forcing the slaves to kneel before him. During the procession a slave collasped in exhaustion, resulting the exhausted slave crushed to death on Pharoah's orders, which Miriam witnessed in horror. And as the procession leaves, Miriam cried to God to send the Deliverer to liberate them. After the devastating Plagues of Egypt, Moses, along with his brother Aaron, stood before Pharoah. Since the plagues convince him to let his people go, what they expected, but instead Ramesses orders that more work would be laid on them. Moses ",
                    "has_answer": false
                },
    ```
- Evaluate the retriever and generates the top-10, top-20 and top-100 accuracy

