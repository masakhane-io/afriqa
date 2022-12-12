# Processing Wiki Dumps

This document contains information on how to convert downloaded XML wikipedia dumps into 100 token long passages stored in JSON files.

For processing, we extract the Wikipedia articles into multiple jsonlines file, The articles are then preprocessed, cleaned and stored in a SQLite database file after which they are chunked into 100 token long passages.
The processing pipeline adopted here is same as described in Section 4.1 of the [Dense Passage Retriever Paper](https://arxiv.org/pdf/2004.04906.pdf).

The pipeline has been bundled into this [script](scripts/download_process_dumps.sh). You can run using the code provided below:

```terminal
bash scripts/generate_process_dumps.sh /path/to/dir_containing_dumps
```

However, below is a step by step break down of the different steps in the processing pipeline for English:

1. Download the dumps into a specified file:

    ```terminal
    wget https://archive.org/download/enwiki-20220501/enwiki-20220501-pages-articles-multistream.xml.bz2 -P /path/to/dir
    ```

2. Use Wikiextractor (bundled into this repo as a submodule) to extract the Wikipedia articles into multiple Jsonlines 

    ```terminal
    git clone https://github.com/attardi/wikiextractor.git
    cd wikiextractor && git checkout e4abb4cbd01

    python3 WikiExtractor.py /path/to/your/enwiki-20220501-pages-articles-multistream.xml.bz2 --filter_disambig_pages --json -o /path/to/output/directory -s
    ```

3. Store data into SQLite database

    ```terminal
    python3 process/retriever/build_db.py /path/to/preprocessed/data/dir /path/to/db/enwiki-20220501.db
    ```

4. Chunk the articles in the database into 100 token long sequences/passages to improve answer extraction

    ```terminal
    python3 preprocess/retriever/wikipedia_generate_context_tsv.py --db_path /path/to/db/enwiki-20220501.db --output_path_100w  /path/to/tsv/enwiki-20220501.tsv --lang en
    ```

5. (Optional) Shard the data into multiple `jsonl` files to make indexing easy

    ```python3
    from utils import shard_tsv_data
    shard_tsv_data(tsv_file_path="/path/to/tsv/enwiki-20220501.tsv", output_file_path="/path/to/jsonl_shards", shard_size="1GB")
    ```

    This produces multiple jsonl files of size 1GB each

    To view the processed files:

    ```
    head -1 /path/to/jsonl_shards/docs-000.jsonl
    ```
    Output:
    ```
    {"docid":809223,"text":" The hockey rink's dimensions are × , ...", "title": "Bolshoy Ice Dome"}
    ```