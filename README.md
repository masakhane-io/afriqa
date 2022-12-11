# Cross-lingual Question Answering for African Languages


## Source Data

The English and French passages for this project are drawn from Wikipedia snapshots of 2022-05-01 and 2022-04-20 respectively, and are downloaded from the [Internet Archive](https://archive.org/) to enable open-domain experiments.
The raw documents can be downloaded from the following URLS:

- https://archive.org/download/enwiki-20220501/enwiki-20220501-pages-articles-multistream.xml.bz2
- https://archive.org/download/frwiki-20220420/frwiki-20220420-pages-articles-multistream.xml.bz2


## Processing dumps

For processing, we extract the Wikipedia articles into multiple jsonlines file, The articles are then preprocessed, cleaned and stored in a SQLite database file after they are Chunked into 100 token long passages. 

The processing pipeline has been bundled into this [script](scripts/generate_process_dumps.sh). You can run using the code provided below:

```terminal
bash scripts/generate_process_dumps.sh /path/to/wiki_dir
```

Here is a step by step break down of the different steps in the processing pipeline for English:

1. Download the dumps into a specified file:

    ```terminal
    wget https://archive.org/download/enwiki-20220501/enwiki-20220501-pages-articles-multistream.xml.bz2 -P /path/to/wiki_dir
    ```

2. Use Wikiextractor (it is bundled into these repo as a submodule) to extract the Wikipedia articles into multiple Jsonlines 

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
    python3 preprocess/retriever/wikipedia_generate_context_tsv.py --db_path /path/to/db/enwiki-20220501.db --output_path_100w  /path/to/tsv/enwiki-20220501.tsv
    ```

5. (Optional) Shard the data into multiple `jsonl` files to make indexing easy

    ```python3
    from utils import shard_tsv_data
    shard_tsv_data(tsv_file_path="/path/to/tsv/enwiki-20220501.tsv", output_file_path="/path/to/jsonl_shards", shard_size="1GB")
    ```

    This produces multiple jsonl files of size 1GB each

## Retriever

### BM25