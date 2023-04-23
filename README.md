# AfriQA: Cross-lingual Open-Retrieval Question Answering for African Languages

[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg


AfriQA is the first cross-lingual question answering (QA) dataset with a focus on African languages. The dataset includes over 12,000 XOR QA examples across 10 African languages, making it an invaluable resource for developing more equitable QA technology.
African languages have historically been underserved in the digital landscape, with far less in-language content available online. This makes it difficult for QA systems to provide accurate information to users in their native language. However, cross-lingual open-retrieval question answering (XOR QA) systems can help fill this gap by retrieving answer content from other languages.
AfriQA focuses specifically on African languages where cross-lingual answer content is the only high-coverage source of information. Previous datasets have primarily focused on languages where cross-lingual QA augments coverage from the target language, but AfriQA highlights the importance of African languages as a realistic use case for XOR QA.

## Languages

There are currently 10 languages covered in AfriQA:

- Bemba (bem)
- Fon (fon)
- Hausa (hau)
- Igbo (ibo)
- Kinyarwanda (kin)
- Swahili (swa)
- Twi (twi)
- Wolof (wol)
- Yorùbá (yor)

## Dataset Download

Question-answer pairs for each language and `train-dev-test` split are in the [data directory](data/queries) in `jsonlines` format.

- Dataset Naming Convention ==> `queries.afriqa.{lang_code}.{en/fr}.{split}.json`
- Data Format:
    - id : Question ID
    - question : Question in African Language
    - translated_question : Question translated into a pivot language (English/French)
    - answers : Answer in African Language
    - lang : Datapoint Language (African Language) e.g `bem`
    - split : Dataset Split
    - translated_answer : Answer in Pivot Language
    - translation_type : Translation type of question and answers


    ```bash
    {   "id": 0, 
        "question": "Bushe icaalo ca Egypt caali tekwapo ne caalo cimbi?", 
        "translated_question": "Has the country of Egypt been colonized before?", 
        "answers": "['Emukwai']", 
        "lang": "bem", 
        "split": "dev", 
        "translated_answer": "['yes']", 
        "translation_type": "human_translation"
        }
    ```

## Environment and Repository Setup

- Set up a virtual environment using Conda or Virtualenv or

    ```bash
    conda create -n xor_qa_venv python=3.9 anaconda
    conda activate xor_qa_venv
    ```
    or
    ```bash
    python3 -m venv xor_qa_venv
    source xor_qa_venv/bin/activate
    ```
- Clone the repo

    ```bash
    git clone https://github.com/ToluClassics/masakhane_xqa --recurse-submodules
    ```
- Install Requirements

    ```bash
    pip install -r requirements.txt
    ```
## Source Data -- WIkipedia

The English and French passages for this project are drawn from Wikipedia snapshots of 2022-05-01 and 2022-04-20 respectively, and are downloaded from the [Internet Archive](https://archive.org/) to enable open-domain experiments.
The raw documents can be downloaded from the following URLS:

- https://archive.org/download/enwiki-20220501/enwiki-20220501-pages-articles-multistream.xml.bz2
- https://archive.org/download/frwiki-20220420/frwiki-20220420-pages-articles-multistream.xml.bz2

### Processing Wikipedia dumps

The already processed dumps are available on HuggingFace 😊, It is recommended to use this exact corpora to be able to reproduce the baseline results.
To download:

- [English](https://huggingface.co/datasets/ToluClassics/masakhane_wiki_100/resolve/main/masakhane_wiki_100-english/corpus.jsonl)
- [French](https://huggingface.co/datasets/ToluClassics/masakhane_wiki_100/resolve/main/masakhane_wiki_100-french/corpus.jsonl)



## BibTeX entry and citation info

```
Coming soon...
```