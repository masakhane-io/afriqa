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

### Processing Wikipedia dumps for Retrieval

The already processed dumps are available on HuggingFace 😊, It is recommended to use this exact corpora to be able to reproduce the baseline results.
To download:

- [English](https://huggingface.co/datasets/ToluClassics/masakhane_wiki_100/resolve/main/masakhane_wiki_100-english/corpus.jsonl)
- [French](https://huggingface.co/datasets/ToluClassics/masakhane_wiki_100/resolve/main/masakhane_wiki_100-french/corpus.jsonl)

However, to prepare the Wikipedia retrieval corpus yourself, consult [docs/process_wiki_dumps.md](docs/process_wiki_dumps.md).

### Data Splits

For all languages, there are three splits.

The original splits were named `train`, `dev` and `test` and they correspond to the `train`, `validation` and `test` splits.

The splits have the following sizes :

| Language        | train | dev | test |
|-----------------|------:|-----------:|-----:|
| Bemba         |  502 | 503 |  314 |
| Fon          |  427 | 428 |  386 |
| Hausa           |  435 | 436 |  300 |
| Igbo            |  417 | 418 |  409 |
| Kinyarwanda     	  |   407 |  409 |  347 |
| Swahili       |  415 |   417 |  302 |
| Twi          |  451 |   452 |  490 |
| Wolof        |  503 |    504 |  334 |
| Yoruba          |  360 |   361 |  332 |
| Zulu        |  387 |    388 |  325 |
| <b>Total</b>    |  <b>4333</b>  |  <b>4346</b>  |<b>3560</b>  |


## BibTeX entry and citation info

```
Coming soon...
```