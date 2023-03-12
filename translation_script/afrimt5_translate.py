"""
Script to translate queries using AfriMT5.
How to: python afrimt5_translate.py  --queries_file <path to csv/tsv file with queries> --source_lang <Source language e.g hausa> --target_lang <Target language e.g hausa> --output_file <path to output csv/tsv file>
"""
import os
import random
import argparse
import pandas as pd
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer,  M2M100Tokenizer, M2M100ForConditionalGeneration

random.seed(65)

lang_2_modelname = {
    "bemba": "",
    "fon": "masakhane/afribyt5_fon_fr_news",
    "hausa": "masakhane/m2m100_418M_hau_en_rel_news_ft",
    "igbo": "masakhane/m2m100_418M_ibo_en_rel_news_ft",
    "kinyarwanda": "",
    "twi": "masakhane/m2m100_418M_twi_en_rel_news_ft",
    "yoruba": "masakhane/m2m100_418M_yor_en_rel_news_ft",
    "swahili": "masakhane/m2m100_418M_hau_en_rel_news_ft",
    "wolof": "masakhane/m2m100_418M_wol_fr_rel_news_ft",
    "zulu": "masakhane/m2m100_418M_zul_en_rel_news_ft",
}

lang_2_lang_code = {
    "hausa": "ha",
    "igbo": "ig",
    "yoruba": "yo",
    "swahili": "sw",
    "zulu": "zu",
    "kinyarwanda": "kin",
    "bemba": "bem",
    "fon": "fa",
    "twi": "tl",
    "wolof": "wo",
    "english": "en",
    "french": "fr",
}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate from Source to Pivot with AfriMT5"
    )

    parser.add_argument(
        "--queries_file",
        type=str,
        required=True,
        help="Path to file containing queries"
    )

    parser.add_argument(
        "--source_lang",
        type=str,
        required=True,
        help="Source language"
    )

    parser.add_argument(
        "--target_lang",
        type=str,
        required=True,
        help="Target language"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="File path to save translations"
    )
    parser.add_argument(
        "--decoding_strategy",
        type=str,
        default="beam",
        help="Decoding strategy to use. Options are beam and nucleus",
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    assert os.path.exists(args.queries_file), f'The queries file path \{args.queries_file}\ does not exist.'
    
    if args.queries_file.endswith('csv'):
        lang_df = pd.read_csv(args.queries_file, sep=",", header=0)
    elif args.queries_file.endswith('tsv'):
        lang_df = pd.read_csv(args.queries_file, sep="\t", header=0)
    queries = lang_df['Original question in African language'].to_list()


    config = AutoConfig.from_pretrained(lang_2_modelname[args.source_lang])
    tokenizer = M2M100Tokenizer.from_pretrained(lang_2_modelname[args.source_lang], use_fast=True)
    model = M2M100ForConditionalGeneration.from_pretrained(lang_2_modelname[args.source_lang], config=config)

     # model.resize_token_embeddings(len(tokenizer))
    tokenizer.src_lang = lang_2_lang_code[args.source_lang]

    # forced_bos_token=lang_2_lang_code[args.target_lang]
    # forced_bos_token_id = (
    #         tokenizer.lang_code_to_id[forced_bos_token] if forced_bos_token is not None else None
    #     )

    # model.config.forced_bos_token_id = forced_bos_token_id
    model.resize_token_embeddings(len(tokenizer))


    inputs = tokenizer(queries, padding=True, truncation=True, return_tensors="pt")


    print("Translating Queries...")
    if args.decoding_strategy == "beam":
        translated_tokens = model.generate(
            **inputs, max_length=64, num_beams=5, early_stopping=True, do_sample = True, forced_bos_token_id=tokenizer.get_lang_id(lang_2_lang_code[args.target_lang])
        )
    else:
        translated_tokens = model.generate(
                **inputs, max_length=64, do_sample=True, top_p = 0.8, num_return_sequences=1, early_stopping=True, decoder_start_token_id = model.config.bos_token_id
            )
    
    translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)  

    lang_df['AfriMT5 Translations'] = translations

    print(f"Saving to {args.output_file}...")
    if args.output_file.endswith('.tsv'):
        lang_df.to_csv(args.output_file, sep="\t", index=False)
    else:
        lang_df.to_csv(args.output_file, index=False)

    print("Done Translating")
            


if __name__ == "__main__":
    main()


    # python translation_script/afrimt5_translate.py --queries_file /home/oogundep/african_qa/queries/official_topics/ibo/test.ibo.tsv --source_lang igbo --target_lang english --output_file queries/afrimt5_topics/test.ibo.tsv --decoding_strategy beam
