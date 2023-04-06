"""
Script to translate queries using NLLB.
How to: python nllb_translate.py --queries_file <path to csv/tsv file with queries> --lang <Source language e.g hausa> --output_file <path to output csv/tsv file>
"""
import os
import argparse
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import NllbTokenizer

flores_codes = {
    "bemba" : "bem_Latn",
    "bem" : "bem_Latn",
    "fon" : "fon_Latn",
    "hausa" : "hau_Latn",
    "hau" : "hau_Latn",
    "igbo" : "ibo_Latn",
    "ibo" : "ibo_Latn",
    "kinyarwanda" : "kin_Latn",
    "kin" : "kin_Latn",
    "twi" : "twi_Latn",
    "yoruba" : "yor_Latn",
    "yor" : "yor_Latn",
    "swahili" : "swh_Latn",
    "swa" : "swh_Latn",
    "wolof" : "wol_Latn",
    "wol" : "wol_Latn",
    "zulu" : "zul_Latn",
    "zul" : "zul_Latn",
    "french" : "fra_Latn",
    "fr" : "fra_Latn",
    "english" : "eng_Latn",  
    "en" : "eng_Latn",
}

def nllb_translate(translation_sentence: list, source_lang: str, target_lang) -> str:
    """
    Translates a sentence from a source language to English using NLLB.
    Args:
        translation_sentence (str): Sentence to be translated.
        lang (str): Source language.
    Returns:
        str: Translated sentence.
    """
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B", 
        src_lang=flores_codes[source_lang])
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-1.3B")
    inputs = tokenizer(translation_sentence, padding=True, truncation=True, return_tensors="pt")

    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[flores_codes[target_lang]], max_length=30
    )
    translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)  
    return translations

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate from Source to Pivot with NLLB"
    )

    parser.add_argument(
        "--queries_file",
        type=str,
        required=True,
        help="Path to file containing queries"
    )

    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Source language"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="File path to save translations"
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    assert os.path.exists(args.queries_file), f'The queries file path \{args.queries_file}\ does not exist.'

    if args.queries_file.endswith('csv'):
        lang_df = pd.read_csv(args.queries_file, sep=",", header=0, dtype=object)
    elif args.queries_file.endswith('tsv'):
        lang_df = pd.read_csv(args.queries_file, sep="\t", header=0, dtype=object)
    queries = lang_df['Original question in African language'].to_list()

    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B", 
        src_lang=flores_codes[args.lang])
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-1.3B")

    inputs = tokenizer(queries, padding=True, truncation=True, return_tensors="pt")

    print("Translating Queries...")
    if args.lang == 'fon' or args.lang == 'wolof':
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fra_Latn"], max_length=30
        )
    else:
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=30
        )
    
    translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)  
    lang_df['NLLB Translations'] = translations

    print(f"Saving to {args.output_file}...")
    if args.output_file.endswith('.tsv'):
        lang_df.to_csv(args.output_file, sep="\t", index=False)
    else:
        lang_df.to_csv(args.output_file, index=False)

    print("Done Translating")
            


if __name__ == "__main__":
    main()


