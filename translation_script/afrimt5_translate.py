"""
Script to translate queries using AfriMT5.
How to: python afrimt5_translate.py --model_name <model name or path> --queries_file <path to csv/tsv file with queries> --source_lang <Source language e.g hausa> --target_lang <Target language e.g hausa> --output_file <path to output csv/tsv file>
"""
import os
import argparse
import pandas as pd
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate from Source to Pivot with AfriMT5"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path"
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

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, config=config)

    model.resize_token_embeddings(len(tokenizer))
    tokenizer.src_lang = args.source_lang
    tokenizer.tgt_lang = args.target_lang


    inputs = tokenizer(queries, padding=True, truncation=True, return_tensors="pt")

    print("Translating Queries...")
    translated_tokens = model.generate(
            **inputs, max_length=30
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
