import argparse
import json
import os
import re
import jsonlines
import pandas as pd


def process_string(sentence: str):
    sentence = sentence.strip().lower()
    sentence = sentence = " ".join(re.split("\s+", sentence, flags=re.UNICODE))
    return sentence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_gold_data_split", type=str, required=True)
    parser.add_argument("--input_translation_directory", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--translation", type=str, required=True)
    args = parser.parse_args()

    # Open the gold passages
    with jsonlines.open(args.input_gold_data_split, "r") as f:
        gold_passages = list(f)
    
    # Open the translation files
    trans_df = pd.read_csv(os.path.join(args.input_translation_directory, f"{args.split}.{args.lang}.tsv"), sep="\t")

    for passage in gold_passages:
        row = trans_df.loc[trans_df["Question translated into pivot language"] == passage['question_translated']]

        if not 'human' in args.translation:
            try:
                passage['question_translated'] = row.iloc[0]['Translated Question in English']
            except KeyError as e:
                passage['question_translated'] = row.iloc[0]['NLLB Translations']


    # Write the output file
    with jsonlines.open(os.path.join(args.output_directory, f"{args.split}.{args.lang}.{args.translation}.json"), "w") as f:
        for passage in gold_passages:
            f.write(passage)
    


if __name__ == "__main__":
    main()