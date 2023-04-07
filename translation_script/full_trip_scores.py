import argparse
import json
import logging
import os
import pandas as pd
from tqdm import tqdm
from nllb_translate import nllb_translate
from translate_queries_gmt import google_translate

source_lang_2_pivot_lang = {
    "hausa": "english",
    "hau": "english",
    "igbo": "english",
    "ibo": "english",
    "kinyarwanda": "english",
    "kin": "english",
    "twi": "english",
    "yoruba": "english",
    "yor": "english",
    "swahili": "english",
    "swa": "english",
    "wolof": "french",
    "wol": "french",
    "zulu": "english",
    "zul": "english",
    "french": "english",
    "bemba": "english",
    "bem": "english",
    "fon": "french",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reader_file",
        type=str,
        required=True,
        help="Path to file containing reader predictions"
    )
    parser.add_argument(
        "--query_file",
        type=str,
        required=True,
        help="Path to file containing queries"
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Source language"
    )
    parser.add_argument(
        "--translation_type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Split of the dataset"
    )
    parser.add_argument(
        "--retriever",
        type=str,
        required=True,
        help="Retriever used"
    )
    args = parser.parse_args()

    assert os.path.exists(args.reader_file), f'The reader file path \{args.reader_file}\ does not exist.'
    assert os.path.exists(args.query_file), f'The query file path \{args.query_file}\ does not exist.'

    query_dataframe = pd.read_csv(args.query_file, sep="\t", header=0, dtype=object)

    with open(args.reader_file, 'r') as f:
        reader_predictions = json.load(f)

    updated_reader_predictions = []
    
    for i in tqdm(range(len(reader_predictions))):
        new_dict_reader = {}
        try:
            new_dict_reader["question"] = reader_predictions[i]['question']
            prediction = reader_predictions[i]['prediction']

            #row = query_dataframe.loc[query_dataframe["Question translated into pivot language"].str.strip() == reader_predictions[i]['question'].strip()]

            if args.translation_type.startswith("nllb"):
                row = query_dataframe.loc[query_dataframe["NLLB Translations"].str.strip() == reader_predictions[i]['question'].strip()]
            elif args.translation_type.startswith("google"):
                row = query_dataframe.loc[query_dataframe["Translated Question in English"].str.strip() == reader_predictions[i]['question'].strip()]

            new_dict_reader['answers'] = [row.iloc[0]['Answer translated into African language']]

            all_answers = list(prediction["DPR"].values())

            if args.translation_type.startswith("nllb"):
                all_answers = nllb_translate(all_answers, source_lang_2_pivot_lang[args.lang], args.lang)
            elif args.translation_type.startswith("google"):
                all_answers = google_translate(all_answers, source_lang_2_pivot_lang[args.lang], args.lang)


            new_prediction = { "DPR": {} }
            for i, key_item in enumerate(prediction["DPR"].keys()):
                new_prediction["DPR"][key_item] = all_answers[i]

            new_dict_reader['prediction'] = new_prediction
        except IndexError as e:
            print(reader_predictions[i])
            print(f'IndexError: {e}')
            continue
        
        updated_reader_predictions.append(new_dict_reader)
        # break

    with open(os.path.join(args.output_directory, f'{args.split}_{args.translation_type}_{args.lang}_{args.retriever}_reader_predictions.json'), 'w') as f:
        json.dump(updated_reader_predictions, f)
        


if __name__ == '__main__':
    main()