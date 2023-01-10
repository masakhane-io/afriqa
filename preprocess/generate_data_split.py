import os
import jsonlines
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

TARGET_LANG={
    "bem" : "en", 
    "fon" : "en", 
    "hau" : "en", 
    "ibo" : "en", 
    "kin" : "en", 
    "twi" : "en", 
    "yor" : "en", 
    "swa" : "en", 
    "wol" : "fr", 
    "zul" : "en"
}

train_test_split_ratio = {}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Data Split"
    )
    parser.add_argument(
        "--input_annotation_file",
        type=str,
        required=True,
        help="File path to the file containing the annotations",
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="language",
        choices=["bem", "fon", "hau", "ibo", "kin", "twi", "yor", "swa", "wol", "zul"]
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="path to store output processed file"
    )
    parser.add_argument(
        "--output_file_extension", type=str, default="json", choices=["tsv", "txt"], help="path to store output processed file"
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    assert os.path.exists(args.input_annotation_file), f"The input annotation file: \'{args.input_annotation_file}\' does not exist"

    if args.input_annotation_file.endswith("csv"):
        lang_df = pd.read_csv(args.input_annotation_file, sep=",", header=0)
    elif args.input_annotation_file.endswith("tsv"):
        lang_df = pd.read_csv(args.input_annotation_file, sep="\t", header=0)
    
    negatives_df = lang_df[lang_df[f'Answer in pivot language'] == "No Gold Paragraph"]
    positives_df = lang_df[lang_df[f'Answer in pivot language'] != "No Gold Paragraph"]

    assert len(negatives_df) + len(positives_df) == len(lang_df)

    train_dev_positives, test_positives = train_test_split(positives_df, test_size=0.4, random_state=1, shuffle=True)
    train_positives, dev_positives = train_test_split(train_dev_positives, test_size=0.5, random_state=1, shuffle=True)

    if len(negatives_df) > 250:
        train_dev_negatives, _ = train_test_split(negatives_df, test_size=0.2, random_state=1, shuffle=True)
        train_negatives, dev_negatives = train_test_split(train_dev_negatives, test_size=0.5, random_state=1, shuffle=True)
    else:
        train_negatives, dev_negatives = train_test_split(negatives_df, test_size=0.5, random_state=1, shuffle=True)

    train_df = pd.concat([train_positives, train_negatives])
    dev_df = pd.concat([dev_positives, dev_negatives])
    test_df = test_positives

    os.makedirs(os.path.join(args.output_dir, "topics", args.lang), exist_ok=True)

    train_df.drop(columns=[f'Answer in pivot language', 'Answer translated into African language', 'Action for data processing (answer)'], inplace=True)
    dev_df.drop(columns=[f'Answer in pivot language',  'Answer translated into African language', 'Action for data processing (answer)'], inplace=True)
    test_df.drop(columns=[f'Answer in pivot language', 'Answer translated into African language', 'Action for data processing (answer)'], inplace=True)

    train_df.to_csv(os.path.join(args.output_dir, "topics", args.lang ,f"train.{args.lang}.{args.output_file_extension}"), sep="\t", index=False)
    dev_df.to_csv(os.path.join(args.output_dir, "topics", args.lang ,f"dev.{args.lang}.{args.output_file_extension}"), sep="\t", index=False)
    test_df.to_csv(os.path.join(args.output_dir, "topics", args.lang ,f"test.{args.lang}.{args.output_file_extension}"), sep="\t", index=False)


    print(f"Train: {len(train_df)}")
    print(f"Dev: {len(dev_df)}")
    print(f"Test: {len(test_df)}")

    print(f"Train Positives: {len(train_positives)}")
    print(f"Train Negatives: {len(train_negatives)}")
    print(f"Dev Positives: {len(dev_positives)}")
    print(f"Dev Negatives: {len(dev_negatives)}")
    print(f"Test Positives: {len(test_positives)}")


    
if __name__ == "__main__":
    main()
    
