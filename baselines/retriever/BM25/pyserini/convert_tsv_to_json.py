import os
import jsonlines
import argparse
import pandas as pd
from tqdm import tqdm

TARGET_LANG={
    "bemba" : "english", 
    "fon" : "english", 
    "hausa" : "english", 
    "igbo" : "english", 
    "kinyarwanda" : "english", 
    "twi" : "english", 
    "yoruba" : "english", 
    "swahili" : "english", 
    "wolof" : "english", 
    "zulu" : "english"
}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse Annotation File into JSON"
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
        choices=["bemba", "fon", "hausa", "igbo", "kinyarwanda", "twi", "yoruba", "swahili", "wolof", "zulu"]
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


    # write strings to jsonl file
    with open(
        os.path.join(args.output_dir, f"queries.xqa.{args.lang}" + f".{args.output_file_extension}"), mode="w"
    ) as writer:
        for index, row in lang_df.iterrows():
            if pd.isnull(row[0]):
                question = ""
                print(f"Question at row {index} is empty")
            else:
                question = row[0]

            if str(row[3]).strip().lower() == "no gold paragraph" or pd.isnull(row[3]):
                answer = "[]"
            else:
                answer = f"['{row[3]}']"
                
            writer.write(question + "\t" + answer + "\n")
    

    with open(
        os.path.join(args.output_dir, f"queries.xqa.{args.lang}.{TARGET_LANG[args.lang]}" + f".{args.output_file_extension}"), mode="w"
    ) as writer:
        for index, row in lang_df.iterrows():
            if pd.isnull(row[1]):
                question = ""
                print(f"Question at row {index} is empty")
            else:
                question = row[1]

            if row[2].strip().lower() == "no gold paragraph" or pd.isnull(row[2]):
                answer = "[]"
            else:
                answer = f"['{row[2]}']"
            
            writer.write(question + "\t" + answer + "\n")


if __name__ == "__main__":
    main()