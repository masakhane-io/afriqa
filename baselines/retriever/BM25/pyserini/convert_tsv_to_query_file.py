import os
import jsonlines
import argparse
import pandas as pd
from tqdm import tqdm

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
        choices=["bem", "fon", "hau", "ibo", "kin", "twi", "yor", "swa", "wol", "zul"]
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="path to store output processed file"
    )
    parser.add_argument(
        "--translation_type", type=str, required=True, help="translation type e.g gmt or hmt"
    )
    parser.add_argument(
        "--output_file_extension", type=str, default="txt", choices=["tsv", "txt"], help="path to store output processed file"
    )
    parser.add_argument(
        "--split", type=str, required=True, choices=["train", "dev", "test"], help="data split"
    )

    parser.add_argument(
        "--row_index", type=int, help="index of the row to be processed", default=1
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


    # # write strings to jsonl file
    # with open(
    #     os.path.join(args.output_dir, f"queries.xqa.{args.lang}.{args.split}" + f".{args.output_file_extension}"), mode="w"
    # ) as writer:
    #     for index, row in lang_df.iterrows():
    #         if pd.isnull(row[0]):
    #             question = ""
    #             print(f"Question at row {index} is empty")
    #         else:
    #             question = row[0]

    #         if str(row[3]).strip().lower() == "no gold paragraph" or pd.isnull(row[3]):
    #             answer = "[]"
    #         else:
    #             answer = f"['{row[3]}']"
                
    #         writer.write(question + "\t" + answer + "\n")
    

    with open(
        os.path.join(args.output_dir, f"queries.xqa.{args.lang}.{args.split}.{TARGET_LANG[args.lang]}.{args.translation_type}" + f".{args.output_file_extension}"), mode="w"
    ) as writer:
        for index, row in lang_df.iterrows():
            if pd.isnull(row[args.row_index]):
                question = ""
                print(f"Question at row {index} is empty")
            else:
                question = row[args.row_index]

            if str(row[2]).strip().lower() == "no gold paragraph" or pd.isnull(row[2]):
                answer = "[]"
            else:
                answer = f"['{row[2]}']"
            
            writer.write(question + "\t" + answer + "\n")


if __name__ == "__main__":
    main()



# python3 baselines/retriever/BM25/pyserini/convert_tsv_to_query_file.py  --input_annotation_file /home/oogundep/african_qa/queries/gmt_topics/test.kin.tsv --lang kin --output_dir /home/oogundep/african_qa/queries/processed_topics --translation_type google_machine_translation --split test --row_index -1 

# python3 baselines/retriever/BM25/pyserini/convert_tsv_to_query_file.py  --input_annotation_file queries/nllb_topics/test.zul.tsv --output_dir /home/oogundep/african_qa/queries/processed_topics --translation_type nllb_1_3b_translation --split test --row_index -1 --lang zul