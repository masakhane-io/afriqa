import os
from answer_parsing import answers_from_cell, serialize_answers
import jsonlines
import argparse
import pandas as pd

TARGET_LANG={
    "bem" : "en", 
    "fon" : "fr", 
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
        "--translation_row_index", type=int, help="index of the row to be processed", default=1
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    assert os.path.exists(args.input_annotation_file), f"The input annotation file: \'{args.input_annotation_file}\' does not exist"

    extension = os.path.splitext(args.input_annotation_file)[1].lower()
    if extension not in {".csv", ".tsv"}:
        parser.error("Input annotations must be CSV or TSV")
    lang_df = pd.read_csv(args.input_annotation_file,
                          sep="," if extension == ".csv" else "\t", header=0, dtype=object)
    if lang_df.shape[1] < 4 or not -lang_df.shape[1] <= args.translation_row_index < lang_df.shape[1]:
        parser.error("Annotation columns or translation_row_index are invalid")
    os.makedirs(args.output_dir, exist_ok=True)
    native = os.path.join(args.output_dir, f"queries.xqa.{args.lang}.{args.split}.{args.output_file_extension}")
    pivot = os.path.join(args.output_dir, f"queries.xqa.{args.lang}.{args.split}.{TARGET_LANG[args.lang]}.{args.translation_type}")
    with open(native, "w", encoding="utf-8") as writer, open(
        pivot + f".{args.output_file_extension}", "w", encoding="utf-8"
    ) as translation_writer, jsonlines.open(pivot + ".json", mode="w") as json_writer:
        for index, row in lang_df.iterrows():
            question = "" if pd.isnull(row.iloc[0]) else str(row.iloc[0])
            translated_question = "" if pd.isnull(row.iloc[args.translation_row_index]) else str(row.iloc[args.translation_row_index])
            answer = serialize_answers(answers_from_cell(row.iloc[3]))
            translated_answer = serialize_answers(answers_from_cell(row.iloc[2]))
            writer.write(question + "\t" + answer + "\n")
            translation_writer.write(translated_question + "\t" + translated_answer + "\n")
            json_writer.write({"id": index, "question": question, "translated_question": translated_question,
                "answers": answer, "lang": args.lang, "split": args.split,
                "translated_answer": translated_answer, "translation_type": args.translation_type})


if __name__ == "__main__":
    main()



# python3 baselines/retriever/BM25/pyserini/convert_tsv_to_query_file.py  --input_annotation_file /home/oogundep/african_qa/queries/gmt_topics/test.kin.tsv --lang kin --output_dir /home/oogundep/african_qa/queries/processed_topics --translation_type google_machine_translation --split test --translation_row_index -1 

# python3 baselines/retriever/BM25/pyserini/convert_tsv_to_query_file.py  --input_annotation_file queries/nllb_topics/test.zul.tsv --output_dir /home/oogundep/african_qa/queries/processed_topics --translation_type nllb_1_3b_translation --split test --translation_row_index -1 --lang zul