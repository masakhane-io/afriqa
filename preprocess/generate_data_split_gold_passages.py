"""
{
    "answers": {
        "answer_start": [94, 87, 94, 94],
        "text": ["10th and 11th centuries", "in the 10th and 11th centuries", "10th and 11th centuries", "10th and 11th centuries"]
    },
    "context": "\"The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave thei...",
    "id": "56ddde6b9a695914005b9629",
    "question": "When were the Normans in Normandy?",
    "title": "Normans"
}
"""

import json
import os
import re
import argparse
import pandas as pd

def process_string(sentence: str):
    sentence = sentence.strip().lower()
    sentence = sentence = " ".join(re.split("\s+", sentence, flags=re.UNICODE))
    return sentence.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_split", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--gold_passages", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)

    args = parser.parse_args()

    assert args.split in args.input_data_split, "The split must be contained in the input data split file"
    assert args.lang in  args.gold_passages, "The language must be contained in the gold passages file"

    # Open the gold passages
    with open(args.gold_passages, "r") as f:
        gold_passages = json.load(f)
    
    # Open the data split
    split_df = pd.read_csv(args.input_data_split, sep="\t", header=0, dtype=object)

    outputs = []

    for index, row in split_df.iterrows():
        span_dict = None
        question_lang = row[0]
        question_translated = row[1]
        answer_pivot = row[2] if row[2] else None
        answer_lang = row[3] if row[3] else None

        # print(question_lang)
        # print(question_translated)
        # print(answer_pivot)
        # print(answer_lang)
        # print("=====================================")
        # print("Processing question: {}".format(question_translated))


        if str(answer_pivot).strip() == "No Gold Paragraph" or answer_lang is None:
            span_dict = {
                        "answer_pivot": {
                            "answer_start": [],
                            "text": []
                        },
                        "context": None,
                        "id": str(index),
                        "question_lang": question_lang,
                        "question_translated": question_translated,
                        "title": None,
                        "answer_lang": None
                    }
            outputs.append(span_dict)
        else:
            for passage_query in gold_passages["data"]:
                if (process_string(passage_query["question"]) == process_string(question_translated)) and (passage_query["answer"]["spans"] or passage_query["answer"]["yes_no_answer"] != ""):
                    for i in range(len(passage_query["context"])):
                        if passage_query["answer"]["yes_no_answer"] != "":
                            answer_start = {
                                        "answer_start": [-1],
                                        "text": [passage_query["answer"]["yes_no_answer"]]
                                    }
                        else:
                            try:
                                answer_start = {
                                        "answer_start": [passage_query["context"][i]["paragraph_text"].lower().strip().index(answer_pivot.lower().strip())],
                                        "text": [answer_pivot]
                                    }
                            except:
                                answer_start = {
                                        "answer_start": [passage_query["context"][i]["paragraph_text"].lower().strip().index(passage_query["answer"]["spans"][0].lower().strip())],
                                        "text": passage_query["answer"]["spans"]
                                    }
                        span_dict = {
                            "answer_pivot":answer_start,
                            "context": passage_query["context"][i]["paragraph_text"],
                            "id": str(index),
                            "question_lang": question_lang,
                            "question_translated": question_translated,
                            "title": passage_query["context"][i]["title"],
                            "answer_lang": answer_lang
                        }
                        outputs.append(span_dict)
                    break

            if not span_dict:
                print("[INFO] No gold passage found for question: {}".format(question_translated))
                span_dict = {
                    "answer_pivot": {
                        "answer_start": [],
                        "text": [answer_pivot]
                    },
                    "context": None,
                    "id": str(index),
                    "question_lang": question_lang,
                    "question_translated": question_translated,
                    "title": None,
                    "answer_lang": answer_lang
                }

            outputs.append(span_dict)


    # Write the output
    with open(os.path.join(args.output_directory, "data_split_gold_passages_{}_{}.json".format(args.lang, args.split)), "w") as f:
        json.dump(outputs, f, indent=4)

if __name__ == "__main__":
    main()