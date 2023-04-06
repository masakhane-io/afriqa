import os
import math
import json
import argparse
import logging
import subprocess
from tqdm import tqdm
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_context(question: str, answer_text: str, passage_id:str, paragraph_id: str, gold_passages: pd.DataFrame ):
    """
    Fetch the context from the gold passages
    """
    context_list = []
    
    if passage_id in gold_passages["passage_id"].values:
        gold_passage_filtered = gold_passages[gold_passages["passage_id"] == passage_id]
        if answer_text.strip() != "":
            id = paragraph_id.replace('-body','')
            try:
                if int(id) in gold_passage_filtered["paragraph_id"].values:
                    gold_passage_filtered = gold_passage_filtered[gold_passage_filtered["paragraph_id"] == int(id)]
                    for index, row in gold_passage_filtered.iterrows():
                        if answer_text.lower() in row["paragraph_text"].lower():
                            context_list.append(dict(row))
            except ValueError as e:
                if id in gold_passage_filtered["paragraph_id"].values:
                    gold_passage_filtered = gold_passage_filtered[gold_passage_filtered["paragraph_id"] == id]
                    for index, row in gold_passage_filtered.iterrows():
                        if answer_text.lower() in row["paragraph_text"].lower():
                            context_list.append(dict(row))
        else:
            for index, row in gold_passage_filtered.iterrows():
                context_list.append(dict(row))
                break
    
    return context_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)

    args = parser.parse_args()

    """download the data"""
    logger.info(f"Downloading the data for {args.lang} language")
    os.makedirs(os.path.join(args.input_directory, f"{args.lang}", "collections"), exist_ok=True)
    if not len(os.listdir(os.path.join(args.input_directory, f"{args.lang}", "collections"))) > 2:
        print(f"[INFO] Downloading the data for {args.lang} language")
        with open(os.path.join(args.input_directory, f"{args.lang}", f"{args.lang}_real_links.txt"), "r") as f:
            real_links = f.read().splitlines()
            for i, link in tqdm(enumerate(real_links)):
                process = subprocess.Popen(['wget', f'{link}', '-P', f'{os.path.join(args.input_directory, f"{args.lang}", "collections")}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                process.wait()
    
    """
    generate the gold passages
    """
    logger.info(f"Generating the gold passages for {args.lang} language")
    if not os.path.exists(os.path.join(args.input_directory, f"{args.lang}", f"{args.lang}_gold_passages.tsv")):
        gold_list = []
        for file in tqdm(os.listdir(os.path.join(args.input_directory, f"{args.lang}", "collections"))):
            print(f"[INFO] Processing file: {file}")
            with open(os.path.join(args.input_directory, f"{args.lang}", "collections", file), "r") as f:
                data = json.load(f)
                for key, value in data.items():
                    for passage in value["passage"]:
                        for paragraph in passage["paragraphs"]:
                            question_list = []
                            if paragraph["paragraphs"][0]["bert_ranked"] == "gold":
                                question_list.append(key)
                                question_list.append(value["question"])
                                question_list.append(passage["title"])
                                question_list.append(paragraph["section_title"])
                                question_list.append(paragraph["paragraphs"][0]["paragraph_id"])
                                question_list.append(paragraph["paragraphs"][0]["paragraph_text"])
                                gold_list.append(question_list)

        df = pd.DataFrame(gold_list, columns=["passage_id", "question", "title", "section_title", "paragraph_id", "paragraph_text"])
        df.to_csv(os.path.join(args.input_directory, f"{args.lang}", f"{args.lang}_gold_passages.tsv"), sep="\t" ,index=False)
    else:
        print(f"[INFO] {args.lang}_gold_passages.tsv already exists. Skipping...")
        df = pd.read_csv(os.path.join(args.input_directory, f"{args.lang}", f"{args.lang}_gold_passages.tsv"), sep="\t")

    # """generate the gold passages"""
    output_data = {"data": []}
    with open(os.path.join(args.input_directory, f"{args.lang}", f"{args.lang.capitalize()}QAAnswers.json"), "r") as f:
        qa_dict = json.load(f)
        for passage_dict in tqdm(qa_dict):
            for qa_pairs_set in passage_dict["passages"]:
                for qa_pair in qa_pairs_set["question_answer_pairs"]:
                    data_dict = {}
                    if len(qa_pair["answer"]["spans"]) > 0:
                        context_list = fetch_context(qa_pair["question"], qa_pair["answer"]["spans"][0], qa_pair["passageID"], qa_pair["answer"]["gold_paragraph"], df)
                    else:
                        context_list = fetch_context(qa_pair["question"], "", qa_pair["passageID"], "", df)
                    # Generate data dictionary
                    data_dict["question"] = qa_pair["question"]
                    data_dict["answer"] = qa_pair["answer"]
                    try:
                        data_dict["context"] = [{"title": context["title"], "section_title": context["section_title"] if not math.isnan(context["section_title"]) else "", "paragraph_text": context["paragraph_text"]} for context in context_list]
                    except:
                        data_dict["context"] = [{"title": context["title"], "section_title": context["section_title"], "paragraph_text": context["paragraph_text"]} for context in context_list]
                    output_data["data"].append(data_dict)

    with open(os.path.join(args.input_directory, f"{args.lang}", f"{args.lang}_gold_passages.json"), "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Total number of questions: {len(output_data['data'])}")
    


if __name__ == "__main__":
    main()