import os
import json
import argparse

from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_gold_passages_file", type=str, required=True)
    parser.add_argument("--input_retrieval_run_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.input_gold_passages_file, "r") as f:
        gold_passages = json.load(f)

    with open(args.input_retrieval_run_file, "r") as f:
        retrieval_run = json.load(f)
    
    mod_gold_passages = []

    for qa_pair in tqdm(gold_passages):
        ques_translated = qa_pair["question_translated"]
        answer_gold = qa_pair["answer_pivot"]["text"][0] if len(qa_pair["answer_pivot"]["text"]) > 0 else None

        if answer_gold:
            if not qa_pair["context"]:
                breakout_flag = False
                for _, value in retrieval_run.items():
                    if ques_translated.lower().strip() == value['question'].lower().strip() and answer_gold.lower().strip() == value['answers'][0].lower().strip():
                        for ctxt in value['contexts']:
                            if ctxt['has_answer']:
                                qa_pair["context"] = ctxt['text']
                                qa_pair["title"] = ""
                                try:
                                    qa_pair["answer_pivot"]["answer_start"].append(qa_pair["context"].index(answer_gold))
                                except ValueError as e:
                                    print(qa_pair)
                                    print(ctxt)
                                    print("=====================================")
                                breakout_flag = True
                                break

                    if breakout_flag:
                        break

            if qa_pair["context"]:
                mod_gold_passages.append(qa_pair)


    # Write the output
    with open(args.input_gold_passages_file, "w") as f:
        json.dump(mod_gold_passages, f, indent=4)
            



if __name__ == "__main__":
    main()