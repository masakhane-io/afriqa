import jsonlines
import json
import argparse
from datasets import load_dataset


def main(args):
    with open(f"queries/gold_passages/data_split_gold_passages_{args.lang}_{args.split}.json") as f:
        data = json.load(f)

    written_set = set()
    with jsonlines.open(f'queries/json_lines_gold/data_split_gold_passages_{args.lang}_{args.split}.json', mode='w') as writer:
        for i, json_dict in enumerate(data):
            if type(json_dict['answer_pivot']['answer_start']) == int:
                json_dict['answer_pivot']['answer_start'] = [json_dict['answer_pivot']['answer_start']]

            if type(json_dict['answer_lang']) == int:
                json_dict['answer_lang'] = str(json_dict['answer_lang'])
            
            if type(json_dict['answer_lang']) == None:
                json_dict['answer_lang'] = ""

            if json_dict['id'] not in written_set:
                writer.write(json_dict)
            
            written_set.add(json_dict['id'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    args = parser.parse_args()
    main(args)