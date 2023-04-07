import os
import json
import argparse
import collections

from tqdm import tqdm
from helper import ReaderEvaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_reader_file', type=str, required=True)
    args = parser.parse_args()

    evaluator = ReaderEvaluator()

    with open(args.input_reader_file, 'r') as f:
        reader_predictions = json.load(f)

    f1_results = collections.defaultdict(list)
    bleu_results = collections.defaultdict(list)
    em_results = collections.defaultdict(list)

    for i, prediction in tqdm(enumerate(reader_predictions)):
        answer = prediction['answers'][0]

        for cutoff in prediction['prediction']['DPR'].keys():

            cutoff_string = prediction['prediction']['DPR'][cutoff]

            if cutoff_string is None:
                continue

            f1, _, _ = evaluator.f1_score(cutoff_string, answer)
            em = evaluator.exact_match_score(cutoff_string, answer)

            f1_results[cutoff].append(f1)
            em_results[cutoff].append(em)
            bleu_results[cutoff].append(evaluator.bleu_score(cutoff_string, answer))
        
    
    print("=========================================")
    print("f1_results")
    print("=========================================")
    for key, value in f1_results.items():
        print(f"{key}: {sum(value)/len(value)}")
    print("=========================================")
    print("bleu_results")
    for key, value in bleu_results.items():
        print(f"{key}: {sum(value)/len(value)}")
    print("=========================================")
    print("em_results")
    for key, value in em_results.items():
        print(f"{key}: {sum(value)/len(value)}")
    print("=========================================")
            

if __name__ == '__main__':
    main()