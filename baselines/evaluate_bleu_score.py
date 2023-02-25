import argparse
import evaluate 
import pandas as pd
from datasets import load_metric



def compute_bleu(predictions, references):
    """
    Compute BLEU score
    :param y_pred: list of predicted answers
    :param y_true: list of ground truth answers
    :return: BLEU score
    """
    metric = evaluate.load("bleu")
    report = metric.compute(predictions=predictions, references=references)
    bleu = report['bleu']
    return bleu

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the retrieval results with BLEU score.')
    parser.add_argument('--queries_file', required=True, help='Input Translation Document')

    args = parser.parse_args()

    if args.queries_file.endswith('csv'):
        lang_df = pd.read_csv(args.queries_file, sep=",", header=0, dtype=object)
    elif args.queries_file.endswith('tsv'):
        lang_df = pd.read_csv(args.queries_file, sep="\t", header=0, dtype=object)
    
    predictions = []
    references = []
    for index, row in lang_df.iterrows():
        predictions.append(row[-1])
        references.append(row[1])
    
    assert len(predictions) == len(references)
    assert len(predictions) == len(lang_df)

    bleu = compute_bleu(predictions, references)
    print(f"BLEU score: {bleu}")

