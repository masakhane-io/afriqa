from typing import List, Optional
from pathlib import Path
import logging
import json
import numpy as np
import subprocess
import argparse

from pydantic import BaseModel
from helper import DprReader, ReaderEvaluator

from pygaggle.qa.base import Reader, Question, Context
from pygaggle.qa.span_selection import DprSelection, GarSelection, DprFusionSelection, GarFusionSelection
from pygaggle.data.retrieval import RetrievalExample


READER_CHOICES = ('dpr')

def _make_parser_setter(option, key):
    def fn(value):
        option.kwargs[key] = value
        return option
    return fn


class ArgumentParserOption:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return iter((self.args, self.kwargs))

    def __getattr__(self, item: str):
        if item == 'kwargs':
            return self.kwargs
        if item == 'args':
            return self.args
        return _make_parser_setter(self, item)


opt = ArgumentParserOption


class ArgumentParserBuilder(object):
    def __init__(self, **init_kwargs):
        self.parser = argparse.ArgumentParser(**init_kwargs)

    def add_opts(self, *options):
        for args, kwargs in options:
            self.parser.add_argument(*args, **kwargs)
        return self.parser


class PassageReadingEvaluationOptions(BaseModel):
    task: str
    retriever: str
    reader: str
    settings: List[str]
    retrieval_file: Path
    model_name: Optional[str]
    tokenizer_name: Optional[str]
    num_spans: int
    max_answer_length: int
    num_spans_per_passage: int
    device: str
    batch_size: int
    topk_em: List[int]


def construct_dpr(options: PassageReadingEvaluationOptions) -> Reader:
    model = options.model_name
    tokenizer = options.tokenizer_name

    span_selection_rules = [parse_span_selection_rules(setting) for setting in options.settings]

    return DprReader(model,
                     tokenizer,
                     span_selection_rules,
                     options.num_spans,
                     options.max_answer_length,
                     options.num_spans_per_passage,
                     options.batch_size,
                     options.device)


def parse_span_selection_rules(settings):
    settings = settings.split('_')

    settings_map = dict(
        dpr=DprSelection,
        dprfusion=DprFusionSelection,
        gar=GarSelection,
        garfusion=GarFusionSelection,
    )
    return settings_map[settings[0]](*settings[1:])


def main():
    apb = ArgumentParserBuilder()
    apb.add_opts(
        opt('--task',
            type=str,
            default='wikipedia'),
        opt('--retriever',
            type=str,
            required=True,
            help='Retriever score field to rank the input passages to the reader'),
        opt('--reader',
            type=str,
            required=True,
            choices=READER_CHOICES),
        opt('--settings',
            type=str,
            nargs='+',
            default='dpr'),
        opt('--retrieval-file',
            type=Path,
            required=True,
            help='JSON file containing top passages selected by the retrieval model'),
        opt('--model-name',
            type=str,
            default='facebook/dpr-reader-single-nq-base',
            help='Pretrained model for reader'),
        opt('--tokenizer-name',
            type=str,
            default='facebook/dpr-reader-single-nq-base',
            help='Pretrained model for tokenizer'),
        opt('--num-spans',
            type=int,
            default=1,
            help='Number of answer spans to return'),
        opt('--max-answer-length',
            type=int,
            default=10,
            help='Maximum length that an answer span can be'),
        opt('--num-spans-per-passage',
            type=int,
            default=10,
            help='Maximum number of answer spans to return per passage'),
        opt('--output-file',
            type=Path,
            default=None,
            help='File to output predictions for each example; if not specified, this output will be discarded'),
        opt('--device',
            type=str,
            default='cuda:0',
            help='Device for model computations'),
        opt('--batch-size',
            type=int,
            default=16,
            help='batch size of reader inference'),
        opt('--topk-retrieval',
            type=int,
            default=[],
            nargs='+',
            help='Values of k to print the topk accuracy of the retrieval file'),
        opt('--topk-em',
            type=int,
            default=[50],
            nargs='+',
            help='Values of k to print the topk exact match score'),
    )
    args = apb.parser.parse_args()
    options = PassageReadingEvaluationOptions(**vars(args))

    logging.info('Loading the Retrieval File')
    with open(options.retrieval_file) as f:
        data = json.load(f)

    if args.topk_retrieval:
        logging.info('Evaluating Topk Retrieval Accuracies')
        subprocess.call(['python',
                         'tools/scripts/dpr/evaluate_retrieval.py',
                         '--retrieval',
                         options.retrieval_file,
                         '--topk',
                         *map(str, args.topk_retrieval)])

    logging.info('Loading Reader Model and Tokenizer')
    construct_map = dict(
        dpr=construct_dpr,
    )
    reader = construct_map[options.reader](options)

    evaluator = ReaderEvaluator(reader)

    max_topk_passages = max(options.topk_em)
    examples = []
    for _, item in data.items():
        topk_contexts = sorted(item['contexts'], reverse=True, key=lambda context: float(context[options.retriever]))[
                        : max_topk_passages]
        texts = list(map(lambda context: Context(text=context['text'].replace('""', '"'),
                                                 score=float(context[options.retriever])),
                         topk_contexts))
        examples.append(
            RetrievalExample(
                question=Question(text=item['question']),
                contexts=texts,
                ground_truth_answers=item['answers'],
            )
        )
    dpr_predictions = [] if args.output_file is not None else None

    ems, f1s = evaluator.evaluate(examples, options.topk_em, dpr_predictions)

    logging.info('Reader completed')

    for setting in reader.span_selection_rules:
        logging.info(f'Setting: {str(setting)}')
        for k in options.topk_em:
            em = np.mean(np.array(ems[str(setting)][k])) * 100.
            logging.info(f'Top{k}\tExact Match Accuracy: {em}')

            f1 = np.mean(np.array(f1s[str(setting)][k])) * 100.
            logging.info(f'Top{k}\tF1 Score: {f1}')

    if args.output_file is not None:
        with open(args.output_file, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(dpr_predictions, f, indent=4)


if __name__ == '__main__':
    main()