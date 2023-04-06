from typing import List, Optional, Dict
import unidecode
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Union
import torch.nn.functional as F

import re
import string
import evaluate 
from tqdm import tqdm
from collections import Counter
from transformers import DPRReader, DPRReaderTokenizer

from pygaggle.qa.base import Reader, Answer, Question, Context
from pygaggle.qa.span_selection import DprSelection
from pygaggle.data.retrieval import RetrievalExample

from transformers import BertConfig, DPRConfig, BertModel, DPRReader, DPRReaderOutput, DPRReaderTokenizer

import __main__

class UpdatedDPRReader(nn.Module):
    def __init__(self) -> None:
        super(UpdatedDPRReader, self).__init__()

        self.model = DPRReader(DPRConfig(**BertConfig.get_config_dict("bert-base-multilingual-uncased")[0]))
        bert_model = BertModel(BertConfig.from_pretrained("bert-base-multilingual-uncased"))

        self.model.span_predictor.encoder.bert_model = bert_model

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict=None,
    ) -> Union[DPRReaderOutput, Tuple[Tensor, ...]]:

        return self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)


class DprReader(Reader):
    """Class containing the DPR Reader
    Takes in a question and a list of the top passages selected by the retrieval model,
    and predicts a list of the best answer spans from the most relevant passages.
    Parameters
    ----------
    model_name : DPR Reader model name or path
    tokenizer_name : DPR Reader tokenizer name or path
    num_spans : Number of answer spans to return
    max_answer_length : Maximum length that an answer span can be
    num_spans_per_passage : Maximum number of answer spans to return per passage
    """

    def __init__(
            self,
            model_name: str,
            tokenizer_name: str = None,
            span_selection_rules=None,
            num_spans: int = 1,
            max_answer_length: int = 10,
            num_spans_per_passage: int = 10,
            batch_size: int = 16,
            device: str = 'cuda:0'
    ):
        if span_selection_rules is None:
            span_selection_rules = [DprSelection()]
        self.device = device
        try:
            setattr(__main__, "UpdatedDPRReader", UpdatedDPRReader)
            # self.model = UpdatedDPRReader()
            self.model = torch.load(model_name)
            self.model.to(self.device)
        except:
            self.model = DPRReader.from_pretrained(model_name).to(self.device).eval()
        if tokenizer_name:
            self.tokenizer = DPRReaderTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = DPRReaderTokenizer.from_pretrained(model_name)
        self.span_selection_rules = span_selection_rules
        self.num_spans = num_spans
        self.max_answer_length = max_answer_length
        self.num_spans_per_passage = num_spans_per_passage
        self.batch_size = batch_size

    def compute_spans(
            self,
            question: Question,
            contexts: List[Context],
    ):
        spans_for_contexts = []

        for i in range(0, len(contexts), self.batch_size):
            lis_contexts = list(map(lambda t: t.text, contexts[i: i + self.batch_size]))
            encoded_inputs = self.tokenizer(
                questions=[question.text]*len(lis_contexts),
                texts=lis_contexts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=350,
            )
            input_ids = encoded_inputs['input_ids'].to(self.device)
            attention_mask = encoded_inputs['attention_mask'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            outputs.start_logits = outputs.start_logits.cpu().detach().numpy()
            outputs.end_logits = outputs.end_logits.cpu().detach().numpy()
            outputs.relevance_logits = outputs.relevance_logits.cpu().detach().numpy()

            predicted_spans = self.tokenizer.decode_best_spans(
                encoded_inputs,
                outputs,
                self.batch_size * self.num_spans_per_passage,
                self.max_answer_length,
                self.num_spans_per_passage,
            )

            # collect spans for each context
            batch_spans_by_contexts = [[] for _ in range(len(outputs.relevance_logits))]
            for span in predicted_spans:
                batch_spans_by_contexts[span.doc_id].append(span)

            # sort spans by span score
            for k in range(len(outputs.relevance_logits)):
                spans_for_contexts.append(
                    sorted(batch_spans_by_contexts[k], reverse=True, key=lambda span: span.span_score)
                )

        return spans_for_contexts

    def predict(
            self,
            question: Question,
            contexts: List[Context],
            topk_retrievals: Optional[List[int]] = None,
    ) -> Dict[int, List[Answer]]:
        if isinstance(question, str):
            question = Question(question)
        if topk_retrievals is None:
            topk_retrievals = [len(contexts)]

        answers = {str(rule): {} for rule in self.span_selection_rules}
        prev_topk_retrieval = 0
        for rule in self.span_selection_rules:
            rule.reset()

        spans = self.compute_spans(question, contexts)

        for topk_retrieval in topk_retrievals:
            for rule in self.span_selection_rules:
                rule.add_answers(
                    spans[prev_topk_retrieval: topk_retrieval],
                    contexts[prev_topk_retrieval: topk_retrieval]
                )
                answers[str(rule)][topk_retrieval] = rule.top_answers(self.num_spans)

            prev_topk_retrieval = topk_retrieval

        return answers


class ReaderEvaluator:
    """Class for evaluating a reader.
    Takes in a list of examples (query, texts, ground truth answers),
    predicts a list of answers using the Reader passed in, and
    collects the exact match accuracies between the best answer and
    the ground truth answers given in the example.
    Exact match scoring used is identical to the DPR repository.
    """
    def __init__(
        self,
        reader: Reader = None,
    ):
        self.reader = reader

    def evaluate(
        self,
        examples: List[RetrievalExample],
        topk_em: List[int] = [50],
        dpr_predictions: Optional[Dict[int, List[Dict[str, str]]]] = None,
    ):
        ems = {str(setting): {k: [] for k in topk_em} for setting in self.reader.span_selection_rules}
        f1s = {str(setting): {k: [] for k in topk_em} for setting in self.reader.span_selection_rules}

        for example in tqdm(examples):
            answers = self.reader.predict(example.question, example.contexts, topk_em)
            ground_truth_answers = example.ground_truth_answers
            topk_prediction = {str(setting): {} for setting in self.reader.span_selection_rules}
            for setting in self.reader.span_selection_rules:
                for k in topk_em:
                    best_answer = answers[str(setting)][k][0].text
                    em_hit = max([ReaderEvaluator.exact_match_score(best_answer, ga) for ga in ground_truth_answers])
                    ems[str(setting)][k].append(em_hit)
                    topk_prediction[f'{str(setting)}'][f'top{k}'] = best_answer

                    f1_hit = max([ReaderEvaluator.f1_score(best_answer, ga) for ga in ground_truth_answers])
                    f1s[str(setting)][k].append(f1_hit)

            if dpr_predictions is not None:
                dpr_predictions.append({
                    'question': example.question.text,
                    'answers': ground_truth_answers,
                    'prediction': topk_prediction,
                })
        return ems, f1s


    @staticmethod
    def exact_match_score(prediction, ground_truth):
        return ReaderEvaluator._normalize_answer(prediction) == ReaderEvaluator._normalize_answer(ground_truth)

    @staticmethod
    def _normalize_answer(s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        
        def unidecode_str(text):
            return unidecode.unidecode(str(text))
        
        return white_space_fix(remove_articles(remove_punc(lower(unidecode_str(s)))))
    
    @staticmethod
    def f1_score(prediction, ground_truth):
        prediction_tokens = ReaderEvaluator._normalize_answer(prediction).split()
        ground_truth_tokens = ReaderEvaluator._normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall
    
    @staticmethod
    def bleu_score(prediction, ground_truth):
        prediction_tokens = ReaderEvaluator._normalize_answer(prediction)
        ground_truth_tokens = ReaderEvaluator._normalize_answer(ground_truth)

        bleu = sentence_bleu([ground_truth_tokens.split()] ,prediction_tokens.split(), weights=(1, 0, 0, 0)) 
        return bleu



if __name__ == "__main__":
    question = Question("What religion do Muslims believe in?")
    contexts = [
        Context("At-Tawba 29 At-Tawba 29 Verse 29 of chapter 9 of the Qur'an is notable as dealing with the imposition of tribute (\u01e7izya) on non-Muslims who have fallen under Muslim rule (the \"ahl al-\u1e0fimma\"). Most Muslim commentators believe this verse was revealed at the time of the expedition to Tabuk. Text. Sahih International: Fight those who do not believe in Allah or in the Last Day and who do not consider unlawful what Allah and His Messenger have made unlawful and who do not adopt the religion of truth from those who were given the Scripture - [fight] until they give At-Tawba 29 At-Tawba 29 Verse 29 of chapter 9 of the Qur'an is notable as dealing with the imposition of tribute (\u01e7izya) on non-Muslims who have fallen under Muslim rule (the \"ahl al-\u1e0fimma\"). Most Muslim commentators believe this verse was revealed at the time of the expedition to Tabuk. Text. Sahih International: Fight those who do not believe in Allah or in the Last Day and who do not consider unlawful what Allah and His Messenger have made unlawful and who do not adopt the religion of truth from those who were given the Scripture - [fight] until they give "),
        Context("administrative reform is led to similar outbreaks.\u201d Beliefs. Saminist do not see any distinction of religions, therefore Samin people will never deny or hate religion. Though Saminists are generally non-Muslims, some followers abide by the Muslim religion. Most, however, do not believe in the existence of Allah nor heaven or hell, but instead \u201cGod is within me.\u201d Saminists believe in the \u201cFaith of Adam\u201d in which stealing, lying, and adultery are forbidden. However, compliance with laws was voluntary because they recognized no authority and often withdrew from other societal norms. In the after life Saminists believe that if one "),
        Context("called all the sages together and said to them. \"Speak and argue with one another and make clear to me which is the best religion.\" They began to dispute with one another without arriving at any results until the King said to the Christian priest \"What do you think? Of the religion of the Jews and the Muslims, which is to be preferred?\" The priest answered: \"The religion of the Israelites is better than that of the Muslims.\" The King then asked the qadi: \"What do you say? Is the religion of the Israelites, or that of the Christians preferable"),
        Context("travel farther to spread the ashes of the deceased in a holy river. However, if they choose not to do so, the ashes will be spread in the sea or a river nearby. Islam. In the Islam religion, the funeral procession is a virtuous act that typically involves a large amount of participation from other Muslims. Traditions that were begun by the Prophet are what urged Muslims to partake in the procession. Muslims believe that by following the funeral procession, praying over the body, and attending the burial one may receive qu\u012br\u0101ts (rewards) to put them in good favor with "),
        Context("Paris is the capital of France."),
    ]

    reader = DprReader("facebook/dpr-reader-single-nq-base", device="cpu")
    answers = reader.predict(question, contexts)
    print(list(answers['DPR'].values())[0][0].text)
