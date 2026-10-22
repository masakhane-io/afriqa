import importlib.util
import json
from pathlib import Path
import sys

import pytest

FOLDER = Path(__file__).resolve().parents[1] / 'baselines/retriever/BM25/pyserini'
sys.path.insert(0, str(FOLDER))
from answer_parsing import answers_from_cell, parse_answers, serialize_answers


@pytest.mark.parametrize('answers', [[], ["O'Brien"], ['Àjọ', 'quoted "answer"', 'a\\b'], ['line\nbreak']])
def test_round_trip(answers):
    assert parse_answers(serialize_answers(answers)) == answers


@pytest.mark.parametrize('value,expected', [("['yes']", ['yes']), ('["yes"]', ['yes']), ('"[""yes""]"', ['yes']), ([], [])])
def test_legacy_and_csv(value, expected):
    assert parse_answers(value) == expected


@pytest.mark.parametrize('value', ['', '[1]', '"yes"', None, "['broken's']", '__import__("os").system("false")'])
def test_invalid_answers_are_rejected(value):
    with pytest.raises(ValueError): parse_answers(value)


@pytest.mark.parametrize('value', [None, float('nan'), '', 'No Gold Paragraph'])
def test_missing_cells(value):
    assert answers_from_cell(value) == []


def test_annotation_converter_preserves_native_and_pivot_answers(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('converter', FOLDER / 'convert_tsv_to_query_file.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / 'annotations.tsv'
    source.write_text("question\ttranslation\tpivot_answer\tnative_answer\nQ1\tT1\tO'Brien\tÀjọ\nQ2\tT2\tstill valid\tno gold paragraph\n", encoding='utf-8')
    output = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['converter', '--input_annotation_file', str(source), '--lang', 'yor', '--output_dir', str(output), '--translation_type', 'human', '--split', 'test'])
    module.main()
    path = output / 'queries.xqa.yor.test.en.human.json'
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert parse_answers(records[0]['answers']) == ['Àjọ']
    assert parse_answers(records[0]['translated_answer']) == ["O'Brien"]
    assert parse_answers(records[1]['answers']) == []
    assert parse_answers(records[1]['translated_answer']) == ['still valid']
    assert [r['id'] for r in records] == [0, 1]


def test_actual_trec_converter_handles_empty_and_quoted_answers(tmp_path, monkeypatch):
    import runpy
    from types import ModuleType, SimpleNamespace
    search = ModuleType('pyserini.search')
    search.get_topics = lambda _: {0: {'title': 'q', 'answers': '[]'},
                                  1: {'title': 'q2', 'answers': serialize_answers(["O'Brien"])}}
    search.get_topics_with_reader = lambda *args: search.get_topics(None)
    lucene = ModuleType('pyserini.search.lucene')
    class Searcher:
        @classmethod
        def from_prebuilt_index(cls, name): return cls()
        def doc(self, doc_id):
            return SimpleNamespace(raw=lambda: json.dumps({'contents': "O'Brien passage"}))
    lucene.LuceneSearcher = Searcher
    evaluation = ModuleType('pyserini.eval.evaluate_dpr_retrieval')
    calls = []
    def has_answers(text, answers, tokenizer, regex):
        calls.append(answers)
        return any(answer in text for answer in answers)
    evaluation.has_answers = has_answers
    evaluation.SimpleTokenizer = lambda: object()
    for name, module in [('pyserini', ModuleType('pyserini')), ('pyserini.search', search),
                         ('pyserini.search.lucene', lucene), ('pyserini.eval', ModuleType('pyserini.eval')),
                         ('pyserini.eval.evaluate_dpr_retrieval', evaluation)]:
        monkeypatch.setitem(sys.modules, name, module)
    run = tmp_path / 'run.txt'
    run.write_text('0 Q0 doc1 1 1.0 run\n1 Q0 doc2 1 1.0 run\n')
    output = tmp_path / 'output.json'
    monkeypatch.setattr(sys, 'argv', ['convert', '--topics', 'fake-topics', '--index', 'fake-index',
                                    '--input', str(run), '--output', str(output)])
    runpy.run_path(str(FOLDER / 'convert_trec_run_to_dpr_retrieval_run.py'), run_name='__main__')
    converted = json.loads(output.read_text())
    assert calls == [[], ["O'Brien"]]
    assert converted['0']['contexts'][0]['has_answer'] is False
    assert converted['1']['contexts'][0]['has_answer'] is True
