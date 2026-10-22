"""Safe answer conversion shared by annotation and retrieval converters."""
import ast
import csv
import json
import math


def answers_from_cell(value):
    """Treat an annotation cell as one answer, preserving its text exactly."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    value = str(value)
    if not value.strip() or value.strip().casefold() == 'no gold paragraph':
        return []
    return [value]


def serialize_answers(answers):
    """Retain the dataset's string-valued answer-list format, using valid JSON."""
    if not isinstance(answers, list) or any(not isinstance(a, str) for a in answers):
        raise ValueError('Answers must be a list of strings')
    return json.dumps(answers, ensure_ascii=False)


def parse_answers(value):
    """Accept string lists, JSON, legacy Python literals, or CSV-quoted lists.

    Never evaluate executable expressions. Malformed/scalar answers produce
    an explicit error rather than silently changing their content.
    """
    if isinstance(value, list):
        if any(not isinstance(a, str) for a in value):
            raise ValueError('Answers must be a list of strings')
        return list(value)
    if not isinstance(value, str):
        raise ValueError('Answers must be a list or a serialized list')
    candidates = [value]
    if value.startswith('"') and value.endswith('"'):
        try:
            decoded = next(csv.reader([value], strict=True))
            if len(decoded) == 1 and decoded[0] != value:
                candidates.append(decoded[0])
        except csv.Error:
            pass
    for candidate in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(candidate)
            except (ValueError, SyntaxError, TypeError):
                continue
            if isinstance(parsed, list) and all(isinstance(a, str) for a in parsed):
                return parsed
    raise ValueError('Invalid serialized answer list')
