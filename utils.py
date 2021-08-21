import gzip
import re
from collections import Counter, OrderedDict
import re
import string
import warnings

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s: str):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold: str, a_pred: str):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold: str, a_pred: str):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_raw_scores(examples: dict, references: dict):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}
    
    if len(examples) != len(references):
        warnings.warn('The length of the prediction and reference are not the same')
        assert len(examples) < len(references), 'prediction should be a subset'

    for idx, prediction in examples.items():
        reference = references[idx]
        assert isinstance(reference, list), reference
        
        exact_scores[idx] = max(compute_exact(a, prediction) for a in reference)
        f1_scores[idx] = max(compute_f1(a, prediction) for a in reference)

    return OrderedDict(
        [
            ("exact", 100.0 * sum(exact_scores.values()) / len(exact_scores)),
            ("f1", 100.0 * sum(f1_scores.values()) / len(f1_scores)),
            ("total", len(examples)),
        ]
    )

def readGZip(file_name):
    if file_name.endswith('gz'):
        with gzip.GzipFile(file_name, 'r') as fin:    # 4. gzip
            json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)

        json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
        data = json.loads(json_str)                      # 1. data
        return data
    else:
        with open(file_name, 'r') as fin:
            data = json.load(fin)
        return data
