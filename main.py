import datasets
import torch
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML
from transformers import BigBirdTokenizer, BigBirdForQuestionAnswering
import hydra
from omegaconf import DictConfig
from datasets import Dataset
from collections import defaultdict
from tqdm import tqdm
device = torch.device('cuda:2')

def format_dataset(example):
    # the context might be comprised of multiple contexts => me merge them here
    example["context"] = " ".join(("\n".join(example["entity_pages"]["wiki_context"])).split("\n"))
    example["targets"] = example["answer"]["aliases"]
    example["norm_target"] = example["answer"]["normalized_value"]
    return example

def get_sub_answers(answers, begin=0, end=None):
    return [" ".join(x.split(" ")[begin:end]) for x in answers if len(x.split(" ")) > 1]

def expand_to_aliases(given_answers, make_sub_answers=False):
    PUNCTUATION_SET_TO_EXCLUDE = set(''.join(['‘', '’', '´', '`', '.', ',', '-', '"']))
    if make_sub_answers:
        # if answers are longer than one word, make sure a predictions is correct if it coresponds to the complete 1: or :-1 sub word
        # *e.g.* if the correct answer contains a prefix such as "the", or "a"
        given_answers = given_answers + get_sub_answers(given_answers, begin=1) + get_sub_answers(given_answers, end=-1)
    answers = []
    for answer in given_answers:
        alias = answer.replace('_', ' ').lower()
        alias = ''.join(c if c not in PUNCTUATION_SET_TO_EXCLUDE else ' ' for c in alias)
        answers.append(' '.join(alias.split()).strip())

    return set(answers)

def get_best_valid_start_end_idx(start_scores, end_scores, top_k=1, max_size=100):
    best_start_scores, best_start_idx = torch.topk(start_scores, top_k)
    best_end_scores, best_end_idx = torch.topk(end_scores, top_k)

    widths = best_end_idx[:, None] - best_start_idx[None, :]
    mask = torch.logical_or(widths < 0, widths > max_size)
    scores = (best_end_scores[:, None] + best_start_scores[None, :]) - (1e8 * mask)
    best_score = torch.argmax(scores).item()

    return best_start_idx[best_score % top_k], best_end_idx[best_score // top_k]

def evaluate(example, model, tokenizer):
    # encode question and context so that they are seperated by a tokenizer.sep_token and cut at max_length
    encoding = tokenizer(example["question"], example["context"], return_tensors="pt", max_length=4096, padding="max_length", truncation=True)
    input_ids = encoding.input_ids.to(device)

    with torch.no_grad():
        start_scores, end_scores = model(input_ids=input_ids).to_tuple()

    start_score, end_score = get_best_valid_start_end_idx(start_scores[0], end_scores[0], top_k=8, max_size=16)

    # Let's convert the input ids back to actual tokens 
    all_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())
    answer_tokens = all_tokens[start_score: end_score + 1]

    example["output"] = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))

    answers = expand_to_aliases(example["targets"], make_sub_answers=True)
    predictions = expand_to_aliases([example["output"]])

    # if there is a common element, it's a match
    example["match"] = len(list(answers & predictions)) > 0

    return example

@hydra.main(config_name="config")
def main(cfg: DictConfig) -> None:
    print('configuration: ', cfg)
    if cfg.name == 'triviaqa':
        validation_dataset = datasets.load_dataset("trivia_qa", "rc", split="validation[:5%]")  # remove [:5%] to run on full validation set
        validation_dataset = validation_dataset.map(format_dataset, remove_columns=["search_results", "question_source", "entity_pages", "answer", "question_id"])
        validation_dataset = validation_dataset.filter(lambda x: len(x["context"]) > 0)
        validation_dataset = validation_dataset.filter(lambda x: (len(x['question']) + len(x['context'])) < 4 * 4096)
    elif cfg.name == 'tsqa':
        features = datasets.Features({
            "idx": datasets.Value("string"),
            "context": datasets.Value("string"),
            "question": datasets.Value("string"),
            "targets": datasets.features.Sequence(datasets.Value("string"))
        })
        dataset = datasets.load_dataset('json', features=features, data_files={'train': cfg.train_file, 'dev': cfg.dev_file, 'test': cfg.test_file})
        validation_dataset = dataset['dev']
        validation_dataset = validation_dataset.filter(lambda x: len(x["context"]) > 0)
        validation_dataset = validation_dataset.filter(lambda x: (len(x['question']) + len(x['context'])) < 4 * 4096)
    else:
        raise ValueError('Unkown dataset')

    if cfg.model_id == 'triviaqa':
        model_id = "google/bigbird-base-trivia-itc"
    elif cfg.model_id == 'nq':
        model_id = "vasudevgupta/bigbird-roberta-natural-questions"
    else:
        raise ValueError('Unknown model id!')

    tokenizer = BigBirdTokenizer.from_pretrained(model_id)
    model = BigBirdForQuestionAnswering.from_pretrained(model_id).to(device)
    
    out_dict = defaultdict(list)
    results = []
    for example in tqdm(iter(validation_dataset), desc='Testing'):
        outs = evaluate(example, model, tokenizer)
        for k in outs:
            if k != 'context':
                out_dict[k].append(outs[k])

    results_short = Dataset.from_dict(out_dict)

    print("Exact Match (EM): {:.2f}".format(100 * sum(results_short['match'])/len(results_short)))

    with open('output.json', 'w') as f:
        json.dump(results_short, f, indent=2)

if __name__ == "__main__":
    main()
