import datasets
import torch
import pandas as pd
from IPython.display import display, HTML
from transformers import BigBirdTokenizer, BigBirdForQuestionAnswering, BigBirdConfig
from utils import get_raw_scores, readGZip
import hydra
from omegaconf import DictConfig
from datasets import Dataset
from tqdm import tqdm, trange
from functools import partial
import json
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
import logging
from transformers.data.processors.utils import DataProcessor
from multiprocessing import Pool, cpu_count
import numpy as np
from datetime import datetime
from torch.utils.data import TensorDataset
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.big_bird.modeling_big_bird import BigBirdOutput, BigBirdIntermediate
from transformers import PreTrainedModel

import os
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def get_best_valid_start_end_idx(start_scores, end_scores, top_k=1, max_size=100):
    best_start_scores, best_start_idx = torch.topk(start_scores, top_k)
    best_end_scores, best_end_idx = torch.topk(end_scores, top_k)

    widths = best_end_idx[:, None] - best_start_idx[None, :]
    mask = torch.logical_or(widths < 0, widths > max_size)
    scores = (best_end_scores[:, None] + best_start_scores[None, :]) - (1e8 * mask)
    best_score_idx = torch.argmax(scores).item()
    best_score = torch.max(scores).item()

    return best_start_idx[best_score_idx % top_k], best_end_idx[best_score_idx // top_k], best_score

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

class TSQAExample(object):
    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.is_impossible = is_impossible
        
        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

class TSQAProcessor(DataProcessor):
    # Processing the example
    def _create_examples(self, dataset, is_training):
        examples = []
        for entry in tqdm(iter(dataset)):         
            qas_id = entry["idx"]
            context_text = entry["context"]            
            question_text = entry["question"]
            if is_training:
                for target, start_position in zip(entry['targets'], entry['from']):
                    example = TSQAExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=target,
                        start_position_character=start_position,
                        is_impossible=len(target) == 0,
                    )
                    examples.append(example)
            else:
                for target in entry['targets']:
                    example = TSQAExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=target,
                        start_position_character=None,
                        is_impossible=len(target) == 0,
                    )
                    examples.append(example)
        return examples

class TSQAFeatures(object):
    def __init__(
        self,
        input_ids,
        attention_mask,
        cls_index,
        qas_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.cls_index = cls_index
        self.qas_id = qas_id

        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position

        self.is_impossible = is_impossible

def convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training):
    features = []

    with Pool(cpu_count(), initializer=convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert examples to features",
            )
        )

    # Gather the feature outputs
    mapping = {}
    new_features = []
    for example_features in tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            
            if example_feature.qas_id not in mapping:
                mapping[example_feature.qas_id] = len(mapping)
            example_feature.qas_id = mapping[example_feature.qas_id]

            new_features.append(example_feature)
    features = new_features
    del new_features

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    # all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.long)
    
    if not is_training:
        all_qas_ids = torch.tensor([f.qas_id for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_qas_ids
        )
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)

        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_start_positions,
            all_end_positions,
            all_is_impossible,
        )

    return mapping, dataset

def convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, is_training):
    features = []
    assert isinstance(example, TSQAExample), str(type(example))
    
    if is_training and not example.is_impossible:
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        if actual_text.find(example.answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s' in '%s'", actual_text, example.answer_text, example.qas_id)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []
    truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, max_length=max_query_length)
    sequence_added_tokens = tokenizer.model_max_length - tokenizer.max_len_single_sentence

    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):
        encoded_dict = tokenizer.encode_plus(
            truncated_query,
            span_doc_tokens,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            padding='max_length',
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            truncation='only_second'
        )
        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or len(encoded_dict["overflowing_tokens"]) <= 0:
            if "overflowing_tokens" in encoded_dict:
                del encoded_dict["overflowing_tokens"]
            break
        
        span_doc_tokens = encoded_dict["overflowing_tokens"]
        del encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)
        assert cls_index == 0, cls_index

        span_is_impossible = example.is_impossible
        
        # Setting start/end position to the last + 1 position to be ignored
        start_position = len(span['input_ids'])
        end_position = len(span['input_ids'])
        
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                span_is_impossible = True
            else:
                doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        features.append(
            TSQAFeatures(
                input_ids=span["input_ids"],
                attention_mask=span["attention_mask"],
                cls_index=cls_index,
                qas_id=example.qas_id,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
            )
        )
    return features

class BigBirdNullHead(nn.Module):
    """Head for question answering tasks."""

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intermediate = BigBirdIntermediate(config)
        self.output = BigBirdOutput(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self, encoder_output):
        hidden_states = self.dropout(encoder_output)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.output(hidden_states, encoder_output)
        logits = self.qa_outputs(hidden_states)
        return logits

class BigBirdForQuestionAnsweringWithNull(PreTrainedModel):
    def __init__(self, config, model_id):
        super().__init__(config)
        self.bertqa = BigBirdForQuestionAnswering.from_pretrained(model_id,
            config=self.config, add_pooling_layer=True)
        self.null_classifier = BigBirdNullHead(self.bertqa.config)

    def forward(self, **kwargs):
        if self.training:
            null_labels = kwargs['is_impossible']
            del kwargs['is_impossible']
            outputs = self.bertqa(**kwargs)
            pooler_output = outputs.pooler_output
            null_logits = self.null_classifier(pooler_output)
            loss_fct = CrossEntropyLoss()
            null_loss = loss_fct(null_logits, null_labels)

            outputs.loss = outputs.loss + null_loss

            return outputs.to_tuple()
        else:
            outputs = self.bertqa(**kwargs)
            pooler_output = outputs.pooler_output
            null_logits = self.null_classifier(pooler_output)

            return (outputs.start_logits, outputs.end_logits, null_logits)

@hydra.main(config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg)

    if cfg.cuda:
        assert cfg.n_gpu == 1, "If you specify cuda id, the n_gpu needs to be set to 1."
        device = torch.device(f'cuda:{cfg.cuda}')
    else:
        device = torch.device('cuda')

    if cfg.model_id == 'triviaqa':
        model_id = "google/bigbird-base-trivia-itc"
    elif cfg.model_id == 'nq':
        model_id = "vasudevgupta/bigbird-roberta-natural-questions"
    else:
        raise ValueError('Unknown model id!')

    tokenizer = BigBirdTokenizer.from_pretrained(model_id)
    config = BigBirdConfig.from_pretrained(model_id)
    model = BigBirdForQuestionAnsweringWithNull(config, model_id)
    model = model.to(device)
    print(config)

    if cfg.model_path:
        logger.info('loading model from {}'.format(cfg.model_path))
        state_dict = torch.load(os.path.join(cfg.model_path, 'pytorch_model.bin'))
        #if hasattr(state_dict, 'module'):
        #    state_dict = state_dict.module
        model.load_state_dict(state_dict)

    if cfg.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model.to(f'cuda:{model.device_ids[0]}')

    if cfg.mode == 'eval':
        model.eval()
        dataset = datasets.load_dataset('json', data_files={'dev': cfg.dataset.dev_file, 'test': cfg.dataset.test_file})        
        dataset = dataset['dev']
        logger.info("original dataset: {}".format(len(dataset)))

        references = {}
        for entry in iter(dataset):
            references[entry['idx']] = entry['targets']

        processor = TSQAProcessor()
        examples = processor._create_examples(dataset, is_training=False)
        logger.info('Finished processing the examples')

        mapping, dataset = convert_examples_to_features(
            examples, tokenizer, cfg.max_sequence_length, 
            cfg.doc_stride, cfg.max_query_length, False
        )
        imapping = {v:k for k, v in mapping.items()}

        # validation_dataset = validation_dataset.filter(lambda x: len(x["context"]) > 0)

        logger.info('Finished converting the examples')

        batch_size = cfg.per_gpu_train_batch_size * max(1, cfg.n_gpu)
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Instantaneous batch size per GPU = %d", cfg.per_gpu_train_batch_size)

        outputs = {}
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
            }

            with torch.no_grad():
                scores = model(**inputs)
                if len(scores) == 3:
                    start_scores, end_scores, null_scores = scores
                elif len(scores) == 2:
                    start_scores, end_scores = scores
                else:
                    raise ValueError(scores)

                for i in range(start_scores.size(0)):
                    is_impossible = null_scores[i].argmax().item()
                    qas_id = batch[2][i].item()

                    if not is_impossible:
                        start_index, end_index, score = get_best_valid_start_end_idx(start_scores[i], end_scores[i], top_k=8, max_size=16)
                        input_ids = inputs["input_ids"][i].tolist()
                        answer_ids = input_ids[start_index: end_index + 1]
                        answer = tokenizer.decode(answer_ids)

                        if imapping[qas_id] not in outputs:
                            outputs[imapping[qas_id]] = (answer, score)
                        else:
                            if score > outputs[imapping[qas_id]][1]:
                                outputs[imapping[qas_id]] = (answer, score)
                    else:
                        outputs[imapping[qas_id]] = ('', -10000)

        outputs = {k: v[0] for k, v in outputs.items()}
        scores = get_raw_scores(outputs, references)
        print('evaluation results', scores)

        with open('output.json', 'w') as f:
            json.dump(outputs, f, indent=2)

    if cfg.mode == 'train':
        tb_writer = SummaryWriter(log_dir='')
        
        dataset = datasets.load_dataset('json', data_files={'train': cfg.dataset.train_file})[cfg.mode]
        logger.info("original dataset: {}".format(len(dataset)))

        processor = TSQAProcessor()
        examples = processor._create_examples(dataset, is_training=True)
        logger.info('Finished processing the examples')

        _, dataset = convert_examples_to_features(
            examples, tokenizer, cfg.max_sequence_length, 
            cfg.doc_stride, cfg.max_query_length, True
        )

        logger.info('Finished converting the examples')

        batch_size = cfg.per_gpu_train_batch_size * max(1, cfg.n_gpu)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate, eps=cfg.adam_epsilon)
        t_total = len(dataloader) // cfg.num_train_epochs

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Num Epochs = %d", cfg.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", cfg.per_gpu_train_batch_size)
        logger.info("  Total optimization steps = %d", t_total)
        
        global_step = 1
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()

        iterator = trange(0, int(cfg.num_train_epochs), desc="Epoch")

        for epoch in iterator:
            for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
                model.train()
                batch = tuple(t.to(device) for t in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "start_positions": batch[2],
                    "end_positions": batch[3],
                    "is_impossible": batch[4]
                }

                outputs = model(**inputs)

                loss = outputs[0]
                if cfg.n_gpu > 1:
                    loss = loss.mean()

                loss.backward()

                tr_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                
                optimizer.step()
                model.zero_grad()
                global_step += 1

                # Log metrics
                if cfg.logging_steps > 0 and global_step % cfg.logging_steps == 0:
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / cfg.logging_steps, global_step)
                    logging_loss = tr_loss

            # Save Model to a new Directory
            output_dir = "checkpoint-epoch-{}".format(epoch)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        tb_writer.close()


if __name__ == "__main__":
    main()
