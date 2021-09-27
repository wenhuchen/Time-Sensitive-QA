import time
import sys
import torch
import transformers
import numpy as np
import random
import json
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import os
import hydra
from tqdm import tqdm, trange
from omegaconf import DictConfig
from transformers import AdamW
import pathlib
sys.path.append(os.path.dirname(__file__))
from model import FiDT5
from utils import get_raw_scores
from omegaconf import OmegaConf
import gzip

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def write_output(glob_path, output_path):
    files = list(glob_path.glob('*.txt'))
    files.sort()
    with open(output_path, 'w') as outfile:
        for path in files:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    outfile.write(line)
            path.unlink()
    glob_path.rmdir()

class TSQADataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        assert 'targets' in example
        if 'targets' in example:
            target = random.choice(example['targets'])
            return target + ' </s>'

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        target = self.get_target(example)

        if 'paragraphs' in example and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            contexts = example['paragraphs'][:self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
        else:
            passages, scores = None, None


        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
        }

    def get_example(self, index):
        return self.data[index]


def load_data(data_path: str):
    data = []
    if data_path.endswith('gzip'):
        with gzip.open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))

    examples = []
    for k, example in enumerate(data):
        if not 'id' in example:
            example['id'] = k
        examples.append(example)

    return examples

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example, max_paragraphs):
            if example['passages'] is None:
                return [example['question'] + " " for i in range(max_paragraphs)]

            results = []
            for i in range(max_paragraphs):
                if i < len(example['passages']):
                    results.append(example['question'] + " " + example['passages'][i])
                else:
                    results.append(example['question'])
            return results

        max_paragraphs = max([len(example['passages']) for example in batch])
        text_passages = [append_question(example, max_paragraphs) for example in batch]

        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        return (index, target_ids, target_mask, passage_ids, passage_masks)

@hydra.main(config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.cuda:
        cfg.n_gpu = 1
        device = torch.device(f'cuda:{cfg.cuda}')
    else:
        cfg.n_gpu = torch.cuda.device_count()
        device = torch.device('cuda')
    print(cfg)

    OmegaConf.save(config=cfg, f='config.yaml')
    
    assert cfg.model_id == 'base', 'model id can only be base'
    model_name = 't5-' + cfg.model_id

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    config = T5Config.from_pretrained(model_name)
    print(config)

    if cfg.model_path:
        logger.info('loading model from {}'.format(cfg.model_path))
        model = FiDT5.from_pretrained(cfg.model_path, config=config, use_checkpoint=cfg.use_checkpoint)
    else:
        t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        model = FiDT5(config)
        model.load_t5(t5.state_dict())
    model = model.to(device)

    if cfg.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model.to(f'cuda:{model.device_ids[0]}')

    root_folder = os.path.dirname(os.path.dirname(__file__))
    if cfg.mode == 'train':
        tb_writer = SummaryWriter(log_dir='')

        train_examples = load_data(os.path.join(root_folder, cfg.dataset.train_file))
        dataset = TSQADataset(train_examples, cfg.n_context)
        logger.info("original dataset: {}".format(len(dataset)))

        collator = Collator(cfg.text_maxlength, tokenizer, answer_maxlength=cfg.answer_maxlength)
        
        step, best_dev_em = 0, 0.0

        optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
        
        batch_size = cfg.per_gpu_train_batch_size * max(1, cfg.n_gpu)

        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, 
            drop_last=True, collate_fn=collator)
        t_total = len(dataloader) * cfg.num_train_epochs

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

                (idx, labels, _, context_ids, context_mask) = batch

                inputs = {
                    'input_ids': context_ids,
                    'attention_mask': context_mask,
                    'labels': labels
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

    if cfg.mode == 'eval':
        model.eval()
        dev_examples = load_data(os.path.join(root_folder, cfg.dataset.dev_file))

        dataset = TSQADataset(dev_examples, cfg.n_context)
        logger.info("original dataset: {}".format(len(dataset)))

        references = {}
        for entry in dev_examples:
            references[entry['idx']] = entry['targets']

        batch_size = cfg.per_gpu_eval_batch_size * max(1, cfg.n_gpu)
        
        sampler = SequentialSampler(dataset)
        collator = Collator(cfg.text_maxlength, tokenizer)

        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, 
            drop_last=False, collate_fn=collator)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Instantaneous batch size per GPU = %d", cfg.per_gpu_eval_batch_size)

        results = {}
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            (idx, _, _, context_ids, context_mask) = batch

            inputs = {
                'input_ids': context_ids,
                'attention_mask': context_mask,
                'max_length': 50
            }

            with torch.no_grad():
                outputs = model.generate(**inputs)

                for k, o in enumerate(outputs):
                    ans = tokenizer.decode(o, skip_special_tokens=True)
                    example = dataset.data[idx[k]]
                    assert example['idx'] not in results
                    results[example['idx']] = ans

        scores = get_raw_scores(results, references)
        print('evaluation results', scores)

        with open('output.json', 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
