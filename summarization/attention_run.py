
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import sys
import pickle
import requests
import torch
import bleu
import json
import random
import logging
import re
import markdown
from bs4 import BeautifulSoup
import argparse
import numpy as np
import pandas as pd
from io import open
from itertools import cycle
import torch.nn as nn
from attention_model import Seq2Seq
from typing import Union, List
from requests import Session
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler,TensorDataset
from dataset_sum import Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from sklearn.model_selection import ShuffleSplit, train_test_split

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def get_about(repo_name):
    headers = {"connection" : "keep-alive","keep-alive" : "timeout=10, max=1000"}
    repo_name = repo_name.replace('_', '/', 1)
    html = requests.get(f"https://github.com/{repo_name}",
                        proxies={'http': 'http://proxy.jpmchase.net:10443',
                                 'https': 'http://proxy.jpmchase.net:10443'},
                        headers=headers)
    soup = BeautifulSoup(html.content, 'html.parser')
    about = soup.find('p', class_='mt-3').text.replace('\n', '')
    encoded_string = about.encode("ascii", 'ignore')
    about = encoded_string.decode()
    about = re.sub(r'^https?:\/\/.*[\r\n]*', '', about).strip()
    return about

def explore_repository(repo_path: str) -> List[str]:
    repo_paths = []
    for (dir_path, dirname, filename) in os.walk(os.path.join('dataset', repo_path)):
        for file in filename:
            if file.endswith('.py'):
                repo_paths.append(os.path.join(dir_path, file))
    return repo_paths

def read_examples_readme(dataset) -> Union[List[Example], None]:
    """Read examples from filename."""
    examples=[]
    for repo in list(set(dataset['repo'].tolist())):
        result = []
        #dataset_repo = dataset[dataset['repo'] == repo]
        if 'README.md' in os.listdir(os.path.join('dataset', repo)):
            result = []
            with open(os.path.join('dataset', repo, 'README.md'), 'r', encoding='utf-8') as f:
                html = markdown.markdown(f.read())
        else:
            s = Session()
            repo_name = repo.replace('_', '/', 1)
            path = 'README.md'
            headers = {"connection" : "keep-alive","keep-alive" : "timeout=10, max=1000"}
            proxies = {'http': 'http://proxy.jpmchase.net:10443',
                       'https': 'http://proxy.jpmchase.net:10443'}
            r = s.get(f"https://github.com/{repo_name}/blob/master/{path}",
                      proxies=proxies,
                      params={'raw': 'true'},
                      headers=headers)
            if r.status_code != 200:
                continue
            else:
                html = markdown.markdown(r.content)
        soup = BeautifulSoup(html, features='html.parser')
        all_p = soup.findAll('p')
        for p in all_p:
            if len(p.findAll('code'))>0 or '|' in p.text or len(re.findall(r'[\u4e00-\u9fff]+', p.text)):
                pass
            else:
                result.append(p.text)
        readme = " ".join(result)
        encoded_string = readme.encode("ascii", 'ignore')
        readme = encoded_string.decode()
        readme = re.sub(r'[^A-Za-z,!?\d\.\"\'\- ]+', " ", readme).strip()
        if len(readme) == 0:
            continue
        code = []
        repo_files = explore_repository(repo)
        for file in repo_files:
            try:
                with open(file, 'r', encoding="utf-8") as f:
                    code.append(f.read())
            except:
                pass
        if len(code)<5:
            continue
        example = Example(
            idx=repo,
            source=code,
            target=readme,
        )
        examples.append(example)
    return examples

def read_examples_about(dataset) -> Union[List[Example], None]:
    """Read examples from filename."""
    examples=[]
    for repo in list(set(dataset['repo'].tolist())):
        if 'about' not in dataset.columns or dataset[dataset['repo']==repo] is None:
            about = get_about(repo)
        else:
            about = dataset[dataset['repo']==repo]['about'].tolist()[0]
        if len(about) == 0:
            continue
        code = []
        repo_files = explore_repository(repo)
        for file in repo_files:
            try:
                with open(file, 'r', encoding="utf-8") as f:
                    code.append(f.read())
            except:
                pass

        example = Example(
            idx=repo,
            source=code,
            target=about,
        )
        if len(code) > 0:
            examples.append(example)
        else:
            pass
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_masks,
                 target_mask,

                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_masks = source_masks
        self.target_mask = target_mask



def convert_examples_to_features(examples, tokenizer, args,stage=None):
    features = []
    for example_index, example in enumerate(examples):
        source_tokens = [tokenizer.tokenize(source)[:args.max_source_length-2] for source in example.source]
        source_tokens_spe =[[tokenizer.cls_token]+source+[tokenizer.sep_token] for source in source_tokens]
        source_ids = [tokenizer.convert_tokens_to_ids(source) for source in source_tokens_spe]
        source_mask = [[1] * (len(source)) for source in source_tokens_spe]
        padding_length = [args.max_source_length - len(source) for source in source_ids]
        source_ids_padded = [source + [tokenizer.pad_token_id]*pad for source, pad in zip(source_ids, padding_length)]
        source_mask_padded =[source + [0]*pad for source, pad in zip(source_mask, padding_length)]
        if len(example.source)<args.len_sequence:
            padding_sequence_length = args.len_sequence - len(example.source)
            padding_sequence_ids = [[tokenizer.pad_token_id for i in range(args.max_source_length)] for j in range(padding_sequence_length)]
            padding_sequence_mask =  [[0 for i in range(args.max_source_length)] for j in range(padding_sequence_length)]
            source_ids_padded.extend(padding_sequence_ids)
            source_mask_padded.extend(padding_sequence_mask)
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+= [tokenizer.pad_token_id]*padding_length
        target_mask += [0]*padding_length

        if len(example.source)>args.len_sequence:
            zip_example = list(zip(source_ids_padded, source_mask_padded))
            random.shuffle(zip_example)
            seq_examples = [zip_example[i:i+args.len_sequence]
                            for i in range(0, len(zip_example)//args.len_sequence, args.len_sequence)]
            for s in seq_examples:
                s_id, s_mask = zip(*s)
                features.append(
                    InputFeatures(
                        example_index,
                        s_id,
                        target_ids,
                        s_mask ,
                        target_mask,
                    )
                )

        else:
            features.append(
                InputFeatures(
                    example_index,
                    source_ids_padded,
                    target_ids,
                    source_mask_padded,
                    target_mask,
                )
            )
    return features



def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default='roberta', type=str,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default="C:\ds\senatus-code-s4\graphcodebert-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base" )
    parser.add_argument("--output_dir", default='summarization_results', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files" )
    ## Other parameters
    parser.add_argument("--dataset_filename", default='C:\ds\senatus-code-s4\\processed_dataset\\full_resources.pkl', type=str,
                        help="The train filename. Should contain the csv files for this task.")
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="C:\ds\senatus-code-s4\graphcodebert-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--readme", action='store_true', default = False, help='if called, will use readmes as groundtruth else will use About section')
    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=200, type=int,
                        help="")
    parser.add_argument("--train_steps", default=400, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--len_sequence", type=int, default=10)
    # print arguments
    args = parser.parse_args()
    logger.info(args)
    args.n_gpu = 0
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    dataset = pd.read_pickle(args.dataset_filename)
    train_dataset, test_dataset = train_test_split(dataset, test_size = 0.5)
    test_dataset, eval_dataset = train_test_split(test_dataset, test_size=0.5)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)

    #budild model
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))




    if args.do_train:
        # Prepare training data loader
        if args.readme:
            train_examples = read_examples_readme(train_dataset)
        else:
            train_examples = read_examples_about(train_dataset)
        train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_masks for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=2)#args.train_batch_size//args.gradient_accumulation_steps)
        num_train_optimization_steps = args.train_steps
        with open('full_sum_train.pkl', 'wb') as f:
            pickle.dump(train_dataloader, f)
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)


        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", num_train_optimization_steps*args.train_batch_size//len(train_dataloader))

        model.train()
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6
        bar = tqdm(range(num_train_optimization_steps),total=num_train_optimization_steps)
        train_dataloader=cycle(train_dataloader)
        eval_flag = True
        for step in bar:
            batch = next(train_dataloader)
            source_ids,source_mask,target_ids,target_mask = batch
            loss,_,_ = model(source_ids=source_ids,source_masks=source_mask,target_ids=target_ids,target_mask=target_mask)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
            bar.set_description("loss {}".format(train_loss))
            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                #Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                eval_flag = True

            if args.do_eval and ((global_step + 1) %args.eval_steps == 0) and eval_flag:
                #Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                eval_flag=False
                if 'dev_loss' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples_about(eval_dataset)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_masks for f in eval_features], dtype=torch.long)
                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                    all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)
                    dev_dataset['dev_loss'] = eval_examples,eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                #Start Evaling model
                model.eval()
                eval_loss,tokens_num = 0,0
                for batch in eval_dataloader:
                    source_ids,source_mask,target_ids,target_mask = batch

                    with torch.no_grad():
                        _,loss,num = model(source_ids=source_ids,source_masks=source_mask,
                                           target_ids=target_ids,target_mask=target_mask)
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                #Pring loss of dev dataset
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss),5),
                          'global_step': global_step+1,
                          'train_loss': round(train_loss,5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)

                #save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                if eval_loss<best_loss:
                    logger.info("  Best ppl:%s",round(np.exp(eval_loss),5))
                    logger.info("  "+"*"*20)
                    best_loss=eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)


                    #Calculate bleu
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples_about(test_dataset)
                    eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_masks for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids,all_source_mask)
                    dev_dataset['dev_bleu']=eval_examples,eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
                model.eval()
                p=[]
                for batch in eval_dataloader:
                    source_ids,source_mask= batch
                    with torch.no_grad():
                        preds = model(source_ids=source_ids,source_masks=source_mask)
                        for pred in preds:
                            t=pred[0].cpu().numpy()
                            t=list(t)
                            if 0 in t:
                                t=t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions=[]
                with open(os.path.join(args.output_dir,"dev_output.txt"),'w') as f, open(os.path.join(args.output_dir,"dev_gold.txt"),'w') as f1:
                    for ref,gold in zip(p,eval_examples):
                        predictions.append(str(gold.idx)+'\t'+ref)
                        f.write(str(gold.idx)+'\t'+ref+'\n')
                        f1.write(str(gold.idx)+'\t'+gold.target+'\n')

                (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev_gold.txt"))
                dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  "+"*"*20)
                if dev_bleu>best_bleu:
                    logger.info("  Best bleu:%s",dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu=dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

    if args.do_test:
        files=[]
        if dev_dataset is not None:
            files.append(dev_dataset)
        if test_dataset is not None:
            files.append(test_dataset)
        for idx,file in enumerate(files):
            if args.readme:
                eval_examples = read_examples_readme(file)
            else:
                eval_examples = read_examples_about(file)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_masks for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_source_ids,all_source_mask)
            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            with open('mean_test_dataloader.pkl', 'wb') as f:
                pickle.dump(eval_dataloader, f)
            model.eval()
            p=[]
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                source_ids,source_mask = batch
                with torch.no_grad():
                    preds = model(source_ids=source_ids,source_masks=source_mask)
                    for pred in preds:
                        t=pred[0].cpu().numpy()
                        t=list(t)
                        if 0 in t:
                            t=t[:t.index(0)]
                        text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                        p.append(text)
            model.train()
            predictions=[]
            with open(os.path.join(args.output_dir,"test_{}_output.txt".format(str(idx))),'w') as f, open(os.path.join(args.output_dir,"test_{}_gold.txt".format(str(idx))),'w') as f1:
                for ref,gold in zip(p,eval_examples):
                    predictions.append(str(gold.idx)+'\t'+ref)
                    f.write(str(gold.idx)+'\t'+ref+'\n')
                    f1.write(str(gold.idx)+'\t'+gold.target+'\n')

            (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "test_{}_gold.txt".format(idx)))
            dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
            logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
            logger.info("  "+"*"*20)

if __name__ == "__main__":
    main()