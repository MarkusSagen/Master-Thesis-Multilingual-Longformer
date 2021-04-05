#!/usr/bin/env python3

# TODO make AutoModel and AutoTokenizer

from __future__ import print_function
import argparse
from collections import Counter
import dataclasses
import datetime
from dataclasses import dataclass, field
import logging
import json
import os
from pprint import pprint
import re
import string
import sys
import time
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from typing import TypeVar
from typing import Dict, List, Optional

import datasets
from datasets import logging as DSlogging
import datasets as nlp
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import logging as hf_logging
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)



# helper
class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def check_positive_concats(nr_concats):
    """ Helper funtion for argparse
    Instructs how many contexts to concatinate together.
    Defualt for longer contexts are three.
    More can be used, but then it requires larger GPUs.

    *NOTE* this is only used when using the datasets:
    - squad_long or
    - xquad_long
    """
    try:
        nr_concats_int = int(nr_concats)
        if nr_concats_int <= 0:
            raise argparse.ArgumentTypeError(f"--nr_concats expects a positive int as a value, not {nr_concats}")
    except ValueError as e:
        if hasattr(e, "message"):
            print(e.message)
        else: 
            print(e)
    return nr_concats_int


parser = argparse.ArgumentParser()
parser.add_argument(
    "--nr_concats",
    default=3,
    type=check_positive_concats,
    help="Define how many context to concatinate together when using a `long` QA dataset.\n3 is default and yields an average context lenght of 2048 tokens",
)
parser.add_argument(
    "--model_name",
    default=None,
    type=str,
    # required=True,
    help="Name to save the model as.",
)
parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    help="The output directory where the model checkpoints and predictions will be written.",
)
parser.add_argument(
    "--model_type",
    default=None,
    type=str,
    help="Model type selected in the list from Huggingface ex: `bert, roberta, xlm-roberta, ...`",
)
# xlm-roberta-base
# ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
# 'xlm-roberta-base'
parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pretrained model from huggingface.co/models. Only tested on `xlm-roberta-base` and `roberta-base`.",
)
parser.add_argument(
    "--datasets",
    default=None,
    type=str,
    required=True,
    help="Define one of Huggingface Datasets Question Answering Tasks. Example: `squad` and `trivia-qa`.",
)
parser.add_argument(
    "--train_file_path",
    default=None,
    type=str,
    help="File path to where torch training file is stored (.pt files).",
)
parser.add_argument(
    "--valid_file_path",
    default=None,
    type=str,
    help="File path to where torch validation file is stored (.pt files).",
)
parser.add_argument(
    "--data_dir",
    default=None,
    type=str,
    help="Directory to where training and validation torch files will be stored."
)
parser.add_argument(
    "--logging_dir",
    default=None,
    type=str,
    help="The output directory where the model loss, epochs and evaluations are written. Logger info also stored here",
)
parser.add_argument(
    "--max_length",
    default=512,
    type=int,
    choices=[
        512,
        1024,
        2048,
        4096,
    ],
    help="The maxiumum position of the model",
)
parser.add_argument(
    "--attention_window",
    default=512,
    type=int,
    help="Size of attention window",
)
parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
parser.add_argument(
    "--do_eval", action="store_true", help="Whether to run eval on the dev set."
)
parser.add_argument(
    "--evaluate_during_training",
    action="store_true",
    help="Run evaluation during training at each logging step.",
)
parser.add_argument(
    "--per_device_train_batch_size",
    default=8,
    type=int,
    help="Batch size per GPU/CPU for training.",
)
parser.add_argument(
    "--per_device_eval_batch_size",
    default=8,
    type=int,
    help="Batch size per GPU/CPU for evaluation.",
)
parser.add_argument(
    "--learning_rate",
    default=5e-5,
    type=float,
    help="The initial learning rate for Adam.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
)
parser.add_argument(
    "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
)
parser.add_argument(
    "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
)
parser.add_argument(
    "--num_train_epochs",
    default=3.0,
    type=float,
    help="Total number of training epochs to perform.",
)
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
)
parser.add_argument(
    "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
)
parser.add_argument(
    "--verbose_logging",
    action="store_true",
    help="If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.",
)
parser.add_argument(
    "--lang_id",
    default=0,
    type=int,
    help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
)
parser.add_argument(
    "--logging_steps", type=int, default=500, help="Log every X updates steps."
)
parser.add_argument(
    "--save_steps",
    type=int,
    default=500,
    help="Save checkpoint every X updates steps.",
)
parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
)
parser.add_argument(
    "--overwrite_output_dir",
    action="store_true",
    help="Overwrite the content of the output directory",
)
parser.add_argument(
    "--seed", type=int, default=42, help="random seed for initialization"
)
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="local_rank for distributed training on gpus",
)
parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html",
)
parser.add_argument(
    "--prediction_loss_only",
    action="store_true",
    help="If only prediciton loss should be returned",
)
parser.add_argument(
    "--eval_steps",
    type=int,
    default=500,
    help="If input should be tokenized to only lowercase",
)
# https://github.com/huggingface/transformers/blob/6494910f2741befae281388db0d9eacfbe82fad3/src/transformers/data/processors/squad.py#L304
# https://huggingface.co/transformers/model_doc/bert.html?highlight=do_lower_case
# SQUAD default:
# FIXME:
#          --max_seq_length 384
#          --doc_stride 128
#          --do_lowercase
#          --truncate
#          --label_names
#          --cache_dir

#
parser.add_argument(
    "--do_lowercase",
    action="store_true",
    help="If input should be lowercase or not when tokenizing",
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=384,
    help="FIXME: NOT USED!",
)
parser.add_argument(
    "--doc_stride",
    type=int,
    default=128,
    help="FIXME: NOT USED!",
)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=None,
    help="FIXME: NOT USED!",
)
#parser.add_argument(
#    "--label_names",
#    type=['str'],
#    default=["start_positions", "end_positions"],
#    help="FIXME: Add good thext! Needed for QA, since default is incorrect:  HF issue 8390!",
#)








# Create args and make huggingface transformers print out its operations
args = parser.parse_args()

hf_logging.enable_default_handler()
hf_logging.set_verbosity_info()
hf_logging.enable_explicit_format()

# Setup logging
tb_writer = SummaryWriter(log_dir=args.logging_dir)

logger = logging.getLogger("")
logger.setLevel(logging.INFO)

fh = logging.FileHandler(f"{args.logging_dir}.log")
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "[%(asctime)s], %(levelname)s %(message)s",
    datefmt="%a, %d %b %Y %H:%M:%S",
)
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)
logger.info("\n --> Starting logger:\n" + "=" * 55 + "\n")

logger.warning(
    f"Process rank: {args.local_rank}, distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}"
)


# Initialize
logger.info("=" * 50)
logger.info("=" + "\t" * 6 + " =")
logger.info("=" + "\tInitialization" + "\t" * 4 + " =")
logger.info("=" + "\t" * 6 + " =")
logger.info("=" * 50 + "\n\n")



# Set up tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    cache_dir=args.cache_dir,
    do_lowercase=args.do_lowercase,
    pad_to_max_length=True,
    max_length=args.max_length,
    truncation=True,
    use_fast=True,
)
model = AutoModelForQuestionAnswering.from_pretrained(
    args.model_name_or_path,
    cache_dir=args.cache_dir,
)




#########################################
#
# SQuADs Evaluation metrics and helper functions
#
#########################################

## SQuAD evaluation script. Modifed slightly for this notebook
# https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
# https://github.com/huggingface/transformers/tree/master/examples/question-answering

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(gold_answers, predictions):
    f1 = exact_match = total = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
        total += 1
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths
        )
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}



####################################################
#
# Evaluation
#
####################################################

def get_squad_evaluation(valid_dataset, model, tokenizer, dataset_name, batch_size):
    """
    Makes a prediction and evaluates it based on the trained model
    The evaluation is based on the SQuAD evaluation metric:
    - F1 and Exact Match
    valdid_datset is expected to be converted to a torch Tensor type:

    Ex:
        columns = ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
        valid_dataset.set_format(type='torch', columns=columns)
    """
    # TODO: returns torch, 
    #assert valid_dataset.format['type'] == 'torch'
    

    logging.info(f"Generating perdictions for dataset '{dataset_name}'")
    dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    # predictions
    start = time.time()
    predicted_answers = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            start_scores, end_scores = model(
                input_ids=batch["input_ids"].cuda(),
                attention_mask=batch["attention_mask"].cuda(),
            )
            for i in range(start_scores.shape[0]):
                all_tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][i])
                answer = " ".join(
                    all_tokens[
                        torch.argmax(start_scores[i]) : torch.argmax(end_scores[i]) + 1
                    ]
                )
                ans_ids = tokenizer.convert_tokens_to_ids(answer.split())
                answer = tokenizer.decode(ans_ids)
                predicted_answers.append(answer)

    # logging.info(f"Generating perdictions took: {time.time() - start}s")
    # logging.info("Computing F1 and Exact Match:")

    # Return to dataset from tensor
    valid_dataset.reset_format()
    predictions = []
    references = []
    # valid_dataset = nlp.load_dataset('squad', split='validation')
    for ref, pred_answer in zip(valid_dataset, predicted_answers):
        actual_answer = ref["answers"]["text"]
        predictions.append(pred_answer)
        references.append(actual_answer)

    eval_score = evaluate(references, predictions)
    print(eval_score)

    #logging.info(f"Evaluation on dataset: {dataset_name}")
    logging.info(f"Results from prediction:\n{eval_score}\n" + "="*55 + "\n")




#########################################
#
# Convert train and validation to
#  prefconfigured tensors
#
#########################################

def get_correct_alignement(context: str, answer):
    """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
    gold_text = answer["text"][0]
    start_idx = answer["answer_start"][0]
    end_idx = start_idx + len(gold_text)
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx  # When the gold label position is good
    elif context[start_idx - 1 : end_idx - 1] == gold_text:
        return start_idx - 1, end_idx - 1  # When the gold label is off by one character
    elif context[start_idx - 2 : end_idx - 2] == gold_text:
        return start_idx - 2, end_idx - 2  # When the gold label is off by two character
    else:
        raise ValueError()


# Tokenize our training dataset
def convert_to_features(example):
    # Tokenize contexts and questions (as pairs of inputs)
    encodings = tokenizer.encode_plus(
        example["question"],
        example["context"],
        pad_to_max_length=True,
        max_length=args.max_length,
        truncation=True,
    )

    context_encodings = tokenizer.encode_plus(example["context"])

    # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
    # this will give us the position of answer span in the context text
    start_idx, end_idx = get_correct_alignement(example["context"], example["answers"])
    start_positions_context = context_encodings.char_to_token(start_idx)
    end_positions_context = context_encodings.char_to_token(end_idx - 1)

    # FIXME: UGLY HACK because of XLM-R tokenization, works fine with monolingual
    # 2 training examples returns incorrect positions
    sep_idx = encodings["input_ids"].index(tokenizer.sep_token_id)
    try:
        # here we will compute the start and end position of the answer in the whole example
        # as the example is encoded like this <s> question</s></s> context</s>
        # and we know the postion of the answer in the context
        # we can just find out the index of the sep token and then add that to position + 1 (+1 because there are two sep tokens)
        # this will give us the position of the answer span in whole example

        start_positions = start_positions_context + sep_idx + 1
        end_positions = end_positions_context + sep_idx + 1

        #if end_positions > 4096:
        #     start_positions, end_positions = 0, 0

    # Returned None for start or end position index
    except:
        #print(f"{example}")
        #print(f"Start_idx: {start_idx} \t End_idx: {end_idx}")
        #print(f"Sep_idx: {sep_idx}")
        #print(f"with start: {start_positions_context} \t end: {end_positions_context}\n")
        start_positions = None
        end_positions = None

    encodings.update(
        {
            "start_positions": start_positions,
            "end_positions": end_positions,
            "attention_mask": encodings["attention_mask"],
        }
    )
    return encodings

MAX_CONTEXT_LENGTH = 0

# Tokenize our training dataset
def convert_to_features(example):
    # Tokenize contexts and questions (as pairs of inputs)
    encodings = tokenizer.encode_plus(
        example["question"],
        example["context"],
        pad_to_max_length=True,
        max_length=args.max_length,
        truncation=True,
    )
    context_encodings = tokenizer.encode_plus(example["context"])

    # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
    # this will give us the position of answer span in the context text
    start_idx, end_idx = get_correct_alignement(example["context"], example["answers"])
    start_positions_context = context_encodings.char_to_token(start_idx)
    end_positions_context = context_encodings.char_to_token(end_idx - 1)

    # FIXME: UGLY HACK because of XLM-R tokenization, works fine with monolingual
    # 2 training examples returns incorrect positions
    sep_idx = encodings["input_ids"].index(tokenizer.sep_token_id)
    try:
        # here we will compute the start and end position of the answer in the whole example
        # as the example is encoded like this <s> question</s></s> context</s>
        # and we know the postion of the answer in the context
        # we can just find out the index of the sep token and then add that to position + 1 (+1 because there are two sep tokens)
        # this will give us the position of the answer span in whole example

        start_positions = start_positions_context + sep_idx + 1
        end_positions = end_positions_context + sep_idx + 1

        if end_positions > 4096:
            start_positions, end_positions = None, None

    # Returned None for start or end position index
    except:
        #print(f"{example}")
        #print(f"Start_idx: {start_idx} \t End_idx: {end_idx}")
        #print(f"Sep_idx: {sep_idx}")
        #print(f"with start: {start_positions_context} \t end: {end_positions_context}\n")
        start_positions = None
        end_positions = None

    encodings.update(
        {
            "start_positions": start_positions,
            "end_positions": end_positions,
            "attention_mask": encodings["attention_mask"],
        }
    )
    return encodings











def convert_dataset_to_torch_format(data):
    data = data.map(convert_to_features).filter(lambda example: (example['start_positions'] is not None) and (example['end_positions'] is not None))

    # set the tensor type and the columns which the dataset should return
    columns = ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
    data.set_format(type='torch', columns=columns)
    print(max(data['start_positions']))
    print(data.shape)
    return data





##################
#
#  Training
#
##################

class DummyDataCollator:
    def __call__(self, batch):
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example['input_ids'] for example in batch])
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        start_positions = torch.stack([example['start_positions'] for example in batch])
        end_positions = torch.stack([example['end_positions'] for example in batch])

        return {
            'input_ids': input_ids,
            'start_positions': start_positions,
            'end_positions': end_positions,
            'attention_mask': attention_mask
        }



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    # TODO UNused!
    # ['--model_type', 'roberta', '--max_seq_length', '384', '--doc_stride', '128']
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    do_lowercase: bool = field(
        default=False,
        metadata={"help": "If tokenizer should make all to lowercase."},
    )
    max_seq_length: Optional[int] = field(
        default=384,
        metadata={"help": "TODO"},
    )
    doc_stride: Optional[int] = field(
        default=128,
        metadata={"help": "TODO"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "TODO"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # TODO fix to List of strings
    datasets: str = field(
        metadata={
            "help": "TODO"
        }
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the dataset containing train and eval datasets."},
    )
    train_file_path: Optional[str] = field(
        default="train_data.pt",
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        default="valid_data.pt",
        metadata={"help": "Path for cached valid dataset"},
    )
    max_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    nr_concats: Optional[int] = field(
        default=3,
        metadata={"help": "Number of contexts to concatinate"},
    )




#################################################################
#
# Data loading functions
#
#################################################################



def concatinate_squad_data(d, span=3):
    """
    Concatinate "SPAN" number of SQuAD questions together
    """

    def get_span(index, span):
        """
        Returns the value in a range for whole numbers

        Ex: index=4, span=5
            lower=0, upper=5

            index=5, span=5
            lower=0, upper=5

            index=8, span=5
            lower=5, upper=10
        """
        lower_bound = (index)//span
        lower_bound = lower_bound*span
        upper_bound = lower_bound+span
        return lower_bound, upper_bound


    def set_start_pos(example, idx):
        """
        Get correct new starting position when concatinating SQuAD datasets
        """
        low, high = get_span(idx, span)

        # Get new starting position
        prev_start=0
        if idx != low:
            prev_start = len(''.join(data['context'][low:idx]))

        start_pos = data['answers'][idx]['answer_start'][0]
        if not isinstance(start_pos, int):
            start_pos = start_pos[0]
        new_start = [prev_start + start_pos]
        example['answers']['answer_start'] = new_start
        return example


    def set_context(example, idx):
        """
        Concatinate "SPAN" number of SQuAD samples
        """
        low, high = get_span(idx, span)

        # Get new context
        example['context'] = ''.join(data['context'][low:high])
        return example


    # First filters out questions using the same context but different questions and starts
    data = d.filter(lambda example, idx: example['context'] != d['context'][idx-1], with_indices=True)
    data = data.map(lambda example, idx: set_start_pos(example, idx), with_indices=True)
    data = data.map(lambda example, idx: set_context(example, idx), with_indices=True)
    print("Data length =", len(data))
    return data







#################################################################
#
#  Main function
#
#################################################################

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Needed for getting eval_loss for QA with Trainer::  transformer release between 3.0.2 and 4.0.0
    training_args.label_names = ["start_positions", "end_positions"]
    
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )


    if data_args.data_dir is None:
        data_args.data_dir = "."

    if (data_args.train_file_path is None or data_args.valid_file_path is None):
         data_args.train_file_path = f"{data_args.data_dir}/train_data.pt"
         data_args.valid_file_path = f"{data_args.data_dir}/val_data.pt"


    logger.info("Model parameters set: \n", model_args)
    logging.info(f"Logging to file: {training_args.logging_dir}.log")

    # Set seed
    set_seed(training_args.seed)


    
    # Set up tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        do_lowercase=args.do_lowercase,
        pad_to_max_length=True,
        max_length=args.max_length,
        truncation=True,
        use_fast=True,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )


    ##########################################
    #
    # Configure and configure the data
    #
    ##########################################


    if data_args.datasets == "xquad":
        xquad_ar = nlp.load_dataset('xquad', 'xquad.ar', split="validation")# Arabic
        xquad_ar = convert_dataset_to_torch_format(xquad_ar)

        xquad_de = nlp.load_dataset('xquad', 'xquad.de', split="validation")# German
        xquad_de = convert_dataset_to_torch_format(xquad_de)

        xquad_el = nlp.load_dataset('xquad', 'xquad.el', split="validation")# Greek
        xquad_el = convert_dataset_to_torch_format(xquad_el)

        xquad_en = nlp.load_dataset('xquad', 'xquad.en', split="validation")# English
        xquad_en = convert_dataset_to_torch_format(xquad_en)

        xquad_es = nlp.load_dataset('xquad', 'xquad.es', split="validation")# Spanish
        xquad_es = convert_dataset_to_torch_format(xquad_es)

        xquad_hi = nlp.load_dataset('xquad', 'xquad.hi', split="validation")# Hindi
        xquad_hi = convert_dataset_to_torch_format(xquad_hi)

        xquad_ru = nlp.load_dataset('xquad', 'xquad.ru', split="validation")# Russian
        xquad_ru = convert_dataset_to_torch_format(xquad_ru)

        xquad_th = nlp.load_dataset('xquad', 'xquad.th', split="validation")# Thai
        xquad_th = convert_dataset_to_torch_format(xquad_th)

        xquad_tr = nlp.load_dataset('xquad', 'xquad.tr', split="validation")# Turkish
        xquad_tr = convert_dataset_to_torch_format(xquad_tr)

        xquad_vi = nlp.load_dataset('xquad', 'xquad.vi', split="validation")# Vietnamese
        xquad_vi = convert_dataset_to_torch_format(xquad_vi)

        xquad_zh = nlp.load_dataset('xquad', 'xquad.zh', split="validation")# Chinese
        xquad_zh = convert_dataset_to_torch_format(xquad_zh)


    if data_args.datasets == "mlqa":
        mlqa_ar = nlp.load_dataset('mlqa', 'mlqa.ar.ar', split="validation")# Arabic
        mlqa_ar = convert_dataset_to_torch_format(mlqa_ar)

        mlqa_de = nlp.load_dataset('mlqa', 'mlqa.de.de', split="validation")# German
        mlqa_de = convert_dataset_to_torch_format(mlqa_de)

        mlqa_en = nlp.load_dataset('mlqa', 'mlqa.en.en', split="validation")# English
        mlqa_en = convert_dataset_to_torch_format(mlqa_en)

        mlqa_es = nlp.load_dataset('mlqa', 'mlqa.es.es', split="validation")# Spanish
        mlqa_es = convert_dataset_to_torch_format(mlqa_es)

        mlqa_hi = nlp.load_dataset('mlqa', 'mlqa.hi.hi', split="validation")# Hindi
        mlqa_hi = convert_dataset_to_torch_format(mlqa_hi)

        mlqa_vi = nlp.load_dataset('mlqa', 'mlqa.vi.vi', split="validation")# Vietnamese
        mlqa_vi = convert_dataset_to_torch_format(mlqa_vi)

        mlqa_zh = nlp.load_dataset('mlqa', 'mlqa.zh.zh', split="validation")# Chinese
        mlqa_zh = convert_dataset_to_torch_format(mlqa_zh)


    # FIXME
    if data_args.datasets == "tydiqa":
        raise ValueError("Not yet implemented")


    # Define long context training and how many contexts to concatinate together
    SPAN = args.nr_concats
    if data_args.datasets == "xquad_long":
        xquad_ar = nlp.load_dataset('xquad', 'xquad.ar', split="validation")# Arabic
        xquad_ar = concatinate_squad_data(xquad_ar, SPAN)
        xquad_ar = convert_dataset_to_torch_format(xquad_ar)

        xquad_de = nlp.load_dataset('xquad', 'xquad.de', split="validation")# German
        xquad_de = concatinate_squad_data(xquad_de, SPAN)
        xquad_de = convert_dataset_to_torch_format(xquad_de)

        xquad_el = nlp.load_dataset('xquad', 'xquad.el', split="validation")# Greek
        xquad_el = concatinate_squad_data(xquad_el, SPAN)
        xquad_el = convert_dataset_to_torch_format(xquad_el)

        xquad_en = nlp.load_dataset('xquad', 'xquad.en', split="validation")# English
        xquad_en = concatinate_squad_data(xquad_en, SPAN)
        xquad_en = convert_dataset_to_torch_format(xquad_en)

        xquad_es = nlp.load_dataset('xquad', 'xquad.es', split="validation")# Spanish
        xquad_es = concatinate_squad_data(xquad_es, SPAN)
        xquad_es = convert_dataset_to_torch_format(xquad_es)

        xquad_hi = nlp.load_dataset('xquad', 'xquad.hi', split="validation")# Hindi
        xquad_hi = concatinate_squad_data(xquad_hi, SPAN)
        xquad_hi = convert_dataset_to_torch_format(xquad_hi)

        xquad_ru = nlp.load_dataset('xquad', 'xquad.ru', split="validation")# Russian
        xquad_ru = concatinate_squad_data(xquad_ru, SPAN)
        xquad_ru = convert_dataset_to_torch_format(xquad_ru)

        xquad_th = nlp.load_dataset('xquad', 'xquad.th', split="validation")# Thai
        xquad_th = concatinate_squad_data(xquad_th, SPAN)
        xquad_th = convert_dataset_to_torch_format(xquad_th)

        xquad_tr = nlp.load_dataset('xquad', 'xquad.tr', split="validation")# Turkish
        xquad_tr = concatinate_squad_data(xquad_tr, SPAN)
        xquad_tr = convert_dataset_to_torch_format(xquad_tr)

        xquad_vi = nlp.load_dataset('xquad', 'xquad.vi', split="validation")# Vietnamese
        xquad_vi = concatinate_squad_data(xquad_vi, SPAN)
        xquad_vi = convert_dataset_to_torch_format(xquad_vi)

        xquad_zh = nlp.load_dataset('xquad', 'xquad.zh', split="validation")# Chinese
        xquad_zh = concatinate_squad_data(xquad_zh, SPAN)
        xquad_zh = convert_dataset_to_torch_format(xquad_zh)


    if data_args.datasets == "squad_long" or data_args.datasets == "xquad_long":
        squad_train = nlp.load_dataset('squad', split='train') # convert val dataset to long here
        squad_train = concatinate_squad_data(squad_train, SPAN)

        squad_valid = nlp.load_dataset('squad', split='validation') # convert val dataset to long here
        squad_valid = concatinate_squad_data(squad_valid, SPAN)

        train_dataset = convert_dataset_to_torch_format(squad_train)
        valid_dataset = convert_dataset_to_torch_format(squad_valid)


    if data_args.datasets == "xquad" or data_args.datasets == "mlqa"  or  data_args.datasets == "squad":
        # When using XQuAD
        squad_train, squad_valid = nlp.load_dataset('squad', split=['train', 'validation'])
        
        train_dataset = convert_dataset_to_torch_format(squad_train)
        valid_dataset = convert_dataset_to_torch_format(squad_valid)


    # TODO: Fix so this wont crash!
    # cache the dataset, so we can load it directly for training
    torch.save(train_dataset, data_args.train_file_path)
    torch.save(valid_dataset, data_args.valid_file_path)

    # Get datasets
    print('loading data')
    train_dataset  = torch.load(data_args.train_file_path)
    valid_dataset = torch.load(data_args.valid_file_path)
    print('loading done')

    # FIXME - add as a param
    # Needed to plot the loss
    training_args.label_names = ["start_positions", "end_positions"] # fixes missing eval_loss logging

    # Gives option to train and eval o
    # Skip training and only eval a fine-tuned model
    if training_args.do_train:

        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=DummyDataCollator(),
            prediction_loss_only=True,
        )


        print(f"\n\n{model_args.model_name_or_path}\n")
        # Training
        if training_args.do_train:
            trainer.train(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            )
            trainer.save_model()
            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            if trainer.is_world_process_zero():
                tokenizer.save_pretrained(training_args.output_dir)



        # Evaluation
        results = {}
        if training_args.do_eval and training_args.local_rank in [-1, 0]:
            logger.info("*** Evaluate ***")

            eval_output = trainer.evaluate()
            output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
            print("\n'==========================================\n")
            print("Eval output:     ", eval_output)
            print("\n'==========================================\n")

            # TODO add steps back in
            #if training_args.max_steps == -1: # if evaluating based on epochs
            #    global_step = eval_output["step"]

            #else: # if training based on max_steps
            #    global_step = eval_output['epoch']

            # TODO!!!  -->>>  Test to not add it first
            # tb_writer.add_scalar("eval_loss", eval_output["eval_loss"] , global_step)
            # TODO: Label_issue might make it possible to add custom metric for F1 and EM???
            # tb_writer.add_scalar("eval_f1", eval_output["loss"], global_step)
            # tb_writer.add_scalar("eval_exact_match", eval_output["loss"], global_step)

            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(eval_output.keys()):
                    logger.info("  %s = %s", key, str(eval_output[key]))
                    writer.write("%s = %s\n" % (key, str(eval_output[key])))
                    print(key, str(eval_output[key]))

            results.update(eval_output)

        logging.info("=" * 45)
        logging.info("Results from evaluation:")
        logging.info(results)
        logging.info("\n")

    logging.info("" * 45)




    ####################################
    #
    # Evaluate the trained model
    #
    ####################################

    # Load the fine-tuned model and tokenizer
    if training_args.do_train:
        tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir, use_fast=True, do_lowercase=args.do_lowercase)
        model = AutoModelForQuestionAnswering.from_pretrained(training_args.output_dir, )
    else:
        # Try loading model from where it assumes prev. trained model is stored
        # Else load from the set `model_name_or_path`
        try:
            model_path = training_args.output_dir
            tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir, use_fast=True, do_lowercase=args.do_lowercase)
            model = AutoModelForQuestionAnswering.from_pretrained(training_args.output_dir,) 
        except:
            model_path = model_args.model_name_or_path
            tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir, use_fast=True, do_lowercase=args.do_lowercase)
            model = AutoModelForQuestionAnswering.from_pretrained(training_args.output_dir,)


    model = model.cuda()
    model.eval()

    # TODO: Make better loader and evaluation, ex: Dict of dataset names and the dataset
    # { "k" : v }

    if data_args.datasets == "xquad" or data_args.datasets == "xquad_long":
        get_squad_evaluation(valid_dataset, model, tokenizer, "SQuAD 1.1 validation dataset", training_args.per_device_eval_batch_size)
        get_squad_evaluation(xquad_ar, model, tokenizer, "XQuAD Arabic validation", training_args.per_device_eval_batch_size)
        get_squad_evaluation(xquad_de, model, tokenizer, "XQuAD German validation", training_args.per_device_eval_batch_size)
        get_squad_evaluation(xquad_el, model, tokenizer, "XQuAD Greek validation", training_args.per_device_eval_batch_size)
        get_squad_evaluation(xquad_en, model, tokenizer, "XQuAD English validation", training_args.per_device_eval_batch_size)
        get_squad_evaluation(xquad_es, model, tokenizer, "XQuAD Spanish validation", training_args.per_device_eval_batch_size)
        get_squad_evaluation(xquad_hi, model, tokenizer, "XQuAD Hindi validation", training_args.per_device_eval_batch_size)
        get_squad_evaluation(xquad_ru, model, tokenizer, "XQuAD Russian validation", training_args.per_device_eval_batch_size)
        get_squad_evaluation(xquad_th, model, tokenizer, "XQuAD Thai validation", training_args.per_device_eval_batch_size)
        get_squad_evaluation(xquad_tr, model, tokenizer, "XQuAD Turkish validation", training_args.per_device_eval_batch_size)
        get_squad_evaluation(xquad_vi, model, tokenizer, "XQuAD Vietnamese validation", training_args.per_device_eval_batch_size)
        get_squad_evaluation(xquad_zh, model, tokenizer, "XQuAD Chinese validation", training_args.per_device_eval_batch_size)
    

    elif data_args.datasets == "mlqa":
        get_squad_evaluation(valid_dataset, model, tokenizer, "SQuAD 1.1 validation dataset", training_args.per_device_eval_batch_size)
        get_squad_evaluation(mlqa_ar, model, tokenizer, "MLQA Arabic validation", training_args.per_device_eval_batch_size)
        get_squad_evaluation(mlqa_de, model, tokenizer, "MLQA German validation", training_args.per_device_eval_batch_size)
        get_squad_evaluation(mlqa_en, model, tokenizer, "MLQA English validation", training_args.per_device_eval_batch_size)
        get_squad_evaluation(mlqa_es, model, tokenizer, "MLQA Spanish validation", training_args.per_device_eval_batch_size)
        get_squad_evaluation(mlqa_hi, model, tokenizer, "MLQA Hindi validation", training_args.per_device_eval_batch_size)
        get_squad_evaluation(mlqa_vi, model, tokenizer, "MLQA Vietnamese validation", training_args.per_device_eval_batch_size)
        get_squad_evaluation(mlqa_zh, model, tokenizer, "MLQA Chinese validation", training_args.per_device_eval_batch_size)


    elif data_args.datasets == "squad" or data_args.datasets == "squad_long":
        get_squad_evaluation(valid_dataset, model, tokenizer, "SQuAD 1.1 validation dataset", training_args.per_device_eval_batch_size)


    elif data_args.datasets == "trivia_qa":
        pass

    else:
        print("Not a valid eval dataset...\n Exiting")

if __name__ == '__main__':
    main()


