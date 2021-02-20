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
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)




logger = logging.getLogger(__name__)
DSlogging.set_verbosity_warning()


parser = argparse.ArgumentParser()
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
    "--dataset",
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
    "--val_file_path",
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
    "--max_pos",
    default=512,
    type=int,
    choices=[
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
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
#          --max_seq_length 384 --doc_stride 128 --do_lower_case
#
parser.add_argument(
    "--do_lower_case",
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


# Predictions
#
#
def get_correct_alignement(context: str, answer) -> Tuple[int, int]:
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
        max_length=512,
        truncation=True,
    )
    context_encodings = tokenizer.encode_plus(example["context"])

    # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
    # this will give us the position of answer span in the context text
    start_idx, end_idx = get_correct_alignement(example["context"], example["answers"])
    start_positions_context = context_encodings.char_to_token(start_idx)
    end_positions_context = context_encodings.char_to_token(end_idx - 1)

    # here we will compute the start and end position of the answer in the whole example
    # as the example is encoded like this <s> question</s></s> context</s>
    # and we know the postion of the answer in the context
    # we can just find out the index of the sep token and then add that to position + 1 (+1 because there are two sep tokens)
    # this will give us the position of the answer span in whole example
    sep_idx = encodings["input_ids"].index(tokenizer.sep_token_id)
    start_positions = start_positions_context + sep_idx + 1
    end_positions = end_positions_context + sep_idx + 1

    if end_positions > 512:
        start_positions, end_positions = 0, 0

    encodings.update(
        {
            "start_positions": start_positions,
            "end_positions": end_positions,
            "attention_mask": encodings["attention_mask"],
        }
    )
    return encodings


##################
#
#  Training
#
##################


class DummyDataCollator:
    def __call__(self, batch):
        # def collate_batch(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example["input_ids"] for example in batch])
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        start_positions = torch.stack([example["start_positions"] for example in batch])
        end_positions = torch.stack([example["end_positions"] for example in batch])

        return {
            "input_ids": input_ids,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "attention_mask": attention_mask,
        }


class DataCollector:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([example["input_ids"] for example in batch])
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        start_positions = torch.stack([example["start_positions"] for example in batch])
        end_positions = torch.stack([example["end_positions"] for example in batch])

        return {
            "input_ids": input_ids,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "attention_mask": attention_mask,
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
    do_lower_case: bool = field(
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

    dataset: str = field(
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
    val_file_path: Optional[str] = field(
        default="valid_data.pt",
        metadata={"help": "Path for cached valid dataset"},
    )
    max_len: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )



parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

# we will load the arguments from a json file,
# make sure you save the arguments in at ./args.json
# TODO replace

# model_args, data_args, training_args = parser.parse_json_file(
#     json_file=os.path.abspath("args.json")
# )

model_args, data_args, training_args = parser.parse_args_into_dataclasses()
# model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath('args.json'))

# Setup logging
if training_args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter(log_dir=training_args.logging_dir)

logger = logging.getLogger("")
logger.setLevel(logging.INFO if training_args.local_rank in [-1,0] else logging.WARNING)

fh = logging.FileHandler(f"{training_args.logging_dir}.log")
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "[%(asctime)s], %(levelname)s %(message)s",
    datefmt="%a, %d %b %Y %H:%M:%S",
)
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)
logging.info("\n --> Starting logger:\n" + "=" * 55 + "\n")



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

if (data_args.train_file_path is None or data_args.val_file_path is None):
    data_args.train_file_path = f"{data_args.data_dir}/train_data.pt"
    data_args.val_file_path = f"{data_args.data_dir}/val_data.pt"


logging.info(f"Logging to file: {training_args.logging_dir}.log")
logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    training_args.local_rank,
    training_args.device,
    training_args.n_gpu,
    bool(training_args.local_rank != -1),
    training_args.fp16,
)
logger.info("Training/evaluation parameters %s", training_args)

# Set seed
set_seed(training_args.seed)


#if os.path.isfile(training_args.logging_dir):
#    logging.warning("Previous run detected... Removing file for clean run log")
#    os.remove(f"{training_args.logging_dir}".log)

# TODO check that dont give error
#logging.basicConfig(
#    level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
#    format="%(asctime)s [%(levelname)s] %(message)s",
#    handlers=[
#        logging.FileHandler(f"{training_args.logging_dir}.log"),
#        logging.StreamHandler(),
#    ],
#)

# Initialize
logger.info("=" * 50)
logger.info("=" + "\t" * 6 + " =")
logger.info("=" + "\tInitialization" + "\t" * 4 + " =")
logger.info("=" + "\t" * 6 + " =")
logger.info("=" * 50 + "\n\n")
# TODO logg better, gives warning
logger.info("Model parameters set: \n", model_args)


# Load pretrained model and tokenizer
# TODO: workaround for longformer
use_fast = True
tokenizer_base = model_args.model_name_or_path
if model_args.model_name_or_path == "allenai/longformer-base-4096":
    tokenizer_base = "roberta-base"


tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_base, do_lower_case=model_args.do_lower_case, use_fast=use_fast
)
model = AutoModelForQuestionAnswering.from_pretrained(model_args.model_name_or_path)


# tokenizer = LongformerTokenizerFast.from_pretrained(
#    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
#    cache_dir=model_args.cache_dir,
# )
# model = LongformerForQuestionAnswering.from_pretrained(
#    model_args.model_name_or_path,
#    cache_dir=model_args.cache_dir,
# )


# LOAD MODELS
# TODO - Fix checking if has this file
# if ( "train_file_path": f"{SQUAD_DIR}/train_data.pt",
#    "val_file_path": f"{SQUAD_DIR}/valid_data.pt",)
# load train and validation split of squad
#
# TODO findout how to cache and not need to parse this every time!
train_dataset = datasets.load_dataset(data_args.dataset, split="train")
valid_dataset = datasets.load_dataset(data_args.dataset, split="validation")

train_dataset = train_dataset.map(convert_to_features)
valid_dataset = valid_dataset.map(convert_to_features)

# Makes it a datasets.arrow_dataset.Dataset to make to tensor for training
columns = ["input_ids", "attention_mask", "start_positions", "end_positions"]
train_dataset.set_format(type="torch", columns=columns)
valid_dataset.set_format(type="torch", columns=columns)


# Save data
if not os.path.isdir(data_args.data_dir):
    os.mkdir(DATA_DIR)
    logging.info(f"Creating data dir: {DATA_DIR}")

# TODO try not load dataset first if .pt files exists
torch.save(train_dataset, data_args.train_file_path)
torch.save(valid_dataset, data_args.val_file_path)

# TODO
# Now load model
#print("loading data")
#from functools import partial
#import pickle

#pickle.load = partial(pickle.load, encoding="latin1")
#pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
#train_dataset = torch.load(
#    data_args.train_file_path,
#    map_location=lambda storage, loc: storage,
#    pickle_module=pickle,
#)
#valid_dataset = torch.load(
#    data_args.val_file_path,
#    map_location=lambda storage, loc: storage,
#    pickle_module=pickle,
#)

train_dataset = torch.load(data_args.train_file_path)
valid_dataset = torch.load(data_args.val_file_path)
print("loading done")



# TRAIN MODEL
training_args.label_names = ["start_positions", "end_positions"] # fixes missing eval_loss logging
# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=DummyDataCollator(),
    prediction_loss_only=True,
    # compute_metrics=compute_metrics,
)

# Training
if training_args.do_train:
    trainer.train(
        model_path=model_args.model_name_or_path
        if os.path.isdir(model_args.model_name_or_path)
        else None
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
    print(eval_output)


    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(eval_output.keys()):
            logger.info("  %s = %s", key, str(eval_output[key]))
            writer.write("%s = %s\n" % (key, str(eval_output[key])))

    results.update(eval_output)

logging.info("=" * 45)
logging.info("Results from evaluation:")
logging.info(results)
logging.info("\n")
logging.info("" * 45)


# EVALUATE
#
## SQuAD evaluation script. Modifed slightly for this notebook
# https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
# https://github.com/huggingface/transformers/tree/master/examples/question-answering
#
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


# BEGIN EVAL
#
tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(training_args.output_dir)
model = model.cuda()
model.eval()


logging.info("Generating perdictions")
# TODO
# def collate_fn(examples):
#     return tokenizer.pad(examples, return_tensors='pt')
# dataloader = torch.utils.data.DataLoader(encoded_dataset['train'], collate_fn=collate_fn, batch_size=8)

dataloader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=training_args.per_device_eval_batch_size
)


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
logging.info(f"Generating perdictions took: {time.time() - start}s")


logging.info("Computing F1 and Exact Match:")
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

logging.info("=" * 45)
logging.info(f"Results from prediction:\n{eval_score}\n")
logging.info(eval_score)
logging.info("Closing file...\n\n" + "=" * 55 + "\n" * 3)
logging.shutdown()
