#!/usr/bin/env python3


import torch
import datasets as nlp
from transformers import LongformerTokenizerFast
from transformers import XLMRobertaTokenizerFast, AutoTokenizer
from transformers import XLMRobertaForQuestionAnswering
from transformers import RobertaForQuestionAnswering

from transformers.utils import logging as hf_logging


hf_logging.enable_default_handler()
hf_logging.set_verbosity_info()
hf_logging.enable_explicit_format()


#tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
#tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)
#tokenizer

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)
#tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')


"""
English
Spanish, German, Greek,
Russian, Turkish, Arabic,
Vietnamese, Thai, Chinese,
and Hindi.
"""
# Each of these are validation datasets
xquad_en = nlp.load_dataset('xquad', 'xquad.en', split="validation")
xquad_ru = nlp.load_dataset('xquad', 'xquad.ru', split="validation")
xquad_ar = nlp.load_dataset('xquad', 'xquad.ar', split="validation")
xquad_en


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
        max_length=512,
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

        if end_positions > 512:
            start_positions, end_positions = 0, 0

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






# When using only SQuAD

"""
## load train and validation split of squad
train_dataset  = nlp.load_dataset('squad', split=nlp.Split.TRAIN)
valid_dataset = nlp.load_dataset('squad', split=nlp.Split.VALIDATION)
print("Unchanged: ==>", len(valid_dataset), len(xquad_en))
#valid_dataset = xquad_en


# Add
train_dataset = train_dataset.map(convert_to_features).filter(lambda example: (example['start_positions'] is not None) and (example['end_positions'] is not None))
valid_dataset = valid_dataset.map(convert_to_features).filter(lambda example: (example['start_positions'] is not None) and (example['end_positions'] is not None))

# set the tensor type and the columns which the dataset should return
columns = ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
train_dataset.set_format(type='torch', columns=columns)
valid_dataset.set_format(type='torch', columns=columns)


#print(len(train_dataset), len(squad_train))
print("Modded: ==>",len(valid_dataset), len(xquad_en))
"""

# When using XQuAD
# train dataset
squad_train, squad_valid = nlp.load_dataset('squad', split=['train', 'validation'])
train_dataset = squad_train.map(convert_to_features).filter(lambda example: (example['start_positions'] is not None) and (example['end_positions'] is not None))
print("\n\n")
valid_dataset = squad_valid.map(convert_to_features).filter(lambda example: (example['start_positions'] is not None) and (example['end_positions'] is not None))


# set the tensor type and the columns which the dataset should return
columns = ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
train_dataset.set_format(type='torch', columns=columns)
valid_dataset.set_format(type='torch', columns=columns)

"""

/usr/local/lib/python3.6/dist-packages/transformers/tokenization_utils_base.py:2022:
FutureWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version,
use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch,
or use `padding='max_length'` to pad to a max length.
In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or
leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).
  FutureWarning,

"""








# cach the dataset, so we can load it directly for training

torch.save(train_dataset, 'train_data.pt')
torch.save(valid_dataset, 'valid_data.pt')



import json



args_dict = {
    #'xlm-roberta-base',
    "n_gpu": 1,
    "model_name_or_path": 'xlm-roberta-base',
    "max_len": 512 ,
    "output_dir": './models',
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    #"train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 1e-4,
    "num_train_epochs": 3,
    "do_train": True,
    "do_eval": True,
    "max_steps": 100,
    "logging_steps": 50,
    "eval_steps": 50,
    "prediction_loss_only": True,
    "seed": 42,
    "max_seq_length": 384,
    "doc_stride": 128,
    "evaluate_during_training": True,
    #"evaluation_strategy": "steps",
    "fp16": True,
    "do_lower_case": True,
}

with open('args.json', 'w') as f:
    json.dump(args_dict, f)


"""
/usr/local/lib/python3.6/dist-packages/transformers/tokenization_utils_base.py:1944: FutureWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert
"""


import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from transformers import LongformerForQuestionAnswering, LongformerTokenizerFast, EvalPrediction
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)

# @dataclass
class DummyDataCollator:
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
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
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: Optional[str] = field(
        default='train_data.pt',
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        default='valid_data.pt',
        metadata={"help": "Path for cached valid dataset"},
    )
    max_len: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # we will load the arguments from a json file,
    # make sure you save the arguments in at ./args.json
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath('args.json'))

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
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

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    print('loading data')
    train_dataset  = torch.load(data_args.train_file_path)
    valid_dataset = torch.load(data_args.valid_file_path)
    print('loading done')

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DummyDataCollator(),
    )

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
        print(eval_output)
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))

        results.update(eval_output)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

    args.prediction_loss_only=True













####################
# Main
####################
main()
