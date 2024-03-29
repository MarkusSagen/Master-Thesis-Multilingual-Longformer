#!/usr/bin/env python

import argparse
import datetime
from dataclasses import dataclass, field
import functools
import logging
import math
import os
import pickle
import re
import sys
import time
import threading
from typing import Optional

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
import tqdm
from transformers import logging as hf_logging
from transformers.modeling_longformer import LongformerSelfAttention
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForMaskedLM,
    RobertaForMaskedLM,
    XLMRobertaForMaskedLM,
    AutoTokenizer,
)

from transformers import (
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


class color:
    """Help print colors to terminal."""
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


def is_roberta_based_model(model_name: str) -> str:
    """Validate if the model to pre-train is of roberta architecture."""

    r = re.compile('(.*)roberta(.*)')
    matches = r.findall(model_name)
    base_name = 'none'
    if len(matches) > 0:
        base_name = '-'.join(model_name.split('-')[:-1])

    return base_name


##########################################
#
# Arguments
#
##########################################

"""Helper function: Define argparser and args."""
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    default=None,
    type=str,
    help="Name to save the model as.",
)
parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    help="The output directory for the trained model.",
)
parser.add_argument(
    "--model_type",
    default=None,
    type=str,
    help="Model type selected in the list from Huggingface ex:"
    " `bert, roberta, xlm-roberta, ...`",
)
parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pretrained model from huggingface.co/models. "
    "Only tested on `xlm-roberta-base` and `roberta-base`.",
)
parser.add_argument(
    "--logging_dir",
    default=None,
    type=str,
    help="Where logs are stored.",
)
parser.add_argument(
    "--model_max_length",
    default=4096,
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
parser.add_argument(
    "--evaluation_strategy",
    default="no",
    type=str,
    help="How evaluation should be logged, 'steps', 'epochs', 'no'.",
)
parser.add_argument(
    "--do_train",
    action="store_true",
    help="Whether to run training."
)
parser.add_argument(
    "--do_eval",
    action="store_true",
    help="Whether to run eval on the dev set."
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
    help="Number of gradient updates to perform before updating the weights",
)
parser.add_argument(
    "--weight_decay",
    default=0.0,
    type=float,
    help="Weight decay if we apply some."
)
parser.add_argument(
    "--adam_epsilon",
    default=1e-8,
    type=float,
    help="Epsilon for Adam optimizer."
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
    help="If > 0: set total number of training steps to perform. "
    "Override num_train_epochs.",
)
parser.add_argument(
    "--warmup_steps",
    default=0,
    type=int,
    help="Linear warmup over warmup_steps."
)
parser.add_argument(
    "--verbose_logging",
    action="store_true",
    help="If true, log all information when loading datasets.",
)
parser.add_argument(
    "--cache_dir",
    default=None,
    help="Where do you want to store the pretrained models.",
)
parser.add_argument(
    "--lang_id",
    default=0,
    type=int,
    help="language id of input for language-specific xlm models "
    "(see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
)
parser.add_argument(
    "--logging_steps",
    type=int,
    default=500,
    help="Log every X updates steps."
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
    help="Evaluate all checkpoints starting with the same prefix as model_name"
    "ending and ending with step number",
)
parser.add_argument(
    "--overwrite_output_dir",
    action="store_true",
    help="Overwrite the content of the output directory",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="random seed for initialization"
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
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex)",
)
parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in"
    "['O0', 'O1', 'O2', and 'O3'].",
)
parser.add_argument(
    "--train_file_path",
    type=str,
    default="/workspace/data/wikitext-103/wiki.train.raw",
    help="File path to language model training file",
)
parser.add_argument(
    "--val_file_path",
    type=str,
    default="/workspace/data/wikitext-103/wiki.valid.raw",
    help="File path to language model training file",
)
parser.add_argument(
    "--eval_steps",
    type=int,
    default=None,
    help="File path to language model training file",
)

args = parser.parse_args()

hf_logging.enable_default_handler()
hf_logging.set_verbosity_info()
hf_logging.enable_explicit_format()

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
    f"Process rank: {args.local_rank}, \
    distributed training: {bool(args.local_rank != -1)}, \
    16-bits training: {args.fp16}"
)


##########################################
#
# Replace Huggingface - TextDataset
#
##########################################

# https://github.com/tqdm/tqdm/issues/458
def provide_progress_bar(
    function, estimated_time, tstep=0.2, tqdm_kwargs={}, args=[], kwargs={}
):
    ret = [None]  # Mutable var so the function can store its return value

    def myrunner(function, ret, *args, **kwargs):
        ret[0] = function(*args, **kwargs)

    thread = threading.Thread(
        target=myrunner, args=(function, ret) + tuple(args), kwargs=kwargs
    )
    pbar = tqdm.tqdm(total=estimated_time, **tqdm_kwargs)

    thread.start()
    while thread.is_alive():
        thread.join(timeout=tstep)
        pbar.update(tstep)
    pbar.close()
    return ret[0]


def progress_wrapped(estimated_time, tstep=0.2, tqdm_kwargs={}):
    def real_decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return provide_progress_bar(
                function,
                estimated_time=estimated_time,
                tstep=tstep,
                tqdm_kwargs=tqdm_kwargs,
                args=args,
                kwargs=kwargs,
            )

        return wrapper
    return real_decorator


class TextDataset(Dataset):
    # Ugly HACK on older transformers
    # Use same code as Huggingface TextDataset
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):
        assert os.path.isfile(
            file_path), f"Input file path {file_path} not found"
        block_size = block_size - \
            tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            "cached_lm_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        @progress_wrapped(estimated_time=200)
        def tokenize_text(text):
            return tokenizer.tokenize(text)

        @progress_wrapped(estimated_time=300)
        def convert_tokens_to_ids(tokenized_text):
            return tokenizer.convert_tokens_to_ids(tokenized_text)

        if os.path.exists(cached_features_file) and not overwrite_cache:
            start = time.time()
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
            logger.info(
                f"Loading features from cached file {cached_features_file} [took %.3f s]",
                time.time() - start,
            )

        else:
            logger.info(
                f"Creating features from dataset file at {directory}\n\n")

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            # For large texts and models, this could take a long time
            # Done i two steps, since each part can take between 5-10 min
            start = time.time()
            text = tokenize_text(text)
            logger.info("Tokenizing text [took %.3f s]", time.time() - start)
            start = time.time()
            tokenized_text = convert_tokens_to_ids(text)
            logger.info(
                "Converting text to id [took %.3f s]\n", time.time() - start)

            start = time.time()
            for i in range(
                0, len(tokenized_text) - block_size + 1, block_size
            ):  # Truncate in block of block_size
                self.examples.append(
                    tokenizer.build_inputs_with_special_tokens(
                        tokenized_text[i: i + block_size]
                    )
                )
            logger.info(
                "Build tokenizer inputs by block_size length [took %.3f s]",
                time.time() - start,
            )

            start = time.time()
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(
                "Saving features into cached file %s [took %.3f s]",
                cached_features_file,
                time.time() - start,
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


###########################################################
#
# Longformer conversion
#
###########################################################

# TODO: Huggingface transformers v. >3.5.1 breaks this
class LongModelSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        print()

        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
        )


# Load initial model
MODEL: PreTrainedModel

if is_roberta_based_model(args.model_name_or_path) == "xlm-roberta":
    MODEL = XLMRobertaForMaskedLM
elif is_roberta_based_model(args.model_name_or_path) == "roberta":
    MODEL = RobertaForMaskedLM
else:
    raise NotImplementedError(
        "Currently only supports roberta-based architectures.")


class LongModelForMaskedLM:
    def __init__(self, config):
        super().__init__(config)
        print(f"\n{color.YELLOW}Converting models to Longformer is currently only tested for RoBERTa like architectures.{color.END}")
        for i, layer in enumerate(self.roberta.encoder.layer):
            layer.attention.self = LongModelSelfAttention(config, layer_id=i)


def create_long_model(
        save_model_to,
        model,
        tokenizer,
        attention_window,
        model_max_length
):

    config = model.config
    position_embeddings = model.roberta.embeddings.position_embeddings

    tokenizer.model_max_length = model_max_length
    tokenizer.init_kwargs['model_max_length'] = model_max_length
    current_model_max_length, embed_size = position_embeddings.weight.shape

    # NOTE: RoBERTa has positions 0,1 reserved
    # embedding size is max position + 2
    model_max_length += 2
    config.max_position_embeddings = model_max_length
    assert model_max_length > current_model_max_length, \
        "New model max_length must be longer than current max_length"

    # BUG for XLM: Need to make all zeros sice too large base model
    new_pos_embed = position_embeddings.weight.new_zeros(
        model_max_length, embed_size
    )

    k = 2
    step = current_model_max_length - 2
    while k < model_max_length - 1:
        new_pos_embed[k:(
            k + step)] = position_embeddings.weight[2:]
        k += step

    # HACK for Huggingface transformers >=3.4.0 and < 4.0
    # https://github.com/huggingface/transformers/issues/6465#issuecomment-719042969
    position_embeddings.weight.data = new_pos_embed
    model.roberta.embeddings.position_embeddings.num_embeddings = len(
        new_pos_embed.data
    )
    num_model_embeddings = position_embeddings.num_embeddings
    model.roberta.embeddings.position_ids = torch.arange(
        0, num_model_embeddings
    )[None]

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = layer.attention.self.query
        longformer_self_attn.key_global = layer.attention.self.key
        longformer_self_attn.value_global = layer.attention.self.value

        layer.attention.self = longformer_self_attn

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer


def copy_proj_layers(model):
    for _, layer in enumerate(model.roberta.encoder.layer):
        layer.attention.self.query_global = layer.attention.self.query
        layer.attention.self.key_global = layer.attention.self.key
        layer.attention.self.value_global = layer.attention.self.value
    return model


def pretrain_and_evaluate(
    training_args, data_args, model, tokenizer, eval_only, model_path
):
    val_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=data_args.val_file_path,
        block_size=tokenizer.max_len,
    )
    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info(
            f"Loading and tokenizing training data is usually slow: {data_args.train_file_path}"
        )
        train_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=data_args.train_file_path,
            block_size=tokenizer.max_len,
        )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        prediction_loss_only=True,
    )

    eval_loss = trainer.evaluate()
    eval_loss = eval_loss["eval_loss"]
    print(f"Initial eval bpc: {color.GREEN}{eval_loss/math.log(2)}{color.END}")
    logger.info(f"Initial eval bpc: {eval_loss/math.log(2)}")

    if not eval_only:
        trainer.train(model_path=model_path)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss["eval_loss"]
        print(
            f"Eval bpc after pretraining: \
            {color.GREEN}{eval_loss/math.log(2)}{color.END}"
        )
        logger.info(f"Eval bpc after pretraining: {eval_loss/math.log(2)}")


@dataclass
class ModelArguments:
    """Huggingface parameters for the model training."""

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Name of pretrained model to load for model and tokenizer"
            ", based on huggingface.co/models, ex 'roberta-base'"
        },
    )
    model_name: str = field(
        default="roberta-base-long-4096-lm",
        metadata={"help": "Name to use when saving model."},
    )
    attention_window: int = field(
        default=512,
        metadata={"help": "Size of attention window"}
    )
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum position"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models."
        },
    )


@dataclass
class DataTrainingArguments:
    """Training and validation data arguments."""

    val_file_path: str = field(
        default="/workspace/data/wikitext-103-raw/wiki.valid.raw",
        metadata={"help": "File for training a Language Model"},
    )
    train_file_path: str = field(
        default="/workspace/data/wikitext-103-raw/wiki.train.raw",
        metadata={"help": "File for evaluating a Language Model"},
    )


def main():
    ############################################
    #
    # Define model params
    #
    ############################################

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) \
            already exists and is not empty.\
            Use --overwrite_output_dir to overcome."
        )

    ###########################################
    #
    # RUN
    #
    ###########################################

    start = time.time()
    print("---------------------------------------------------------")
    print(
        f"\nLoading from Huggingface pretrained model: \
        `{color.BOLD}{color.GREEN}\
        {model_args.model_name_or_path}\
        {color.END}{color.END}` \
        with name: {model_args.model_name}\n"
    )

    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=model_args.model_max_length,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )

    print(f"{color.RED}Creating Longformer model{color.END}")
    model_path = training_args.output_dir
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger.info(
        f"Converting {model_args.model_name_or_path} \
        into {model_args.model_name}"
    )
    model, tokenizer = create_long_model(
        save_model_to=model_path,
        model=model,
        tokenizer=tokenizer,
        attention_window=model_args.attention_window,
        model_max_length=model_args.model_max_length,
    )

    print(f"{color.RED}Loading Model{color.END}")
    logger.info(f"Loading the model from {model_path}")
    model = LongModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=model_args.model_max_length,
        use_fast=True
    )

    print(f"{color.RED}Evaluate{color.END}")
    logger.info(
        f"Pretraining \
        {model_args.model_name_or_path}-{model_args.model_max_length}... "
    )
    pretrain_and_evaluate(
        training_args,
        data_args,
        model,
        tokenizer,
        eval_only=False,
        model_path=training_args.output_dir,
    )

    print(
        f"{color.PURPLE}TIME elapsed{color.END}: {datetime.datetime.fromtimestamp(time.time()-start).strftime('%d days, %H:%M:%S')}"
    )

    logger.info(
        "Copying local projection layers into global projection layers..."
    )
    model = copy_proj_layers(model)
    logger.info(f"Saving model to {model_path}")
    model.save_pretrained(model_path)

    print(f"{color.RED}Loading Done model{color.END}")

    logger.info(f"Loading the model from {model_path}")
    model = LongModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)


if __name__ == "__main__":
    main()
