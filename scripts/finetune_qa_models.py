#!/usr/bin/env python3

from __future__ import print_function
import argparse
from collections import Counter
from dataclasses import dataclass, field
import logging
import os
import re
import string
import sys
from typing import Any, Dict, List, Optional, Union

import datasets
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import logging as hf_logging
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollator,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
    Trainer,
    TrainingArguments,
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


@dataclass
class QADataset:
    """Collection for the language to load in HF datasets

    args:
    - langs: includes the number of languages to load,
    - text_on_eval: the print statements when evaluating the datasets
    - data: the tokenized datasets
    """
    langs: List[str]
    text_on_eval: List[str]
    data: List[Any] = None


SQUAD = QADataset(
    ["squad"],
    [
        "SQuAD 1.1 validation dataset"
    ]
)


# base xquad
XQUAD = QADataset(
    ["ar", "de", "el", "en", "es", "hi", "ru", "th", "tr", "vi", "zh", ],
    [
        "XQuAD Arabic validation",
        "XQuAD German validation",
        "XQuAD Greek validation",
        "XQuAD English validation",
        "XQuAD Spanish validation",
        "XQuAD Hindi validation",
        "XQuAD Russian validation",
        "XQuAD Thai validation",
        "XQuAD Turkish validation",
        "XQuAD Vietnamese validation",
        "XQuAD Chinese validation",
    ]
)

# base mlqa
MLQA = QADataset(
    ["ar", "de", "en", "es", "hi", "vi", "zh"],
    [
        "SQuAD 1.1 validation dataset",
        "MLQA Arabic validation",
        "MLQA German validation",
        "MLQA English validation",
        "MLQA Spanish validation",
        "MLQA Hindi validation",
        "MLQA Vietnamese validation",
        "MLQA Chinese validation",
    ]
)


def check_positive_concats(nr_concats):
    """Helper funtion for argparse
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
            raise argparse.ArgumentTypeError(
                f"--nr_concats expects a positive int as a value, \
                not {nr_concats}"
            )
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
    help="How many context to concatinate when using a `long` QA dataset.\n"
    "3 is default and yields an average context lenght of 2048 tokens",
)
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
    help="The output directory for the model checkpoints and predictions.",
)
parser.add_argument(
    "--model_type",
    default=None,
    type=str,
    help="Model type selected from Huggingface ex: `roberta, xlm-roberta`",
)
parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pretrained model from huggingface.co/models. \n"
    "Only tested on `xlm-roberta-base` and `roberta-base`.",
)
parser.add_argument(
    "--datasets",
    default=None,
    type=str,
    required=True,
    help="Define one of Huggingface Datasets Question Answering Tasks.",
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
    help="Directory to where to store training and validation torch files.",
)
parser.add_argument(
    "--logging_dir",
    default=None,
    type=str,
    help="The output directory where the the loggs are stored.",
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
    help="Number of updates to acummulate the gradient for before updating.",
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
    "--max_grad_norm",
    default=1.0,
    type=float,
    help="Max gradient norm."
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
    help="If > 0: set total number of training steps to perform."
    " Override num_train_epochs.",
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
    help="If true, display all logging messages from huggingface libraries."
    "A number of warnings are expected for a normal SQuAD evaluation.",
)
parser.add_argument(
    "--lang_id",
    default=0,
    type=int,
    help="language id of input for language-specific xlm models.",
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
    help="Evaluate all checkpoints starting with the same prefix as model_name",
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
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex).",
)
parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in "
    "['O0', 'O1', 'O2', and 'O3']."
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
parser.add_argument(
    "--do_lowercase",
    action="store_true",
    help="If input should be lowercase or not when tokenizing",
)


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
    f"Process rank: {args.local_rank}, \
    distributed training: {bool(args.local_rank != -1)}, \
    16-bits training: {args.fp16}"
)


logger.info("=" * 50)
logger.info("=" + "\t" * 6 + " =")
logger.info("=" + "\tInitialization" + "\t" * 4 + " =")
logger.info("=" + "\t" * 6 + " =")
logger.info("=" * 50 + "\n\n")


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
#                                       #
#       SQuADs Evaluation metrics       #
#                                       #
#########################################

def normalize_answer(s: str) -> str:
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


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(
    gold_answers: List[str],
    predictions: List[str]
) -> Dict[Union[str, float]]:

    f1 = exact_match = total = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
        total += 1
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths
        )
        f1 += metric_max_over_ground_truths(f1_score,
                                            prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}


####################################################
#
# Evaluation
#
####################################################


def get_squad_evaluation(
        valid_dataset: DataCollator,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str,
        batch_size: int
) -> None:
    """
    Makes a prediction and evaluates it based on the trained model
    The evaluation is based on the SQuAD evaluation metric:
    valdid_datset is expected to be converted to a torch Tensor type:
    """

    logging.info(f"Generating perdictions for dataset '{dataset_name}'")
    dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size)

    # predictions
    predicted_answers = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            start_scores, end_scores = model(
                input_ids=batch["input_ids"].cuda(),
                attention_mask=batch["attention_mask"].cuda(),
            )
            for i in range(start_scores.shape[0]):
                all_tokens = tokenizer.convert_ids_to_tokens(
                    batch["input_ids"][i])
                answer = " ".join(
                    all_tokens[
                        torch.argmax(start_scores[i]):
                        torch.argmax(end_scores[i]) + 1
                    ]
                )
                ans_ids = tokenizer.convert_tokens_to_ids(answer.split())
                answer = tokenizer.decode(ans_ids)
                predicted_answers.append(answer)

    valid_dataset.reset_format()
    predictions = []
    references = []
    for ref, pred_answer in zip(valid_dataset, predicted_answers):
        actual_answer = ref["answers"]["text"]
        predictions.append(pred_answer)
        references.append(actual_answer)

    eval_score = evaluate(references, predictions)
    logging.info(f"Results from prediction:\n{eval_score}\n" + "=" * 55 + "\n")


#########################################
#                                       #
# Convert train and validation datasets #
#                                       #
#########################################

def get_correct_alignement(context: str, answer):
    """Some original examples in SQuAD have indices wrong by 1 or 2 character.
    """
    gold_text = answer["text"][0]
    start_idx = answer["answer_start"][0]
    end_idx = start_idx + len(gold_text)
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx
    elif context[start_idx - 1: end_idx - 1] == gold_text:
        return start_idx - 1, end_idx - 1
    elif context[start_idx - 2: end_idx - 2] == gold_text:
        return start_idx - 2, end_idx - 2
    else:
        raise ValueError()


MAX_CONTEXT_LENGTH = 0


def convert_to_features(example):

    encodings = tokenizer.encode_plus(
        example["question"],
        example["context"],
        pad_to_max_length=True,
        max_length=args.max_length,
        truncation=True,
    )
    context_encodings = tokenizer.encode_plus(example["context"])

    start_idx, end_idx = get_correct_alignement(
        example["context"], example["answers"])
    start_positions_context = context_encodings.char_to_token(start_idx)
    end_positions_context = context_encodings.char_to_token(end_idx - 1)

    # FIXME: UGLY HACK because of XLM-R tokenization, works fine with roberta
    sep_idx = encodings["input_ids"].index(tokenizer.sep_token_id)
    try:
        start_positions = start_positions_context + sep_idx + 1
        end_positions = end_positions_context + sep_idx + 1

        # if end_positions > 4096:
        #    start_positions, end_positions = None, None
    except:
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
    data = data.map(convert_to_features).filter(
        lambda example: (example["start_positions"] is not None)
        and (example["end_positions"] is not None)
    )

    # set the tensor type and the columns which the dataset should return
    columns = ["input_ids", "attention_mask",
               "start_positions", "end_positions"]
    data.set_format(type="torch", columns=columns)
    print(max(data["start_positions"]))
    print(data.shape)
    return data


##################
#
#  Training
#
##################


class DummyDataCollator:
    def __call__(self, batch):

        input_ids = torch.stack([example["input_ids"] for example in batch])
        attention_mask = torch.stack(
            [example["attention_mask"] for example in batch])
        start_positions = torch.stack(
            [example["start_positions"] for example in batch])
        end_positions = torch.stack(
            [example["end_positions"] for example in batch])

        return {
            "input_ids": input_ids,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "attention_mask": attention_mask,
        }


@ dataclass
class ModelArguments:

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier"
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models"
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

    datasets: str = field(metadata={"help": "The dataset name to load."})
    data_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the dataset containing train and eval datasets."},
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


def load_datasets(
        languages: QADataset,
        base_dataset: str = None,
        concatinate: bool = False,
        split: str = 'validation',
):
    """Loads a Huggingface dataset based on the `base` dataset
    (squad/xquad/mlqa)."""

    dataset: List[Any] = []

    data: List
    dataset: str
    for lang in languages.langs:
        if base_dataset is not None:
            dataset = f"{base_dataset}.{lang}"
            if base_dataset == "mlqa":
                dataset = f"{dataset}.{lang}"

            data = datasets.load_dataset(base_dataset, dataset, split=split)
        else:
            data = datasets.load_dataset(lang, split=split)

        if concatinate:
            data = concatinate_squad_data(data, args.nr_concats)
        data = convert_dataset_to_torch_format(data)
        dataset.append(data)

    return dataset


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
        lower_bound = (index) // span
        lower_bound = lower_bound * span
        upper_bound = lower_bound + span
        return lower_bound, upper_bound

    def set_start_pos(example, idx):
        """
        Get correct new starting position when concatinating SQuAD datasets
        """
        low, high = get_span(idx, span)

        # Get new starting position
        prev_start = 0
        if idx != low:
            prev_start = len("".join(data["context"][low:idx]))

        start_pos = data["answers"][idx]["answer_start"][0]
        if not isinstance(start_pos, int):
            start_pos = start_pos[0]
        new_start = [prev_start + start_pos]
        example["answers"]["answer_start"] = new_start
        return example

    def set_context(example, idx):
        """
        Concatinate "SPAN" number of SQuAD samples
        """
        low, high = get_span(idx, span)

        # Get new context
        example["context"] = "".join(data["context"][low:high])
        return example

    # Filters out questions using the same context but different questions
    data = d.filter(
        lambda example, idx: example["context"] != d["context"][idx - 1],
        with_indices=True,
    )

    data = data.map(
        lambda example, idx: set_start_pos(example, idx),
        with_indices=True
    )
    data = data.map(
        lambda example, idx: set_context(example, idx),
        with_indices=True
    )

    return data


#################################################################
#
#  Main function
#
#################################################################


def main():

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Needed for getting eval_loss for QA in transformer v. 3.0.2 and 4.0.0
    training_args.label_names = ["start_positions", "end_positions"]

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) \
            already exists and is not empty. \
            Use --overwrite_output_dir to overcome."
        )

    if data_args.data_dir is None:
        data_args.data_dir = "."

    if data_args.train_file_path is None or data_args.valid_file_path is None:
        data_args.train_file_path = f"{data_args.data_dir}/train_data.pt"
        data_args.valid_file_path = f"{data_args.data_dir}/val_data.pt"

    logger.info("Model parameters set: \n", model_args)
    logging.info(f"Logging to file: {training_args.logging_dir}.log")

    set_seed(training_args.seed)

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

    if data_args.datasets == "xquad":
        XQUAD.data = load_datasets(XQUAD, base_dataset="xquad")

    if data_args.datasets == "mlqa":
        MLQA.data = load_datasets(MLQA, base_dataset="mlqa")

    if data_args.datasets == "tydiqa":
        raise ValueError("Not yet implemented")

    if data_args.datasets == "xquad_long":
        XQUAD.data = load_datasets(XQUAD, "xquad", concatinate=True)

    if data_args.datasets in ["squad_long", "xquad_long"]:
        train_dataset = load_datasets(
            SQUAD, split="train", concatinate=True)[0]
        valid_dataset = load_datasets(SQUAD, concatinate=True)[0]
        SQUAD.data = valid_dataset

    if (data_args.datasets in ["xquad", "mlqa", "squad"]):
        train_dataset = load_datasets(
            SQUAD, split="train", concatinate=True)[0]
        valid_dataset = load_datasets(SQUAD, concatinate=True)[0]
        SQUAD.data = valid_dataset

    torch.save(train_dataset, data_args.train_file_path)
    torch.save(valid_dataset, data_args.valid_file_path)

    train_dataset = torch.load(data_args.train_file_path)
    valid_dataset = torch.load(data_args.valid_file_path)

    ####################################
    #
    # Train the model
    #
    ####################################

    if training_args.do_train:

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=DummyDataCollator(),
            prediction_loss_only=True,
        )

        if training_args.do_train:
            trainer.train(
                model_path=model_args.model_name_or_path
                if os.path.isdir(model_args.model_name_or_path)
                else None
            )
            trainer.save_model()
            if trainer.is_world_process_zero():
                tokenizer.save_pretrained(training_args.output_dir)

        results = {}
        if training_args.do_eval and training_args.local_rank in [-1, 0]:
            logger.info("*** Evaluation ***")

            eval_output = trainer.evaluate()
            output_eval_file = os.path.join(
                training_args.output_dir, "eval_results.txt"
            )
            print("\n'==========================================\n")
            print("Eval output:     ", eval_output)
            print("\n'==========================================\n")

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

    if training_args.do_train:
        tokenizer = AutoTokenizer.from_pretrained(
            training_args.output_dir,
            use_fast=True,
            do_lowercase=args.do_lowercase
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            training_args.output_dir,
        )
    else:
        try:
            model_path = training_args.output_dir
            tokenizer = AutoTokenizer.from_pretrained(
                training_args.output_dir,
                use_fast=True,
                do_lowercase=args.do_lowercase
            )
            model = AutoModelForQuestionAnswering.from_pretrained(
                training_args.output_dir,
            )
        except:
            model_path = model_args.model_name_or_path
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=True, do_lowercase=args.do_lowercase
            )
            model = AutoModelForQuestionAnswering.from_pretrained(
                model_path
            )

    model = model.cuda()
    model.eval()

    get_squad_evaluation(
        SQUAD.data,
        model,
        tokenizer,
        SQUAD.text_on_eval,
        training_args.per_device_eval_batch_size,
    )
    if data_args.datasets == "xquad" or data_args.datasets == "xquad_long":
        for i, _ in enumerate(XQUAD.langs):
            get_squad_evaluation(
                XQUAD.data[i],
                model,
                tokenizer,
                XQUAD.text_on_eval[i],
                training_args.per_device_eval_batch_size,
            )
    elif data_args.datasets == "mlqa":
        for i, _ in enumerate(MLQA.langs):
            get_squad_evaluation(
                MLQA.data[i],
                model,
                tokenizer,
                MLQA.text_on_eval[i],
                training_args.per_device_eval_batch_size,
            )

    elif data_args.datasets == "trivia_qa":
        pass

    else:
        print("Not a valid eval dataset...\n Exiting")


if __name__ == "__main__":
    main()
