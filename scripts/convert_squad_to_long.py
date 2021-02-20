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

"""

This script converts datsets following the SQuAD format
Into a concatinated longer format

TODO
- Insert argparser to better control the params and dataset to be converted
- Insert how many documents to concat together or a max length of the combined documents
- Change `MAX_LEN` to a argparse variable

"""

MAX_LEN = 4096

def concatinate_squad_data(data):
    """
    Concatinate 5 SQuAD questions together
    """

    def get_span(index, span=5):
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
        low, high = get_span(idx, span=5)

        # Get new starting position
        if idx != low:
            prev_start = len(''.join(data['context'][low:idx]))
            start_pos = data['answers'][idx]['answer_start'][0]
            example['answers']['answer_start'] = [prev_start + start_pos]

        return example


    def set_context(example, idx):
        """
        Concatinate 5 SQuAD samples
        """
        low, high = get_span(idx, span=5)

        # Get new context
        example['context'] = ''.join(data['context'][low:high])
        return example


    # First filters out questions using the same context but different questions and starts
    data = data.filter(lambda example, indice: indice % 5 == 0, with_indices=True)
    data = data.map(lambda example, idx: set_start_pos(example, idx), with_indices=True)
    data = data.map(lambda example, idx: set_context(example, idx), with_indices=True)
    return data


tokenizer = AutoTokenizer.from_pretrained('roberta-base', cache_dir=None, do_lowercase=True,
                                          pad_to_max_length=True, max_length=MAX_LEN,
                                          truncation=True, use_fast=True)



# Change the dataset to the one you want to convert
dataset_name = "squad"
dataset_type = "validation"
data = nlp.load_dataset(dataset_name, split=dataset_type)


# Concatinate the dataset
data = concatinate_squad_data(data)

# Save the dataset
data.save_to_disk(f"{dataset_name}_long_{dataset_type}")
