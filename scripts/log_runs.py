#!/usr/bin/env python 

import os
from os import path
import wget
from zipfile import ZipFile
import logging
import math
import random
import time

from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, field
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser
from transformers.modeling_longformer import LongformerSelfAttention






"""
class RobertaLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)


class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)
            
            
def create_long_model(save_model_to, attention_window, max_pos):
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', model_max_length=max_pos)
    config = model.config

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.roberta.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.roberta.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.roberta.embeddings.position_embeddings.weight[2:]
        k += step
    model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed

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
    for i, layer in enumerate(model.roberta.encoder.layer):
        layer.attention.self.query_global = layer.attention.self.query
        layer.attention.self.key_global = layer.attention.self.key
        layer.attention.self.value_global = layer.attention.self.value
    return model
"""

def pretrain_and_evaluate(args, model, tokenizer, eval_only, model_path):
    # set_trace()
    val_dataset = TextDataset(tokenizer=tokenizer,
                              file_path=args.val_datapath,
                              block_size=tokenizer.max_len)

    if eval_only:
        train_dataset = val_dataset
    else:
        print("Tokenizing takes a long time")
        start = time.time()
        logger.info(f'Loading and tokenizing training data is usually slow: {args.train_datapath}')
        train_dataset = TextDataset(tokenizer=tokenizer,
                                    file_path=args.train_datapath,
                                    block_size=tokenizer.max_len)

        end = time.time()
        print("Took {:.2f}s to tokenize".format(end-start))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    trainer = Trainer(model=model, args=args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=val_dataset, prediction_loss_only=True,)

    eval_loss = trainer.evaluate()
    eval_loss = eval_loss['eval_loss']
    logger.info(f'Initial eval bpc: {eval_loss/math.log(2)}')
    
    if not eval_only:
        trainer.train(model_path=model_path)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        logger.info(f'Eval bpc after pretraining: {eval_loss/math.log(2)}')





def get_data(model, tokenizer, eval_only, model_path):
    # set_trace()
    val_dataset = TextDataset(tokenizer=tokenizer,
                              file_path=args.val_datapath,
                              block_size=tokenizer.max_len)

    return val_dataset












@dataclass
class ModelArgs:
    attention_window: int = field(default=512, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=4096, metadata={"help": "Maximum position"})

        
        
def download_and_unzip(url):
    zipfile = url.rsplit('/', 1)[1]
    if os.path.exists(zipfile):
        return
    filename = wget.download(url)
    datafolder = zipfile.rsplit('.')[0]
    if path.exists(datafolder):
        return
    with ZipFile(filename, 'r') as zipObj:
       # Extract all the contents of zip file in current directory
       zipObj.extractall()
    








if __name__ == '__main__':
    # Download and unzip
    # TODO make as an ARG
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
    download_and_unzip(url)
    #DATA_DIR = ""
    DATA_DIR = "/workspace/data/"
    LOG_DIR = "/workspace/logs/"

    # FIXME - Gets the path of the file
    # dir_path = os.path.dirname(os.path.realpath(__file__))


    # Initialize Logger
    # save logfile as name of running file + log
    #logger_filename = str(__file__).rstrip('.py') + '.log'
    logger = logging.getLogger(__name__)
    logger_filename = LOG_DIR +  "log_runs.log" # str(__file__).rstrip(".py" + ".log")
    logging.basicConfig(level=logging.INFO, filename=logger_filename) # , filename=logger_filename)

    writer = SummaryWriter(log_dir=f'/workspace/logs/logruns')
    
    parser = HfArgumentParser((TrainingArguments, ModelArgs,))
    training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
        '--output_dir', 'tmp',
        '--logging_dir', '/workspace/logs/logruns',
        '--warmup_steps', '500',
        '--learning_rate', '0.00003',
        '--weight_decay', '0.01',
        '--adam_epsilon', '1e-6',
        '--max_steps', '3000',
        '--logging_steps', '500',
        '--save_steps', '500',
        '--max_grad_norm', '5.0',
        '--per_gpu_eval_batch_size', '1', #'8',
        '--per_gpu_train_batch_size', '1', #'2', # 32GB gpu with fp32
        '--gradient_accumulation_steps', '8',#'32',
        '--evaluate_during_training',
        '--do_train',
        '--do_eval',
    ])


    """
    # HERE are args for model training passed
    parser = HfArgumentParser((TrainingArguments, ModelArgs,))
    training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
        '--output_dir', 'tmp',
        #'--logging_dir', '/workspace/logs/longformer',
        '--warmup_steps', '500',
        '--learning_rate', '0.00003',
        '--weight_decay', '0.01',
        '--adam_epsilon', '1e-6',
        '--max_steps', '3000',
        '--logging_steps', '500',
        '--save_steps', '500',
        '--max_grad_norm', '5.0',
        #'--num_train_epochs', '3',
        #'--per_gpu_eval_batch_size', '1', #'8',
        '--per_device_eval_batch_size', '4', #'8',
        #'--per_gpu_train_batch_size', '1', #'2', # 32GB gpu with fp32
        # Makes it crash! Allocate 2 will try to make it allocate 32 GB, but Peltarion capacity is 23.65 GB
        '--per_device_train_batch_size', '1', #'2' # 32GB gpu with fp32
        '--gradient_accumulation_steps', '2',#'32',
        '--evaluate_during_training',
        '--do_train',
        '--do_eval',
    ])
    """


    training_args.val_datapath = 'wikitext-103-raw/wiki.valid.raw'
    training_args.train_datapath = 'wikitext-103-raw/wiki.train.raw'
    #
    # Choose GPU
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    roberta_base = RobertaForMaskedLM.from_pretrained('roberta-base')
    roberta_base_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    logger.info('Evaluating roberta-base (seqlen: 512) for refernece ...')
    

    data = get_data(training_args, roberta_base, roberta_base_tokenizer, eval_only=True, model_path=None)
    print(data)
    print('\n==========================================\n')
    print("length: ", len(data))
    print(data[:2])

    
    #print("Convert model to long")
    #model_path = f'{training_args.output_dir}/roberta-base-{model_args.max_pos}'
    #if not os.path.exists(model_path):
    #    os.makedirs(model_path)
    
    #logger.info(f'Converting roberta-base into roberta-base-{model_args.max_pos}')
    #model, tokenizer = create_long_model(
    #    save_model_to=model_path, attention_window=model_args.attention_window, max_pos=model_args.max_pos)

    #print("Loading model")
    #logger.info(f'Loading the model from {model_path}')
    #tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    #model = RobertaLongForMaskedLM.from_pretrained(model_path)

    #print("Start pretraining")
    #logger.info(f'Pretraining roberta-base-{model_args.max_pos} ... ')
    ##training_args.max_steps = 3   ## <<<<<<<<<<<<<<<<<<<<<<<< REMOVE THIS <<<<<<<<<<<<<<<<<<<<<<<<
    #pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path=training_args.output_dir)

    #print("Save projection layers")
    #logger.info(f'Copying local projection layers into global projection layers ... ')
    #model = copy_proj_layers(model)
    #logger.info(f'Saving model to {model_path}')
    #model.save_pretrained(model_path)

    #print("Load new model")
    #logger.info(f'Loading the model from {model_path}')
    #tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    #model = RobertaLongForMaskedLM.from_pretrained(model_path)
