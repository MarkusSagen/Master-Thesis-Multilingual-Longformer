# Pre-Training Details 

### Models
Converting transformer models are based on the [Longformer conversion script](https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb). The script can be run for any pre-trained RoBERTa based model and can be extended to be used with other pre-traned models.  

Training with these parameters on a 48GB GPU takes ~5 days   
We pre-trained both a monolingual RoBERTa and multilingual XLM-R model using the Longformer pre-training scheme to extend the context's of the models. These models were trained on the same datasets and same hyper-parameters and only trained with one seed because of the long training time.   

The arguments `MAX_POS` indicate how many tokens the model should learn to attend. The number of tokens it can learn to attend to must be of the form $2^x$ and be larger than $512$.   

The `MODEL_NAME_OR_PATH` indicated the pre-trained model that the Longformer can be extended from. The names of the models must be pre-trained model names available at [Huggingface](https://huggingface.co/models), such as `roberta-base`, `xlm-roberta-base` or similar. The pre-training scheme should in theory work for all encoder-type Transformers, such as BERT, RoBERTa, Alberta, etc. However, we have only tested it for RoBERTa and XLM-R, so the training script may need to be changed if used for BERT.   

We refer to these models that we have trained using the Longformer pre-training scheme as:   

1. `RoBERTa-Long`   
2. `XLM-Long`   
   


