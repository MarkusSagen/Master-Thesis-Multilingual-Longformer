
# Multilingual Longformer
Master thesis work for investigating if and how well multilingual models in low-resource languages (such as Swedish) can incorporate longer contexts without re-training the models from scratch on long-context datasets in each language. The goal was to investigate if a multilingual model, such as XLM-R, could be extended into a Longformer model and then pre-train only on English, while still having a long-context model in several languages.    
   
The script provided includes the necessary steps to reproduce the result presented in the master thesis. We convert pre-trained monolingual and multilingual language models on English to Longformer models to a maximum model length of 4096.

**We call the pre-trained models using the Longformer pre-training**:  
1. RoBERTa-Long  
2. XLM-Long  

Based on a RoBERTa and XLM-R model that has been pre-trained using the Longformer pre-training scheme.   

Training of all models are done through docker containers for reproducability  

## Usage and Setup  
Example of how to build, start, run and shutdown the docker container and the training script  
If you encounter problems, toggle the `Technical Requirement` and `Pre-Requisites` links to verify that you have a sufficiently large GPU and the pre-requisite applications/libraries installed.  

<details><summary><b>Technical Requirements</b></summary>
<p>
**Please Note**:
Running the following project is quite computationally expensive. It is required to have a Docker container with at least 90GB of RAM allocated for the pre-training and a CUDA enabled GPU with 48GB of memory!    
     
For the Fine-tuning on QA tasks, 32GB of RAM is sufficient and a smaller GPU can be used when fine-tuning on regular or multilingual SQuAD. However, for the datasets created with a longer context, it requires at least 32GB of RAM    
</p>
</details>


<details><summary><b>Pre-Requisites</b></summary>
<p>
The following applications and libraries needs to be installed in order to run the application
- [Docker](https://docs.docker.com/get-docker/)  
- [Docker Compose](https://docs.docker.com/compose/install/)  
- Miniconda or Anaconda with Python3  
- make (terminal command)  
- wget (terminal command)  
- unzip (terminal command)  
- tmux (terminal command)  
- CUDA enabled GPU (check if set up correctly by entering `nvidia-smi` in your terminal)  
- [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html) installed and linked to your Docker container (Needed if encountering error: ```ERROR: for XXX_markussagen_repl1  Cannot create container for service repl: Unknown runtime specified nvidia```)   
</p>
</details>


1. **Download the repo**   
    
        git@github.com:MarkusSagen/Master-Thesis-Multilingual-Longformer.git   
        cp .env.template .env
        
2.  **Download the dataset**   
    Unzip the dataset and then place it in a suitable location
    
        wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
        unzip wikitext-103-raw-v1.zip   

3.  **Change your model and dataset paths**   
    Open the `.env` file and change the `DATA_DIR` and the `MODEL_DIR` to the relative path to where you have you want your models stored and where you downloaded the dataset. Make sure that the folders you set exist on your system.   
    For instance:  
    
        DATA_DIR=/Users/admin/data/wikitext-103-raw
        MODEL_DIR=/Users/admin/model
4.  **Start the docker container** 
    
        make build && make up
5.  **Start tmux**  
    In your terminal start tmux. This will ensure that your runs are not stopped if you are disconnected from an ssh connection  
    
        tmux
6.  **Run the script**   
    Here is an example of how a training script might look like for pre-training a XLM-R model into a Longformer. The general format follows the parameters of [Huggingface Transformer's TrainingArgument](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments). 
    
        export SEED=42
        export MAX_POS=4096
        export MODEL_NAME_OR_PATH=xlm-roberta-base
        export MODEL_NAME=test-the-gpu
        export MODEL_DIR=/workspace/models
        export DATA_DIR=/workspace/data
        export LOG_DIR=/workspace/logs
        
        make repl run="scripts/run_long_lm.py \
            --model_name_or_path $MODEL_NAME_OR_PATH \
            --model_name $MODEL_NAME \
            --output_dir $MODEL_DIR/$MODEL_NAME \
            --logging_dir $LOG_DIR/$MODEL_NAME \
            --val_file_path $DATA_DIR/wiki.valid.raw \
            --train_file_path $DATA_DIR/wiki.train.raw \
            --seed $SEED \
            --max_pos $MAX_POS \
            --adam_epsilon 1e-8 \
            --warmup_steps 500 \
            --learning_rate 3e-5 \
            --weight_decay 0.01 \
            --max_steps 6000 \
            --evaluate_during_training \
            --logging_steps 50 \
            --eval_steps 50 \
            --save_steps 6000  \
            --max_grad_norm 1.0 \
            --per_device_eval_batch_size 2 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 64 \
            --overwrite_output_dir \
            --fp16 \
            --do_train \
            --do_eval
        "
7.  **Shutdown run and container** 
    
        make down

8. **(Optional) terminate tmux**   
   
        exit   
            
## Training and Evaluation in-depth    
The training of these models were done in two steps:   
1. Pre-train `RoBERTa-base` and `XLM-R-base` models into Longformer models   
2. Fine-tune regular RoBERTa and XLM-R models on SQuAD formated dataset. Compare the results of these with our Longformer trained models and a Longformer model released by the Longformer authors. We train these models with multiple different seeds, datasets and context length.   
    
We have grouped each model trained and evaluated based on:    
- The dataset and language used for each model    
- Then based on what model that were trained    

## Pre-train: Transfer Long-Context of Language Models   

The models were trained according to this structure   

    English Pre-training
    |-- Wikitext-103
        |-- RoBERTa-Long (4096)
        |-- XLM-Long (4096)



Each fine-tuning are grouped based on the dataset, language and context length and then evaluated for each model.  For more in-depth explanation of the pre-training script and parameters, see [Here](Pretraining_Details.md)   

<details><summary><b>Runs:</b></summary>
<p>   

<details><summary><b>Wikitext-103</b></summary>
<p>   

##### RoBERTa   

    export SEED=42
    export MAX_POS=4096
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=roberta-base
    export MODEL_NAME=$MODEL_NAME_OR_PATH-long
    export DATA_DIR=/workspace/data
    export LOG_DIR=/workspace/logs
    
    make repl run="scripts/run_long_lm.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --model_name $MODEL_NAME \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --val_file_path $DATA_DIR/wiki.valid.raw \
        --train_file_path $DATA_DIR/wiki.train.raw \
        --seed $SEED \
        --max_pos $MAX_POS \
        --adam_epsilon 1e-8 \
        --warmup_steps 500 \
        --learning_rate 3e-5 \
        --weight_decay 0.01 \
        --max_steps 6000 \
        --evaluate_during_training \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 500  \
        --max_grad_norm 1.0 \
        --per_device_eval_batch_size 2 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 64 \
        --overwrite_output_dir \
        --fp16 \
        --do_train \
        --do_eval
    "


##### XLM-R   

    export SEED=42
    export MAX_POS=4096
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=xlm-roberta-base
    export MODEL_NAME=$MODEL_NAME_OR_PATH-long
    export DATA_DIR=/workspace/data
    export LOG_DIR=/workspace/logs
    
    make repl run="scripts/run_long_lm.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --model_name $MODEL_NAME \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --val_file_path $DATA_DIR/wiki.valid.raw \
        --train_file_path $DATA_DIR/wiki.train.raw \
        --seed $SEED \
        --max_pos $MAX_POS \
        --adam_epsilon 1e-8 \
        --warmup_steps 500 \
        --learning_rate 3e-5 \
        --weight_decay 0.01 \
        --max_steps 6000 \
        --evaluate_during_training \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 500  \
        --max_grad_norm 1.0 \
        --per_device_eval_batch_size 2 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 64 \
        --overwrite_output_dir \
        --fp16 \
        --do_train \
        --do_eval
    "

</p>
</details>

</p>
</details>   


## Fine-Tune on Question Answering Tasks    

    
    English QA Fine-Tuning:
    |-- SQuAD
        |-- RoBERTa (512)
        |-- Longformer (4096)
        |-- RoBERTa-Long (4096)
        |-- XLM-R (512)
        |-- XLM-Long (4096)
    |-- SQ3 (512)
        |-- RoBERTa (512)
        |-- Longformer (4096)
        |-- RoBERTa-Long (4096)
        |-- XLM-R (512)
        |-- XLM-Long (4096)
    |-- SQ3 (2048)
        |-- RoBERTa (512)
        |-- Longformer (4096)
        |-- RoBERTa-Long (4096)
        |-- XLM-R (512)
        |-- XLM-Long (4096)
    |-- TriviaQA = TODO
    
    Multilingual QA Fine-Tuning:
    |-- XQuAD
        |-- RoBERTa (512)
        |-- XLM-R (512)
        |-- XLM-Long (4096)
    |-- XQ3 (512)
        |-- XLM-R (512)
        |-- XLM-Long (4096)
    |-- XQ3 (2048)
        |-- XLM-R (512)
        |-- XLM-Long (4096)
    |-- MLQA
        |-- XLM-R (512)
        |-- XLM-Long (4096)


We fine-tune the models on SQuAD-formated extractive question-answer datasets in English and multiple other languages. We also create a concatenated dataset with longer context for both the SQuAD and XQuAD (multilingual SQuAD). The datasets are provided through Huggingface's Datasets library.   

For more more in-depth information regarding how the fine-tuning scripts, parameters and evaluation setup, see [Here](Finetuning_Details.md)     

<details><summary><b>Runs:</b></summary>
<p>   

Each fine-tuning are grouped based on the dataset, language and context length and then evaluated for each model.   

### English    

<details><summary><b>SQuAD</b></summary>   
<p>  
  
##### RoBERTa   

    export SEED=42
    export DATASET=squad
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=roberta-base
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 8 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --max_length 512 \
    "

   
##### Longformer   

    export SEED=42
    export DATASET=squad
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=allenai/longformer-base-4096
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 8 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --max_length 512 \
    "


##### RoBERTa-Long  

    export SEED=42
    export DATASET=squad
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=$MODEL_DIR/roberta-base-long
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 8 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --max_length 512 \
    "


##### XLM-R   

    export SEED=42
    export DATASET=squad
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=xlm-roberta-base
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 8 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --max_length 512 \
    "


##### XLM-Long   

    export SEED=42
    export DATASET=squad
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=$MODEL_DIR/xlm-roberta-base-long
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 8 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --max_length 512 \
    "


</p>
</details>

<details><summary><b>SQ3 (512)</b></summary>
<p>   
  
##### RoBERTa   

    export SEED=42
    export MAX_LENGTH=512
    export NR_CONCATS=1
    export DATASET=squad_long
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=roberta-base
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 8 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --nr_concats $NR_CONCATS \
        --max_length $MAX_LENGTH \
    "

##### Longformer   

    export SEED=42
    export MAX_LENGTH=512
    export NR_CONCATS=1
    export DATASET=squad_long
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=allenai/longformer-base-4096
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 8 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --nr_concats $NR_CONCATS \
        --max_length $MAX_LENGTH \
    "


##### RoBERTa-Long   

    export SEED=42
    export MAX_LENGTH=512
    export NR_CONCATS=1
    export DATASET=squad_long
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=$MODEL_DIR/roberta-base-long
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 8 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --nr_concats $NR_CONCATS \
        --max_length $MAX_LENGTH \
    "

##### XLM-R   

    export SEED=42
    export MAX_LENGTH=512
    export NR_CONCATS=1
    export DATASET=squad_long
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=xlm-roberta-base
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 8 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --nr_concats $NR_CONCATS \
        --max_length $MAX_LENGTH \
    "

##### XLM-Long  
   
    export SEED=42
    export MAX_LENGTH=512
    export NR_CONCATS=1
    export DATASET=squad_long
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=$MODEL_DIR/xlm-roberta-base-long
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 8 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --nr_concats $NR_CONCATS \
        --max_length $MAX_LENGTH \
    "


</p>
</details>

<details><summary><b>SQ3 (4096)</b></summary>
<p>   

##### Longformer   

    export SEED=42
    export MAX_LENGTH=2048
    export NR_CONCATS=3
    export DATASET=squad_long
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=allenai/longformer-base-4096
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 32 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --nr_concats $NR_CONCATS \
        --max_length $MAX_LENGTH \
    "


##### RoBERTa-Long   

    export SEED=42
    export MAX_LENGTH=2048
    export NR_CONCATS=3
    export DATASET=squad_long
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=$MODEL_DIR/roberta-base-long
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 32 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --nr_concats $NR_CONCATS \
        --max_length $MAX_LENGTH \
    "


##### XLM-Long  

    export SEED=42
    export MAX_LENGTH=2048
    export NR_CONCATS=3
    export DATASET=squad_long
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=$MODEL_DIR/xlm-roberta-base-long
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 32 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --nr_concats $NR_CONCATS \
        --max_length $MAX_LENGTH \
    "


</p>
</details>

<details><summary><b>TODO TriviaQA (4096)</b></summary>
<p>
</p>
</details>
        
### Multilingual    
<details><summary><b>XQuAD</b></summary>
<p>   

##### RoBERTa   

    export SEED=42
    export DATASET=xquad
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=roberta-base
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 8 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --max_length 512 \
    "

##### XLM-R   

    export SEED=42
    export DATASET=xquad
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=xlm-roberta-base
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 8 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --max_length 512 \
    "

##### XLM-Long   

    export SEED=42
    export DATASET=xquad
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=$MODEL_DIR/xlm-roberta-base-long
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 8 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --max_length 512 \
    "


</p>
</details>

<details><summary><b>XQ3 (512)</b></summary>
<p>  

##### XLM-R   

    export SEED=42
    export MAX_LENGTH=512
    export NR_CONCATS=1
    export DATASET=xquad_long
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=xlm-roberta-base
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 8 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --nr_concats $NR_CONCATS \
        --max_length $MAX_LENGTH \
    "


##### XLM-Long   

    export SEED=42
    export MAX_LENGTH=512
    export NR_CONCATS=1
    export DATASET=xquad_long
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=$MODEL_DIR/xlm-roberta-base-long
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 8 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --nr_concats $NR_CONCATS \
        --max_length $MAX_LENGTH \
    "


</p>
</details>

<details><summary><b>XQ3 (4096)</b></summary>
<p>  

##### XLM-Long   

    export SEED=42
    export MAX_LENGTH=2048
    export NR_CONCATS=3
    export DATASET=xquad_long
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=$MODEL_DIR/xlm-roberta-base-long
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 32 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --nr_concats $NR_CONCATS \
        --max_length $MAX_LENGTH \
    "


</p>
</details>


<details><summary><b>MLQA</b></summary>
<p>   

##### XLM-R   

    export SEED=42
    export DATASET=mlqa
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=xlm-roberta-base
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 8 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --max_length 512 \
    "


##### XLM-Long   

    export SEED=42
    export DATASET=mlqa
    export MODEL_DIR=/workspace/models
    export MODEL_NAME_OR_PATH=$MODEL_DIR/xlm-roberta-base-long
    export MODEL_NAME=$MODEL_NAME_OR_PATH-seed-$SEED-on-$DATASET
    export LOG_DIR=/workspace/logs
    export DATA_DIR=/workspace/data
    # Debugging
    CUDA_LAUNCH_BLOCKING=1
    # model args
    make repl run="scripts/finetune_qa_models.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $MODEL_DIR/$MODEL_NAME \
        --logging_dir $LOG_DIR/$MODEL_NAME \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_steps 1000  \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 8 \
        --overwrite_output_dir \
        --evaluate_during_training \
        --fp16 \
        --do_train \
        --do_eval \
        --do_lowercase \
        --max_length 512 \
    "




</p>
</details>


</p>
</details>

## Acknowledgment   
Many thanks to the [Longformer Authors](https://github.com/allenai/longformer) for providing reproducible training scripts and Huggingface for open-sourcing their models and frameworks. I would like to thank my supervisor at Peltarion Philipp Eisen for his invaluable feedback, insight and availability. Thank you Professor Joakim Nivre for insightful and thorough feedback and for taking the time out of your busy schedule. A massive thank you to all the wonderful people at Peltarion for the opportunity to work on such an interesting project.   

## TODO Citation 

## Contact   

For questions regarding the code or the master thesis in general add an issue in the repo or contact:   

[markus.john.sagen@gmail.com](mailto:markus.john.sagen@gmail.com)

## TODO
- Include images
- Include link to report
- Include plots
