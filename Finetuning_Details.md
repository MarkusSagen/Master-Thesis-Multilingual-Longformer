# Fine-Tuning Details   


We fine-tune and evaluate on these datasets using several pre-trained models released by Huggingface and compare it with the long-context models (Longformer type models) we have trained.   

We have divided the models firstly based the number of languages, then on the specific dataset and finally which model was fine-tuned. The datasets SQ3 and XQ3 are the long context variants (with concatenated context) of the SQuAD and XQuAD datasets. And to better understand and evaluate how the performance was effected when creating a new dataset, we chose fine-tune on the SQ3 and XQ3 dataset using either the regular attention window (512 tokens) or the attention window learned by the Longformer trained models (4096 tokens). These datasets were denoted SQ3 (512) and SQ3 (2048) respectively for the English dataset and XQ3 (512) and XQ3 (2048) for the multilingual datasets.   

The long context models are trained on a longer context than 2048, but we restricted the long context datasets to this many tokens at time, since the models did not manage to fit in memory on a 48GB GPU otherwise.   

#### Context lengths   
Depending on the number of contexts one choses to concatinate together, the maximum number of tokens the model can attend to also changes. The maximum number of contexts and tokens we managed to run on a 48GB GPU was 3 concatinated context, and corresponded to that the average number of tokens for each context were slightly below 2048. Therefore, for the concatinated long datasets, we set the hyper-parameters --nr\_concats=3 and --max\_length=2048. If you want to test out other values, we suggest the following pairings:   

concats=1, max\_length=512   
concats=3, max\_length=2048   
concats=5, max\_length=4098   


#### Seeds
Each model is trained with 5 different SEEDS. To replicate our experiments, re-run each code segment and replace the SEED with the following seeds:

- 42
- 1337
- 1729
- 165
- 758241

