# ET-DeepSDD: An Efficient Neural Probabilistic Logical Solution to Entity Typing

A CUDA-enabled PyTorch and pysdd implemention for ET-DeepSDD: "An Efficient Neural Probabilistic Logical Solution to Entity Typing"

pysdd package is downloaded from https://github.com/wannesm/PySDD

A google colab script to play with our model is attached as well.


## Setting up
First place the BERT model folder into bert/. To run the mention-level model, start Bert as a Service via the command:

```
bert-serving-start -model_dir cased_L-12_H-768_A-12 -num_worker=1 -max_seq_len=10
```

Then, open config.json and modify it to suit your experiments. The main values to change are:  

```
"dataset": either "bbn_original", "figer", or "ontonotes_original"
"model": <the name of your model (for your reference)>
"use_hierarchy": either "false" or "true"
"use_bilstm": either "false" or "true"
```


## Running the model


First, build the data loaders using:

```
python3 build_data.py

```
The code will transform the relevant dataset in the /data/datasets directory into a numerical format. Then, to train the model:

```
python3 train.py
```

The model will be evaluated during training on the dev set, and then evaluated on the test set once training is complete.
# ET-DeepSDD