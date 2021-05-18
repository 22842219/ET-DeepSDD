# ET-DeepSDD: An Efficient Neural Probabilistic Logical Solution to Entity Typing

A CUDA-enabled PyTorch and pysdd implemention for ET-DeepSDD: "An Efficient Neural Probabilistic Logical Solution to Entity Typing"

pysdd package is downloaded from https://github.com/wannesm/PySDD

# Setup 

This repository uses [pysdd](https://github.com/wannesm/PySDD) to convert logical formula, and use [hugging_face pre-trained bert model](https://huggingface.co/transformers/model_doc/bert.html) to convert tokens into embeded vectors.
Firstly, construct sdd tree based on customized logical formula.

```
python sdd.py --dataset bbn_modified --label_size 48
```
Secondly, Configure arguments in config.json.

Then, build the data loader.

```
python biuld_data.py
```

Then, train the model.

```
python train.py
```
