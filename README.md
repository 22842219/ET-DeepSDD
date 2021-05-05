# ET-DeepSDD: An Efficient Neural Probabilistic Logical Solution to Entity Typing

A CUDA-enabled PyTorch and pysdd implemention for ET-DeepSDD: "An Efficient Neural Probabilistic Logical Solution to Entity Typing"

pysdd package is downloaded from https://github.com/wannesm/PySDD

Firstly, construct sdd tree based on customized logical formula.

```
python sdd.py --dataset bbn_modified --label_size 48
```

Secondly, build the data loader.

```
python biuld_data.py
```

Then, train the model.

``
python train.py
```
