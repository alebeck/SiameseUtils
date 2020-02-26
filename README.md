# SiameseUtils

This repository provides a sampler to construct uniform batches from a dataset consisting of multiple labels with
multiple samples for each label. In addition, it provides a function `select_triplets` which - given a batch of
embeddings - selects triplets according to the *batch hard* strategy (https://omoindrot.github.io/triplet-loss#batch-hard-strategy).

## Installation
Make sure you have numpy and torch installed, then install via:

```bash
pip install git+https://github.com/alebeck/SiameseUtils
```

## Example usage
```python
from torch.utils.data import DataLoader
from siamese_utils import SiameseSampler, select_triplets

train_sampler = SiameseSampler(dataset, num_labels, num_samples, validation_size=val_size)
train_loader = DataLoader(dataset, batch_sampler=train_sampler)

for x, labels in train_loader:
  embeddings = model(x)
  loss = loss_fn(*select_triplets(embeddings, labels))
  ...

```

## Dataset requirements
For the SiameseSampler to work, the dataset should have a property `labels` which is an array of length `len(dataset)` and
which specifies the integer label of each element. The SiameseSampler automatically creates a train/val split based
on the arguments provided to `validate` and `validation_size`. In addition, for the triplet selection strategy to work 
properly, `dataset[i]` should return a 2-tuple consisting of the i-th element as well as the label of the i-th element.
