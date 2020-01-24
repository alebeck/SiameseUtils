# SiameseUtils

This repository provides a sampler to construct uniform batches from a dataset consisting of multiple identities with multiple samples for each identity. In addition, it provides a function `select_triplets` which - given a batch of embeddings - selects triplets according to the *batch hard* strategy (https://omoindrot.github.io/triplet-loss#batch-hard-strategy).

## Installation
Make sure you have numpy and torch installed, then install via:

```bash
pip install git+https://github.com/alebeck/SiameseUtils
```

## Example usage
```python
from torch.utils.data import DataLoader
from siamese_utils import SiameseSampler, select_triplets

train_sampler = SiameseSampler(dataset, num_ids, num_samples, validation_size=val_size)
train_loader = DataLoader(dataset, batch_sampler=train_sampler)

for x, y in train_loader:
  embeddings = model(x)
  loss = loss_fn(*select_triplets(embeddings, y))
  ...

```
