import random

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


def get_positive_mask(ids):
    ids = ids.unsqueeze(0)
    mask = ids == ids.t()
    idx = torch.arange(ids.shape[1])
    mask[idx, idx] = 0
    return mask.float()


def get_negative_mask(ids):
    ids = ids.unsqueeze(0)
    mask = ids != ids.t()
    return mask.float()


def select_triplets(embeddings, ids):
    ids = ids.detach()
    # calculate distance matrix
    dot = torch.mm(embeddings, embeddings.t()).detach()
    square_norm = dot.diagonal()
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    dist = square_norm.expand_as(dot).t() - 2 * dot + square_norm.expand_as(dot)
    dist = dist.clamp(min=0)
    
    # get hardest positive for each embedding
    pos_mask = get_positive_mask(ids)
    pos_dist = dist.clone()
    pos_dist *= pos_mask
    pos_indices = pos_dist.argmax(dim=1)
    
    # hardest negative
    neg_mask = get_negative_mask(ids)
    neg_dist = dist.clone()
    neg_dist[neg_mask == 0.] += neg_dist.max()
    neg_indices = neg_dist.argmin(dim=1)
    
    positives = embeddings[pos_indices].contiguous()
    negatives = embeddings[neg_indices].contiguous()
    
    return embeddings, positives, negatives


class BatchHardSampler(Sampler):
    """
    Samples by the batch hard strategy (See https://omoindrot.github.io/triplet-loss#batch-hard-strategy)
    The batch size is defined as num_ids * num_samples.
    
    :param dataset: the dataset to sample over
    :param num_ids: number of disctinct identities to be included in one batch
    :param num_samples: number of samples for each identity
    :param validate: use as validation set
    :param validation_size
    """
    
    def __init__(self, dataset, num_ids, num_samples, validate=False, validation_size=0.2):
        super(BatchHardSampler, self).__init__(dataset)
        
        self.dataset = dataset
        self.num_ids = num_ids
        self.num_samples = num_samples
        self.ids = np.array(dataset.ids)
        
        split = int(validation_size * len(dataset.ids_unique))
        if validate:
            self.ids_unique = dataset.ids_unique[:split]
        else:
            self.ids_unique = dataset.ids_unique[split:]
            
            
        self.id_sample_map = {_id : np.where(self.ids == _id)[0] for _id in self.ids_unique}
        
        self.num_iter = len(self.ids_unique) // num_ids
        
    def __iter__(self):
        # shuffle
        random.shuffle(self.ids_unique)
        for _id in self.id_sample_map:
            random.shuffle(self.id_sample_map[_id])
            
        for i in range(self.num_iter):
            pos = i * self.num_samples
            ids_batch = self.ids_unique[pos:pos + self.num_ids]
            
            indices = []
            for _id in ids_batch:
                replace = len(self.id_sample_map[_id]) < self.num_samples
                indices += list(np.random.choice(self.id_sample_map[_id], self.num_samples, replace))
                
            yield indices
            
    def __len__(self):
        return self.num_iter