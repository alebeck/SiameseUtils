import random

import numpy as np
from torch.utils.data.sampler import Sampler


class SiameseSampler(Sampler):
    """
    Samples batches containing num_ids identities with num_samples samples each.
    Thus, the batch size is defined as num_ids * num_samples.
    
    :param dataset: the dataset to sample over
    :param num_ids: number of disctinct identities to be included in one batch
    :param num_samples: number of samples for each identity
    :param validate: use as validation set
    :param validation_size
    """
    
    def __init__(self, dataset, num_ids, num_samples, validate=False, validation_size=0.2):
        super(SiameseSampler, self).__init__(dataset)
        
        self.dataset = dataset
        self.num_ids = num_ids
        self.num_samples = num_samples
        self.ids = np.array(dataset.ids)

        ids_unique = list(set(self.ids))
        split = int(validation_size * len(ids_unique))
        if validate:
            self.ids_unique = ids_unique[:split]
        else:
            self.ids_unique = ids_unique[split:]
            
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