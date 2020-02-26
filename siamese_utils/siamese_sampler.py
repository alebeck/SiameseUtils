import numpy as np
from torch.utils.data.sampler import Sampler


class SiameseSampler(Sampler):
    """
    Samples batches containing num_ids identities with num_samples samples each.
    Thus, the batch size is defined as num_ids * num_samples.
    
    :param dataset: the dataset to sample over
    :param num_ids: number of distinct identities to be included in every batch
    :param num_samples: number of samples for each identity in every batch
    :param validate: use as validation set
    :param validation_size: fraction of unique ids to use for validation
    :param
    """
    
    def __init__(self, dataset, num_ids, num_samples, validate=False, validation_size=0.2, split_seed=42):
        super(SiameseSampler, self).__init__(dataset)
        
        self.dataset = dataset
        self.num_ids = num_ids
        self.num_samples = num_samples
        self.ids = np.array(dataset.ids)

        ids_unique = np.array(list(set(self.ids)))
        ids_unique = np.sort(ids_unique)  # enforce determinism at this point

        np.random.seed(split_seed)
        np.random.shuffle(ids_unique)

        split = int(validation_size * len(ids_unique))
        if validate:
            self.ids_unique = ids_unique[:split]
        else:
            self.ids_unique = ids_unique[split:]
            
        self.id_sample_map = {_id: np.where(self.ids == _id)[0] for _id in self.ids_unique}

        self.num_iter = len(self.ids_unique) // num_ids
        
    def __iter__(self):
        # every epoch, randomize order in which ids and samples are presented
        np.random.shuffle(self.ids_unique)
        for _id in self.id_sample_map:
            np.random.shuffle(self.id_sample_map[_id])
            
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
