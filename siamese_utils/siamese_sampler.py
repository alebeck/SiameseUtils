import numpy as np
from torch.utils.data.sampler import Sampler


class SiameseSampler(Sampler):
    """
    Samples batches containing <num_labels> labels with <num_samples> samples per label.
    Thus, the batch size is defined as num_labels * num_samples. Note: An epoch is completed when all
    distinct labels - not data points - have been presented.
    
    :param labels: integer labels list for dataset elements, of length len(dataset)
    :param num_labels: number of distinct labels to be included in every batch
    :param num_samples: number of samples for each label in every batch
    :param validate: use as validation set
    :param validation_size: fraction of unique labels to use for validation
    :param split_seed: random seed wich controls train/val split
    """
    
    def __init__(self, labels, num_labels, num_samples, validate=False, validation_size=0.2, split_seed=42):
        self.num_labels = num_labels
        self.num_samples = num_samples
        self.labels = np.array(labels)

        labels_unique = np.array(list(set(self.labels)))
        labels_unique = np.sort(labels_unique)  # enforce determinism at this point

        np.random.seed(split_seed)
        np.random.shuffle(labels_unique)

        split = int(validation_size * len(labels_unique))
        if validate:
            self.labels_unique = labels_unique[:split]
        else:
            self.labels_unique = labels_unique[split:]
            
        self.label_sample_map = {l: np.where(self.labels == l)[0] for l in self.labels_unique}

        self.num_iter = len(self.labels_unique) // num_labels
        
    def __iter__(self):
        # every epoch, randomize order in which labels and samples are presented
        np.random.shuffle(self.labels_unique)
        for label in self.label_sample_map:
            np.random.shuffle(self.label_sample_map[label])
            
        for i in range(self.num_iter):
            pos = i * self.num_samples
            labels_batch = self.labels_unique[pos:pos + self.num_labels]
            
            indices = []
            for label in labels_batch:
                replace = len(self.label_sample_map[label]) < self.num_samples
                indices += list(np.random.choice(self.label_sample_map[label], self.num_samples, replace))
                
            yield indices
            
    def __len__(self):
        return self.num_iter
