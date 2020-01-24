import torch


def _get_positive_mask(ids):
    ids = ids.unsqueeze(0)
    mask = ids == ids.t()
    idx = torch.arange(ids.shape[1])
    mask[idx, idx] = 0
    return mask.float()


def _get_negative_mask(ids):
    ids = ids.unsqueeze(0)
    mask = ids != ids.t()
    return mask.float()


def select_triplets(embeddings, ids):
    """
    Select embedding triplets according to the batch hard strategy
    (https://omoindrot.github.io/triplet-loss#batch-hard-strategy).
    """
    ids = ids.detach()
    # calculate distance matrix
    dot = torch.mm(embeddings, embeddings.t()).detach()
    square_norm = dot.diagonal()
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    dist = square_norm.expand_as(dot).t() - 2 * dot + square_norm.expand_as(dot)
    dist = dist.clamp(min=0)
    
    # get hardest positive for each embedding
    pos_mask = _get_positive_mask(ids)
    pos_dist = dist.clone()
    pos_dist *= pos_mask
    pos_indices = pos_dist.argmax(dim=1)
    
    # hardest negative
    neg_mask = _get_negative_mask(ids)
    neg_dist = dist.clone()
    neg_dist[neg_mask == 0.] += neg_dist.max()
    neg_indices = neg_dist.argmin(dim=1)
    
    positives = embeddings[pos_indices].contiguous()
    negatives = embeddings[neg_indices].contiguous()
    
    return embeddings, positives, negatives
