import numpy as np

def squash_siblings(paternity_probs, partitions):
    """
    Take an array of paternity likelihoods across fathers for each offspring, and multiply rows over putative siblings.
    
    ARGUMENTS
    paternity_probs: array of posterior probabilities of paternity.
    
    partitions: vector assigning a full sibship label to each individual.
    
    RETURNS
    An array with a row for every full sibship and a column for every candidate father.
    """
    partitions  = partitions.astype('int')
    labels      = np.unique(partitions)
    if len(labels) == 1:
        return paternity_probs.sum(0)
    counts      = np.bincount(partitions)[labels]
    fullsibliks = np.array([paternity_probs[partitions == labels[f]].sum(0) for f in range(len(labels))])
    
    return fullsibliks
