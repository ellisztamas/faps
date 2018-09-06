import numpy as np

def squash_siblings(paternity_probs, partitions):
    """
    Take an array of paternity likelihoods across fathers for each offspring, and multiply rows over putative siblings.
    
    Parameters
    ----------
    paternity_probs: array
        Array of posterior probabilities of paternity. Usually, this is the
        output of `prob_array` from a `paternityArray` object.
    
    partitions: array-like
        vector assigning a full sibship label to each individual.
    
    Returns
    -------
    An array with a row for every full sibship and a column for every candidate father.
    """
    partitions  = partitions.astype('int')
    labels      = np.unique(partitions)
    if len(labels) == 1:
        return paternity_probs.sum(0)[np.newaxis]
    else:
        fullsibliks = np.array([paternity_probs[partitions == f].sum(0) for f in labels])
        return fullsibliks

