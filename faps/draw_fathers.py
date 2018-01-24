import numpy as np
from warnings import warn
from alogsumexp import alogsumexp
from unique_rows import unique_rows
from squash_siblings import squash_siblings

def draw_fathers(partition, paternity_probs=None, null_probs = None, ndraws=1000):
    """
    Draws a sample of compatible fathers for each family in a single partition.
    Candidates are drawn proportional to their posterior probability of paternity.

    Optionally, a sample of candidates can be drawn at random, or proportional
    to some other distribution, such as a function of distance.

    Parameters
    ----------
    partition: list
        A 1-d array of integers labelling individuals into families. This should
        have as many elements as there are individuals in paternity_probs.
    paternity_probs: array
        Log probabilities of the paternity of individuals. This should usually be a
        prob_array from a paternityArray object. If values do not sum to one, they
        will be normalised to do so.
    null_probs: array, optional
        1-d vector of log probabilities that each candidate is the sire of each
        full sibship in the partition. Probabilities are assumed to be the same
        for each partition. If values do not sum to one, they will be normalised
        to do so.
    ndraws: int
        Number of Monte Carlo draws for each family.

    Returns
    -------
    A list of candidates compatible with the genetic data, and a second list of
    candidates drawn under random mating if specified.
    """
    # number of sibships and compatible fathers
    nfamilies = len(np.unique(partition))

    if isinstance(paternity_probs, np.ndarray):
        nfathers  = paternity_probs.shape[1]
        # multiply likelihoods for individuals within each full sibship, then normalise rows to sum to 1.
        prob_array = squash_siblings(paternity_probs, partition)
        if nfamilies == 1:
            prob_array = np.exp(prob_array - alogsumexp(prob_array))
            prob_array = prob_array[np.newaxis] # add extra dimension so looping is still possible.
        if nfamilies >  1:
            prob_array = np.exp(prob_array - alogsumexp(prob_array,1)[:, np.newaxis])

    elif isinstance(null_probs, np.ndarray):
        if len(null_probs.shape) > 1:
            raise ValueError("null_probs is supplied should be a one dimensional vector.")  
        nfathers   = null_probs.shape[0]
        prob_array = null_probs - alogsumexp(null_probs)
        prob_array = np.tile(prob_array, nfamilies).reshape([nfamilies, nfathers])
        prob_array = np.exp(prob_array)

    else:
        raise TypeError("Supply an array for either paternity_probs or null_probs.")
    if isinstance(paternity_probs, np.ndarray) and isinstance(null_probs, np.ndarray):
        warn('Values supplied for both paternity_probs and null_probs. null_probs will be ignored.')
        
    output = []
    counter = 0
    #while len(output) < 1:
    # generate a sample of possible paths through the matrix of candidate fathers.
    path_samples = np.array([np.random.choice(nfathers, ndraws, replace=True, p = prob_array[i]) for i in range(nfamilies)])
    path_samples = path_samples.T
    # identify samples with two or more famililies with shared paternity
    counts = [np.unique(path_samples[i], return_counts=True)[1] for i in range(len(path_samples))]
    valid  = [all((counts[i] == 1) & (counts[i] != nfathers))   for i in range(len(counts))]
    path_samples = np.array(path_samples)[np.array(valid)]
    output = [val for sublist in path_samples for val in sublist]

    return np.array(output)