import numpy as np
from alogsumexp import alogsumexp
from unique_rows import unique_rows
from squash_siblings import squash_siblings

def draw_fathers(partition, paternity_probs, ndraws=1000, max_tries=100):
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
        Information on the paternity of individuals. This should usually be a
        prob_array from a paternityArray object.
    null_probs: array-lik, optional
        If a sample of fathers is to be drawn proportional to some distribution
        other than the probabilities of paternity, supply this as a 1-d vector.
        To draw candidates at random (i.e. each with equal probability) supply
        the string 'uniform'.
    ndraws: int
        Number of Monte Carlo draws for each family.

    Returns
    -------
    A list of candidates compatible with the genetic data, and a second list of
    candidates drawn under random mating if specified.
    """
    # check the partition is the correct length!
    if paternity_probs.shape[0] != len(partition):
        print "Error in lik_partition: paternity_probs and partition are of uneven length."
        return None
    # number of sibships and compatible fathers
    nfamilies = len(np.unique(partition))
    nfathers  = paternity_probs.shape[1]

    # multiply likelihoods for individuals within each full sibship, then normalise rows to sum to 1.
    prob_array = squash_siblings(paternity_probs, partition)
    if nfamilies == 1:
        prob_array = np.exp(prob_array - alogsumexp(prob_array))
        prob_array = prob_array[np.newaxis] # add extra dimension so looping is still possible.
    if nfamilies >  1:
        prob_array = np.exp(prob_array - alogsumexp(prob_array,1)[:, np.newaxis])

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
