import numpy as np
from warnings import warn
from faps.alogsumexp import alogsumexp
from faps.unique_rows import unique_rows
from faps.squash_siblings import squash_siblings
from faps.paternityArray import paternityArray


def draw_fathers(partition, paternity_array=None, null_probs = None, ndraws=1000, use_covariates=False):
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
    paternity_array: paternityArray
        Object listing information on paternity of individuals.
    null_probs: array, optional
        1-d vector of log probabilities that each candidate is the sire of each
        full sibship in the partition. Probabilities are assumed to be the same
        for each partition. If values do not sum to one, they will be normalised
        to do so.
    ndraws: int
        Number of Monte Carlo draws for each family.
    use_covariates: logical, optional
        If True, information on prbabilities associated with covariates stored
        in paternityArray objects are incorporated into weights for drawing likely
        fathers.

    Returns
    -------
    A list of candidates compatible with the genetic data, and a second list of
    candidates drawn under random mating if specified.
    """
    # number of sibships and compatible fathers
    nfamilies = len(np.unique(partition))

    if isinstance(paternity_array, paternityArray):
        if use_covariates is True:
            if paternity_array.covariate is 0:
                covar = 0
            elif isinstance(paternity_array.covariate, np.ndarray):
                if len(paternity_array.covariate.shape) > 1:
                    raise ValueError("covariate should be a 1-d array, but has shape {}".format(covariate.shape))
                if paternity_array.prob_array.shape[1] != len(paternity_array.covariate):
                    raise ValueError("Length of vector of covariates ({}) does not match the number of fathers ({})".format(len(paternity_array.candidates), paternity_array.covariate.shape[0]))
                if not all(paternity_array.covariate <= 0):
                    warn("Not all values in covariate are less or equal to zero. Is it possible probabilities have not been log transformed?")
                covar = paternity_array.covariate[np.newaxis]
            else:
                raise TypeError("If use_covariates is set to True, the covraiate attribute of the paternityArray object should be a 1-d array.")
        else:
            covar = 0

        nfathers  = len(paternity_array.candidates)+1
        # multiply likelihoods for individuals within each full sibship, then normalise rows to sum to 1.
        prob_array = squash_siblings(paternity_array.prob_array, partition)
        prob_array = prob_array + covar
        prob_array = np.exp(prob_array - alogsumexp(prob_array,1)[:, np.newaxis])

    elif isinstance(null_probs, np.ndarray):
        if len(null_probs.shape) > 1:
            raise ValueError("null_probs is supplied should be a one dimensional vector for a single half-sibs array, but has shape {}.format(null_probs.shape}.")
        if not all(null_probs <= 0):
            warn("Not all values in null_probs are less or equal to zero. Is it possible probabilities have not been log transformed?")
        nfathers   = null_probs.shape[0]
        prob_array = null_probs - alogsumexp(null_probs)
        prob_array = np.tile(prob_array, nfamilies).reshape([nfamilies, len(null_probs)])
        prob_array = np.exp(prob_array)
    else:
        raise TypeError("Supply an array for either paternity_probs or null_probs.")
    if paternity_array is not None and null_probs is not None:
        warn('Values supplied for both paternity_probs and null_probs. null_probs will be ignored.')

    # generate a sample of possible paths through the matrix of candidate fathers.
    path_samples = np.array([np.random.choice(range(nfathers), ndraws, replace=True, p = prob_array[i]) for i in range(nfamilies)])
    path_samples = path_samples.T
    # identify samples with two or more famililies with shared paternity
    counts = [np.unique(path_samples[i], return_counts=True)[1] for i in range(len(path_samples))]
    valid  = [all((counts[i] == 1) & (counts[i] != nfathers))   for i in range(len(counts))]
    path_samples = np.array(path_samples)[np.array(valid)]
    output = [val for sublist in path_samples for val in sublist]

    return np.array(output)
