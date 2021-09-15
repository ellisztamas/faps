import numpy as np
from warnings import warn
from faps.alogsumexp import alogsumexp
from faps.unique_rows import unique_rows
from faps.squash_siblings import squash_siblings
from faps.paternityArray import paternityArray


def draw_fathers(partition, genetic=None, covariate = None, ndraws=10000, use_covariates=False, covariates_only=False):
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
    genetic: numpy array
        2-D matrix listing information on paternity of individuals.
    covariate: array, optional
        1-d vector of log probabilities that each candidate is the sire of each
        full sibship in the partition. Probabilities are assumed to be the same
        for each partition. If values do not sum to one, they will be normalised
        to do so.
    ndraws: int
        Number of Monte Carlo draws for each family.
    use_covariates: boolean, optional
        If True, information on prbabilities associated with covariates stored
        in paternityArray objects are incorporated into weights for drawing likely
        fathers.
    covariates_only: boolean, optional
        If True, candidates are drawn based on covariate probabilities only 
        (i.e. ignoring genetic data)

    Returns
    -------
    A list of candidates compatible with the genetic data, and a second list of
    candidates drawn under random mating if specified.
    """
    # number of sibships and compatible fathers
    nfamilies = len(np.unique(partition))

    if use_covariates is True or covariates_only is True:
        if use_covariates is False:
            warn("You have set use_covariates to False, but covariates_only to True. Covariates will be used.")
        if covariate is 0:
            warn("You have requested to use covariate information, but none is given in the sibshipCluster object. Check that `use_covariates` is set to True in your call to `sibshi_clustering`.")
            covar = 0
        if isinstance(covariate, np.ndarray):
            covariate = np.array(covariate.squeeze())
            if len(covariate.squeeze().shape) > 1:
                raise ValueError("covariate should be a 1-d array, but has shape {}".format(covariate.shape))
            if genetic.shape[1] != len(covariate):
                raise ValueError("Length of vector of covariates ({}) does not match the number of fathers ({})".format(covariate.shape[0], genetic.shape[1]))
            if not all(covariate <= 0):
                warn("Not all values in covariate are less or equal to zero. Is it possible probabilities have not been log transformed?")
            covar = covariate[np.newaxis]
    else:
        covar = 0

    # Simulate from genetic data, including covariates if `use_covariates` is set to True
    if covariates_only is False:    
        nfathers  = genetic.shape[1]
        # multiply likelihoods for individuals within each full sibship, then normalise rows to sum to 1.
        prob_array = squash_siblings(genetic, partition)
        prob_array = prob_array + covar
        prob_array = np.exp(prob_array - alogsumexp(prob_array,1)[:, np.newaxis])
    # Simulate from covariates only
    elif covariates_only:
        if isinstance(covariate, int) and covariate == 0:
            raise ValueError('Requested drawing fathers from covariate probabilities, but covariates are set to 0.')
        covariate  = covariate[:-1]
        nfathers   = covariate.shape[0]
        prob_array = covariate - alogsumexp(covariate)
        prob_array = np.tile(prob_array, nfamilies).reshape([nfamilies, len(covariate)])
        prob_array = np.exp(prob_array)
    
    # generate a sample of possible paths through the matrix of candidate fathers.
    path_samples = np.array([np.random.choice(range(nfathers), ndraws, replace=True, p = prob_array[i]) for i in range(nfamilies)])
    path_samples = path_samples.T
    # identify samples with two or more famililies with shared paternity
    counts = [np.unique(i, return_counts=True)[1] for i in path_samples]
    valid  = [all((i == 1) & (i != nfathers))     for i in counts]
    if not any(valid):
        warn("Could not find a combination of fathers that were valid for one or more partitions, but that were found by sibship_clustering. Consider increasing `ndraws`.")
        return []
    else:
        path_samples = np.array(path_samples)[np.array(valid)]
        output = [val for sublist in path_samples for val in sublist]
        # output is currently of size ndraws * n families
        # subsample down to ndraws
        output = np.random.choice(output, size=ndraws, replace=True)

        return output
