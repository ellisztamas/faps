import numpy as np
from faps.alogsumexp import alogsumexp
from faps.unique_rows import unique_rows
from faps.paternityArray import paternityArray
from faps.sibshipCluster import sibshipCluster
from faps.sibship_partitions import sibship_partitions
from warnings import warn

def do_clustering(paternity_array, ndraws=1000, exp=False, use_covariates=False):
    """
    Cluster offspring into full sibship groups using hierarchical clustering.

    This first builds a dendrogram of relatedness between individuals, and pulls
    out every possible partition structure compatible with the dendrogram. The
    likelihood for each partition is also estimated by Monte Carlo simulations.

    If the paternityArray object contains only a single individual, clustering
    is meaningless, but an artifical sibshipCluster object is created with a
    single partition with log likelihood of 0.

    Parameters
    ----------
    paternity_array: paternityArray
        Object listing information on paternity of individuals.
    ndraws: int
        Number of Monte Carlo simulations to run for each partition.
    exp: logical, optional
        Indicates whether the probabilities of paternity should be exponentiated
        before calculating pairwise probabilities of sibships. This gives a
        speed boost if this is to be repeated many times in simulations, but
        there may be a cost to accuracy. Defaults to False for this reason.
    use_covariates: logical, optional
        If True, information on prbabilities associated with covariates stored
        in paternityArray objects are incorporated into sibship clustering.

    Returns
    -------
    A sibshipCluster object.

    Example
    -------
    import numpy as np
    from faps import *

    # generate 4 families of 5 offspring
    allele_freqs = np.random.uniform(0.3, 0.5, 50)
    males = make_parents(200, allele_freqs)
    offspring = make_sibships(males, 0, range(1,5), 5)

    # Add muations
    mu = 0.0013
    males = males.dropouts(0.01).mutations(mu)
    offspring= offspring.dropouts(0.01).mutations(mu)

    mothers = males.subset(offspring.parent_index('m', males.names))
    patlik = paternity_array(offspring, mothers, males, mu)

    # compare the effect of exponentiating likelihoods
    sc1 = do_clustering(patlik, exp=True)
    sc2 = do_clustering(patlik, exp=False)

    sc1.accuracy(offspring, males)
    sc2.accuracy(offspring, males)
    """
    if not isinstance(paternity_array, paternityArray):
        TypeError("paternity_array should be a paternityArray object.")

    if use_covariates is True:
        if paternity_array.covariate is None:
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

    # # If there is only one offspring in the paternityArray, there is no need to do any clustering.
    # if paternity_array.offspring.shape[0] == 0:
    #     output = sibshipCluster(
    #         paternity_array = paternity_array,
    #         linkage_matrix  = 'Linkage matrix not available because there is only one offspring',
    #         partitions      = np.array([[0]]),
    #         lik_partitions  = np.array([[0]]),
    #         paths           = {0: np.array([[0]])},
    #         path_likelihoods = {0: np.array([[0]])}
    #     )
    #     return output

    # # Otherwise, generate a sample of plausible partitions from the dendrogram.
    # else:
    partition_sample, z = sibship_partitions(paternity_array, exp=exp, use_covariates=use_covariates)

    #get a list of all unique families.
    unique_families = [np.where(p == val)[0] for p in partition_sample for val in np.unique(p)]
    unique_families = set([tuple(up) for up in unique_families]) # coerce to tuple to create set
    unique_families = [np.array(up) for up in unique_families] # coerce back to numpy arrays

    # Calculate likelihoods of paternity for each family.
    siblik = np.array([paternity_array.prob_array[uf].sum(0) for uf in unique_families])
    siblik = siblik + covar
    sibprob= siblik - alogsumexp(siblik, 1)[:, np.newaxis]
    sibprob= np.exp(sibprob)

    # Draw ndraws fathers for each full sibship.
    candidates = np.arange(paternity_array.prob_array.shape[1])
    draw_dads = np.array([np.random.choice(candidates, ndraws, replace=True, p=s) for s in sibprob], dtype=np.int64)

    # Loop over each partition and pull out the rows in siblik for each full sibship within it.
    # Then, remove duplicate and invalid configurations.
    # Finally, extract probabilities for each candidate, multiply across families, and sum across valid paths.
    up_str = np.array([str(up) for up in unique_families])
    lik_partitions = np.zeros(len(partition_sample))
    paths = {}
    path_likelihoods = {}

    for i in range(len(partition_sample)):
        p = partition_sample[i]
        these_families = np.array([str(np.where(p == val)[0]) for val in np.unique(p)]) # families in this partition.
        # pull out the sublists in up_str for this partition
        ix = np.array([np.where(f == up_str)[0][0] for f in these_families])
        these_dads = draw_dads[ix]
        # remove invalid partitions
        these_dads = unique_rows(these_dads.T) # remove duplicates
        these_dads = np.array([v for v in these_dads if len(set(v)) == len(v)]) # remove partitions with duplicate candidates
        paths[i] = these_dads
        # If no valid partitions are found, return likelihood of zero.
        if len(these_dads) == 0:
            with np.errstate(divide='ignore'):
                lik_partitions[i] = np.array([np.log(0)])
                path_likelihoods[i] =  np.array([np.log(0)])
        # Otherwise, pull out the likelihoods
        else:
            this_path_lik       = siblik[ix, these_dads].sum(1)
            path_likelihoods[i] = this_path_lik
            lik_partitions[i]    = alogsumexp(this_path_lik)

    # Normalise likelihoods of individual paths over all partitions to sum to one.
    total_lik = alogsumexp(lik_partitions)[np.newaxis]
    path_probs = {k: v - total_lik for k,v in path_likelihoods.items()}

    output = sibshipCluster(
        paternity_array  = paternity_array,
        linkage_matrix   = z,
        partitions       = partition_sample,
        lik_partitions   = lik_partitions,
        paths            = paths,
        path_likelihoods = path_likelihoods,
        path_probs       = path_probs
    )
    return output
