import numpy as np
from faps.paternityArray import paternityArray
from faps.alogsumexp import alogsumexp

def pairwise_lik_fullsibs(paternity_array, use_covariates = False, exp = False):
    """
    Create a matrix of probabilities that each pair of individuals in a
    half-sibling array are full siblings.

    If two individuals are full siblings, the likelihood is the product of the
    likelihoods that each is fathered by father i, or father j, etc. We therefore
    take the product of likelihoods for paternity for each candidtae, and sum
    across the products for each father to account for uncertainty.

    The majority of fathers will not contribute to paternity, but will cause
    noise in the estimation of pairwise sibship probability. For this reason the
    function works much better if the likelihood array has had most unlikely
    fathers, for example by setting the likelihood for fathers with three or
    more opposing homozygous loci to zero.

    Parameters
    ----------
    paternity_array: paternityArray
        Object listing information on paternity of individuals.
    exp: bool, optional
        If True, probabilities of paternity are exponentiated before calculating
        pairwise probabilities of sibships. This gives a speed boost if this is
        to be repeated many times, but there may be a cost to accuracy (this is
        untested!).
    use_covariates: logical, optional
        If True, information on prbabilities associated with covariates stored
        in paternityArray objects are incorporated into sibship clustering.

    Returns
    -------
    A noffspring*noffspring matrix of log likelihoods of
    being full siblings.

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
    patlik = paternity_array(offspring, mothers, males, allele_freqs, mu)

    # Matrix of pairwise probabilities of being full siblings.
    fullpairs = pairwise_lik_fullsibs(patlik)
    """
    paternity_probs = paternity_array.prob_array
    # pull out the number of offspring and parents
    noffs     = paternity_probs.shape[0]
    nparents  = paternity_probs.shape[1]

    if exp is False:
        # Set up use of covariates
        if use_covariates is True:
            covar = paternity_array.covariate[np.newaxis, :, np.newaxis]
        else: # If they are not to be used, use a log prob of 0.
            covar = 0

        # this section of code calculates the matrix in log space, but I found it quicker to exponentiate (below).
        paternity_probs = paternity_probs[:,:, np.newaxis]
        # take all pairwise products of sharing fathers.
        pairwise_lik = np.array([alogsumexp(i + paternity_probs.T + covar, 1) for i in paternity_probs]).squeeze()

        return pairwise_lik
    if exp is True:
        # Set up use of covariates
        if use_covariates is True:
            covar = paternity_array.covariate[np.newaxis]
        else: # If they are not to be used, use a log prob of 0.
            covar = 0

        # the sum can be quicker if you exponentiate, but this may harm precision.
        paternity_probs = np.exp(paternity_probs)
        covar = np.exp(covar)
        # for each pair of offspring, the likelihood of not sharing each father.
        pairwise_lik = np.matmul(paternity_probs*covar, paternity_probs.T)
        pairwise_lik = np.array(pairwise_lik).reshape([noffs, noffs]) # reshape

        return np.log(pairwise_lik)
