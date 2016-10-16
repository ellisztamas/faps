import numpy as np
from alogsumexp import alogsumexp

def paternity_array(offspring, mothers, males, allele_freqs, mu, missing_parents=None, mother_index=None):
    """
    Construct an array of log posterior probabilities that each offspring is sired
    by each of the candidate males in the sample, or that the true father is not
    present in the sample. The latter is weighted by the prior probability that a
    true father is missing, and each row sums to one.
    
    ARGUMENTS:
    offspring: A genotypeArray of observed genotype data for the offspring.

    mothers: an array of observed genotype data for the mothers. Data on mothers need
        to be in the same order as those for the offspring. If a whole parent array is
        used, this can be subsetted into the correct order using the option
        mother_index.
    
    males: A genotypeArray of observed genotype data for the candidate males.
    
    allele_freqs: vector of population allele frequencies for the parents.
    
    mu: point estimate of the genotyping error rate.
    
    threshold: maximum array size (n. offspring x n. parents) at which arrays can be
        considered as a whole without overloading the RAM. Above this threshold, the
        function starts looping over rows.
    
    missing_parents: value between zero and one indicating the proportion of
        individuals whose father has not been sampled. This is used to weight the
        probabilties of paternity for each father relative to the probability that a
        father was not sampled. The function will return and error message if this is
        set to 1. If this is given as None, no weighting is performed.
    
    threshold: the maximum value of the product of the numbers of loci, offspring and
        candidate males to be considered before switching to looping over loci.
    
    RETURNS:
    An array with a row for each offspring individual and column for each
    candidate male, with an extra final column for the probability that the offspring
    is drawn from population allele frequencies. Each element is a log
    probability, and as such each row sums to one.
    """
    # If a list of indices for the mothers has been given, subset the maternal data
    if mother_index is not None:
        mothers = mothers.subset(mother_index)
    
    # arrays of log likelihoods of paternity for sampled and unsampled fathers
    paternity_liks = lik_sampled_fathers(offspring, mothers, males, mu)
    missing_liks   = lik_unsampled_fathers(offspring, mothers, allele_freqs, mu)
    # concatenate an normalise
    lik_array     = np.append(paternity_liks, missing_liks[:,np.newaxis], 1)
    lik_array     = lik_array - alogsumexp(lik_array,1)[:, np.newaxis]
    
    if missing_parents is not None:
        # apply correction for the prior on number of missing parents.
        if missing_parents < 0 or missing_parents >1:
            print "missingDads must be between 0 and 1!"
            return None
        # If all fathers are missing, return nothing!    
        if missing_parents ==1:
            print "Error in ppr_paternities(): missing_dads set to 100%!"
            return None
        # if missing_parents is between zero and one, correct the likelihoods.
        if missing_parents >0 and missing_parents <1:
            lik_array[:, -1] = lik_array[:, -1] + np.log(  missing_parents)
            lik_array[:,:-1] = lik_array[:,:-1] + np.log(1-missing_parents)
        # if missing_parents is 0, set the term for unrelated fathers to zero.
        if missing_parents == 0:
            with(np.errstate(divide='ignore')): lik_array[:,-1] = np.log(0)

    # normalise so rows sum to one.
    lik_array = lik_array - alogsumexp(lik_array, axis=1)[:,np.newaxis]
    
    return lik_array