import numpy as np
from paternityArray import paternityArray
from genotypeArray import genotypeArray
from transition_probability import transition_probability
from incompatibilities import incompatibilities
from warnings import warn

def paternity_array(offspring, mothers, males, mu, return_by_locus = True, purge=None, missing_parents=None, selfing_rate=None, max_clashes=None, covariate=None, allele_freqs=None):
    """
    Construct a paternityArray object for the offspring given known mothers and a set of candidate fathers using genotype data. Currently only SNP 
    data is supported.

    Additional information about paternity from non-genetic sources can be provided through the argument `covariate`. For example, this might be a function of spatial distance between individuals, or social dominance.

    Parameters
    ---------
    offspring: genotypeArray, or list of genotypeArrays
        Observed genotype data for the offspring.
    mothers: genotypeArray, or list of genotypeArrays
        Observed genotype data for the offspring. Data on mothers need to be in the same order as those for the offspring.
    males: genotypeArray
        Observed genotype data for the candidate males.
    mu: float between zero and one
        Point estimate of the genotyping error rate. Clustering is unstable if
        mu_input is set to exactly zero. Any zero values will therefore be set
        to a very small number close to zero (10^-12).
    return_by_locus: bool, optional
        If True, the intermediate 3-d array for log likelihoods for each
        offspring-candidate-locus combination are returned before the
        multiplication step over loci. Defaults to True. In general, this
        provides better inference when used in sibship clustering, but may use
        large amounts of memory.
    purge: float between zero or one, int, array-like, optional
        Individuals who can be removed from the paternity array a priori. If
        a float is given, that proportion of individuals is removed from the
        array at random. Alternatively an integer or vector of integers
        indexing specific individuals can be supplied.
    missing_parents : float between zero and one, or 'NA', optional
        Input value for the proportion of adults who are missing from the sample.
        This is used to weight the probabilties of paternity for each father
        relative to the probability that a father was not sampled. If this is
        given as 'NA', no weighting is performed.
    selfing_rate: float between zero and one, optional
        Input value for the prior probability of self-fertilisation.
    max_clashes: int, optional
        Maximum number of opposing homozygous loci for each parent-offspring.
    covariate: 1-d array, or list of 1-d arrays, optional
        Vector of (log) probabilities of paternity based on non-genetic 
        information, with one element for every candidate father. If this is a
        function of multiple sources they should be multiplied and included in
        this vector. If a list of offspring arrays have been supplied, this
        should be a list of vectors.
    allele_freqs: Deprecated
        Deprecated, but retained for backwards compatibility.

    Returns
    -------
    A paternityArray, or a list of paternityArray objects.

    If covariate is not given this will be returned as a vector of zeros.
    """
    if mu == 0:
        mu = 10**-12
        warn('Setting error rate to exactly zero causes clustering to be unstable. mu set to 10e-12')

    #If a single halfsib family is given.
    if isinstance(offspring, genotypeArray) & isinstance(mothers, genotypeArray):
        # array of opposing homozygous genotypes.
        incomp = incompatibilities(males, offspring)
        # take the log of transition probabilities, and assign dropout_masks.
        prob_f, prob_a, prob_m = transition_probability(offspring, mothers, males, mu)
        output = paternityArray(likelihood=prob_f, lik_absent=prob_a, by_locus=prob_m, offspring=offspring.names, mothers=offspring.mothers, fathers=offspring.fathers, candidates=males.names, mu=mu, purge=purge, missing_parents=missing_parents, selfing_rate=selfing_rate, clashes=incomp, max_clashes=max_clashes)

        if covariate is not None:
            output.add_covariate(covariate)

        return output
    



    # If a list of genotype arrays for separate halfsib families are given:
    elif isinstance(offspring, list) & isinstance(mothers, list):
        if len(offspring) != len(mothers):
            raise ValueError('Lists of genotypeArrays are of different lengths.')

        # Set up input of covariates if necessary. 
        if isinstance(covariate, list):
            if len(offspring) != len(covariate):
                raise ValueError("If a list of arrays of probabilities for covariates are supplied, this should have a row for every offspring genotypeArray.")
            cov = covariate
        elif covariate is None:
            cov = np.zeros(males.size * len(offspring)).reshape([len(offspring), males.size])
        else:
            raise TypeError("If covariates are supplied for multiple half-sib arrays, this should be a list of 1-d vectors, each with an element for each candidate.")


        output = [None] * len(offspring)
        for i in range(len(offspring)):
            # array of opposing homozygous genotypes.
            incomp = incompatibilities(males, offspring[i])
            # take the log of transition probabilities, and assign dropout_masks.
            prob_f, prob_a, by_locus = transition_probability(offspring[i], mothers[i], males, mu)
            # create paternityArray and send to output
            patlik = paternityArray(likelihood=prob_f,lik_absent=prob_a, by_locus=by_locus, offspring=offspring[i].names, mothers=offspring[i].mothers, fathers=offspring[i].fathers, candidates=males.names, mu=mu, purge=purge, missing_parents=missing_parents, selfing_rate=selfing_rate, clashes=incomp, max_clashes=max_clashes)

            patlik.add_covariate(cov[i])
            output[i] = patlik


        return output
    
    else:
        raise TypeError('offspring and mothers should be genotype arrays, or else lists thereof.')