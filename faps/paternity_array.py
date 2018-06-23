import numpy as np
from paternityArray import paternityArray
from genotypeArray import genotypeArray
from transition_probability import transition_probability
from incompatibilities import incompatibilities
from warnings import warn

def paternity_array(offspring, mothers, males, allele_freqs, mu, return_by_locus = True, purge=None, missing_parents=None, selfing_rate=None, max_clashes=None):
    """
    Construct a paternityArray object for the offspring given known mothers and a set of
    candidate fathers using genotype data. Currently only SNP data is supported.

    Parameters
    ---------
    offspring: genotypeArray, or list of genotypeArrays
        Observed genotype data for the offspring.
    mothers: genotypeArray, or list of genotypeArrays
        Observed genotype data for the offspring. Data on mothers need
        to be in the same order as those for the offspring.
    males: genotypeArray
        Observed genotype data for the candidate males.
    allele_freqs: array-like
        Vector of population allele frequencies for the parents.
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

    Returns
    -------
    A paternityArray, or a list of paternityArray objects.
    """
    if mu == 0:
        mu = 10**-12
        warn('Setting error rate to exactly zero causes clustering to be unstable. mu set to 10e-12')

    #If a single halfsib family is given.
    if isinstance(offspring, genotypeArray) & isinstance(mothers, genotypeArray):
        # array of opposing homozygous genotypes.
        incomp = incompatibilities(males, offspring)
        # take the log of transition probabilities, and assign dropout_masks.
        prob_f, prob_a, prob_m = transition_probability(offspring, mothers, males, allele_freqs, mu)
        return paternityArray(likelihood=prob_f, lik_absent=prob_a, by_locus=prob_m, offspring=offspring.names, mothers=offspring.mothers, fathers=offspring.fathers, candidates=males.names, mu=mu, purge=purge, missing_parents=missing_parents, selfing_rate=selfing_rate, clashes=incomp, max_clashes=max_clashes)
    
    # If a litst of genotype arrays for separate halfsib families are given:
    elif isinstance(offspring, list) & isinstance(mothers, list):
        if len(offspring) != len(mothers):
            raise ValueError('Lists of genotypeArrays are of different lengths.')
        output = []
        for i in range(len(offspring)):
            # array of opposing homozygous genotypes.
            incomp = incompatibilities(males, offspring[i])
            # take the log of transition probabilities, and assign dropout_masks.
            prob_f, prob_a, by_locus = transition_probability(offspring[i], mothers[i], males, allele_freqs, mu)
            # create paternityArray and send to output
            patlik = paternityArray(likelihood=prob_f, lik_absent=prob_a, by_locus=by_locus, offspring=offspring[i].names, mothers=offspring[i].mothers, fathers=offspring[i].fathers, candidates=males.names, mu=mu, purge=purge, missing_parents=missing_parents, selfing_rate=selfing_rate, clashes=incomp, max_clashes=max_clashes)
            output = output + [patlik]
        return output
    
    else:
        raise TypeError('offspring and mothers should be genotype arrays, or else lists thereof.')