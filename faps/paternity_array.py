import numpy as np
from paternityArray import paternityArray
from lik_sampled_fathers import lik_sampled_fathers
from lik_unsampled_fathers import lik_unsampled_fathers

def paternity_array(offspring, mothers, males, allele_freqs, mu, purge=None, missing_parents=None, selfing_rate=None):
    """
    Construct a paternityArray object for the offspring given known mothers and a set of
    candidate fathers using genotype data. Currently only SNP data is supported.

    Parameters
    ---------
    offspring: genotypeArray
        Observed genotype data for the offspring.
    mothers: genotypeArray
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
    A paternityArray.
    """
    # arrays of log likelihoods of paternity for sampled and unsampled fathers
    paternity_liks = lik_sampled_fathers(offspring, mothers, males, mu)
    missing_liks   = lik_unsampled_fathers(offspring, mothers, allele_freqs, mu)

    return paternityArray(paternity_liks, missing_liks, offspring.names, offspring.mothers, offspring.fathers, males.names, mu, purge, missing_parents, selfing_rate)
