import numpy as np
from faps.pr_transition import pr_transition
from faps.genotypeArray import genotypeArray

def lik_sampled_fathers(offspring, mothers, males, mu, mother_index=None):
    """
    Calculate a matrix of multilocus transition probabilities for all possible triplets
    of offspring, their known mothers, and a set of candidate males. This sums over all
    possible offspring and parental genotypes.

    Parameters
    ----------
    offspring: genotypeArray
        Array of observed genotype data for the offspring.

    mothers: genotypeArray
        An array of observed genotype data for the mothers. Data on mothers need
        to be in the same order as those for the offspring. If a whole parent
        array is used, this can be subsetted into the correct order using the
        option mother_index.

    fathers: genotypeArray
        Array of observed genotype data for the candidate males.

    mu: float
        Point estimate of the genotyping error rate. Note that sibship clustering
        is unstable if mu_input is set to exactly zero. Any zero values will
        therefore be set to a very small number close to zero (10^-12).

    mother_index: list, optional
        List of integers indexing the position of the mother of
        each offspring in the maternal_genotypes array. If left blank, the whole array
        is used.

    Returns
    -------
    A matrix of log likelihoods of paternity, with a row for each offspring and a
    column for each candidate male.
    """
    if not isinstance(offspring, genotypeArray):
        raise TypeError('offspring data not given as a genotypeArray.')
    elif not isinstance(mothers, genotypeArray):
        raise TypeError('mothers not given as a genotypeArray.')
    elif not isinstance(males, genotypeArray):
        raise TypeError('male genotype data not given as a genotypeArray.')

    # If a list of indices for the mothers has been given, subset the maternal data
    if mother_index is not None:
        mothers = mothers.subset(mother_index)
    # if mu is given as zero, set this to a very small number for sibship clustering
    if mu == 0: mu = 1.0 / 10**12

    # Diploid genotypes of each dataset.
    male_diploid = males.geno.sum(2)
    moth_diploid = mothers.geno.sum(2)
    offs_diploid = offspring.geno.sum(2)

    # empty array to store probabilities.
    lik_trans = np.zeros([offspring.size, males.size, males.nloci])
    # loop over all possible gemotype combinations and get transitions probs for each.
    for o in [0,1,2]:
        for m in [0,1,2]:
            for f in [0,1,2]:
                lik_trans += pr_transition(offs_diploid, moth_diploid, male_diploid, o, m, f, mu)
    lik_trans = lik_trans # correct for the number of combinations.
    with np.errstate(divide='ignore'): lik_trans = np.log(lik_trans) # convert to log.

    # sum diploid genotypes to identify where the negative values (i.e. dropouts) are.
    dropout_mask = offs_diploid[:, np.newaxis] + moth_diploid[:, np.newaxis] + male_diploid[np.newaxis]
    # array of the number of loci for each trio for which one or more individual had a dropout.
    valid_loci   = males.nloci - (dropout_mask < 0).sum(2)

    # Correct for dropouts
    lik_trans[dropout_mask < 0] = 0 # set loci with one or more dropouts to zero to allow summing.
    lik_trans = lik_trans.sum(2) # sum over loci.
    lik_trans = lik_trans * (float(males.nloci) / valid_loci) # scale by the number of valid SNPs.

    return lik_trans
