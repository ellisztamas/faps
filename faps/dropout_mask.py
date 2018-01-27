from genotypeArray import genotypeArray
import numpy as np

def dropout_mask(offspring, mothers, males):
    """
    Define an array indexing marker comparisons between offspring, known mothers
    and candidate fathers which are not possible due to missing marker data for
    one or more of the three individuals.

    Parameters
    ----------
    offspring: genotypeArray, or list of genotypeArrays
        Observed genotype data for the offspring.
    mothers: genotypeArray, or list of genotypeArrays
        Observed genotype data for the offspring. Data on mothers need
        to be in the same order as those for the offspring.
    males: genotypeArray
        Observed genotype data for the candidate males.

    Returns
    -------
    Arrays matching those returned by transition_probability:
    0. Array indexing offspring x candidates x number of loci for per-locus
    transition probabilities
    1. Array indexing offspring x number of loci for transition probabilities
    from population allele frequencies
    """
    # arrays of diploid genotypes
    offspring_diploid = offspring.geno.sum(2)
    maternal_diploid = mothers.geno.sum(2)
    male_diploid = males.geno.sum(2)
    # arrays giving positions of dropouts
    dropout_mask1 = (male_diploid == -18)[np.newaxis] + (offspring_diploid == -18)[:, np.newaxis] + (maternal_diploid == -18)[:,np.newaxis]
    # same for array for missing fathers.
    dropout_mask2 = (offspring_diploid == -18) + (maternal_diploid == -18)

    return dropout_mask1, dropout_mask2
