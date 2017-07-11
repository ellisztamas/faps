import numpy as np
from faps import paternityArray

def paternity_list(offspring, mothers, males, allele_freqs, mu):
    """
    Calculate a matrix of paternity matrices for each of a set of half-sib arrays.

    This divides up offspring and maternal genotypeArrays into half-sib families
    based on information about the mother contained in the offspring genotypeArray.
    It then calculates a paternityArray for each, and returns these as a list.

    ARGUMENTS:
    offspring: A genotypeArray of observed genotype data for the offspring.

    mothers: an array of observed genotype data for the mothers. Data on mothers need
        to be in the same order as those for the offspring. If a whole parent array is
        used, this can be subsetted into the correct order using the option
        mother_index.

    males: A genotypeArray of observed genotype data for the candidate males.

    allele_freqs: vector of population allele frequencies for the parents.

    mu: point estimate of the genotyping error rate.

    RETURNS:
    A list of paternityArrays.
    """
    unique_families = np.unique(offspring.mothers)
    family_list     = []
    for i in range(len(unique_families)):
        this_family = offspring.subset(np.where(offspring.mothers == unique_families[i]))
        this_mother = mothers.subset(np.where(mothers.names == unique_families[[i]]))
        family_list+=  [paternityArray(this_family, this_mother, males, allele_freqs, mu)]
    return family_list
