import numpy as np
from faps.genotypeArray import genotypeArray

def incompatibilities(parents, progeny):
    """
    Calculate the number of loci with Mendelian incompatibilities for arrays
    of offspring and candidate paternal genotypes.

    For each candidate-offspring dyad, this counts the number of loci at
    which the two individuals have opposing homozygous genotypes.

    Parameters
    ----------
    parents: genotypeArray
        Array of observed genotype data on the candidate fathers.
    progeny: genotypeArray
        Array of observed genotype data on the offspring.

    Returns
    -------
    An nxm array of integers, where n is the number of progeny and m is
    the number of candidate fathers.
    """
    # diploid genotypes of the parental and offspring diploid genotypes
    pdip = parents.geno.sum(2)
    odip = progeny.geno.sum(2)[:, np.newaxis]
    # Get number of incompatibilities for each offspring, for both pairs of possible incompatible genotypes
    arr1 = np.array([((pdip == 2) * (odip[i] == 0)).sum(1) for i in range(progeny.size)])
    arr2 = np.array([((pdip == 0) * (odip[i] == 2)).sum(1) for i in range(progeny.size)])
    # sum of incompatibilities for each possible combintation of incompatible genotypes
    return arr1 + arr2
