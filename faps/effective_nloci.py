import numpy as np
from faps.genotypeArray import genotypeArray

def effective_nloci(progeny, mothers, adults):
    """
    Calculate the effective number of loci for all possible triplets in a comparison
    of offspring, known maternal, and candidate father genotypeArrays.

    Effective number of loci are the number of loci which for which all three
    individuals in the triplet have no missing genotype data.

    Parameters
    ----------
    progeny, mothers, adults: genotypeArray
        Genotype data on the progeny, their known mothers and the sample of candidate
        fathers

    Returns
    -------
    An array listing the number of valid loci for each possible triplet, with a row
    for each offspring and a column for each candidate father.
    """
    # confirm data are genotypeArrays
    if isinstance(progeny, genotypeArray) and isinstance(mothers, genotypeArray) and isinstance(adults, genotypeArray):
        if progeny.nloci != mothers.nloci and progeny.nloci != adults.nloci:
            raise ValueError("Numbers of loci in each genotypeArray do not match.")

        else:
            # Arrays indexing the positions of missing data in each genotypeArray
            prog_na = (progeny.geno[:,:,0] == -9)[np.newaxis]
            moth_na = (mothers.geno[:,:,0] == -9)[np.newaxis]
            adult_na= (adults.geno[ :,:,0] == -9)[:,np.newaxis]
            # sum positions
            mat = prog_na + moth_na + adult_na
            # subtract from total for each triplet
            missing = progeny.nloci - (mat > 0).sum(2).T
            return missing
    else:
        raise TypeError("progeny, mothers and adults should be of type genotypeArray.")
        return None
