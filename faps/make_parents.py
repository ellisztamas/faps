import numpy as np
from faps.genotypeArray import genotypeArray

def make_parents(size, allele_freqs, family_name = 'base'):
    """
    Draw a base population of reproductive individuals from population allele frequencies.

    Parameters
    ----------
    size: integer
        Number of individuals to create.

    allele_freqs: array-like
        Vector of allele frequencies.

    family_name: str, optional
        String denoting the name for this family.

    Returns
    -------
    A genotypeArray object.
    """
    if size < 2 or not isinstance(size, int):
        raise ValueError("Size must be an integer of 2 or more.")
    if np.any(allele_freqs > 1) or np.any(allele_freqs < 0):
        raise ValueError("One or more allele frequency values exceeds zero or one.")
    else:
        # draw alleles for each individual.
        genomes = np.array([np.reshape(np.random.binomial(1,p,2*size), [2,size]) for p in allele_freqs])
        genomes = np.rollaxis(genomes, 2, 0) # reorder the array.

        offspring_names   = np.array([family_name+'_'+str(a) for a in np.arange(size)])
        maternal_names    = np.repeat('NA', size)
        paternal_names    = np.repeat('NA', size)

        return genotypeArray(genomes, offspring_names, maternal_names, paternal_names, np.arange(len(allele_freqs)))
