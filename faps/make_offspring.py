import numpy as np
from faps.genotypeArray import genotypeArray

def make_offspring(parents, noffs=None, dam_list=None, sire_list=None, family_name='offs'):
    """
    Mate individuals in a base population to create simulated offspring. Lists of
    specific sires and dams can be provided with the options dam_list and
    sire_list. If only the number of offspring are specified parents are mated at
    random from the base population.

    Parameters
    ----------
    parents: genotypeArray
        Genotype information on the parents to be mated.

    noffs: int
        Number of offspring to be produced. If specific dams and sires are
        specified, this is ignored.

    dam_list, sire_list: lists
        Integer lists of positions of sires and dams to be mated.
        Pairs are mated in order (i.e. the first dam with the first sire, and so
        forth). If used these two lists must be of the same length. If no
        arguments are given for either list, parents are mated at random with
        replacement, and the possibility of self-fertilisation.

    family_name: str, optional
        String denoting the name for this family.

    Returns
    -------
    A genotypeArray object.
    """
    if dam_list is None and sire_list is None and noffs is None:
        raise ValueError("Either noffs needs to be a positive integer, or else lists of dams and sires should be given.")

    # If parents haven't been specified, choose these at random.
    if dam_list is None and sire_list is None:
        if noffs < 1 or not isinstance(noffs, int):
            raise ValueError("noffs should be a positive integer.")
        nparents = parents.geno.shape[0]
        dam_list  = np.random.choice(range(nparents), noffs, replace=True).tolist()
        sire_list = np.random.choice(range(nparents), noffs, replace=True).tolist()
    # if parents have been specified, set noffs to the length of sires and dams.
    if dam_list is not None or sire_list is not None:
        noffs = len(dam_list)
        if len(dam_list) != len(sire_list):
            raise ValueError("List of sires must be the same length as the list of dams.")

    nloci = parents.geno.shape[1] # pull out the number of loci
    offs_genotypes= np.zeros([noffs, nloci, 2]) # empty array to store offspring genotypes.

    # pull out arrays of genotype data for the dams and sires.
    dam_genotypes  = parents.subset(dam_list).geno
    sire_genotypes = parents.subset(sire_list).geno

    # draw an array of indices for whether the first or second allele should be drawn.
    dam_alleles  = np.random.binomial(1, 0.5, nloci*noffs).reshape([noffs, nloci])
    sire_alleles = np.random.binomial(1, 0.5, nloci*noffs).reshape([noffs, nloci])
    # loop over every mating pair and send the selected alleles to offs_genotypes.
    for o in range(noffs):
        offs_genotypes[o,:,0] = np.array([dam_genotypes [o,l][dam_alleles [o,l]] for l in range(nloci)])
        offs_genotypes[o,:,1] = np.array([sire_genotypes[o,l][sire_alleles[o,l]] for l in range(nloci)])
    offs_genotypes = offs_genotypes.astype('int')

    # extra information on names.
    offspring_names   = np.array([family_name+'_'+str(a) for a in np.arange(noffs)])
    maternal_names    = parents.subset(dam_list).names
    paternal_names    = parents.subset(sire_list).names

    return genotypeArray(
        geno = offs_genotypes,
        names = offspring_names,
        mothers = maternal_names,
        fathers = paternal_names,
        markers = np.arange(nloci)
    )
