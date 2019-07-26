import numpy as np
from faps import make_offspring

def make_sibships(parents, dam, sires, family_size, family_name='offs'):
    """
    Mate parents in a base population to create half- or full-sibling families.

    This relies on the indices for the desired parents within the genotype object
    'parents'. These can be found from genotype objects using the function
    parent_index().

    Parameters
    ----------
    parents: genotypeArray
        Genotype information on the parents to be mated.

    dam: int
        Position of the mother in the parental genotypeArray object, to whom
        multiple sires are mated.

    sires: list
        A list indexing the position of the sires to be mated to the dam(s)
        within the parental genotype object. To create a full sibship, supply
        only a single father.

    family_size: int, list
        The sizes of each full sibship. If famillies are to be of the
        same size give and integer. Alternatively, if different sizes are
        desired, supply a list of sizes for each parternal familes. This list
        must be the same length as the list of sires.

    family_name: str
        String denoting the name for this family.

    Returns
    -------
    A genotypeArray object.
    """
    # if there is only one sire, turn the integer into a list of length one.
    if isinstance(sires, int):
        sires = [sires]

    # Multiply each sire ID by the number of his offspring.
    if isinstance(family_size, int): # for equal family sizes.
        sire_list = np.sort(sires*family_size).tolist()

    # if family_size is given as a list
    if isinstance(family_size, list) or isinstance(family_size, np.ndarray):
        # return an error if the two lists are of unequal length.
        if len(family_size) != len(sires):
            raise ValueError("family_size must either be an integer, or a list of the same length as the list of sires.")
        else:
            sire_list = [[sires[x]] * family_size[x] for x in range(len(sires))]
            sire_list = [item for sublist in sire_list for item in sublist]

    # Replicate the dam ID to match the length of sire_list
    dam_list = [dam] * len(sire_list)

    genotypes = make_offspring(parents, dam_list=dam_list, sire_list=sire_list, family_name=family_name)

    return genotypes
