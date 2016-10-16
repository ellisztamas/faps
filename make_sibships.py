import numpy as np

def make_sibships(parents, dam, sires, family_size, family_name='offs'):
    """
    Mate parents in a base population to create half- or full-sibling families.
    
    This relies on the indices for the desired parents within the genotype object
    'parents'. These can be found from genotype objects using the function
    parent_index().
    
    ARGUMENTS:
    parents: A genotype object of parents to be mated.
    
    dam: An integer giving the position of the mother in the parental
        genotype object, to which multiple sires are mated.
    
    sires: A list indexing the position of the sires to be mated to the dam(s) within
        the parental genotype object. To create a full sibship, supply only a single
        father.
        
    family_size: The sizes of each full sibship. If famillies are to be of the same
        size give and integer. Alternatively, if different sizes are desired, supply
        a list of sizes for each parternal familes. This list must be the same length
        as the list of sires.
    
    family_name: an optional string denoting the name for this family.
    
    RETURNS:
    A genotype object of four components:
    1. A three-dimensional array of genotype information of size, with axes denoting
        number of individuals, number of loci, and the two alleles.
    2. Names of each individual.
    3. Names of the mother of each individual.
    4. Names of the mother of each individual.
    """
    # if there is only one sire, turn the integer into a list of length one.
    if isinstance(sires, int):
        sires = [sires]

    # Multiply each sire ID by the number of his offspring.
    if isinstance(family_size, int): # for equal family sizes.
        sire_list = np.sort(sires*family_size).tolist()
    
    # if family_size is given as a list
    if isinstance(family_size, list):
        # return an error if the two lists are of unequal length.
        if len(family_size) is not len(sires):
            print "family_size must either be an integer, or a list of the same length"
            print "as the list of sires."
            return None
        else:
            sire_list = [[sires[x]] * family_size[x] for x in range(len(sires))]
            sire_list = [item for sublist in sire_list for item in sublist]
    
    # Replicate the dam ID to match the length of sire_list
    dam_list = [dam] * len(sire_list)

    genotypes = make_offspring(parents, dam_list=dam_list, sire_list=sire_list, family_name=family_name)

    return genotypes