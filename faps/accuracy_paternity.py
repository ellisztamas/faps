import numpy as np
from faps.genotypeArray import genotypeArray

def accuracy_paternity(progeny, parents, paternity_array, partition, target_parent = 'father'):
    """
    Summarise how well a sibship partition returns information on the posterior
    probability of paternity of true sires. For individuals whose fathers have not been
    sampled, the probability that the father is drawn from population allele frequencies
    is returned.

    Deprecated in favour of a function within sibshipCluster.

    Parameters
    ----------
    progeny: genotypeArray
        Genotype information on offspring individuals included in paternity_array.

    parents: genotypeArray
        Genotype information on the candidate parents included in paternity_array.

    paternity_array: A matrix of probabilities that each candidate parent is the true
        parent of each individual. This array can be generated with paternity_array, or
        else supplied independently. Rows show index offspring, and columns parents.

    target_parent: String denoting whether the paternity_array indexes fathers, mothers,
        or parents pairs. Valid arguments are 'mother', 'father' and 'parents',  or
            equivalently 'm' and 'f' respectively.

    Returns
    -------
    A list of two elements:
    1. The mean posterior probability of sampled true sires, conditioned on the father
        having been included in the sampled of candidate parents.
    2. The mean posterior probability that the sire is drawn from population allele
        frequencies, conditioned on the true sire not having been included in the sampled
        of candidate parents.
    """
    # PROBABILITIES OF PATERNITY FOR THE TRUE SIRES
    # Check which individuals still have their father in the parent array, and which do not.
    if target_parent is 'mother' or target_parent is 'm':
        parent_present = [progeny.mothers[x]     in parents.names for x in range(progeny.size)]
        parent_missing = [progeny.mothers[x] not in parents.names for x in range(progeny.size)]
    elif target_parent is 'father' or target_parent is 'f':
        parent_present = [progeny.fathers[x]     in parents.names for x in range(progeny.size)]
        parent_missing = [progeny.fathers[x] not in parents.names for x in range(progeny.size)]
    elif target_parent is 'parent' or target_parent is 'p':
        parent_present = [progeny.parents[x]     in parents.names for x in range(progeny.size)]
        parent_missing = [progeny.parents[x] not in parents.names for x in range(progeny.size)]
    else:# target_parent not in ['mother', 'father', 'parent', 'm', 'f', 'p']:
        raise ValueError("target_parent must be 'mother', 'father', or 'parents'.\nThis should be whichever is indexed in the columns of paternity_array.")

    # If there are no individuals with fathers present, return
    if sum(parent_present) == 0:
        mean_patprob = -9
    if sum(parent_present) >= 1:
        # subset the progeny to select just those non-orphaned offspring
        not_orphans = progeny.subset(np.arange(progeny.size)[np.array(parent_present)])
        # get the positions of the true fathers in the probability matrix for these individuals.
        father_pos  = not_orphans.parent_index(target_parent, parents.names)
        # Get the posterior probability of the true sire of each individual within its FSG
        # for those individuals with a father in the array.
        pat_probs = np.zeros(sum(parent_present)) # empty vector to store probabilities
        for x in range(sum(parent_present)):
            offs_pos     = np.arange(progeny.size)[np.array(parent_present)][x] # position of the offspring
            # paternity probabilities multiplied across individuals in the sibship
            this_gamma   = paternity_array[partition == partition[offs_pos]].sum(0)
            this_gamma   = this_gamma - alogsumexp(this_gamma) # normalise to sum to one.
            # get the value for the true sire
            pat_probs[x] = this_gamma[father_pos[x]]
        mean_patprob = pat_probs.mean()

    # POSTERIOR PROBABILITIES THAT A SIRE WAS NOT SAMPLED.
    # If there are no individuals with missing fathers, return -9
    if sum(parent_missing) == 0:
        mean_absprob = -9
    # Otherwise, get the posterior probabilities that each sire is missing
    if sum(parent_missing) >= 1:
        # subset the progeny to select just those orphaned offspring
        orphans     = progeny.subset(np.arange(progeny.size)[np.array(parent_missing)])
        # Get the posterior probability that the true sire is missing for each individual
        abs_probs   = np.zeros(sum(parent_missing)) # empty vector to store probabilities
        for x in range(sum(parent_missing)):
            offs_pos     = np.arange(progeny.size)[np.array(parent_missing)][x] # position of the offspring
             # paternity probabilities for this sibship
            this_gamma   = paternity_array[partition == partition[offs_pos]].sum(0)
            this_gamma   = this_gamma - alogsumexp(this_gamma) # normalise to sum to one.
            # The probability of an unsampled father is given in the final column
            abs_probs[x] = this_gamma[-1]
        mean_absprob = abs_probs.mean()

    return [mean_patprob, mean_absprob]
