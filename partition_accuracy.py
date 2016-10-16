import numpy as np

def partition_accuracy(true, proposed, rtype='all'):
    """
    Returns the accuracy of a proposed partition with reference to a known true partition.
    
    Pr(A|B) is the probability of inferring relationship of type A between two individuals
    given that the true relationship is type B. This function estimates the total 
    proportion of pairwise relationships inferred correctly, as well as Pr(FS|FS), and 
    Pr(HS|HS). For half sibling arrays, Pr(FS|HS) = 1-Pr(HS|HS) and Pr(HS|FS) = 1-Pr(FS|FS).
    
    #ARGUMENTS
    true: true partition structure.
    
    proposed: the proposed partition structure.
    
    rtype: the measure of accuracy to be determine. This can be 'all' for overal accuracy,
    'fs' for Pr(FS|FS) or 'hs' for Pr{HS|HS).
    
    RETURNS
    A float between zero and one.
    """
    if rtype not in ['all', 'fs', 'hs']:
        print "rtype must be one of 'all', 'fs' or 'hs'."
        #return None

    real_FS = relation_matrix(true)
    part_FS = relation_matrix(proposed)
    r_given_r = real_FS == part_FS

    if rtype is 'all':
        correct   = (r_given_r * np.triu(np.ones(real_FS.shape), 1)).sum()
        total     = np.triu(np.ones(real_FS.shape), 1).sum()
        accuracy  = correct/total

    if rtype is 'fs':
        # Pr(FS|FS) depends on the correct relationships, conditioned on being a true FS relationship.
        fs_given_fs = r_given_r * real_FS
        correct     = (fs_given_fs * np.triu(real_FS,1)).sum()
        total       = np.triu(real_FS,1).sum()
        accuracy    = correct/total

    if rtype is 'hs':
        real_HS     = 1- real_FS # real halfsibling relationships
        hs_given_hs = r_given_r * real_HS
        correct     = (hs_given_hs * np.triu(real_HS, 1)).sum()
        total       = np.triu(real_HS,1).sum()
        accuracy    = correct/total

    return accuracy