import numpy as np

def relation_matrix(partitions):
    """
    Create a matrix of pairwise relationships within a half-sibling array
    given a real or hypothesised vector of partitions.
    Relationships are coded as 1 if they are full siblings and 0 if not.
    """
    sibshipsize = len(partitions) # total offspring number.
    relmat      = np.zeros([sibshipsize, sibshipsize]) # empty matrix of zeros.
    siblabel    = np.unique(partitions) # vector of unique labels for each full sib family.
    index       = np.arange(sibshipsize) # indices for each sib
    
    # for each full sibship fill in the pairwise elements with ones.
    for i in range(len(siblabel)):
        b                        = partitions == siblabel[i] # truth array
        rowvals                  = np.repeat(index[b], sum(b))  
        colvals                  = np.tile(  index[b], sum(b))
        relmat[rowvals, colvals] = 1
    
    return relmat