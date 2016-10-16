import numpy as np
from alogsumexp import alogsumexp

class sibshipCluster(object):
    """
    Information on  the results of hierarchical clustering of an offspring array
    into full sibling groups.
    
    RETURNS
    linkage_matrix: Matrix describing distances between individuals for creating
        dendrograms.
        
    partitions: Array of possible partition structures from the linkage matrix.
    
    loglik: log likelihood of ecah partition structure.
    
    posterior: log posterior probabilities of each partition structure.
    
    mlpartition: maximum-likelihood partition structure.
    """
    def __init__(self, linkage_matrix, partitions, loglik):
        self.partitions     = partitions
        self.linkage_matrix = linkage_matrix
        self.loglik         = loglik
        self.posterior      = self.loglik - alogsumexp(self.loglik)
        self.mlpartition    = self.partitions[np.where(self.loglik == self.loglik.max())[0][0]]
