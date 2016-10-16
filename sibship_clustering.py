import numpy as np
from alogsumexp import alogsumexp
from scipy.cluster import hierarchy

def sibship_clustering(prob_paternities, MC_draws=1000, exp=False):
    """
    Cluster offspring into full sibship groups using hierarchical clustering.
    
    This first builds a dendrogram of relatedness between individuals, and pulls out every
    possible partition structure compatible with the dendrogram. The likelihood for each
    partition is also estimated by Monte Carlo simulations.
    
    ARGUMENTS
    prob_paternities: an array of probabilities of paternity from paternity_array().
    
    MC_draws: number of Monte Carlo simulations to run for each partition.
    
    exp: Indicates whether the probabilities of paternity should be exponentiated before
        calculating pairwise probabilities of sibships. This gives a speed boost if this
        is to be repeated many times in simulations, but there may be a cost to accuracy.
        Defaults to False for this reason.
    
    RETURNS
    A sibshipCluster object.
    
    """
    # Matrix of pairwise probabilities of being full siblings.
    fullpairs = pairwise_lik_fullsibs(prob_paternities, exp)
    # Clustering matrix z.
    z= hierarchy.linkage(abs(fullpairs[np.triu_indices(fullpairs.shape[0], 1)]), method='average')
    # A list of thresholds to slice through the dendrogram
    thresholds = np.append(0,z[:,2])
    # store all possible partitions from the dendrogram
    partition_sample = unique_rows([hierarchy.fcluster(z, thresholds[t], criterion='distance')
                                    for t in range(thresholds.shape[0])])
    # likelihoods for each partition
    partition_liks  = np.array([lik_partition(prob_paternities, partition_sample[i], ndraws=MC_draws)
                                for i in range(partition_sample.shape[0])])
    # Get the parition with the highest likelihood.
    ml_parition = partition_sample[np.where(partition_liks == partition_liks.max())][0]
    
    return sibshipCluster(z, partition_sample, partition_liks)