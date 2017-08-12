import numpy as np
import fastcluster
from scipy.cluster import hierarchy
from alogsumexp import alogsumexp
from sibshipCluster import sibshipCluster
from pairwise_lik_fullsibs import pairwise_lik_fullsibs
from unique_rows import unique_rows
from lik_partition import lik_partition

def sibship_clustering(paternity_array, MC_draws=1000, exp=False):
    """
    Cluster offspring into full sibship groups using hierarchical clustering.

    This first builds a dendrogram of relatedness between individuals, and pulls out every
    possible partition structure compatible with the dendrogram. The likelihood for each
    partition is also estimated by Monte Carlo simulations.

    Parameters
    ----------
    paternity_array: paternityArray
        Object listing information on paternity of individuals.
    MC_draws: int
        Number of Monte Carlo simulations to run for each partition.
    exp: logical, optional
        Indicates whether the probabilities of paternity should be exponentiated before
        calculating pairwise probabilities of sibships. This gives a speed boost if this
        is to be repeated many times in simulations, but there may be a cost to accuracy.
        Defaults to False for this reason.

    Returns
    -------
    A sibshipCluster object.
    """

    prob_paternities = paternity_array.prob_array
    # Matrix of pairwise probabilities of being full siblings.
    fullpairs = pairwise_lik_fullsibs(prob_paternities, exp)
    # Clustering matrix z.
    z= fastcluster.linkage(abs(fullpairs[np.triu_indices(fullpairs.shape[0], 1)]), method='average')
    z = np.clip(z, 0, 10**12)
    # A list of thresholds to slice through the dendrogram
    thresholds = np.append(0,z[:,2])
    # store all possible partitions from the dendrogram
    partition_sample = unique_rows([hierarchy.fcluster(z, thresholds[t], criterion='distance')
                                    for t in range(thresholds.shape[0])])
    # likelihoods for each partition
    partition_liks  = np.array([lik_partition(prob_paternities, partition_sample[i], ndraws=MC_draws)
                                for i in range(partition_sample.shape[0])])
    return sibshipCluster(paternity_array, z, partition_sample, partition_liks)
