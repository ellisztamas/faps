import numpy as np
from paternityArray import paternityArray
from sibshipCluster import sibshipCluster
from do_clustering import do_clustering

def sibship_clustering(paternity_array, ndraws=1000, exp=False):
    """
    Cluster offspring into full sibship groups using hierarchical clustering.

    This first builds a dendrogram of relatedness between individuals, and pulls
    out every possible partition structure compatible with the dendrogram. The
    likelihood for each partition is also estimated by Monte Carlo simulations.

    This function is really just a wrapper for do_cluster() that allows for
    lists of paternityArray objects.

    Parameters
    ----------
    paternity_array: paternityArray or list
        Either a single paternityArray object, or a list of paternityArrays for
        separate half-sibling arrays.
    ndraws: int
        Number of Monte Carlo simulations to run for each partition.
    exp: logical, optional
        Indicates whether the probabilities of paternity should be exponentiated
        before calculating pairwise probabilities of sibships. This gives a
        speed boost if this is to be repeated many times in simulations, but
        there may be a cost to accuracy. Defaults to False for this reason.

    Returns
    -------
    A sibshipCluster object.
    """
    if isinstance(paternity_array, paternityArray):
        return do_clustering(paternity_array, ndraws, exp)

    elif isinstance(paternity_array, list):
        if not all([isinstance(x, paternityArray) for x in paternity_array]):
            raise TypeError('Not all items in paternity_array are paternityArray objects.')
        return [do_clustering(pa, ndraws, exp) for pa in paternity_array]
    
    else:
        TypeError("paternity_array should be a paternityArray object, or list of paternityArray objects.")