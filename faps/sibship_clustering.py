import numpy as np
from faps.paternityArray import paternityArray
from faps.sibshipCluster import sibshipCluster
from faps.do_clustering import do_clustering
from warnings import warn

def sibship_clustering(paternity_array, ndraws=1000, use_covariates = False, exp=False):
    """
    Cluster offspring into full sibship groups using hierarchical clustering.

    This first builds a dendrogram of relatedness between individuals, and pulls
    out every possible partition structure compatible with the dendrogram. The
    likelihood for each partition is also estimated by Monte Carlo simulations.

    This function is really just a wrapper for do_cluster() that allows for
    lists of paternityArray objects.

    If any paternityArray object contains only a single individual, clustering
    is meaningless, but an artifical sibshipCluster object is created with a
    single partition with log likelihood of 0.

    Parameters
    ----------
    paternity_array: paternityArray or dict
        Information on the likelihood of paternity for individuals in a maternal
        family. To analyse multiple maternal families in parallel,
        give a dictionary containing multiple named paternityArray objects.
    ndraws: int
        Number of Monte Carlo simulations to run for each partition.
    use_covariates: logical, optional
        If True, information on prbabilities associated with covariates stored
        in paternityArray objects are incorporated into sibship clustering.
    exp: logical, optional
        Indicates whether the probabilities of paternity should be exponentiated
        before calculating pairwise probabilities of sibships. This gives a
        speed boost if this is to be repeated many times in simulations, but
        there may be a cost to accuracy. Defaults to False for this reason.

    Returns
    -------
    A sibshipCluster object.

    Examples
    --------
    from faps import *
    import numpy as np

    # Generate a population of adults
    allele_freqs = np.random.uniform(0.3,0.5,50)
    adults = make_parents(20, allele_freqs)

    # Ecample with a single family
    # Mate the first adult to the next three.
    mother = adults.subset(0)
    progeny = make_sibships(adults, 0, [1,2,3], 5, 'x')
    # Create paternityArray
    patlik = paternity_array(progeny, mother, adults, mu=0.0013)
    # compare the effect of exponentiating likelihoods on the accuracy of sibship
    sc1 = sibship_clustering(patlik, exp=True)
    sc2 = sibship_clustering(patlik, exp=False)
    # see help(sibship_clustering) for explanation of the output
    sc1.accuracy(progeny, adults)
    sc2.accuracy(progeny, adults)

    # Example with multiple half-sib families
    progeny = make_offspring(parents = adults, dam_list=[7,7,1,8,8,0], sire_list=[2,4,6,3,0,7])
    # Split mothers and progeny up by half-sib array.
    mothers = adults.split(by=progeny.mothers, return_dict=True)
    progeny = progeny.split(by=progeny.mothers, return_dict=True)
    # Create paternity array for each family using dictionaries
    patlik = paternity_array(progeny, mothers, adults, mu=0.0013)
    # The dictionary is pass straight to sibship_clustering.
    sc = sibship_clustering(patlik)
    # Pull out info on individual families
    sc['base_7'].mlpartition
    # Since family 'base_0' contains only a single offspring, there is no
    # information about clustering. Compare with family `base_7`:
    sc['base_7'].linkage_matrix
    sc['base_0'].linkage_matrix
    """
    # If a single paternity array is given
    if isinstance(paternity_array, paternityArray):
        output = do_clustering(paternity_array,ndraws=ndraws, use_covariates= use_covariates, exp=exp)
    elif isinstance(paternity_array, list):
        if not all([isinstance(x, paternityArray) for x in paternity_array]):
            raise TypeError('Not all items in paternity_array are paternityArray objects.')
        output = [do_clustering(pa, ndraws=ndraws, use_covariates= use_covariates, exp=exp) for pa in paternity_array]
    # If dictionary is given
    elif(paternity_array, dict):
        # Check data types.
        if not all([isinstance(x, paternityArray) for x in paternity_array.values()]):
            raise TypeError('Not all items in paternity_array are paternityArray objects.')
        # Cluster each paternityArray.
        output = {}
        for pa in paternity_array.keys():
            if paternity_array[pa].offspring.shape[0] == 1:
                warn("Family {} has only one individual. A sibship array will be returned, but without a linkage matrix.".format(pa))
            sc = do_clustering(paternity_array = paternity_array[pa],
                               ndraws=ndraws,
                               use_covariates=use_covariates,
                               exp=exp)
            output[pa] = sc
    else:
        TypeError("paternity_array should be a paternityArray object, or list of paternityArray objects.")

    return output
