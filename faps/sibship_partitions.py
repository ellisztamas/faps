import numpy as np
from faps.unique_rows import unique_rows
import fastcluster
from scipy.cluster.hierarchy import fcluster
from faps.pairwise_lik_fullsibs import pairwise_lik_fullsibs

def sibship_partitions(paternity_array, exp=False, method='average', criterion = 'distance', use_covariates=False):
    """
    Generate a sample of partition structures from an array of likelihoods that
    each pair of individuals in a half-sib array are really full sibs.

    This performs hierarchical clustering using the fastcluster and scipy
    libraries. The default algorithm is UPGMA, but linkage and clustering
    functions can be tweaked using the method and criterion functions.

    Parameters
    ----------
    paternity_array: paternityArray
        Object listing information on paternity of individuals.
    method: str
        Distance function passed to linkage. See fastcluster.linkage for
        available inputs. Defaults to 'average'.
    criterion: str
        Clustering function used to bisect the dendrogram. See
        scipy.hierarchy.fcluster for available inputs. Defaults to 'distance'.
    use_covariates: logical, optional
        If True, information on prbabilities associated with covariates stored
        in paternityArray objects are incorporated into sibship clustering.

    Returns
    -------
    An array of plausible partitions that group individuals into full sibships.
    In each row of the output individuals are labelled with an integer that
    groups them in a full sibship with all other individuals with that label.

    Example
    -------
    import numpy as np
    from faps import *

    # generate 4 families of 5 offspring
    allele_freqs = np.random.uniform(0.3, 0.5, 50)
    males = make_parents(200, allele_freqs)
    offspring = make_sibships(males, 0, range(1,5), 5)

    # Add muations
    mu = 0.0013
    males = males.dropouts(0.01).mutations(mu)
    offspring= offspring.dropouts(0.01).mutations(mu)

    mothers = males.subset(offspring.parent_index('m', males.names))
    patlik = paternity_array(offspring, mothers, males, allele_freqs, mu)

    # Generate possible sets of partitions
    partition_sample = sibship_partitions(patlik)
    """
    if len(paternity_array.offspring) == 1:
        partition_sample = np.array([[0]])
        z = 'Linkage matrix not available because there is only one offspring'

    else:
        fullpairs = pairwise_lik_fullsibs(paternity_array, use_covariates=use_covariates, exp = exp)
        # Clustering matrix z.
        z= fastcluster.linkage(abs(fullpairs[np.triu_indices(fullpairs.shape[0], 1)]), method)
        z = np.clip(z, 0, 10**12)
        # A list of thresholds to slice through the dendrogram
        thresholds = np.append(0,z[:,2])
        # store all possible partitions from the dendrogram
        partition_sample = [fcluster(z, t, criterion) for t in thresholds]
        partition_sample = unique_rows(partition_sample)

    return partition_sample, z
