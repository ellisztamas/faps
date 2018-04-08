from unique_rows import unique_rows
import fastcluster
from scipy.cluster.hierarchy import fcluster

def sibship_partitions(pairwise_likelihoods, method='average', criterion = 'distance'):
    """
    Generate a sample of partition structures from an array of likelihoods that each pair of individuals in a half-sib array are really full sibs.

    This performs hierarchical clustering using the fastcluster and scipy libraries. The default algorithm is UPGMA, but linkage and clustering functions can be tweaked using the method and criterion functions.
    
    Parameters
    ----------
    pairwise_likelihoods: array
        A square array of log-likelihoods that each pair of individuals are full sibs.
    method: str
        Distance function passed to linkage. See fastcluster.linkage for available inputs. Defaults to 'average'.
    criterion: str
        Clustering function used to bisect the dendrogram. See scipy.hierarchy.fcluster for available inputs. Defaults to 'distance'.
    
    Returns
    -------
    An array of plausible partitions that group individuals into full sibships. In each row of the output individuals are labelled with an integer that groups them in a full sibship with all other individuals with that label.
    """
    # Clustering matrix z.
    z= fastcluster.linkage(abs(fullpairs[np.triu_indices(fullpairs.shape[0], 1)]), method)
    z = np.clip(z, 0, 10**12)
    # A list of thresholds to slice through the dendrogram
    thresholds = np.append(0,z[:,2])
    # store all possible partitions from the dendrogram
    partition_sample = [hierarchy.fcluster(z, t, criterion) for t in thresholds]
    partition_sample = unique_rows(partition_sample)
    
    return partition_sample