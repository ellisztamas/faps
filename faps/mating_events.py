import numpy as np
from paternityArray import paternityArray
from sibshipCluster import sibshipCluster
from matingEvents import matingEvents


def mating_events(sibship_clusters, paternity_arrays, family_draws = 1000, total_draws  = 10000, n_subsamples = 1000, subsample_size = None, null_probs = None, family_weights=None, use_covariates=False):
    """
    Sample plausible mating events for a list of half-sibling arrays.
    
    This creates a single matingEvents object, where every half-sib family
    represents a unit, from which n=family_draws likely fathers are sampled.
    By default, the contribution of each half-sib family to the total sample
    is determined by the expected number of full-sib families in each array, but
    this can be changed by modifying family_weights.
    
    This is essentially a wrapper to perform sibshipCluster.mating_events for
    multiple half-sibling arrays at once. See the documentation for that function
    for details on sampling individual families.
    
    Parameters
    ----------
    sibship_clusters: list
        List sibshipCluster objects for each half-sib array.
    paternity_arrays: list
        List of paternityArray objects for each half-sib array. These should be the
        same paternityArray objects used to construct the sibshipCluster objects,
        or else results will be meaningless.
    family_draws: int
        Number of mating events to sample for each partition.
    total_draws: int
        Total number of mating events to resample for each partition.
    n_subsamples: int, optional
        Number of subsamples to draw from the total mating events.
    subsample_size: int, optional
        Number of mating events in each subsample. Defaults to 0.1*total_draws.
    null_probs: list or array, optional
        Array of probabilities for paternity if this were not based on marker
        data. If the same probabilities apply to each half-sib array, this can
        be a vector with an element for each candidate father. Alternatively, if
        the vector for each array is different, supply an array with a row for each
        array, and a column for each candidate. If null_probs is supplied, genetic
        information is ignored.
    family_weights: array
        Vector of weights to give to each half-sib array. Defaults to the weighted
        mean number of families in the array. If values do not sum to one, they
        will be normalised to do so.

    Returns
    -------
    A matingEvents object. If null_probs is supplied samples for null mating are returned.
    
    See also
    --------
    sibshipCluster.mating_events()
    matingEvents()
    """
    if not isinstance(sibship_clusters, list) or not all([isinstance(x, sibshipCluster) for x in sibship_clusters]):
        raise TypeError('sibship_clusters should be a list of sibshipCluster objects.')
    if not isinstance(paternity_arrays, list) or not all([isinstance(x, paternityArray) for x in paternity_arrays]):
        raise TypeError('paternity should be a list of paternityArray objects.')
    elif len(sibship_clusters) != len(paternity_arrays):
        raise ValueError('Lists of sibshipCluster and paternityArray objects are different lengths.')
    elif len(sibship_clusters) == 1:
        warnings.warn('Lists of sibshipCluster and paternityArray are of length 1. If there is only one array to analyse it is cleaner to call mating_events dircetly from the sibshipCluster object.')
        
    # labels for each array, and each candidate
    unit_names = np.array([x.mothers[0] for x in paternity_arrays])
    candidates = np.append(paternity_arrays[0].candidates[0], 'father_missing')
    
    # Determine how to draw the correct number of samples for each half-sib array.
    if family_weights is np.ndarray:
        if len(family_weights.shape) > 1:
            raise ValueError("family_weights should be a 1-d vector, but has shape {}.".format(family_weights.shape))
        if len(family_weights) != len(sibship_clusters):
            raise ValueError('family_weights has length {}, but there are {} half-sib arrays sibship_clusters.'.format(len(family_weights), len(sibship_clusters)))
            family_weights = family_weights / family_weights.sum()
    elif family_weights is None:
        # weight each family by the expected number of mating events
        family_weights = np.array([x.mean_nfamilies() for x in sibship_clusters]) # expected number of mating events for each array
        family_weights = family_weights / family_weights.sum() # normalise to sum to one
        family_weights = np.around(family_weights*total_draws).astype('int') # get integer value
    else:
        raise TypeError('family_weights is type {}. If supplied, it should be a NumPy array.').format(type(family_weights))
        
    if null_probs is None:
        null_probs = [None] * len(sibship_clusters)
    elif isinstance(null_probs, np.ndarray):
        if len(null_probs.shape) == 1:
            null_probs = np.repeat(null_probs[np.newaxis], len(sc), 0)
        if null_probs.shape[0] != len(sibship_clusters):
            raise ValueError('null_probs has {} rows, but there are {} half-sib arrays; if null_probs is supplied it should have a row for each half-sib array, or be a 1-d to be applied to each half-sib array.'.format(null_probs.shape[0], len(sibship_clusters)))
    else:
        raise TypeError("null_probs is type {}. If supplied, this should be a NumPy array.").format(type(null_probs))
    
    # draw events for each half-sib array
    ix = range(len(sibship_clusters))
    family_events = [sibship_clusters[x].mating_events(
        paternity_array = paternity_arrays[x],
        unit_draws=family_draws,
        total_draws=total_draws,
        n_subsamples=n_subsamples,
        subsample_size=subsample_size,
        null_probs=null_probs[x],
        use_covariates=use_covariates)
                     for x in ix]
    family_events = np.array([x.total_events for x in family_events])
    # resample me_list proportional to the prob of each unit.
    total_events = [np.random.choice(family_events[i], family_weights[i], replace=True) for i in ix if len(family_events[i])>0]
    total_events = [item for sublist in total_events for item in sublist]
    total_events = np.array(total_events)

    #create matingEvents object
    me = matingEvents(unit_names, candidates, family_weights, family_events, total_events)
    # draw subsamples.
    me.subsamples = me.draw_subsample(n_subsamples, subsample_size)
    
    return me