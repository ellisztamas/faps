import numpy as np
from faps.paternityArray import paternityArray
from faps.sibshipCluster import sibshipCluster
from faps.matingEvents import matingEvents


def mating_events(sibship_clusters, paternity_arrays, family_draws = 1000, total_draws  = 10000, n_subsamples = 0, subsample_size = None, null_probs = None, family_weights=None, use_covariates=False, return_names=True):
    """
    Sample plausible mating events for multiple half-sibling arrays in parallel.

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
    sibship_clusters: dict
        Dictionary of sibshipCluster objects for multiple maternal families.
    paternity_arrays: dict
        Dictionary of paternityArray objects for multiple maternal families.
        Keys should match those for sibship_clusters.
    family_draws: int
        Number of mating events to sample for each maternal family.
    total_draws: int
        Total number of mating events to draw across maternal families.
    n_subsamples: int, optional
        Number of subsamples to draw from the total mating events.
    subsample_size: int, optional
        Number of mating events in each subsample. Defaults to 0.1*total_draws.
    null_probs: vector or dict, optional
        Array of (log) probabilities for paternity if this were not based on marker
        data. If the same probabilities apply to each half-sib array, this can
        be a vector with an element for each candidate father. Alternatively, if
        the vector for each array is different, supply an dictionary of 1-d
        vectors, with keys that match those for sibship_clusters. Probabilities
        in a given vector should sum to one.
    family_weights: array
        1-dimensional vector of weights to give to each half-sib array. If None,
        weights will be calculated as the mean number of full-sibships in each
        maternal family over possible partitions, weighted by the probability
        of each partition.
    return_names: logical, optional
        If True, unit_events and total_events are returned as the names of
        the candidates drawn for each maternal family. If False, their integer
        positions in the list of candidates is returned.

    Returns
    -------
    A matingEvents object. If null_probs is supplied samples for null mating are
    returned.

    See also
    --------
    sibshipCluster.mating_events()
    matingEvents()
    """
    if not isinstance(sibship_clusters, dict) or not all([isinstance(x, sibshipCluster) for x in sibship_clusters.values()]):
        raise TypeError('sibship_clusters should be a dictionary of sibshipCluster objects.')
    if not isinstance(paternity_arrays, dict) or not all([isinstance(x, paternityArray) for x in paternity_arrays.values()]):
        raise TypeError('paternity should be a dictionary of paternityArray objects.')
    elif len(sibship_clusters) != len(paternity_arrays):
        raise ValueError('Lists of sibshipCluster and paternityArray objects are different lengths.')
    elif len(sibship_clusters) == 1:
        warnings.warn('Lists of sibshipCluster and paternityArray are of length 1. If there is only one array to analyse it is cleaner to call mating_events dircetly from the sibshipCluster object.')

    # labels for each array, and each candidate
    candidates = list(paternity_arrays.values())[0].candidates
    candidates = np.append(candidates, 'father_missing')

    # Determine how to draw the correct number of samples for each half-sib array.
    # For a user-supplied vector of weights check this is the right format
    if family_weights is np.ndarray:
        if len(family_weights.shape) > 1:
            raise ValueError("family_weights should be a 1-d vector, but has shape {}.".format(family_weights.shape))
        if len(family_weights) != len(sibship_clusters):
            raise ValueError('family_weights has length {}, but there are {} half-sib arrays sibship_clusters.'.format(len(family_weights), len(sibship_clusters)))
            family_weights = family_weights / family_weights.sum()
    # If no vector is given, use the expected mean number of full sibships per
    # maternal family.
    elif family_weights is None:
        family_weights = {k:v.mean_nfamilies() for k,v in sibship_clusters.items()} # expected number of mating events for each array
        total_weight   = total_draws / np.array(list(family_weights.values())).sum() # correction factor to sum to total_draws
        family_weights = {k : v * total_weight for k,v in family_weights.items()} # normalise to sum to total_draws
        family_weights = {k : np.around(v).astype('int') for k,v in family_weights.items()}
    else:
        raise TypeError('family_weights is type {}. If supplied, it should be a NumPy array.').format(type(family_weights))

    # If nothing is given for null_probs
    if null_probs is None:
        null_probs = {k: None for k in paternity_arrays.keys()}
    # If a NumPy array is given, coerce to a dictionary
    elif isinstance(null_probs, np.ndarray):
        # For 1-d vectors.
        if(null_probs.ndim == 1):
            if len(null_probs != len(candidates)-1):
                ValueError("If null_probs is a 1-d vector it should have the same length as the number of candidate fathers.")
            null_probs = {k:null_probs for k in paternity_arrays.keys()}
        # Throw an error if there is more than 1 dimension.
        elif(null_probs.ndim > 1):
            ValueError("null_probs should be a 1-d array or a dictionary of 1-d arrays.")
    # If a dictionary is given for null_probs, check it.
    elif isinstance(null_probs, dict):
        if any([k not in paternity_arrays.keys() for k in null_probs.keys()]):
            ValueError("One or more entries in null_probs does not have a matching entry in paternity_arrays.")
        if any([v.ndim > 1 for v in null_probs.values()]):
            ValueError("One or more entries in null_probs has more than one dimension.")
        if any([len(v) != len(candidates)-1 for k,v in null_probs.items()]):
            ValueError("One or more entries in null_probs is a different length to the number of candidates")
        if not all([isinstance(v, np.ndarray) for k,v in null_probs.items()]):
            TypeError("If null_probs is a dictionary, all values should be 1-d NumPy arrays.")
    else:
        raise TypeError("null_probs is type {}.\nIf supplied, this should be a NumPy array, or dictionary.").format(type(null_probs))

    # draw events for each half-sib array
    family_events = {}
    total_events  = {}
    for id in sibship_clusters.keys():
        me = sibship_clusters[id].mating_events(
            paternity_array = paternity_arrays[id],
            unit_draws=family_draws,
            total_draws=total_draws,
            n_subsamples=n_subsamples,
            subsample_size=subsample_size,
            null_probs=null_probs[id],
            use_covariates=use_covariates,
            return_names = return_names
        )
        family_events[id] = me.total_events
        # Random subset of fathers from each maternal family, with sample size in
        # proportion to the weight of the family
        total_events[id] = np.random.choice(me.total_events, family_weights[id], replace=True)
    # Concatenate total_events into a single list.
    total_events = [item for sublist in list(total_events.values()) for item in sublist]
    total_events = np.array(total_events)

    # Create a matingEvents object.
    me = matingEvents(
        unit_names   = np.array(list(family_events.keys())),
        candidates   = candidates,
        unit_weights = family_weights,
        unit_events  = family_events,
        total_events = total_events
    )
    # draw subsamples.
    me.subsamples = me.draw_subsample(n_subsamples, subsample_size)


    return me
