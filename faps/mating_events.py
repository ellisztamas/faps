import numpy as np
from paternityArray import paternityArray
from sibshipCluster import sibshipCluster
from matingEvents import matingEvents


def mating_events(sibships, paternity, family_draws = 1000, total_draws  = 10000, n_subsamples = 1000, subsample_size = None, null_probs = None, family_weights=None):
    """
    Sample plausible mating events from a list of sibshipCluster objects. The
    contribution of each half-sib family is weighted by the expected number of
    mating events in that family, but this can be changed
    
    This is essentially a wrapper to perform sibshipCluster.mating_events for
    multiple half-sibling arrays at once. See the documentation for that function.
    
    Parameters
    ----------
    sibships: list
        List sibshipCluster objects for each half-sib array.
    paternity: list
        Probabilities of paternity used to construct the sibshipCluster
        objects.
    family_draws: int
        Number of mating events to sample for each partition.
    total_draws: int
        Total number of mating events to resample for each partition.
    n_subsamples: int, optional
        Number of subsamples to draw from the total mating events.
    subsample_size: int, optional
        Number of mating events in each subsample. Defaults to 0.1*total_draws.
    null_probs: array, optional
        Array of probabilities for paternity if this were not based on marker
        data. This should be the same shape as the prob_array in the paternity
        argument, i.e. have a row for every offspring and a column for every
        candidate.
    family_weights: array
        Vector of weights to give to each half-sib array. Defaults to the weighted
        mean number of families in the array. If values do not sum to one, they
        will be normalised to do so.

    Returns
    -------
    A matingEvents object. If null_probs is supplied samples for null mating are returned.
    """
    if not isinstance(sibships, list) or not all([isinstance(x, sibshipCluster) for x in sibships]):
        raise TypeError('sibships should be a list of sibshipCluster objects.')
    if not isinstance(paternity, list) or not all([isinstance(x, paternityArray) for x in paternity]):
        raise TypeError('paternity should be a list of paternityArray objects.')
    elif len(sibships) != len(paternity):
        raise ValueError('Lists of sibshipCluster and paternityArray objects are different lengths.')
    elif len(sibships) == 1:
        warnings.warn('Lists of sibshipCluster and paternityArray are of length 1. If there is only one array to analyse it is cleaner to call mating_events dircetly from the sibshipCluster object.')
        
    # labels for each array, and each candidate
    unit_names = np.array([x.mothers[0] for x in paternity])
    candidates = np.append(paternity[0].candidates[0], 'father_missing')
    
    if family_weights is np.ndarray:
        if len(family_weights) != len(sibships):
            raise ValueError('Length of family_weights ({}) does not match length of sibships ().'.format(len(family_weights), len(sibships)))
            family_weights = family_weights / family_weights.sum()
    elif family_weights is None:
        # weight each family by the expected number of mating events
        family_weights = np.array([x.mean_nfamilies() for x in sibships]) # expected number of mating events for each array
        family_weights = family_weights / family_weights.sum() # normalise to sum to one
        family_weights = np.around(family_weights*total_draws).astype('int') # get integer value
    else:
        raise TypeError('family_weights should be an array, or else None.')

    # draw events for each half-sib array
    ix = range(len(sibships))
    family_events = [sibships[x].mating_events(paternity[x], family_draws, family_draws, None, None, null_probs=null_probs) for x in ix]
    family_events = [x.total_events for x in family_events]
    # resample me_list proportional to the prob of each unit.
    total_events = [np.random.choice(family_events[i], family_weights[i], replace=True) for i in ix if len(family_events[i])>0]
    total_events = [item for sublist in total_events for item in sublist]
    total_events = np.array(total_events)

    #create matingEvents object
    me = matingEvents(unit_names, candidates, family_weights, family_events, total_events)
    # draw subsamples.
    me.subsamples = me.draw_subsample(n_subsamples, subsample_size)
    
    return me