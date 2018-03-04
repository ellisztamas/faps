import numpy as np

class matingEvents(object):
    """
    Information about a sample of possible of mating events drawn proportionally
    to their probability. These can then be used to infer mating patterns using
    and array of phenotypes for each male.

    Indices for possible fathers are drawn based on genetic data, other pertinent
    information (such as distance between parents), or a combination of these
    types. Mating events are drawn for 'units', which usually means partitions
    in a `sibshipCluster` object. These samples are then resampled in proportion
    to a weight on each unit. To assess uncertainty, the aggregate mating events
    can be subsampled.

    `matingEvent` objects are usually created from sibshipCluster objects using
    the fucntion `sibshipCluster.mating_event`. They could also be generated
    from a `paternityArray` if full sibship structure was not an issue, but this
    is not implented yet.

    Parameters
    ----------
    unit_names: array or list
        Names for each unit.
    candidates: array or list
        Names for each candidate.
    unit_weights: vector
        Relative contribution of each unit. If the matingEvent object is derived
        from a sibshipCluster object, this is the relative probability of each
        partition that had a non-zero probability.
    unit_events: array or list
        List of arrays, each of which documents mating events for a single unit.
    total_events: array
        Mating events for the whole sample. This is a resample of elements in
        each sublist of unit_events, weighted by the probability of each
        unit.
    subsamples: array, optional
        Array of subsamples of total_mating events.
    """

    def __init__(self, unit_names, candidates, unit_weights, unit_events, total_events, subsamples = None):
        self.unit_names   = unit_names
        self.candidates   = candidates
        self.unit_weights = unit_weights,
        self.unit_events  = unit_events
        self.total_events = total_events
        self.subsamples   = subsamples
    
    #if len(self.unit_weights) != len(self.unit_events):
    #    raise ValueError('Length of unit_weights ({}) does not match that of unit_events ({}).'.format(len(self.unit_weights), len(self.unit_events)))

    def draw_subsample(self, n_subsamples = 1000, subsample_size = None):
        """
        Subsample total_events.

        Parameters
        ----------
        n_subsamples: int
            Number of subsamples.
        subsample_size: int, optional
            Size fo each subsample. Defaults to 0.1 time the length of
            total_events.

        Returns
        -------
        An array of integers indexing sires.
        """
        if not isinstance(n_subsamples, int):
            raise ValueError('n_subsamples should be an integer.')
        if not isinstance(subsample_size, int) and subsample_size is not None:
            raise ValueError('subsample_size should be an integer.')
        if subsample_size is None:
            subsample_size = np.around(0.1*len(self.total_events)).astype('int')

        sub_events = np.random.choice(self.total_events, n_subsamples*subsample_size, replace=True)
        sub_events = sub_events.reshape([n_subsamples, subsample_size])

        return sub_events
