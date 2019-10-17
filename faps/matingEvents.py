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
        Draw many subsamples from `total_events` at random, with replacement.

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

    def as_trait(self, trait):
        """
        Recast a matingEvents object to return the phenotype values of the
        candidates drawn.

        Parameters
        ----------
        trait: 1-d array or dict
            Phenotypes are usually a 1-d vector of the phenotypes of each candidate
            father. Alternatively, if the phenotype depends on both the mother and the
            father, for example a Euclidean distance between the pair, then this can
            be supplied as a dictionary of 1-d vectors.

        Returns
        -------
        A matingEvents object with entries for unit_events, total_events and
        subsamples given as the phenotype of each candidate. Other attributes are
        unchanged.
        """
        if self.total_events.dtype != int:
            raise TypeError("Values in matingEvents should be integers indexing candidate fathers, but are type {}.".format(self.total_events.dtype))

        # For a vector of phenotypes with an entry for each mother.
        if isinstance(trait, np.ndarray):
            if trait.ndim != 1:
                raise ValueError("`trait` should be a 1-d vector, or else a dictionary, but has shape {}".format(trait.shape))
            output = matingEvents(
                unit_names   = self.unit_names,
                candidates   = self.candidates,
                unit_weights = self.unit_weights,
                unit_events  = {k: trait[v] for k,v in self.unit_events.items()},
                total_events = trait[self.total_events],
                subsamples   = trait[self.subsamples]
            )

        # For dictionaries of trait values with an key for each mother.
        elif isinstance(trait, dict):
            if not all([x in self.unit_events.keys() for x in trait.keys()]):
                raise ValueError("Not all keys for traits have a matching key in unit_events of the matingEvents object.")
            if not all([x in trait.keys() for x in self.unit_events.keys()]):
                raise ValueError("Not all keys for unit_events have a matching key in traits.")

            # Apply phenotypes to unit_events
            unit_events  = {k: trait[k][v] for k,v in self.unit_events.items()}
            # Resample total_events
            total_events = []
            for i in self.unit_events.keys():
                total_events = total_events + [np.random.choice(
                    a    = unit_events[i],
                    size = self.unit_weights[0][i], # For some reason, unit_weights is returned as a tuple. Index [0] allows for that.
                    replace=False
                )]
            # Flatten and coerce to array
            total_events = [x for y in total_events for x in y]
            total_events = np.array(total_events)

            # Recreate matingEvents object.
            output = matingEvents(
                unit_names   = self.unit_names,
                candidates   = self.candidates,
                unit_weights = self.unit_weights,
                unit_events  = unit_events,
                total_events = total_events,
                subsamples   = None
            )
            # Add subsamples
            output.subsamples = output.draw_subsample(
                n_subsamples=self.subsamples.shape[0],
                subsample_size=self.subsamples.shape[1]
            )
        else:
            raise TypeError("`trait` should be a 1-d vector, or else a dictionary.")

        return output

    # def reweight_units(self):
    #     """
    #     Draw a single subsample without replacement from each element in
    #     `unit_events` in proportion to its weight. This is equivalent to
    #     recreating `total_events`, but without concatenating the output.
    #
    #     Parameters
    #     ----------
    #     None.
    #
    #     Returns
    #     -------
    #     A dictionary of vectors indexing candidate fathers that were drawn as
    #     mates for each unit.
    #     """
    #     output = {}
    #     for i in self.unit_events.keys():
    #         output[i] = np.random.choice(
    #             a    = self.unit_events[i],
    #             size = self.unit_weights[0][i], # For some reason, unit_weights is returned as a tuple. Index [0] allows for that.
    #             replace=False
    #         )
    #     return(output)
