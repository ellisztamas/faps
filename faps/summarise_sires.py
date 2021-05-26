import pandas as pd
import numpy as np
from faps.sibshipCluster import sibshipCluster


def summarise_sires(sibships, drop_na=False):
    """
    Summarise mating events across maternal families.

    This is a wrapper function to call `sibshipCluster.sires()` across
    multiple sibshipCluster objects stored in a dictionary. It is assumed that
    each object represents a maternal family.

    Parameters
    ----------
    sibships: dict
        Dictionary of sibshipCluster objects for multiple maternal families.
        Keys should normally indicate maternal family name.

    Returns
    -------
    DataFrame giving mother (taken from the keys of the input dictionary),
    fathers (inherited from each sibshipCluster object), probabilties of
    having sired at least one offspring, and the expected number of offspring.

    Examples
    --------
    from faps import *
    import numpy as np

    # Generate a population of adults
    allele_freqs = np.random.uniform(0.3,0.5,50)
    adults = make_parents(20, allele_freqs)

    # Mate the first adult to the next three.
    mother = adults.subset(0)
    progeny = make_sibships(adults, 0, [1,2,3], 5, 'x')
    patlik = paternity_array(progeny, mother, adults, mu=0.0013)
    # Split offspring by paternal family
    patlik = patlik.split(by=progeny.fathers)
    # Cluster each paternal family into sibships
    sc = sibship_clustering(patlik)

    summarise_sires(sc)

    """

    if not isinstance(sibships, dict):
        raise TypeError("`sibships` should be a dictionary of sibshipCluster onjects.")
    if not all([isinstance(x, sibshipCluster) for x in sibships.values()]):
        raise TypeError("Not all elements of `sibships` are sibshipCluster objects.")
    if isinstance(sibships, sibshipCluster):
        raise TypeError("`summarise_sires` is intended to work on a dictionary of sibshipCluster objects, but a single sibshipCluster object was supplied. In this case, call `sires()` directly on the onject, i.e. object.sires().")

    # For each mother, get a list of plausible mates
    # with the probability of each having sired at least one offspring.
    me_dict = {k: v.sires() for k,v in sibships.items()}
    # Get distances to those sires for each maternal family.
    # Return a list of dataframes for each mother
    me_df = pd.concat(list(me_dict.values()))
    me_df = []
    for k in me_dict.keys():
        df = pd.DataFrame({
            'mother'   : k,
            'father'   : me_dict[k]['label'],
            'log_prob' : me_dict[k]['log_prob'],
            'prob'     : me_dict[k]['prob'],
            'offspring': me_dict[k]['offspring']
        })
        me_df = me_df + [df]
    # Coerce list into a single data frame listing mother, father, distance and log probability
    me_df = pd.concat(me_df)
    me_df = me_df.reset_index(drop=True)
    return me_df
