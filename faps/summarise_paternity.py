import pandas as pd
import numpy as np
from faps.sibshipCluster import sibshipCluster

def summarise_paternity(sibships, n_candidates = 4):
    """
    Pull out the top candidates as fathers for all offspring from across
    many half-sib arrays.

    This is essentially a wrapper for sibshipCluster.paternity.
    
    Parameters
    ----------
    sibships: dict
        Dictionary of sibshipCluster objects
    progeny: dict
        Dictionary of genotypeArray objects. The order of mothers and
        offspring within each family must match the sibshipCluster object you 
        created.
    n_candidates: int
        Maximum number of top candidates to return. Defaults to the top
        four candidates.
    
    Returns
    -------
    Dataframe with a row for every offspring, showing the names and log
    posterior probabilities of paternity for the N most-probable candidates.

    Example
    -------
    import numpy as np
    import faps as fp
    # Example with multiple half-sib families
    allele_freqs = np.random.uniform(0.3,0.5,100)
    adults = fp.make_parents(100, allele_freqs)
    progeny = fp.make_offspring(parents = adults, dam_list=[0,1,7,7,7,7,7,8,8], sire_list=[7,6,2,4,4,4,4,0,3])
    mothers = adults.subset(individuals=progeny.mothers)
    patlik = fp.paternity_array(progeny, mothers, adults, mu = 0.0013, missing_parents = 0.2)
    patlik  = patlik.split(by=progeny.mothers)
    sibships = fp.sibship_clustering(patlik)

    # Pull out the top candidates
    N = 5
    sp = fp.summarise_paternity(sibships)
    """
    if not isinstance(sibships, dict):
        raise TypeError("`sibships` should be a dictionary of sibshipCluster objects")
    if not all([isinstance(v, sibshipCluster) for v in sibships.values()]):
        raise TypeError("`sibships` should be a dictionary of sibshipCluster objects")
    
    out = {}
    for k in sibships.keys():
        dat = sibships[k].paternity(n_candidates = n_candidates)
        dat.insert(loc=0, column = 'mother', value = np.repeat(k, sibships[k].noffspring))
        out[k] = dat
    out = pd.concat(out)
    out = out.reset_index().drop(labels=['level_0', 'level_1'], axis = 1)
    
    return out