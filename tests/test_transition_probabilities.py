import faps as fp
import numpy as np

from faps.paternity_array import paternity_array

def test_transition_probabilities():
    # Generate a population of adults
    allele_freqs = np.random.uniform(0.3,0.5,50)
    adults = fp.make_parents(100, allele_freqs)
    progeny = fp.make_offspring(parents = adults, dam_list=[7,7,1,8,8,0], sire_list=[2,4,6,3,0,7])
    # add genotyping errors
    mu = 0.0015
    progeny = progeny.mutations(mu)
    adults  = adults.mutations(mu)

    # Split mothers and progeny up by half-sib array.
    mothers = adults.subset(progeny.mothers)


    tp = fp.transition_probability(
        progeny,
        mothers,
        adults,
        mu = mu
    )

    assert isinstance(tp, list)
    assert isinstance(tp[0], np.ndarray)
    assert isinstance(tp[1], np.ndarray)
    assert tp[0].shape == (progeny.size, adults.size)
    assert tp[1].shape == (progeny.size, )
    assert all(tp[0].max(axis = 1) > tp[1])