import faps as fp
import numpy as np
import pytest

# Generate a population of adults
allele_freqs = np.random.uniform(0.3,0.5,50)
adults = fp.make_parents(20, allele_freqs)

# Mate the first adult to the next three.
mother = adults.subset(0)
progeny = fp.make_sibships(adults, 0, [1,2,3], 5, 'x')
# Create paternityArray
patlik = fp.paternity_array(progeny, mother, adults, mu=0.0013)

def test_paternityArray():
    # Check rows of prob array sum to 1 (to within machine error)
    p = patlik.prob_array()
    assert (fp.alogsumexp(p, axis = 1) == pytest.approx(0))

    # Check excluding some individuals
    # Set values for candidates 14 to 16 to zero.
    patlik.purge = ['base_14', 'base_15', 'base_16']
    assert (patlik.prob_array()[:, 14:17] == -np.inf).all() # probabilities for those candidates set to -inf
    assert (np.isfinite(patlik.lik_array[:14:17])).all() # lik_array remains unchanged

    # Check changing missing parents
    # By default, the final column of prob_array should be finite (previously this was -inf)
    assert (np.isfinite(patlik.prob_array()[:,-1]).all() 
    # Allow for 10% missing parents, and this should become finite
    patlik.missing_parents = 0.1
    assert np.isfinite(patlik.prob_array()[:, -1]).all()
    # Check that nonsense values raise an error
    patlik.missing_parents = 1.1
    with pytest.raises(ValueError):
        patlik.prob_array()
    patlik.missing_parents = -0.1
    with pytest.raises(ValueError):
        patlik.prob_array()
    patlik.missing_parents = 0.1
