import faps as fp
import numpy as np
import pytest

# Generate a population of adults
allele_freqs = np.random.uniform(0.3,0.5,50)
adults = fp.make_parents(100, allele_freqs)
mother = adults.subset(0)

# Example with multiple half-sib families
progeny = fp.make_offspring(parents = adults, dam_list=[7,7,1,8,8,0], sire_list=[2,4,6,3,0,7])
# Split mothers and progeny up by half-sib array.
mothers = adults.split(by=progeny.mothers)
progeny = progeny.split(by=progeny.mothers)

def test_create_paternityArray():
    # Should work.
    # Create paternity array for a single family
    i = "base_0"
    patlik = fp.paternity_array(progeny[i], mothers[i], adults, mu=0.0013)
    assert isinstance(patlik, fp.paternityArray)
    # Create paternity array for each family using dictionaries
    patlik = fp.paternity_array(progeny, mothers, adults, mu=0.0013)
    assert isinstance(patlik, dict)
    assert all([isinstance(x, fp.paternityArray) for x in patlik.values()])

    # Should fail.
    # Single offspring, dict of mothers
    with pytest.raises(TypeError):
        fp.paternity_array(progeny[i], mothers, adults, mu=0.0013)
