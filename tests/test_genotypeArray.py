import faps as fp
import numpy as np
import pytest

# Generate a population of adults
allele_freqs = np.random.uniform(0.3,0.5,50)
adults = fp.make_parents(100, allele_freqs)
# Change marker names to arbitrary strings
adults.markers = np.array(['m' + str(x) for x in adults.markers])

def test_subsetting_genotypeArray():
    # Subsetting genotype arrays.
    # Check adults.subsetting adults
    x = ['base_' + str(i) for i in range(10)]
    assert all(adults.subset(individuals=adults.names[:10]).names == x)
    assert all(adults.subset(individuals= range(10)).names == x)
    assert adults.subset(individuals= 10).names == ["base_10"]
    assert adults.subset(individuals= 'base_10').names == ["base_10"]
    with pytest.raises(IndexError):
        adults.subset(individuals= 'nobody').names # fails correctly

    # Check adults.subsetting by locus.
    m = ['m' + str(i) for i in range(10)]
    assert all(adults.subset(loci=adults.markers[:10]).markers == m)
    assert all(adults.subset(loci= range(10)).markers == m)
    assert adults.subset(loci= 10).markers == "m10"
    assert adults.subset(loci= 'm0').markers == "m0"
    with pytest.raises(IndexError):
        adults.subset(loci= 'fails').markers # fails correctly

