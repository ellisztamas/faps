import numpy as np
import faps as fp
import pandas as pd

# set up a population
np.random.seed(867)
allele_freqs = np.random.uniform(0.3,0.5,50)
adults = fp.make_parents(100, allele_freqs, family_name='a')
progeny = fp.make_sibships(adults, 0, [1,2,3], 5, 'x')
mothers = adults.subset(progeny.mothers)
patlik  = fp.paternity_array(progeny, mothers, adults, mu = 0.0015, missing_parents=0.01)

def test_clustering():
    # Check basic functionality
    assert isinstance(fp.sibship_clustering(patlik), fp.sibshipCluster)
    sc = fp.sibship_clustering(patlik)
    assert (sc.candidates == patlik.candidates).all()
    assert(sc.covariate ==  0)
    # make up a covariate and check it is inherited
    cov = - np.log(np.random.normal(5, size=adults.size))
    patlik.add_covariate(cov)
    sc2 = fp.sibship_clustering(patlik, use_covariates=True)
    assert (patlik.covariate == sc2.covariate).all()

def test_basic_methods():
    """
    Check functionality of the very simple methods for sibshipCluster objects
    """
    sc = fp.sibship_clustering(patlik)
    assert (sc.accuracy(progeny, adults) == np.array([1, 0, 1,1,1,1,1,0])).all()
    assert len(sc.nfamilies()) == progeny.size

def test_sibshipCluster_paternity():
    """
    Check functionality of sibshipCluster.paternity()
    """
    # Example with a single family
    allele_freqs = np.random.uniform(0.3,0.5,100)
    adults = fp.make_parents(100, allele_freqs)
    progeny = fp.make_sibships(adults, 0, [1,2,3], 5, 'x')
    mothers = adults.subset(progeny.mothers)
    patlik  = fp.paternity_array(progeny, mothers, adults, mu = 0.0015, missing_parents=0.1)
    sc = fp.sibship_clustering(patlik)
    # Pull out the top candidates.
    N = 4
    p = sc.paternity(n_candidates = N)
    # Check function returns a dataframe of the correct shape
    assert isinstance(p, pd.DataFrame)
    assert p.shape == (len(progeny.names), 2*N + 1)
    # Check it found the real fathers
    assert all(p['candidate_1'] == progeny.fathers)
    # Check that the next candidate is 'missing'
    assert all(p['candidate_2'] == "missing")

def test_sires():
    """
    Check functionality of sibshipCluster.sires()
    """
    allele_freqs = np.random.uniform(0.3,0.5,50)
    adults = fp.make_parents(100, allele_freqs, family_name='a')
    # Example with a single family
    progeny = fp.make_sibships(adults, 0, [1,2,3], 5, 'x')
    mothers = adults.subset(progeny.mothers)
    patlik  = fp.paternity_array(progeny, mothers, adults, mu = 0.0015, missing_parents=0.01)
    sc = fp.sibship_clustering(patlik)
    me = sc.sires()
    assert isinstance(me, pd.DataFrame)
    list(me['label'])
