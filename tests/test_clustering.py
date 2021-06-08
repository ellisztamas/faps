import numpy as np
import faps as fp

def test_clustering():
    # set up a population
    np.random.seed(867)
    allele_freqs = np.random.uniform(0.3,0.5,50)
    adults = fp.make_parents(100, allele_freqs, family_name='a')
    progeny = fp.make_sibships(adults, 0, [1,2,3], 5, 'x')
    mothers = adults.subset(progeny.mothers)
    patlik  = fp.paternity_array(progeny, mothers, adults, mu = 0.0015, missing_parents=0.01)
    # Check basic functionality
    assert isinstance(fp.sibship_clustering(patlik), fp.sibshipCluster.sibshipCluster)
    sc = fp.sibship_clustering(patlik)
    assert (sc.candidates == patlik.candidates).all()
    assert(sc.covariates ==  0)
    # make up a covariate and check it is inherited
    cov = - np.log(np.random.normal(5, size=adults.size))
    patlik.add_covariate(cov)
    sc2 = fp.sibship_clustering(patlik, use_covariates=True)
    assert patlik.covariate == sc2.covariate
