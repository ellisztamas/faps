import numpy as np
import faps as fp

def test_draw_fathers():
    np.random.seed(867)
    allele_freqs = np.random.uniform(0.3,0.5,50)
    adults = fp.make_parents(100, allele_freqs, family_name='a')
    progeny = fp.make_sibships(adults, 0, [1,2,3], 5, 'x')
    mothers = adults.subset(progeny.mothers)
    patlik  = fp.paternity_array(progeny, mothers, adults, mu = 0.0015, missing_parents=0.01)
    sc = fp.sibship_clustering(patlik)

    ndraws=1000
    dr = fp.draw_fathers(
        sc.mlpartition,
        genetic = sc.paternity_array,
        ndraws=ndraws)
    assert isinstance(dr, np.ndarray)
    assert len(dr) == ndraws
    assert all([x in [1,2,3] for x in dr])

    # Add a nonsense covariate
    cov = -np.log(np.random.normal(5, size=adults.size+1))
    patlik.add_covariate(cov)
    sc2 = fp.sibship_clustering(patlik)
    dr2 = fp.draw_fathers(sc.mlpartition, genetic = sc.paternity_array, covariate = cov)
    assert isinstance(dr2, np.ndarray)
    assert len(dr2) == ndraws
    # Check that using only the covariate samples more or less at random
    dr3 = fp.draw_fathers(
        sc.mlpartition,
        genetic = sc.paternity_array,
        covariate = cov,
        covariate_only = True)
    assert isinstance(dr3, np.ndarray)
    assert not all([x in [1,2,3] for x in dr3])