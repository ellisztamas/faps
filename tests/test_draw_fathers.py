from faps.draw_fathers import draw_fathers
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
    cov = np.arange(0,adults.size)
    cov = np.log(cov/cov.sum())
    patlik.add_covariate(cov)
    sc2 = fp.sibship_clustering(patlik, use_covariates=True)
    dr2 = fp.draw_fathers(
        sc2.mlpartition,
        genetic = sc2.paternity_array,
        covariate = sc2.covariate,
        ndraws = ndraws,
        use_covariates=True
    )
    assert isinstance(dr2, np.ndarray)
    assert len(dr2) == ndraws
    # Check that using only the covariate samples more or less at random
    dr3 = fp.draw_fathers(
        sc2.mlpartition,
        genetic = sc2.paternity_array,
        covariate = sc2.covariate,
        covariates_only = True
        )
    assert isinstance(dr3, np.ndarray)
    assert not all([x in [1,2,3] for x in dr3])
    # Remove one of the real fathers.
    # Check that he doesn't appear, but the index for missing fathers (100 in
    # this case) does.
    patlik.purge = "a_1"
    sc3 = fp.sibship_clustering(patlik)
    dr4 = fp.draw_fathers(
        sc3.mlpartition,
        genetic = sc3.paternity_array
        )
    assert 1 not in dr4
    assert 100 in dr4


    