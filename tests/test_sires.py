from faps.sibshipCluster import sibshipCluster
import numpy as np
import pandas as pd
import faps as fp

ndraws=1000
np.random.seed(867)

allele_freqs = np.random.uniform(0.3,0.5,50)
adults = fp.make_parents(100, allele_freqs, family_name='a')
def test_sires():
    # Example with a single family
    progeny = fp.make_sibships(adults, 0, [1,2,3], 5, 'x')
    mothers = adults.subset(progeny.mothers)
    patlik  = fp.paternity_array(progeny, mothers, adults, mu = 0.0015, missing_parents=0.01)
    sc = fp.sibship_clustering(patlik)
    me = sc.sires()
    assert isinstance(me, pd.DataFrame)
    list(me['label'])

def test_summarise_sires():
    # Example with multiple half-sib families
    progeny = fp.make_offspring(parents = adults, dam_list=[7,7,7,7,7,1,8,8,0], sire_list=[2,4,4,4,4,6,3,0,7])
    mothers = adults.subset(individuals=progeny.mothers)
    patlik = fp.paternity_array(progeny, mothers, adults, mu = 0.0013 )
    patlik  = patlik.split(by=progeny.mothers)
    sibships = fp.sibship_clustering(patlik)
    me2 = fp.summarise_sires(sibships)
    assert (me2['father'].isin(progeny.fathers)).all()
    # Remove a father
    patlik = fp.paternity_array(progeny, mothers, adults, mu = 0.0013, missing_parents=0.2, purge="base_4")
    patlik  = patlik.split(by=progeny.mothers)
    sibships = fp.sibship_clustering(patlik)
    me3 = fp.summarise_sires(sibships)
    assert me3['father'].str.contains("base_7").any()
    assert not "base_4" in me3['father']