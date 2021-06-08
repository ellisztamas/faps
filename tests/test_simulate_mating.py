from faps.sibshipCluster import sibshipCluster
import numpy as np
import pandas as pd
import faps as fp

ndraws=1000
np.random.seed(867)

def test_method():
    # Simulate a starting population
    allele_freqs = np.random.uniform(0.3,0.5,50)
    adults = fp.make_parents(100, allele_freqs, family_name='a')
    progeny = fp.make_sibships(adults, 0, [1,2,3], 5, 'x')
    mothers = adults.subset(progeny.mothers)
    patlik  = fp.paternity_array(progeny, mothers, adults, mu = 0.0015, missing_parents=0.01)
    sc = fp.sibship_clustering(patlik)
    # Check posterior_mating returns what it should in ideal case
    me = sc.posterior_mating()
    assert isinstance(me, pd.DataFrame)
    assert me['posterior_probability'].sum() == 1.0
    assert all([x in sc.candidates for x in me['father']])
    # Remove one of the fathers and check that a missing dad is sampled.
    patlik.purge = "a_1"
    sc2 = fp.sibship_clustering(patlik)
    me2 = sc2.posterior_mating()
    assert isinstance(me2, pd.DataFrame)
    assert me2['father'].isin(["missing"]).any()
    # Include a nonsense covariate
    cov = np.arange(0,adults.size)
    cov = -cov/cov.sum()
    patlik.add_covariate(cov)
    sc3 = fp.sibship_clustering(patlik, use_covariates=True)
    me3 = sc3.posterior_mating(use_covariates=True)
    assert isinstance(me3, pd.DataFrame)
    assert me3['posterior_probability'].sum() == 1.0
    # Check that 
    sc4 = fp.sibship_clustering(patlik, use_covariates=True)
    me4 = sc4.posterior_mating(use_covariates=True, covariates_only=True)
    assert isinstance(me3, pd.DataFrame)
    assert me3['posterior_probability'].sum() == 1.0

# Generate a population of adults
allele_freqs = np.random.uniform(0.3,0.5,50)
adults = fp.make_parents(20, allele_freqs)
# Example with multiple half-sib families
progeny = fp.make_offspring(parents = adults, dam_list=[7,7,1,8,8,0], sire_list=[2,4,6,3,0,7])
# Split mothers and progeny up by half-sib array.
mothers = adults.split(by=progeny.mothers, return_dict=True)
progeny = progeny.split(by=progeny.mothers, return_dict=True)
# Create paternity array for each family using dictionaries
patlik = fp.paternity_array(progeny, mothers, adults, mu=0.0013, integration='partial')
# The dictionary is passed to sibship_clustering.
sibships = fp.sibship_clustering(patlik)

# Note: this is returning the wrong fathers!
me = fp.posterior_mating(sibships)

me

sibships['base_7'].posterior_mating()