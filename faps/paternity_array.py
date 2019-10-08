import numpy as np
from faps.paternityArray import paternityArray
from faps.genotypeArray import genotypeArray
from faps.transition_probability import transition_probability
from faps.incompatibilities import incompatibilities
from warnings import warn

def paternity_array(offspring, mothers, males, mu, purge=None, missing_parents=None, selfing_rate=None, max_clashes=None, covariate=None):
    """
    Construct a paternityArray object for the offspring given known mothers
    and a set of candidate fathers using genotype data. Currently only SNP
    data is supported.

    Additional information about paternity from non-genetic sources can be
    provided through the argument `covariate`. For example, this might be a
    function of spatial distance between individuals, or social dominance.

    Parameters
    ---------
    offspring: genotypeArray, or dictionary of genotypeArrays
        Observed genotype data for one or more arrays of offspring in a
        single maternal family. To analyse multiple half-sib arrays in parallel,
        give a dictionary containing multiple named genotypeArray objects.
    mothers: genotypeArray, or dictionary of genotypeArrays
        Observed genotype data for the mother of the half-sib array. If multiple
        families of offspring are given, `mothers` should be a dictionary of
        genotypeArray objects with the same key names as for the offspring.
    males: genotypeArray
        Observed genotype data for the candidate males.
    mu: float between zero and one
        Point estimate of the genotyping error rate. Clustering is unstable if
        mu_input is set to exactly zero. Any zero values will therefore be set
        to a very small number close to zero (10^-12).
    purge: float between zero or one, int, array-like, optional
        Individuals who can be removed from the paternity array a priori. If
        a float is given, that proportion of individuals is removed from the
        array at random. Alternatively an integer or vector of integers
        indexing specific individuals can be supplied.
    missing_parents : float between zero and one, or 'NA', optional
        Input value for the proportion of adults who are missing from the sample.
        This is used to weight the probabilties of paternity for each father
        relative to the probability that a father was not sampled. If this is
        given as 'NA', no weighting is performed.
    selfing_rate: float between zero and one, optional
        Input value for the prior probability of self-fertilisation.
    max_clashes: int, optional
        Maximum number of opposing homozygous loci for each parent-offspring.
    covariate: 1-d array, or list of 1-d arrays, optional
        Vector of (log) probabilities of paternity based on non-genetic
        information, with one element for every candidate father. If this is a
        function of multiple sources they should be multiplied and included in
        this vector. If a dictionary for multiple half-sib arrays for have been
        supplied for offspring genotypes, `covariate` should likewise be a
        dictionary with and entry for each half-sib family and key names that
        match those given for the offspring.

    Returns
    -------
    A paternityArray, or a list of paternityArray objects.

    If covariate is not given this will be returned as a vector of zeros.

    Examples
    --------
    # Generate a population of adults
    allele_freqs = np.random.uniform(0.3,0.5,50)
    adults = make_parents(100, allele_freqs)

    # Mate the first adult to the next three.
    mother = adults.subset(0)
    progeny = make_sibships(males, 0, [1,2,3], 5, 'x')
    # Create paternityArray
    patlik = paternity_array(progeny, mother, adults, mu=0.0013)

    # Example with multiple half-sib families
    progeny = make_offspring(parents = adults,
                             dam_list=[7,7,1,8,8,0],
                             sire_list=[2,4,6,3,0,7])
    # Split mothers and progeny up by half-sib array.
    mothers = adults.split(progeny.mothers)
    progeny = progeny.split(progeny.mothers)
    # Create paternity array for each family
    paternity_array(progeny, mothers, adults, mu = 0.0013)
    """
    if mu == 0:
        mu = 10**-12
        warn('Setting error rate to exactly zero causes clustering to be unstable. mu set to 10e-12')

    #If a single halfsib family is given.
    if isinstance(offspring, genotypeArray) & isinstance(mothers, genotypeArray):
        # array of opposing homozygous genotypes.
        incomp = incompatibilities(males, offspring)
        # take the log of transition probabilities, and assign dropout_masks.
        prob_f, prob_a = transition_probability(offspring, mothers, males, mu)
        output = paternityArray(likelihood=prob_f,
                                lik_absent=prob_a,
                                offspring=offspring.names,
                                mothers=offspring.mothers,
                                fathers=offspring.fathers,
                                candidates=males.names, mu=mu,
                                purge=purge,
                                missing_parents=missing_parents,
                                selfing_rate=selfing_rate,
                                clashes=incomp,
                                max_clashes=max_clashes)

        if covariate is not None:
            output.add_covariate(covariate)

        return output




    # If a list of genotype arrays for separate halfsib families are given:
    elif isinstance(offspring, list) & isinstance(mothers, list):
        if len(offspring) != len(mothers):
            raise ValueError('Lists of genotypeArrays are of different lengths.')

        # Set up input of covariates if necessary.
        if isinstance(covariate, list):
            if len(offspring) != len(covariate):
                raise ValueError("If a list of arrays of probabilities for covariates are supplied, this should have a row for every offspring genotypeArray.")
            cov = covariate
        elif covariate is None:
            cov = np.zeros(males.size * len(offspring)).reshape([len(offspring), males.size])
        else:
            raise TypeError("If covariates are supplied for multiple half-sib arrays, this should be a list of 1-d vectors, each with an element for each candidate.")


        output = [None] * len(offspring)
        for i in range(len(offspring)):
            # array of opposing homozygous genotypes.
            incomp = incompatibilities(males, offspring[i])
            # take the log of transition probabilities, and assign dropout_masks.
            prob_f, prob_a = transition_probability(offspring[i], mothers[i], males, mu)
            # create paternityArray and send to output
            patlik = paternityArray(likelihood=prob_f,
                                    lik_absent=prob_a,
                                    offspring=offspring[i].names,
                                    mothers=offspring[i].mothers,
                                    fathers=offspring[i].fathers,
                                    candidates=males.names,
                                    mu=mu,
                                    purge=purge,
                                    missing_parents=missing_parents,
                                    selfing_rate=selfing_rate,
                                    clashes=incomp,
                                    max_clashes=max_clashes)

            patlik.add_covariate(cov[i])
            output[i] = patlik


        return output

    # If a dictionary of genotype arrays for separate halfsib families are given:
    elif isinstance(offspring, dict) & isinstance(mothers, dict):
        if offspring.keys() != mothers.keys():
            raise ValueError('Dictionaries of genotypeArray objects for offspring and mothers do not match')

        # Set up input of covariates if necessary.
        if isinstance(covariate, dict):
            if offspring.keys() != covariate.keys():
                raise ValueError("Key names of genotypeArrays of offspring genotypes and covariates do not match.")
            cov = covariate
        elif covariate is None:
            # Create a dictionary of zero vectors.
            cov = {k: np.zeros(males.size) for k in offspring.keys()}
        else:
            raise TypeError("If offspring and maternal genotypes are supplied as dictionaries, covariates should either be a 1-d of log probabilities, or a dictionary with the same keys as those for genotypes.")

        output = {}
        for i in offspring.keys():
            # array of opposing homozygous genotypes.
            incomp = incompatibilities(males, offspring[i])
            # take the log of transition probabilities, and assign dropout_masks.
            prob_f, prob_a = transition_probability(offspring[i], mothers[i], males, mu)
            # create paternityArray and send to output
            patlik = paternityArray(likelihood=prob_f,
                                    lik_absent=prob_a,
                                    offspring=offspring[i].names,
                                    mothers=offspring[i].mothers,
                                    fathers=offspring[i].fathers,
                                    candidates=males.names,
                                    mu=mu,
                                    purge=purge,
                                    missing_parents=missing_parents,
                                    selfing_rate=selfing_rate,
                                    clashes=incomp,
                                    max_clashes=max_clashes)
            patlik.add_covariate(cov[i])
            output[i] = patlik
        return output

    else:
        raise TypeError("offspring and mothers should be genotypeArray objects for a single half-sib array and its mother, or else lists or dictionaries of half-sibships and mothers.")
