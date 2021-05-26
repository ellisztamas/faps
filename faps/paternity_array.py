import numpy as np
from faps.paternityArray import paternityArray
from faps.genotypeArray import genotypeArray
from faps.transition_probability import transition_probability
from faps.incompatibilities import incompatibilities
from warnings import warn

def paternity_array(offspring, mothers, males, mu=1e-12, missing_parents = 0, purge=None, selfing_rate=None, max_clashes=None, covariate=None, integration="full"):
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
        to a very small number close to zero (1e-12). Defaults to 1e-12
    missing_parents : float between zero and one
        Input value for the proportion of adults who are missing from the sample.
        This is used to weight the probabilties of paternity for each father
        relative to the probability that a father was not sampled. Defaults to
        zero, but you should definitely change this.
    purge: vector of str, or float between zero and one, optional
        Individuals who can be removed from the paternity array a priori. If
        a float is given, that proportion of individuals is removed from the
        array at random. Alternatively a string or vector of strings containing
        names of specific candidate fathers can be supplied.
    selfing_rate: float between zero and one, optional
        Input value for the prior probability of self-fertilisation.
    max_clashes: int, optional
        Maximum number of double-homozygous incompatibilities allowed between
        offspring and candidate fathers. Dyads with more incompatibilities than
        this will have their probability set to zero.
    covariate: 1-d array, or list of 1-d arrays, optional
        Vector of (log) probabilities of paternity based on non-genetic
        information, with one element for every candidate father. If this is a
        function of multiple sources they should be multiplied and included in
        this vector. If a dictionary for multiple half-sib arrays for have been
        supplied for offspring genotypes, `covariate` should likewise be a
        dictionary with and entry for each half-sib family and key names that
        match those given for the offspring.
    integration: str, optional
        From version 2.5 onwards, transition probabilities are calculated by
        integrating fully over all possible offspring, maternal and paternal
        genotypes. For backwards compatibility, the previous implementation
        can be accessed by setting integration to 'partial'.

    Returns
    -------
    A paternityArray, or a list of paternityArray objects.

    If covariate is not given this will be returned as a vector of zeros.

    Examples
    --------
    import faps as fp
    import numpy as np

    # Generate a population of adults
    allele_freqs = np.random.uniform(0.3,0.5,50)
    adults = fp.make_parents(20, allele_freqs)

    # Mate the first adult to the next three.
    mother = adults.subset(0)
    progeny = fp.make_sibships(adults, 0, [1,2,3], 5, 'x')
    # Create paternityArray
    patlik = fp.paternity_array(
        progeny,
        mother,
        adults,
        mu=0.0013,
        missing_parents=0.2
    )
    # View posterior probabilities of paternity (G matrix in the FAPS paper)
    patlik.prob_array()

    # Change some parameters that will affect the G matrix.
    patlik.missing_parents = 0.3
    patlik.selfing_rate = 0
    # Explicitly set some candidates to have probabilities of paternity to zero
    patlik.purge = 'base_0' # Another way to remove the mother
    patlik.purge = ['base_15', 'base_16', 'base_17'] # remove a list of specific candidates
    patlik.purge = 0.2 # Throw two candidates out at random

    # Another way to do the previous steps in a direct call to paternity_array.
    fp.paternity_array(
        progeny,
        mother,
        adults,
        mu=0.0013,
        missing_parents=0.2,
        selfing_rate=0,
        purge = 0.2
    )
    """
    if mu == 0:
        mu = 1e-12
        warn('Setting error rate to exactly zero causes clustering to be unstable. mu set to 10e-12')

    # Check missing_parents
    if not isinstance(missing_parents, (int, float)):
        raise TypeError("missing_parents should be between 0 and 1.")
    if missing_parents < 0 or missing_parents >1:
        raise ValueError("missing_parents must be between 0 and 1!")
    if missing_parents ==1:
        warn("Missing_parents set to 100%. Are you sure this is what you mean?")
    # Check selfing_rate
    if selfing_rate is not None:
        if not isinstance(selfing_rate, (int, float)):
            raise TypeError("selfing_rate should be between 0 and 1.")
        if selfing_rate < 0 or selfing_rate >1:
            raise ValueError("selfing_rate must be between 0 and 1.")
        if selfing_rate == 1: warn("selfing_rate set to 100%. Are you sure that is what you meant?")
    # Check purge
    if purge is not None:
        if isinstance(purge, float):
            if purge <= 0 or purge >= 1:
                raise ValueError("purge must be between zero and one.")
        # If one or more integers is given, remove candidates at those indices
        elif isinstance(purge, (list, np.ndarray, int, str)):
            # If all entries are strings, find the names of the candidates.
            if all([isinstance(x, str) for x in purge]):
                if not all(np.atleast_1d(np.asarray(np.isin(purge, males.names)))):
                    raise ValueError("One or more names in paternityArray.purge are not found in paternityArray.candidates")
        else:
            raise TypeError("Error: purge should be a float or list of floats between zero and one.")

    #If a single halfsib family is given.
    if isinstance(offspring, genotypeArray) & isinstance(mothers, genotypeArray):
        # array of opposing homozygous genotypes.
        incomp = incompatibilities(males, offspring)
        # take the log of transition probabilities, and assign dropout_masks.
        prob_f, prob_a = transition_probability(
            offspring, mothers, males, mu, integration=integration
            )
        output = paternityArray(
            likelihood=prob_f,
            lik_absent=prob_a,
            offspring=offspring.names,
            mothers=offspring.mothers,
            fathers=offspring.fathers,
            candidates=males.names,
            mu=mu,
            purge=purge,
            missing_parents=missing_parents,
            selfing_rate=selfing_rate,
            clashes=incomp,
            max_clashes=max_clashes,
            )

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
            prob_f, prob_a = transition_probability(
                offspring[i], mothers[i], males, mu,integration=integration
                )
            # create paternityArray and send to output
            patlik = paternityArray(
                likelihood=prob_f,
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
                max_clashes=max_clashes
            )

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
            prob_f, prob_a = transition_probability(
                offspring[i], mothers[i], males, mu, integration=integration
                )
            # create paternityArray and send to output
            patlik = paternityArray(
                likelihood=prob_f,
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
