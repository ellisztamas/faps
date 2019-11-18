import numpy as np
from faps.genotypeArray import genotypeArray
from faps.make_parents import make_parents
from faps.make_sibships import make_sibships
from faps.make_generation import make_generation
from time import time, localtime, asctime
from pandas import DataFrame as df
from sys import stdout

def make_power(replicates, nloci, allele_freqs, candidates, sires, offspring, missing_loci, mu_real, mu_input=None, unsampled_real=None, unsampled_input=None, selfing_rate=0, cluster_draws=1000, exp_clusters=True, return_paternities=False, return_clusters=False, verbose=True, progress=True):
    """
    Perform replicate simulation of half sibling arrays, and return accuracies about
    sibship reconstruction and posterior probabilties of paternity.

    This is just calls `make_generation` many times for given values or sets of values
    of parameters. `make_power` checks the input and prints a summary back to the user.
    This can be turned off by setting `verbose` to False. By default the input and
    summary statistics are returned as a Pandas data frame. You can also return a
    list of the `sibshipCluster` objects generated by setting `return clusters` to True.

    Parameters
    ----------
    replicates: int
        Number of replicate simulations to run for each set of parameters.
    nloci: int or list
        Number of loci to simulate.
    allele_freqs: float, or list
        Population allele frequencies for generating reproductive adults.
        This can be a single float for uniform allele frequencies, a list of two
        floats indicating the upper and lower bounds to draw allele frequencies from
        a uniform distribution, or a list of frequencies for each of `nloci` loci.
        Values must be between 0 and 1.
    candidates: int or list
        Adult population size.
    sires: int or list
        Number of males to sire full sibships, or a list of males. Note that the mother
        is always the first candidate in the array, and is indexed as position zero.
    offspring: int or list
        Full sibship sizes. If a single integer has been passed to `sires`,
        this must be an integer indicating full sibship size, which is constant for
        all full sibships. If a list is passed to `sires`, this should be a list of
        the same length indicating the size of each individual full-sib family.
    missing_loci: float or list
        Real diploid allelic dropout rate. Float or list of floats. Values must be
        between 0 and 1.
    mu_real: float or list
        Real haploid genotyping error rate. Float or list of floats between 0 and 1.
        In the latter case, mu_input should be the same length as mu_real.
    mu_input: float or list
        The genotyping error rate used to construct the `paternityArray`. Can be
        a single float or a list of floats values between 0 and 1. In the latter case,
        mu_input should be the same length as mu_real.
    unsampled_real: float or list, optional
        Individuals to remove from the list of candidate males. If a float between
        zero and one is given, this indicates a proportion of candidates which will
        be removed at random, reflecting incomplete sampling. If a list of integers is
        given, this indicates specific indices of candidates to be removed, such as
        a known true sire, or the mother.
    unsampled_input: float or list, optional
        Value for the proportion of unsampled fathers to be used in sibship clustering.
        Can be a single float or a list of floats. Values must be between 0 and 1.
    selfing_rate: float or list, optional
        Value for the selfing rate to be used in sibship clustering. Can be a
        single float or a list of floats. Values must be between 0 and 1.
    cluster_draws: int or list, optional
        Number of Monte Carlo draws in sibship clustering.
    exp_clusters: logical, optional
        If `True` exponentiate pairwise sibship matrix to speed up sibship clustering
        with a potential cost to accuracy. See `sibship_clustering` for details.
    return_paternities: logical, optional
        If `True` the array of paternity probabilities object is returned for each
        simulated array.
    return_clusters: bool, optional
        If true, a `sibshipCluster` object is returned for each simulated array.
    verbose: bool, optional
        If true prints output on parameter input.
    progress: bool, optional
        If True, prints a progress bar to track simulations (only tested in Jupyter).
    Returns
    -------
    A data frame listing:
        0. simulation index
        1. number of loci
        2. allele frequencies
        3. number of candidates
        4. number of sires
        5. number of offspring
        6. genotype dropout rate
        7. real genotype error rate
        8. input genotype error rate
        9. proportion of missing candidates
        10. input value for the presumed number of missing fathers
        11. input selfing rate
        12. time in seconds to create paternityArray
        13. time in seconds to perform clustering
        14. binary indiciator for whether the true partition was included in the
            sample of partitions.
        15. difference in log likelihood for the maximum likelihood partition
            identified and the true partition. Positive values indicate that the
            ML partition had greater support.
        16. posterior probability of the true number of families.
        17. mean probabilities that a pair of full sibs are identified as full sibs.
        18. mean probabilities that a pair of half sibs are identified as half sibs.
        19. mean probabilities that a pair of half or full sibs are correctly
            assigned as such.
        20. mean probability of paternity of the true sires for those sires who
            had been sampled (who had non-zero probability in the paternityArray).
        21. mean probability that the sire had not been sampled for those
            individuals whose sire was truly absent (who had non-zero probability
            in the paternityArray).

    If `return_paternities` is True, the array of paternity probabilities object
    is returned for each simulated array. If `return_clusters` is true, a list is
    returned containing the data frame described nabove and a list of sibshipClusters.

    Examples
    --------
    import numpy as np

    make_power(allele_freqs, 100, [0,2,3], [5,4,3], mu_real=0.0015, mu_input = 0.0015, unsampled_input=0.05, selfing_rate = 0.3)
    """
    if isinstance(replicates, int) and replicates > 0:
        if verbose:
            print("{} of each parameter combination will be performed.".format(replicates))
    else:
        raise TypeError("R should be a positive integer giving the number of replicate simulations to be performed.")
        return None

    if isinstance(nloci, int):
        if verbose: print("Simulating {} diploid loci.".format(nloci))
        nloci = [nloci]
    elif isinstance(nloci, list) or isinstance(nloci, np.ndarray):
        if verbose:
            print("Simulating arrays with multiple number of loci: {}.".format(nloci))
    else:
        raise TypeError("nloci should be a positive integer, or a list of positive integer.")
    if any([nloci[i] < 0 or not isinstance(nloci[i], int) for i in range(len(nloci))]):
        raise TypeError("nloci should be a positive integer, or a list of positive integer.")
    stdout.flush()

    # set up allele frequencies
    if isinstance(allele_freqs, float):
        if allele_freqs <=0 or allele_freqs >=1:
            raise ValueError("Allele frequencies must be between 0 and 1.")
        elif verbose:
            print("Uniform minor allele frequency of {}.format(allele_freqs).")
    elif isinstance(allele_freqs, list) or isinstance(allele_freqs, np.ndarray):
        if any([allele_freqs[i] < 0 and allele_freqs[i] > 1 for i in range(len(allele_freqs))]):
            raise ValueError("Allele frequencies must be between 0 and 1.")
        if len(allele_freqs) == 2 and verbose:
                print("Drawing allele frequencies between {} and {}.".format(allele_freqs[0], allele_freqs[1]))
        elif len(allele_freqs) == nloci and nloci is not 2:
            allele_freqs = allele_freqs
            if verbose:
                print("Allele frequencies supplied by the user.")
    else:
        raise ValueError("Allele frequencies must be given as a single value between 0 and 1, a list of two elements, or else a vector of frequencies for each locus.")
    stdout.flush()

    # set up population parameters
    if isinstance(candidates, int):
        if verbose is True:
            print("Simulating adult populations of {} individuals.".format(candidates))
        candidates = [candidates]
    elif isinstance(candidates, list) or isinstance(candidates, np.ndarray):
        if verbose is True:
            print("Simulating adult populations of multiple sizes: {}.".format(candidates))
    else:
        raise TypeError("candidates should be an integer or list of integers.")
    stdout.flush()

    # Generating the progeny
    if isinstance(sires, int):
        if isinstance(offspring, int):
            if verbose is True:
                print("Simulating {} families of {} offspring.".format(sires, offspring))
        else:
            raise TypeError("If a single integer is given for sires, then offspring must be a single integer.")
    elif isinstance(sires, list) or isinstance(sires, np.ndarray):
        if isinstance(offspring, list) or isinstance(offspring, np.ndarray):
            if len(offspring) != len(sires):
                raise ValueError("offspring should be the same length as sires.")
            if verbose is True:
                print("Simulating {} full-sib families.".format(len(sires)))
        else:
            raise TypeError("sires and offspring should either both be an integer, or should be lists of identical length.")
    stdout.flush()

    # dropout rates
    if isinstance(missing_loci, list) or isinstance(missing_loci, np.ndarray):
        if verbose:
            print("Multiple real genotyping error rates: {}.".format(missing_loci))
    if isinstance(missing_loci, float) or missing_loci ==0:
        if verbose:
            print("{}% of per-locus genotypes will be removed at random.".format(100*missing_loci))
        missing_loci = [missing_loci]
    else:
        raise ValueError("missing_loci should be an float or list of floats between 0 and 1.")
    stdout.flush()

    # Real genotype error rates
    if isinstance(mu_real, list) or isinstance(mu_real, np.ndarray):
        if verbose:
            print("Multiple real genotyping error rates: {}.".format(mu_real))
    elif isinstance(mu_real, float) or mu_real == 0:
        if verbose:
            print("{}% of alleles will be mutated at random.".format(mu_real*100))
        mu_real = [mu_real]
    else:
        raise ValueError("mu_real should be an integer or list of integers.")
    if any([mu_real[i] < 0 or mu_real[i] > 1 for i in range(len(mu_real))]):
        raise ValueError("All values for mu_real should be between zero and one.")
    #if len(mu_real) > 1 and len(missing_parents) > 1 and len(mu_real) > 1 != len(missing_parents) > 1:
        #print("ERROR: If multiple values are given for mu_real and missing_parents supply the same number of arguments for each.
    stdout.flush()

    # input error rates
    if isinstance(mu_input, list) or isinstance(mu_input, np.ndarray):
        if verbose:
            print("Constructing paternity arrays using multiple input values for assumed genotype-error rate: {}.".format(mu_input))
    elif isinstance(mu_input, float) or mu_input==0:
        if verbose:
            print("Genotype error rate of {} will be used to construct paternity arrays.".format(mu_input))
        mu_input = [mu_input]
    elif mu_input is None:
        mu_input = mu_real
        if verbose is True:
            print("Input error rates taken as the real error rates.")
    else:
        print("mu_input should be a float or list of floats between zero and one, or else None.")
    if len(mu_real) > 1 and len(mu_input) > 1 and len(mu_real) != len(mu_input):
        raise TypeError("If multiple values are given for mu_real and mu_input supply the same number of arguments for each.")
    if any([mu_input[i] < 0 and mu_input[i] >= 1 for i in range(len(mu_input))]) and mu_input is not None:
        raise ValueError("All elements in mu_input must be greater or equal to zero and less than one.")

    stdout.flush()

    # remove superfluous individuals
    if unsampled_real == 0 or unsampled_real == 0.0 or unsampled_real is None:
        if verbose is True:
            print("No candidates to be removed.")
    elif isinstance(unsampled_real, float):
        if unsampled_real > 0 and unsampled_real <= 1:
            if verbose is True:
                print("Removing {}% of the candidates at random.".format(np.round(unsampled_real*100, 2)))
        else:
            raise ValueError("If unsampled_real is a float it should be greater or equal to zero and less than one.")
    elif isinstance(unsampled_real, list) or isinstance(unsampled_real, np.ndarray):
        if any([not isinstance(unsampled_real[i], int) or unsampled_real[i] < 0 for i in range(len(unsampled_real))]) :
            raise TypeError("If a list is given for unsampled_real, all values should be positive integers.")
        if verbose is True:
            print("Removing candidates in positions {}.".format(unsampled_real))
    else:
        raise TypeError("unsampled_real should either be a float between zero and one, or else a list indexing candidate individuals to unsampled_real.")
    stdout.flush()

    # input candidate sampling rate
    if unsampled_input is None:
        if unsampled_real == 0 or unsampled_real is [0.0]:
            if verbose is True:
                print("Constructing paternity arrays assuming complete sampling of candidates.")
            unsampled_input = [0.0]
        if isinstance(unsampled_real, float):
            unsampled_input = [unsampled_real]
            if verbose is True:
                print("Constructing paternity arrays assuming sampling proportion is known to be {}.".format(1-unsampled_input))
        if isinstance(unsampled_real, list) or isinstance(unsampled_real, np.ndarray):
            unsampled_input = [float(len(unsampled_real)) / candidates[i] for i in range(len(candidates))]
            if verbose is True:
                print("Constructing paternity arrays assuming {} candidates are missing.".format(len(unsampled_real)))
        else:
            unsampled_input = [unsampled_input]
            print("No prior parameter used for the proportion of missing candidates.")
    elif isinstance(unsampled_input, int) and unsampled_input == -9:
        unsampled_input = [None]
        if verbose:
            print("No prior parameter used for the proportion of missing candidates.")
    else:
        if isinstance(unsampled_input, list) or isinstance(unsampled_input, np.ndarray):
            if verbose is True:
                print("Constructing paternity arrays assuming multiple values of proportions of missing candidates: {}.".format(unsampled_input))
        elif isinstance(unsampled_input, float):
            if verbose is True:
                print("Proportion missing canidates set to {}.".format(unsampled_input))
            unsampled_input = [unsampled_input]
        else:
            raise TypeError("unsampled_input should be an float or list of floats between zero and one.")
        if any([unsampled_input[i] >1 or unsampled_input[i] < 0 for i in range(len(unsampled_input))]):
            raise TypeError("unsampled_input should be an float or list of floats between zero and one.")
    stdout.flush()

    # input selfing rate
    if isinstance(selfing_rate, float) or selfing_rate == 0:
        if verbose is True:
            print("Self-fertilisation rate of {}.".format(selfing_rate))
        selfing_rate = [selfing_rate]
    elif isinstance(selfing_rate, list) or isinstance(selfing_rate, np.ndarray):
        if verbose is True:
            print("Self-fertilisation rate of {}.".format(selfing_rate))
    else:
        raise TypeError("selfing_rate should be an float or list of floats between zero and one.")
        return None
    if any([selfing_rate[i] >1 or selfing_rate[i] < 0 for i in range(len(selfing_rate))]):
        raise ValueError("selfing_rate should be an float or list of floats between zero and one.")
        return None
    stdout.flush()

    # check sibship parameters
    if isinstance(cluster_draws, int):
        if verbose is True:
            print("Performing {} Monte Carlo draws for sibship inference.\n".format(cluster_draws))
    else:
        raise TypeError("Give an integer value for cluster_draws.")
    if not isinstance(exp_clusters, bool):
        raise TypeError("exp_clusters should be logical.")
        return None
    if not isinstance(verbose, bool):
        raise TypeError("exp_clusters should be logical.")
        return None
    stdout.flush()

    if verbose: print("Parameters set. Beginning simulations on {}.".format(asctime(localtime(time()) )))
    stdout.flush()
    t0 = time()
    nreps = len(nloci)*len(candidates)*len(mu_real)*len(missing_loci)*len(unsampled_input)*len(selfing_rate)*replicates
    results  = np.zeros([nreps, 22])
    patarrays= []
    clusters = []

    counter=0

    # set up the progress bar, if required
    if progress:
        from ipywidgets import FloatProgress
        from IPython.display import display
        fp = FloatProgress(min=0, max=nreps) # instantiate the bar
        display(fp) # display the bar

    for l in nloci:
        for c in candidates:
            for mr in mu_real:
                for ml in missing_loci:
                    for th in unsampled_input:
                        for sr in selfing_rate:
                            for r in range(replicates):
                                # set up allele frequencies
                                if isinstance(allele_freqs, float):
                                    af = np.repeat(allele_freqs, l)
                                elif len(allele_freqs) == 2:
                                    af = np.random.uniform(allele_freqs[0], allele_freqs[1], l)
                                elif isinstance(allele_freqs, list) or isinstance(allele_freqs, np.ndarray):
                                    af = allele_freqs
                                # run the simulation.
                                rx = make_generation(
                                    allele_freqs= af,
                                    candidates = c,
                                    sires = sires,
                                    offspring = offspring,
                                    missing_loci = ml,
                                    mu_real = mr,
                                    mu_input = mr,
                                    unsampled_real=unsampled_real,
                                    unsampled_input= th,
                                    selfing_rate=sr,
                                    cluster_draws = cluster_draws,
                                    exp_clusters = exp_clusters,
                                    return_paternities=True,
                                    return_clusters=True,
                                    counter=counter
                                )
                                # send output to tables
                                results[counter] = rx[0]
                                if return_paternities: patarrays = patarrays + [rx[1]]
                                if return_clusters:    clusters = clusters + [rx[2]]

                                counter += 1
                                # update progress bar
                                if progress:
                                    fp.value += 1 # signal to increment the progress bar

    if verbose:
        print("Simulations completed after {} minutes.".format(round((time() - t0) / 60, 2)))

    cn =['rep','nloci','allele_freq','n_adults','n_sires','array_size',
         'missing_loci','mu_real','mu_input',
         'unsampled_real','unsampled_input','selfing_rate',
         'time_paternity','time_cluster',
         'partition_found','delta_loglik',
         'nfamilies',
         'acc_fs','acc_hs','acc_all',
         'prob_sires','prob_absent']
    if return_paternities or return_clusters:
        results = [df(results, columns=cn)]
        if return_paternities: results = results + [patarrays]
        if return_clusters:    results = results + [clusters]
        return results
    else: return df(results, columns=cn)
