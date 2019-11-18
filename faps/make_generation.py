from time import time, localtime, asctime
import numpy as np
from faps.make_parents import make_parents
from faps.make_sibships import make_sibships
from faps.paternityArray import paternityArray
from faps.paternity_array import paternity_array
from faps.sibshipCluster import sibshipCluster
from faps.sibship_clustering import sibship_clustering
from faps.lik_partition import lik_partition
from faps.relation_matrix import relation_matrix

def make_generation(allele_freqs, candidates, sires, offspring, missing_loci=0, mu_real=0, mu_input=0, unsampled_real=None, unsampled_input=0, selfing_rate=0, cluster_draws=1000, exp_clusters=True, return_paternities=False, return_clusters=False, counter='NA'):
    """
    Accuracy of sibship and paternity assignment for a single simulated half-sibling array.

    Simulates a population of adults and generates a half-sibling array. Returns information
    on whether the true partition was among those identified by clustering, accuracy of
    sibling relationships and posterior probabilities of paternity.

    Parameters
    ----------
    allele_freqs: list
        Vector of allele frequencies.
    candidates: int
        Adult population size.
    sires: int or list
        Number of males to sire full sibships, or a list of males. The mother
        is always the first candidate in the array, and is indexed as position zero.
    offspring: int or list
        Full sibship sizes. If a single integer has been passed to `sires`,
        this must be an integer indicating full sibship size, which is constant for
        all full sibships. If a list is passed to `sires`, this should be a list of
        the same length indicating the size of each individual full-sib family.
    missing_loci: float
        Real diploid allelic dropout rate between 0 and 1.
    mu_real: float
        Real haploid genotyping error rate between 0 and 1.
    mu_input: float
        The genotyping error rate used to construct the `paternityArray`. Values must be
        between 0 and 1.
    unsampled_real: float or list
        Individuals to remove from the list of candidate males. If a float between
        zero and one is given, this indicates a proportion of candidates which will
        be removed at random, reflecting incomplete sampling. If a list of integers is
        given, this indicates specific indices of candidates to be removed, such as
        a known true sire, or the mother.
    unsampled_input: float or list
        Value for the proportion of unsampled fathers to be used in sibship clustering.
        Values must be between 0 and 1.
    selfing_rate: float
        Value for the selfing rate to be used in sibship clustering. Values must be
        between 0 and 1.
    cluster_draws: int, optional
        Number of Monte Carlo draws in sibship clustering.
    exp_clusters: logical, optional
        If `True` exponentiate pairwise sibship matrix to speed up sibship clustering
        with a potential cost to accuracy. See `sibship_clustering` for details.
    return_paternities: logical, optional
        If `True` the array of paternity probabilities object is returned for each
        simulated array.
    return_clusters: bool, optional
        If `True`, a `sibshipCluster` object is returned for each simulated array.
    verbose: bool, optional
        If true prints output on parameter input and simulation progress.

    Returns
    -------
    An array listing:
        0. simulation index (when called from `make_power`)
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

    If return_clusters is true, a list is returned containing the array described above and the sibshipCluster.

    Examples
    --------
    import numpy as np

    allele_freqs = np.random.uniform(0.3,0.5,80)
    make_generation(allele_freqs, 100, [0,2,3], [5,4,3], mu_real=0.0015, mu_input = 0.0015, unsampled_real=0.05, selfing_rate = 0.3)
    """
    # simulate population
    adults = make_parents(candidates, allele_freqs) # parents
    # Generate the progeny
    if isinstance(sires, int):
        progeny = make_sibships(adults, 0, np.arange(1,sires+1), offspring)
    elif isinstance(sires, list) or isinstance(sires, np.ndarray):
        progeny = make_sibships(adults, 0, sires, offspring)
    # introduce genotyping errors
    adults = adults.mutations(mu_real).dropouts(missing_loci)
    progeny= progeny.mutations(mu_real).dropouts(missing_loci)
    # reference mothers
    mothers = adults.subset(progeny.parent_index('m', adults.names))

    # Paternity array
    t1      = time()
    patlik  = paternity_array(progeny, mothers, adults, mu_input)
    patlik.prob_array = patlik.adjust_prob_array(unsampled_real, unsampled_input, selfing_rate)
    #Sibship clustering
    t2      = time()
    sc      = sibship_clustering(
                  paternity_array = patlik,
                  ndraws = cluster_draws,
                  exp=exp_clusters,
                  use_covariates=False)
    t3      = time()
    # Determine whether the true partition is among those inferred by clustering.
    # For each partition count up how many pairwise relationships were inferred correct.
    true_part  = progeny.true_partition()
    nmatches   = np.array([(relation_matrix(sc.partitions[x]) == relation_matrix(true_part)).sum()
                           for x in range(sc.npartitions)])
    nmatches   = 1.0*nmatches / true_part.shape[0]**2 # divide by matrix size.
    true_found = int(1 in nmatches) # return 1 if the true partition is in sc.partitions, otherwise zero

    # value to return for the proportion of missing fathers
    if   isinstance(unsampled_real, float): sprop = unsampled_real
    elif isinstance(unsampled_real, list):  sprop = float(len(unsampled_real)) / candidates
    else: sprop = 0

    # a list of simulation data to return.
    output = np.array([counter,
                len(adults.allele_freqs()), # number of loci
                round(adults.allele_freqs().mean(),3), #allele_freqs
                candidates, # number of candidates
                len(np.unique(progeny.fathers)), # sires
                progeny.size, # offspring
                missing_loci,
                mu_real,
                mu_input,
                sprop, # proportion missing fathers
                unsampled_input,
                selfing_rate,
                # timing
                round(t2-t1,3), # time in seconds to create paternityArray
                round(t3-t2,3)  # time in seconds to perform clustering
                ])
    output = np.append(output, sc.accuracy(progeny, adults))

    outlist = [output]
    if return_paternities: outlist = outlist + [patlik]
    if return_clusters:    outlist = outlist + [sc]
    return outlist
