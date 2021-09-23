from warnings import warn
import pandas as pd

from faps.sibshipCluster import sibshipCluster

def posterior_mating(sibships, ndraws=10000, use_covariates=True, covariates_only=False, down_sample = True):
    """
    Simulate plausible mating events from the posterior distribution of all possible
    pairings between mothers and candidate fathers, integrating over uncertainty
    in sibship structure.

    This calls the method `posterior_mating` defined for individual 
    `sibshipCluster` objects over a dictionary of `sibshipCluster` objects.

    For each individual partitition structure, `simulate_mating` draws a sample 
    of putative fathers for each full-sib family in propotion to posterior 
    probabilities of paternity from genetic information. Paternity information 
    from covariate probabilities can be incorporated as well by setting 
    `use_covariates` to `True`. Covariate data are taken from the sibshipCluster
     object directly (see `sibshipCluster.add_covariate`).
    
    Samples are drawn for every partition structure, excluding partitions with zero 
    posterior probability. Samples for each partition are then subsampled in 
    proportion to the posterior probability of each partition to generate a final
    sample of plausible fathers for the whole half-sib family. In this way, fathers
    are drawn in proportion to their probability of paternity, and uncertainty in
    sibship structure is accounted for.

    Samples of fathers can also be drawn in proportion to covariate probabilities
    only by setting `covariates_only` to `True`. This can be used to compare "real"
    mating events inferred using genetic (potentially including covariate information)
    to a null distribution of mating based on covariates only. For example, if 
    covariates describe a model of dispersal, such a comparison might tell you if
    there is non-random mating for some trait other than distance.

    Parameters
    ----------
    ndraws: int
        Number of Monte Carlo draws for each family.
    use_covariates: logical, optional
        If True, information on prbabilities associated with covariates stored
        in paternityArray objects are incorporated into weights for drawing likely
        fathers.
    covariate_only: boolean, optional
        If True, candidates are drawn based on covariate probabilities only 
        (i.e. ignoring genetic data)
    down_sample: boolean, optional
        If True, downsample the output so that there are as many mating
        events as there would be in reality, by downsampling mating events
        to the most-likely number of full sibships in the array. If False,
        ndrawds candidates are returned, which are likely to include most 
        of the candidates in the sample at least once.

    Returns
    -------
    A DataFrame listing plausible mating events between each mother and 
    compatible fathers, plus:

    * `total_offspring`: Number of offspring genotyped for the half-sib array
    * `n_mating_events`: Number of mating events in the half-sib family
      (mean number of full sibships in each partition, weight by the probability
      of each partition)
    * `frequency`: The contribution of each father to each half-sib family.
    * `offspring_sired`: The contribution of each father to the overall pollen pool
      (the product of `frequency` and `n_mating_events`)

    Examples
    --------
    """
    if not isinstance(sibships, dict) or not all([isinstance(x, sibshipCluster) for x in sibships.values()]):
        raise TypeError('sibships should be a dictionary of sibshipCluster objects.')
    elif len(sibships) == 1:
        warn('Lists of sibshipCluster and paternityArray are of length 1. If there is only one array to analyse it is cleaner to call mating_events dircetly from the sibshipCluster object.')

    sims = {}
    for k in sibships.keys():
        this_family = sibships[k].\
            posterior_mating(
                ndraws=ndraws,
                use_covariates=use_covariates,
                covariates_only=covariates_only,
                down_sample = down_sample
                )
        this_family = pd.DataFrame(this_family)
        this_family['mother'] = k
        this_family['total_offspring']       = sibships[k].noffspring
        this_family['n_mating_events']   = sibships[k].mean_nfamilies()
        sims[k] = this_family
    sims = pd.concat(sims).reset_index()
    sims['offspring_sired'] = sims['frequency'] * sims['total_offspring']
    sims = sims[['mother', 'father', 'total_offspring', 'n_mating_events', 'frequency', 'offspring_sired']]

    return sims