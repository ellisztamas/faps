def sibship_transitions(families, paternity_array, male_genotypes, mu, exp=False):
    """
    Transition probabilities (really log likelihoods) that each male is the sire
    of a set of putative full sibships.

    This differs from transition_probabilities in that sibship_transitions
    multiplies over sibships *before* summing over paternal genotypes. This
    ought to be more accurate because any genotyping error in the candidate 
    fathers is not multiplied over offspring, but incurs a loss in speed of
    about one order of magnitude.

    Parameters
    ----------
    families: list of arrays
        A list of vectors, each giving the indices of one or more offspring in
        a putative full sibship. This is usually based on a set of partitions
        generated using sibship_partitions.
    paternity_array: paternityArray
        paternityArray object used to generate families.
    male_genotypes: genotypeArray
        genotypeArray object used to generate families.
    mu: float between zero and one
        Point estimate of the genotyping error rate. Clustering is unstable if
        mu_input is set to exactly zero.
    exp: boolean, optional
        If True, likelihood arrays are exponentiated. This confers a substantial
        speed boost, but is unstable if the sample includes families with very
        low likelihood, because it is necessary to multiply over many small
        numbers.

    Returns
    -------
    An array of likelihoods that each full sibship was sired by each candidate
    male. Output is not normalised over candidates.
    """
    if mu == 0:
        mu = 10**-12
        warn('Setting error rate to exactly zero causes clustering to be unstable. mu set to 10e-12')
    
    male_diploid = male_genotypes.geno.sum(2)
    # create matrix to correct for dropouts.
    drop_f = (male_diploid < 0)
    corr   = 1/(1 - drop_f.mean(1))[np.newaxis]
    # likelihood for each unique family given maternal and offspring genotypes. 
    prob_u = np.array([paternity_array.by_locus[:,up].sum(1) for up in partitions])
    # get the likelihood for each possible paternal genotype, exponentiating likelihoods.
    if exp:
        prob_f = np.zeros([len(partitions), male_genotypes.size, male_genotypes.nloci])
        prob_u = np.exp(prob_u)
        for f in [0,1,2]:
            pr_males = np.zeros(male_diploid.shape)
            pr_males[male_diploid == f] = 1-mu
            pr_males[male_diploid != f] = mu
            pr_males[drop_f] = 1
            prob_f += prob_u[:,f][:, np.newaxis] * pr_males[np.newaxis]
        with np.errstate(divide='ignore'):
            siblik = np.log(prob_f.prod(2))
    
    # get the likelihood for each possible paternal genotype, without exponentiating likelihoods.
    else:
        prob_f = np.zeros([3, len(partitions), male_genotypes.size, male_genotypes.nloci])
        for f in [0,1,2]:
            pr_males = np.zeros(male_diploid.shape)
            pr_males[male_diploid == f] = 1-mu
            pr_males[male_diploid != f] = mu
            pr_males[drop_f] = 1
            prob_f[f] = prob_u[:,f][:, np.newaxis] + np.log(pr_males)[np.newaxis]
        siblik = alogsumexp(prob_f, axis=0).sum(2)
    
    siblik = siblik + np.log(corr)
    return siblik