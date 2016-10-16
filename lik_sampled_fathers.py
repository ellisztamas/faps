import numpy as np

def lik_sampled_fathers(offspring, mothers, males, mu, mother_index=None):
    """
    Calculate a matrix of multilocus transition probabilities for all possible triplets
    of offspring, their known mothers, and a set of candidate males. This sums over all
    possible offspring and parental genotypes.
    
    ARGUMENTS:
    offspring: A genotypeArray of observed genotype data for the offspring.

    mothers: an array of observed genotype data for the mothers. Data on mothers need
        to be in the same order as those for the offspring. If a whole parent array is
        used, this can be subsetted into the correct order using the option
        mother_index.
    
    fathers: A genotypeArray of observed genotype data for the candidate males.
    
    mu: point estimate of the genotyping error rate.
    
    mother_index: an optional list of integers indexing the position of the mother of
        each offspring in the maternal_genotypes array. If left blank, the whole array
        is used.
        
    threshold: the maximum value of the product of the numbers of loci, offspring and
        candidate males to be considered before switching to looping over loci.
        
    verbose: if True, print warnings about the memory threshold.
    
    RETURNS:
    A matrix of log likelihoods of paternity, with a row for each offspring and a
    column for each candidate male.
    """
    # If a list of indices for the mothers has been given, subset the maternal data
    if mother_index is not None:
        mothers = mothers.subset(mother_index)
    
    # Diploid genotypes of each dataset.
    male_diploid = males.geno.sum(2)
    moth_diploid = mothers.geno.sum(2)
    offs_diploid = offspring.geno.sum(2)
    
    # empty array to store probabilities.
    lik_trans = np.zeros([offspring.size, males.size, males.nloci])
    # loop over all possible gemotype combinations and get transitions probs for each.
    for o in [0,1,2]:
        for m in [0,1,2]:
            for f in [0,1,2]:
                lik_trans += pr_transition(offs_diploid, moth_diploid, male_diploid, o, m, f, mu)
    lik_trans = lik_trans / 27 # correct for the number of combinations.
    with np.errstate(divide='ignore'): lik_trans = np.log(lik_trans) # convert to log.
        
    # sum diploid genotypes to identify where the negative values (i.e. dropouts) are.
    dropout_mask = offs_diploid[:, np.newaxis] + moth_diploid[:, np.newaxis] + male_diploid[np.newaxis]
    # array of the number of loci for each trio for which one or more individual had a dropout.
    n_dropouts =(dropout_mask < 0).sum(2)
    
    # Correct for dropouts
    lik_trans[dropout_mask < 0] = 0 # set loci with one or more dropouts to zero to allow summing.
    lik_trans = lik_trans.sum(2) # sum over loci.
    lik_trans = lik_trans - np.log(males.nloci - n_dropouts) # scale by the number of valid SNPs.
    
    return lik_trans