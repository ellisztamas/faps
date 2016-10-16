import numpy as np

def lik_unsampled_fathers(offspring, mothers, allele_freqs, mu, mother_index=None):
    """
    Calculate a matrix of multilocus transition probabilities for an array of offspring
    and known mothers, but where the paternal allele is drawn from population allele
    frequencies.
    
    ARGUMENTS:
    offspring: A genotypeArray of observed genotype data for the offspring.

    mothers: an genotypeArray of observed genotype data for the mothers. Data on mothers
        need to be in the same order as those for the offspring. If a whole parent array
        is used, this can be subsetted into the correct order using the option
        mother_index.
    
    allele_freqs: a vector of population allele frequencies.
    
    mu: point estimate of the genotyping error rate.
    
    mother_index: an optional list of integers indexing the position of the mother of
        each offspring in the maternal_genotypes array. If left blank, the whole array
        is used.
    
    RETURNS:
    A vector of log likelihoods with an elements for each offspring.
    """
    # If a list of indices for the mothers has been given, subset the maternal data
    if mother_index is not None:
        mothers = mothers.subset(mother_index)
    
    # Diploid genotypes of each dataset.
    moth_diploid = mothers.geno.sum(2)
    offs_diploid = offspring.geno.sum(2)
    
    # empty array to store probabilities.
    lik_trans = np.zeros([offspring.size, offspring.nloci])
    # loop over all possible gemotype combinations and get transitions probs for each.
    for o in [0,1,2]:
        for m in [0,1,2]:
            for f in [0,1,2]:
                lik_trans += pr_unsampled(offs_diploid, moth_diploid, allele_freqs, o, m, f, mu)
    lik_trans = lik_trans / 27 # correct for the number of combinations.
    with np.errstate(divide='ignore'): lik_trans = np.log(lik_trans) # convert to log.
        
    # sum diploid genotypes to identify where the negative values (i.e. dropouts) are.
    dropout_mask = offs_diploid + moth_diploid
    # array of the number of loci for each trio for which one or more individual had a dropout.
    n_dropouts =(dropout_mask < 0).sum(1)
    
    # Correct for dropouts
    lik_trans[dropout_mask < 0] = 0 # set loci with one or more dropouts to zero to allow summing.
    lik_trans = lik_trans.sum(1) # sum over loci.
    lik_trans = lik_trans - np.log(offspring.nloci - n_dropouts) # scale by the number of valid SNPs.
    
    return lik_trans
