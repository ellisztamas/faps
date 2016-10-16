import numpy as np

def pr_transition(offspring_diploid, maternal_diploid, male_diploid, offspring_genotype, maternal_genotype, male_genotype, mu):
    """
    Calculate the transition probability for a trio of putative diploid genotypes.

    Transition probabilities are then weight by the probability that the true offspring,
    mothers, and male genotypes match the input genotypes given observed marker data.
    Generally one would need to sum over all 27 possible combinations of genotypes.
    
    This function works with diploid genotypes, rather than a genotypeArray. Generally
    this is not called directly, but through lik_sampled_fathers() or similar.
    
    ARGUMENTS:
    offspring_diploid, maternal_diploid, male_diploid: arrays of diploid genotypes for
        the offspring, mothers and fathers.
    
    offspring_genotype, maternal_genotype, male_genotype: Two-element lists of zeroes
        and ones indicating the diploid genotype of males, mothers and offspring to be
        considered.
    
    mu: point estimate of the genotyping error rate.
    
    RETURNS:
    A 3-dimensional array of probabilities indexing offspring, candidate males, and loci.
    These are given in linear, rather than log space.
    """
    # an array of all possible transition probabilities indexed as [offspring, mother, father].
    trans_prob_array = np.array([[[1,  0.5, 0  ],
                                  [0.5,0.25,0  ],
                                  [0,  0,   0  ]],
                                 [[0,  0.5, 1  ],
                                  [0.5,0.5, 0.5],
                                  [1,  0.5, 0  ]],
                                 [[0,  0,   0  ],
                                  [0,  0.25,0.5],
                                  [0,  0.5, 1  ]]])
    # the transition probability for the given genotypes.
    trans_prob = trans_prob_array[offspring_genotype, maternal_genotype, male_genotype]
    
    # Probabilities that the observed offspring marker data match observed data.
    pr_offs = np.zeros([offspring_diploid.shape[0], offspring_diploid.shape[1]])
    pr_offs[offspring_diploid == offspring_genotype] = 1-mu
    pr_offs[offspring_diploid != offspring_genotype] = mu
    # Probabilities that the observed maternal marker data match observed data.
    pr_mothers = np.zeros([maternal_diploid.shape[0], maternal_diploid.shape[1]])
    pr_mothers[maternal_diploid == maternal_genotype] = 1-mu
    pr_mothers[maternal_diploid != maternal_genotype] = mu
    # Probabilities that the observed candidate male marker data match observed data.
    pr_males = np.zeros([male_diploid.shape[0], male_diploid.shape[1]])
    pr_males[male_diploid == male_genotype] = 1-mu
    pr_males[male_diploid != male_genotype] = mu

    return trans_prob * pr_males[np.newaxis] * pr_mothers[:,np.newaxis] * pr_offs[:,np.newaxis]
