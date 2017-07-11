import numpy as np

def pr_unsampled(offspring_diploid, maternal_diploid, allele_freqs, offspring_genotype, maternal_genotype, male_genotype, mu):
    """
    Calculate the transitions probability for a given set of parental and offspring 
    alleles.

    Transitipn probabilities are then weight by the probability of drawing the allele
    from the population, and the probability that this allele is the true allele, given
    observed genotype data and the error rate mu.
    
    ARGUMENTS:
    offspring_diploid, maternal_diploid, male_diploid: arrays of diploid genotypes for
        the offspring, mothers and fathers.
    
    allele_freqs = vector of population allele frequencies.
    
    offspring_genotype, maternal_genotype, male_genotype: a two-element list of zeroes
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
    
    # Probability of the father is drawn from population allele frequencies.
    if male_genotype is 0: pr_males = allele_freqs**2
    if male_genotype is 1: pr_males = allele_freqs*(1-allele_freqs)
    if male_genotype is 2: pr_males = (1-allele_freqs)**2

    return trans_prob * pr_males * pr_mothers * pr_offs