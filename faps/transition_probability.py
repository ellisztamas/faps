import numpy as np
from paternityArray import paternityArray
from genotypeArray import genotypeArray

def transition_probability(offspring, mothers, males, allele_freqs, mu, inverse=False):
    """
    Calculate per-locus transition probabilities given data on offspring, known
    mothers and candidate fathers. Also returns probabilities that paternal
    alleles are drawn from population allele frequencies.

    Parameters
    ----------
    offspring: genotypeArray, or list of genotypeArrays
        Observed genotype data for the offspring.
    mothers: genotypeArray, or list of genotypeArrays
        Observed genotype data for the offspring. Data on mothers need
        to be in the same order as those for the offspring.
    males: genotypeArray
        Observed genotype data for the candidate males.
    allele_freqs: array-like
        Vector of population allele frequencies for the parents.
    mu: float between zero and one
        Point estimate of the genotyping error rate. Clustering is unstable if
        mu_input is set to exactly zero.
    reciprocal: bool, optional
        If true, function return 1-transition probabilities, or the
        probability of *not* generating the offspring given maternal and
        candidate paternal genotypes
    
    Returns
    -------
    0. Array indexing offspring x candidates x number of loci for per-locus
    transition probabilities
    1. Array indexing offspring x number of loci for transition probabilities
    from population allele frequencies
    """
    trans_prob_array = np.array([[[1,  0.5, 0  ],
                                  [0.5,0.25,0  ],
                                  [0,  0,   0  ]],
                                 [[0,  0.5, 1  ],
                                  [0.5,0.5, 0.5],
                                  [1,  0.5, 0  ]],
                                 [[0,  0,   0  ],
                                  [0,  0.25,0.5],
                                  [0,  0.5, 1  ]]])
    # arrays of diploid genotypes
    offspring_diploid = offspring.geno.sum(2)
    maternal_diploid = mothers.geno.sum(2)
    male_diploid = males.geno.sum(2)
    # array of probabilities for paternal genotypes when drawn from allele frequencies.
    af = np.array([allele_freqs**2,
              allele_freqs * (1-allele_freqs),
              (1-allele_freqs)**2])

    # empty arrays to stores probabilities.
    prob_f = np.zeros([offspring.size, males.size, offspring.nloci])
    prob_a = np.zeros([offspring.size, males.nloci])

    geno =[0,1,2]
    for f in geno:
        prob_m = np.zeros(offspring_diploid.shape)
        for m in geno:
            prob_o = np.zeros(offspring_diploid.shape)
            for o in geno:
                # the transition probability for the given genotypes.
                if inverse: trans_prob = 1-trans_prob_array[o, m, f]
                else:       trans_prob =   trans_prob_array[o, m, f]
                # Probabilities that the observed offspring marker data match observed data.
                pr_offs = np.zeros(offspring_diploid.shape)
                pr_offs[offspring_diploid == o] = 1-mu
                pr_offs[offspring_diploid != o] = mu
                prob_o = prob_o + (trans_prob * pr_offs * 1/3)
            # Probabilities that the observed maternal marker data match observed data.
            pr_mothers = np.zeros(maternal_diploid.shape)
            pr_mothers[maternal_diploid == m] = 1-mu
            pr_mothers[maternal_diploid != m] = mu
            prob_m = prob_m + (prob_o * pr_mothers * 1/3)
        # Probabilities that the observed candidate male marker data match observed data.
        pr_males = np.zeros(male_diploid.shape)
        pr_males[male_diploid == f] = 1-mu
        pr_males[male_diploid != f] = mu

        prob_f += prob_m[:, np.newaxis] * pr_males[np.newaxis]
        prob_a += prob_m * af[f][np.newaxis]
        
    return prob_f, prob_a