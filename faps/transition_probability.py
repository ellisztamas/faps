import numpy as np
from faps.paternityArray import paternityArray
from faps.genotypeArray import genotypeArray

def transition_probability(offspring, mothers, males, mu, inverse=False):
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
    mu: float between zero and one
        Point estimate of the genotyping error rate. Clustering is unstable if
        mu_input is set to exactly zero.
    inverse: bool, optional
        If true, function return 1-transition probabilities, or the
        probability of *not* generating the offspring given maternal and
        candidate paternal genotypes

    Returns
    -------
    0. Array indexing offspring x candidates transition probabilities.
    1. Array indexing offspring only for transition probabilities
    from population allele frequencies.
    """
    if not isinstance(offspring, genotypeArray):
        raise TypeError('offspring is not a genotypeArray')
    if not isinstance(mothers, genotypeArray):
        raise TypeError('mothers is not a genotypeArray')
    if not isinstance(males, genotypeArray):
        raise TypeError('males is not a genotypeArray')

    offspring_diploid = offspring.geno.sum(2)
    maternal_diploid = mothers.geno.sum(2)
    male_diploid = males.geno.sum(2)

    # array of viable transition probabilities
    trans_prob_array = np.array([[[1,  0.5, 0  ],
                                  [0.5,0.25,0  ],
                                  [0,  0,   0  ]],
                                 [[0,  0.5, 1  ],
                                  [0.5,0.5, 0.5],
                                  [1,  0.5, 0  ]],
                                 [[0,  0,   0  ],
                                  [0,  0.25,0.5],
                                  [0,  0.5, 1  ]]])

    # empty arrays to stores probabilities.
    prob_m = np.zeros([3, offspring.size, offspring.nloci])
    # positions of dropouts
    drop_m = (maternal_diploid < 0) + (offspring_diploid < 0) # offspring vs mothers
    # correction factor for dropouts
    corr = 1/(1-drop_m.mean(1))[np.newaxis, :, np.newaxis]

    geno =[0,1,2]
    for f in geno:
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
                prob_o+= (trans_prob * pr_offs * 1/3)
            # Probabilities that the observed maternal marker data match observed data.
            pr_mothers = np.zeros(maternal_diploid.shape)
            pr_mothers[maternal_diploid == m] = 1-mu
            pr_mothers[maternal_diploid != m] = mu

            prob_o[drop_m] = 1
            prob_m[f] += (prob_o * pr_mothers * 1/3)

    prob_m = prob_m ** corr

    # array of probabilities for paternal genotypes when drawn from allele frequencies.
    allele_freqs = males.allele_freqs()
    af = np.array([allele_freqs**2,
                   allele_freqs * (1-allele_freqs),
                   (1-allele_freqs)**2])
    af = af[:,np.newaxis]
    # probability that observed alleles are drawn from the population.
    prob_a = prob_m*af
    prob_a = prob_a.sum(0)
    prob_a = np.log(prob_a).sum(1)


    prob_f = np.zeros([offspring.size, males.size, offspring.nloci])
    drop_f = (male_diploid < 0)
    corr   = 1/(1 - drop_f.mean(1))[np.newaxis, :, np.newaxis]

    for f in [0,1,2]:
        pr_males = np.zeros(male_diploid.shape)
        pr_males[male_diploid == f] = 1-mu
        pr_males[male_diploid != f] = mu
        pr_males[drop_f] = 1

        prob_f += prob_m[f][:, np.newaxis] * pr_males[np.newaxis]

    prob_f = prob_f ** corr
    with np.errstate(divide='ignore'): prob_f = np.log(prob_f)

    output = [prob_f.sum(2), prob_a]
    return output
