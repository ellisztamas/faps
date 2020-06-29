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

    # # If the genotypeArray does not include geno_probs, include this
    # if not hasattr(offspring, 'geno_probs'):
    #     offspring.geno_probs = offspring.calculate_geno_probs(mu = mu)
    # if not hasattr(mothers, 'geno_probs'):
    #     mothers.geno_probs   = mothers.calculate_geno_probs(mu = mu)
    # if not hasattr(males, 'geno_probs'):
    #     males.geno_probs     = males.calculate_geno_probs(mu = mu)

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


    # Array of probabilities of drawing paternal allele if the dad is 0,1,2
    allele_freqs = males.allele_freqs()
    af = np.array([
        (1-allele_freqs)**2,
        allele_freqs * (1-allele_freqs),
        allele_freqs ** 2
    ])

    # Empty arrays to store probabilities of paternity, and of missing dads
    lik_array = np.zeros([offspring.size, males.size, offspring.nloci])
    lik_absent = np.zeros([offspring.size, offspring.nloci])

    geno = [0,1,2]
    for f in geno:
        for m in geno:
            for o in geno:
                # Multiply genotype probabilities by the transition probability
                lik_array  += \
                trans_prob_array[o,m,f] * \
                offspring.geno_probs[:,:,o][:, np.newaxis] * \
                mothers.  geno_probs[:,:,m][:, np.newaxis] * \
                males.    geno_probs[:,:,f][np.newaxis]
                # Multiply transition probs by prob of drawing paternal alleles at random
                lik_absent += \
                trans_prob_array[o,m,f] * \
                offspring.geno_probs[:,:,o] * \
                mothers.  geno_probs[:,:,m] * \
                af[f]

    # Sum log-likelihoods over (valid) loci and correct for dropouts..
    lik_array  = np.log(lik_array)
    lik_absent = np.log(lik_absent)

    lik_array = np.nanmean(lik_array, 2)  * offspring.nloci
    lik_absent = np.nanmean(lik_absent, 1) * offspring.nloci

    return [lik_array, lik_absent]
