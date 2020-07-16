import numpy as np

def calculate_geno_probs(geno, mu):
    """
    Calcuate probabilities that each individual has genotype 0, 1 or 2.

    Parameters
    ----------
    mu: float or 1-d array between 0 and 1
        Per locus genotype error rate; the probability that the called
        genotype is incorrect. Alternatively, supply a vector of error rates
        for each locus.

    Returns
    -------
    A 3-d array of probabilities that each observed marker data point is 0,
    1 or 2 given the input error rate(s).
    Axes index:
    1. Possible genotypes 0, 1 or 2
    2. Individuals
    3. Loci

    Examples
    --------
    from faps import *
    import numpy as np

    # Generate a population of adults
    allele_freqs = np.random.uniform(0.3,0.5,3)
    adults = make_parents(20, allele_freqs)

    # Example with a single error rate
    adults.geno_probs(mu = 0.01)
    # Example with a vector of errors
    adults.geno_probs(mu = np.array([0.01, 0.02, 0.03]))
    """
    if isinstance(mu, float):
        if np.array([mu < 0, mu >1]).any():
            raise ValueError("mu should be between 0 and 1")
    elif isinstance(mu, np.ndarray):
        if (len(mu.shape) != 1):
            raise ValueError("mu should be a 1-dimensional vector or float.")
        if np.array([mu < 0, mu >1]).any():
            raise ValueError("All elements in mu should be between 0 and 1")
        if len(mu) != geno.shape[1]:
            raise ValueError("mu should either be a single float or have the same number of elements as the number of loci.")
    else:
        raise TypeError("mu should either be a float or vector of floats between 0 and 1.")

    diploid = geno.sum(2)
    # 3-d array giving positions that are 0, 1 or 2.
    probs = np.array([diploid == i for i in [0,1,2]])
    # Transpose to index indivudals, loci, and possible genotype
    probs = probs.transpose([1,2,0])
    # Insert mu where probs is True, and mu/2 if False
    # mu/2 because we don't know which other genotype it could be.
    probs = (probs * (1-mu)) + (~probs * mu/2)
    # Where data are not observed, we set this to NaN.
    # probs[diploid < 0] = np.nan

    return probs
