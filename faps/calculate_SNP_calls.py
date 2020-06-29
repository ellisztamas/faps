def calculate_SNP_calls(geno_probs):
    """
    Convert genotype probabilities to hard calls.

    Probabilities in geno_probs.geno_probs are rounded to either
    zero or one, meaning that whichever genotype has probability >0.5
    is called. If all probabilities are < 0.5, no SNP is called
    (returns np.nan)

    Parameters
    ----------
    geno_probs: array
        3-dimensional array indexing individuals, loci, and three
        possible genotypes. This is the same format as the
        `geno_probs` attribute of a genotypeArray object.

    Returns
    -------
    3-dimensional array indexing individuals, loci, and two chromosomes.
    This is the same format as the `geno` attribute of a
    genotypeArray object.

    Example
    -------

    """
    # Round genotype probabilities to zero or one.
    call_probs = np.round(geno_probs.geno_probs)
    # 3-dimensional array of np.nan
    geno = np.full([geno_probs.size, geno_probs.nloci,2], np.nan)
    # Fill in entries called as zero
    geno[:,:,0][call_probs[:,:,0] == 1] = 0
    geno[:,:,1][call_probs[:,:,0] == 1] = 0
    # Fill in entries called as heterozygote
    geno[:,:,0][call_probs[:,:,1] == 1] = 0
    geno[:,:,1][call_probs[:,:,1] == 1] = 1
    # Fill in entries called as 2.
    geno[:,:,0][call_probs[:,:,2] == 1] = 1
    geno[:,:,1][call_probs[:,:,2] == 1] = 1

    return(geno)
