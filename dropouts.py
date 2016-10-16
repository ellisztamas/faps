import numpy as np

def dropouts(genotype_data, dr):
    """
    Add allelic dropouts to an array of genotypic data.
    
    ARGUMENTS
    genotype_data: A genotype array.
    
    dr: diploid dropout rate.
    
    RETURNS
    A copy of the input genotype data, but with dropouts. Alleles which dropout are
    given as -9, in contrast to useful data which can only be 0 or 1 (or 2 for diploid
    genotypes).
    """
    ninds    = genotype_data.geno.shape[0]
    nloci    = genotype_data.geno.shape[1] 

    # pick data points to dropout
    positions = np.reshape(np.random.binomial(1, dr, ninds*nloci),[ninds,nloci])
    positions = np.repeat(positions[:,:,np.newaxis], 2, axis =2)
    
    # make a copy of the genotype data, just in case
    new_alleles = np.copy(genotype_data.geno)
    # insert missing data into parental genotypes
    new_alleles = new_alleles - positions*3 # make all dropped loci < 0
    new_alleles[new_alleles < 0] = -9 # make dropped loci -9, just for tidyness.
    
    output = genotypeArray(new_alleles, genotype_data.names, genotype_data.mothers, genotype_data.fathers)
    
    return output