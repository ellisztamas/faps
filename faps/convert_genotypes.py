import numpy as np

def convert_genotypes(diploid_data):
    """
    Convert a table of diploid genotypes to an array of haploid genotypes for a
    genotypeArray object. Usually, this will be an internal function call when
    importing genotype data from a text file to a genotypeArray.

    This function copies diploid data to each of two columns for haploid genotypes.
    Heterozygotes are assigned a 0 for locus 1 and a 1 for locus 2.
    Major allele homozygotes are assigned as 1 for both loci.
    Minor allele homozygotes are already coded 0 at both loci.

    Parameters
    ----------
    diploid_data: array
        Array containing diploid genotypes of candiate parents, with a row for
        every individual and a column for every SNP. The first column should
        label maternal IDs, or else left blank. The first row should label SNPs,
        or else be left blank.

    Returns
    -------
    Haploid genotype array for genotypeArray objects.
    """
    n_inds = diploid_data.shape[0]         # number of individuals in the array
    n_loci = diploid_data.shape[1]         # number of loci, accounting for the index column.

    parMat = np.zeros([n_inds,n_loci,2]).astype('int')    # empty matrix to store data.

    parMat[:,:,0]                     = np.copy(diploid_data)
    parMat[:,:,0][parMat[:,:,0] == 1] = 0       # for heterozygotes, set this locus to 0.
    parMat[:,:,0][parMat[:,:,0] == 2] = 1       # for minor allele homozygotes, set this locus to 1.

    parMat[:,:,1]                     = np.copy(diploid_data)
    parMat[:,:,1][parMat[:,:,1] == 2] = 1       # for minor allele homozygotes, set this locus to 1.

    return parMat
