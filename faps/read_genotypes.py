import numpy as np
from faps.convert_genotypes import convert_genotypes
from faps.genotypeArray import genotypeArray

def read_genotypes(path, genotype_col = 1, mothers_col=None, fathers_col=None, delimiter=","):
    """
    Import a text file containing of diploid SNP data to FAPS.

    Retrieves the file containing genotype data, and convert this to a genotypeArray.
    If the data are for offspring individuals, optional columns can be used to indicate
    their labels to match up with a file of parental data. Otherwise, names for mothers
    and fathers are given as 'NA' in the output genotypeArray.

    Parameters
    ----------
    path: str
        Filename path to locate the text file. This should contain unique ID
        labels for each individual in the first column, followed by (optional)
        columns of names for the mothers and fathers. After this include diploid
        genotype information for each locus, with a single column for each marker.
        The header row should include names for each locus, or else be left blank.
        Homozygotes should be labelled 0 or 2, and heterozygotes as 1. Missing
        data should be labelled 'NA'.
    genotype_col: int
        indicate the column index where genotype information begins.
    mothers_col: int
        If a column of maternal names has been included, indicate column index
        here.
    fathers_col: If a column of paternal names has been included, indicate column
        index here.
    delimiter The symbol used to separate values in the text file. Defaults to
        commas.

    RETURNS:
    A genotypeArray object.
    """
    # Import genotype data
    geno = np.genfromtxt(path, dtype='int', delimiter=delimiter)[1:, genotype_col:]
    geno[geno == -1] = -9 # set missing data to -9
    geno = convert_genotypes(geno)
    # pull out marker names.
    marker_names = np.genfromtxt(path, dtype=str)[0].split(",")[genotype_col:]

    # Import individual names
    names = np.genfromtxt(path, dtype='str')
    names = [names[i].split(",")[0].replace('"', '') for i in range(1, len(names))]

    # Names for the mothers, if these are given
    if isinstance(mothers_col, int):
        mothers = np.genfromtxt(path, dtype='str')
        mothers = [mothers[i].split(",")[mothers_col].replace('"', '') for i in range(1, len(mothers))]
    # If they aren't, return a list of NAs
    if mothers_col is None:
        mothers = np.repeat('NA', geno.shape[0])

    # Names for the fathers, if given
    if isinstance(fathers_col, int):
        fathers = np.genfromtxt(path, dtype='str')
        fathers = [fathers[i].split(",")[fathers_col].replace('"', '') for i in range(1, len(fathers))]
    # If they aren't, return a list of NAs
    if fathers_col is None:
        fathers = np.repeat('NA', geno.shape[0])

    return genotypeArray(geno, np.array(names), np.array(mothers), np.array(fathers), markers=np.array(marker_names))
