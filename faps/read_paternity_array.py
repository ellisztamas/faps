import numpy as np
from paternityArray import paternityArray

def read_paternity_array(path, likelihood_col = 2, mothers_col=1, fathers_col=None, delimiter=","):
    """
    Import a text file containing of log likelihoods of paternity.

    If the data are for offspring individuals, optional columns can be used to
    indicate their labels to match up with a file of parental data. Otherwise,
    names for mothers and fathers are given as 'NA' in the output.

    Parameters
    ----------
    path: str
        filename path to locate the text file. This should contain unique ID
        labels for each individual in the first column, followed by (optional)
        columns of names for the mothers and fathers. After this include log
        likelihoods that each adult is the sire of each offspring. The header
        row should include names for each candidate, or else be left blank. The
        final column should be a vector of likelihoods that the true sire of
        each offspring has not been sampled.
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
    A paternityArray object.
    """
    # likelihoods of paternity, and off absent fathers.
    likelihood = np.genfromtxt(path, dtype='float', delimiter=delimiter)[1:, likelihood_col: -1]
    lik_absent = np.genfromtxt(path, dtype='float', delimiter=delimiter)[1:, -1]

    # Import individual names
    names = np.genfromtxt(path, dtype='str')
    offspring = np.array([i.split(",")[0].replace('"', '') for i in names])

    # Names for the mothers, if these are given
    if isinstance(mothers_col, int):
        mothers = np.array([i.split(",")[mothers_col].replace('"', '') for i in names])
    # If they aren't, return a list of NAs
    if mothers_col is None:
        mothers = np.repeat('NA', likelihood.shape[0])

    # Names for the fathers, if given
    if isinstance(fathers_col, int):
        fathers = np.array([i.split(",")[fathers_col].replace('"', '') for i in names])
    # If they aren't, return a list of NAs
    if fathers_col is None:
        fathers = np.repeat('NA', likelihood.shape[0])

    # pull out names of candidate males
    candidates = names[0].split(",")[likelihood_col: -1]
    return paternityArray(likelihood, lik_absent, offspring, mothers, fathers, candidates)
