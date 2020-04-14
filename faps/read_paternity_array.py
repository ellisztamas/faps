import numpy as np
from faps.paternityArray import paternityArray
from pandas import read_csv

def read_paternity_array(path, mothers_col=1, likelihood_col = 2, fathers_col=None, delimiter=","):
    """
    Import a text file containing of log likelihoods of paternity.

    This is a shortcut to save having to run paternity_array every time you open
    a data set. Note that only information on the IDs of the offspring and
    mothers, likelihoods of paternity for each candidate and likelihoods that
    fathers are missing can be retrieved; other information; for example
    covariates must be generated again.

    Parameters
    ----------
    path: str
        filename path to locate the text file. This should contain unique ID
        labels for each offspring individual in the first column, followed by
        (optional)columns of names for the mothers and fathers. After this
        include log likelihoods that each adult is the sire of each offspring.
        The header row should include names for each candidate, or else be left
        blank. The final column should be a vector of likelihoods that the true
        sire of each offspring has not been sampled.
    mothers_col: int
        If a column of maternal names has been included, indicate column index
        here. This is usually in the second column, which in Python speak is
        column 1.
    likelihood_col: int
        indicate the column index where genotype information begins. This is
        usually in the third column, which in Python speak is column 2, unless
        you have given a column for fathers.
    delimiter: str
        The symbol used to separate values in the text file. Defaults to
        commas.

    Returns:
    --------
    A paternityArray object with attributes for likehood of paternity,
    likelihoods that father are missing, IDs for the offspring and mothers,
    and a vector of NAs for the fathers. Other possible attributes of a
    paternityArray are set to `None`.

    Examples
    --------
    from faps import *
    import numpy as np

    # Generate a population of adults
    allele_freqs = np.random.uniform(0.3,0.5,50)
    adults = make_parents(20, allele_freqs)
    # Mate the first adult to the next three.
    mother = adults.subset(0)
    progeny = make_sibships(adults, 0, [1,2,3], 5, 'x')
    # Create paternityArray
    patlik = paternity_array(progeny, mother, adults, mu=0.0013)

    # Write the paternityArray object to disk
    patlik.write("my_paternity_array.csv")
    # Read it back in
    patlik2 = read_paternity_array("my_paternity_array.csv")
    # Split the array into paternal families
    patlik2.split(by=progeny.fathers)
    """
    # Import data frame.
    df = read_csv(path, delimiter=delimiter)

    # Create a new paternityArray object.
    pat_array = fp.paternityArray(
        likelihood = df[df.columns[likelihood_col:-1]].to_numpy(),
        lik_absent = df[df.columns[-1]].to_numpy(),
        offspring  = df[df.columns[0]].to_numpy(),
        mothers    = df[df.columns[mothers_col]].to_numpy(),
        fathers    = np.repeat('NA', len(df)),
        candidates = df.columns[likelihood_col:-1]
    )

    return pat_array
