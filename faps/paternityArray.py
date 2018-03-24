from genotypeArray import genotypeArray
from alogsumexp import alogsumexp
import numpy as np

class paternityArray(object):
    """
    Likelihoods of that any of a set of candidate males is the true father of each
    offspring individual, assuming the mother is known.
    
    If you are constructing a paternityArray from genotypeArray objects, it is
    easier to call the wrapper fucntion `paternity_array`.

    Parameters
    ----------
    likelihood: array
        Array of log likelihoods of paternity. Rows index offspring and columns
        candidates. The ij^th element is the likelihood that candidate j is the
        sire of offspring i.
    lik_absent: array
        Vector of log likelihoods that the true sire of each offspring is missing
        from the sample of candidate males, and hence that offspring alleles are
        drawn from population allele frequencies. Should be the same length as
        the number of rows in `likelihood`.
    by_locus: array
        An array of log likelihoods for the offspring genotype given
        the mother. The first axis has three elements, indexing cases for
        the paternal genotype being 0, 1 or 2.
    offspring: array-like
        Identifiers for each offspring individual.
    mothers: array-like
        Identifiers of the mother of each individual, if these are known.
    fathers: array-like
        Identifiers of the father of each individual, if these are known.
    candidates: array-like
        Identifiers of the candidate fathers.
    mu: float
        Point estimate of the genotyping error rate. Note that sibship clustering
        is unstable if mu_input is set to exactly zero. Any zero values will
        therefore be set to a very small number close to zero (10^-12).
    purge: float between zero or one, int, array-like, optional
        Individuals who can be removed from the paternity array a priori. If
        a float is given, that proportion of individuals is removed from the
        array at random. Alternatively an integer or vector of integers
        indexing specific individuals can be supplied.
    missing_parents : float between zero and one, or 'NA', optional
        Input value for the proportion of adults who are missing from the sample.
        This is used to weight the probabilties of paternity for each father
        relative to the probability that a father was not sampled. If this is
        given as 'NA', no weighting is performed.
    selfing_rate: float between zero and one, optional
        Input value for the prior probability of self-fertilisation.
    clashes: array, optional
        Array of the numbers of opposing homozygous incompatibilities for all
        possible parent-offspring dyads. This should have a row for every offspring
        and a column for every candidate father, and hence be the same shape as 
        the `likelihood` array.
    max_clashes: integer
        Maximum number of double-homozygous incompatibilities allowed between
        offspring and candidate fathers. Dyads with more incompatibilities than
        this will have their probability set to zero.
    
    Returns
    -------
    prob_array: array
        Array of probabilities of paternity, accounting for the probability that
        a sire has not been sampled. This is the array `likelihood` with vector
        `lik_absent` appended, with rows normalised to sum to one.
    """

    def __init__(self, likelihood, lik_absent, by_locus, offspring, mothers, fathers, candidates, mu=None, purge=None, missing_parents=None, selfing_rate=None, clashes = None, max_clashes = None):
        self.mu         = mu
        self.offspring  = offspring
        self.mothers    = mothers
        self.fathers    = fathers
        self.candidates = candidates
        self.lik_array  = likelihood
        self.lik_absent = lik_absent
        self.by_locus   = by_locus
        self.clashes    = clashes
        self.prob_array = self.adjust_prob_array(purge, missing_parents, selfing_rate, max_clashes)

    def adjust_prob_array(self, purge=None, missing_parents=None, selfing_rate=None, max_clashes=None):
        """
        Construct an array of log posterior probabilities that each offspring is sired
        by each of the candidate males in the sample, or that the true father is not
        present in the sample. Rows are normalised to some to one.

        Additional arguments can specify the proportion of missing fathers, and the rate
        of self-fertilisation.

        Parameters
        ----------
        purge: float between zero or one, int, array-like, optional
            Individuals who can be removed from the paternity array a priori. If
            a float is given, that proportion of individuals is removed from the
            array at random. Alternatively an integer or vector of integers
            indexing specific individuals can be supplied.
        missing_parents : float between zero and one, or 'NA', optional
            Input value for the proportion of adults who are missing from the sample.
            This is used to weight the probabilties of paternity for each father
            relative to the probability that a father was not sampled. If this is
            given as 'NA', no weighting is performed.
        selfing_rate: float between zero and one, optional
            Input value for the prior probability of self-fertilisation.
        max_clashes: integer
            Maximum number of double-homozygous incompatibilities allowed between
            offspring and candidate fathers. Dyads with more incompatibilities than
            this will have their probability set to zero.

        Returns
        -------
        An array with a row for each offspring individual and column for each
        candidate male, with an extra final column for the probability that the offspring
        is drawn from population allele frequencies. Each element is a log
        probability, and as such each row sums to one.
        """
        new_array = np.append(self.lik_array, self.lik_absent[:,np.newaxis], 1)

        # set log lik of individuals to be purged to -Inf
        if purge is not None:
            nc = len(self.candidates)
            if isinstance(purge, float):
                if purge < 0 or purge > 1:
                    print" Error: purge must be between zero and one."
                    return None
                ix = np.random.choice(range(nc), np.round(purge*nc).astype('int'), replace=False)
                with(np.errstate(divide='ignore')): new_array[:, ix] = np.log(0)
            elif isinstance(purge, list) or isinstance(purge, np.ndarray) or isinstance(purge, int):
                with(np.errstate(divide='ignore')): new_array[:, purge] = np.log(0)
            else:
                raise TypeError("Error: purge should be a float or list of floats between zero and one.")

        # correct for input parameter for proportion of missing fathers.
        if missing_parents is not None and missing_parents is not 'NA':
            # apply correction for the prior on number of missing parents.
            if missing_parents < 0 or missing_parents >1:
                raise ValueError("missing_parents must be between 0 and 1!")
            # if missing_parents is between zero and one, correct the likelihoods.
            if missing_parents >0 and missing_parents <=1:
                if missing_parents ==1: print "Warning: missing_parents set to 100%."
                new_array[:, -1] = new_array[:, -1] + np.log(  missing_parents)
                new_array[:,:-1] = new_array[:,:-1] + np.log(1-missing_parents)
            # if missing_parents is 0, set the term for unrelated fathers to zero.
            if missing_parents == 0:
                with(np.errstate(divide='ignore')): new_array[:,-1] = np.log(0)

        # correct for selfing rate.
        if selfing_rate is not None:
            # apply correction for the prior on number of missing parents.
            if selfing_rate < 0 or selfing_rate >1:
                raise ValueError("Error: selfing_rate must be between 0 and 1.")
            if selfing_rate >=0 and selfing_rate <=1:
                if selfing_rate == 1: print "Warning: selfing_rate set to 100%."
                ix = range(len(self.offspring))
                with np.errstate(divide='ignore'):
                    maternal_pos = [np.where(np.array(self.candidates) == self.mothers[i])[0][0] for i in ix] # positions of the mothers
                    new_array[ix, maternal_pos] += np.log(selfing_rate)

        # set the likelihood dyads with many incompatibilities to zero
        if max_clashes is not None:
            if self.clashes is None:
                raise TypeError("Unable to adjust for number of incompatible homozygous loci because `clashes` is not given.")
            elif self.clashes.shape != self.lik_array.shape:
                raise ValueError("Shape of the likelihood array does not match that of the array of clashes.")
            else:
                inc = np.append(self.clashes, np.zeros(self.lik_absent.shape)[:,np.newaxis], 1) # add an extra column so the shapes match
                with np.errstate(divide='ignore'): 
                    ix = np.log(inc <= max_clashes) # index elements to alter
                new_array = new_array + ix

        # normalise so rows sum to one.
        new_array = new_array - alogsumexp(new_array, axis=1)[:,np.newaxis]

        return new_array

    def split(self, by, purge=None, missing_parents=None, selfing_rate=None, max_clashes = None):
        """
        Split up a paternityArray into groups according to some grouping
        factor. For example, divide an array containing genotype data for
        multiple half-sibling arrays by the ID of their mothers.

        Parameters
        ----------
        by: array-like
            Vector containing grouping labels for each individual.

        Returns
        -------
        A list of paternityArray objects.
        """
        groups = np.unique(by)
        # indices for each family
        ix = [np.where(by == i)[0] for i in groups]
        # create new paternityArray objects for each family.
        if self.clashes is None:
            output = [paternityArray(self.lik_array[i], self.lik_absent[i], self.by_locus[:,i], self.offspring[i], self.mothers[i], self.fathers[i], self.candidates, self.mu, purge, missing_parents, selfing_rate, None, max_clashes) for i in ix]
        else:
            output = [paternityArray(self.lik_array[i], self.lik_absent[i], self.by_locus[:,i], self.offspring[i], self.mothers[i], self.fathers[i], self.candidates, self.mu, purge, missing_parents, selfing_rate, self.clashes[i], max_clashes) for i in ix]

        return output
        
    def write(self, path, decimals=3):
        """
        Write a matrix of (unnormalised) likelihoods of paternity to disk.

        Parameters
        ----------
        path: str
            Path to write to.
        decimals: int
            Number of decimal places to be saved to disk for likleihood values.

        Returns
        -------
        A CSV file indexing offspring ID, mother ID, followed by a matrix of likelihoods
        that each candidate male is the true father of each individual. The final
        column is the likelihood that the paternal alleles are drawn from population
        allele frequencies.
        """
        # append offspring and mother IDs onto the likelihood array.
        # append likelihoods of abset fathers on the back.
        newdata = np.append(self.offspring[:,np.newaxis],
                        np.append(self.mothers[:,np.newaxis],
                                  np.append(np.around(self.lik_array, decimals),
                                            np.around(self.lik_absent[:,np.newaxis], decimals),
                                            1),1),1)
        # headers
        cn = ','.join(self.candidates )
        cn = 'offspringID,motherID,' + cn + ',missing_father'
        # write to disk
        np.savetxt(path, newdata, fmt='%s', delimiter=',', comments='', header=cn)