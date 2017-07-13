from genotypeArray import genotypeArray
from lik_sampled_fathers import lik_sampled_fathers
from lik_unsampled_fathers import lik_unsampled_fathers
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
        
    Returns
    -------
    prob_array: array
        Array of probabilities of paternity, accounting for the probability that
        a sire has not been sampled. This is the array `likelihood` with vector
        `lik_absent` appended, with rows normalised to sum to one.
    """

    def __init__(self, likelihood, lik_absent, offspring, mothers, fathers, candidates, mu=None, purge=None, missing_parents=None, selfing_rate=None):
        self.mu         = mu
        self.offspring  = offspring
        self.mothers    = mothers
        self.fathers    = fathers
        self.candidates = candidates
        self.lik_array  = likelihood
        self.lik_absent = lik_absent
        self.prob_array = self.adjust_prob_array(purge, missing_parents, selfing_rate)

    def adjust_prob_array(self, purge=None, missing_parents=None, selfing_rate=None):
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

        RETURNS:
        An array with a row for each offspring individual and column for each
        candidate male, with an extra final column for the probability that the offspring
        is drawn from population allele frequencies. Each element is a log
        probability, and as such each row sums to one.
        """
        lik_array = np.append(self.lik_array, self.lik_absent[:,np.newaxis], 1)

        # set log lik of individuals to be purged to -Inf
        if purge is not None:
            nc = len(self.candidates)
            if isinstance(purge, float):
                if purge < 0 or purge > 1:
                    print" Error: purge must be between zero and one."
                    return None
                ix = np.random.choice(range(nc), np.round(purge*nc).astype('int'), replace=False)
                with(np.errstate(divide='ignore')): lik_array[:, ix] = np.log(0)
            elif isinstance(purge, list) or isinstance(purge, np.ndarray) or isinstance(purge, int):
                with(np.errstate(divide='ignore')): lik_array[:, purge] = np.log(0)
            else:
                print "Error: purge should be a float or list of floats between zero and one."
                return None

        # correct for input parameter for proportion of missing fathers.
        if missing_parents is not None and missing_parents is not 'NA':
            # apply correction for the prior on number of missing parents.
            if missing_parents < 0 or missing_parents >1:
                print "missing_parents must be between 0 and 1!"
                return None
            # if missing_parents is between zero and one, correct the likelihoods.
            if missing_parents >0 and missing_parents <=1:
                if missing_parents ==1: print "Warning: missing_parents set to 100%."
                lik_array[:, -1] = lik_array[:, -1] + np.log(  missing_parents)
                lik_array[:,:-1] = lik_array[:,:-1] + np.log(1-missing_parents)
            # if missing_parents is 0, set the term for unrelated fathers to zero.
            if missing_parents == 0:
                with(np.errstate(divide='ignore')): lik_array[:,-1] = np.log(0)

        # correct for selfing rate.
        if selfing_rate is not None:
            # apply correction for the prior on number of missing parents.
            if selfing_rate < 0 or selfing_rate >1:
                print "Error: selfing_rate must be between 0 and 1."
                return None
            if selfing_rate >=0 and selfing_rate <=1:
                if selfing_rate == 1: print "Warning: selfing_rate set to 100%."
                ix = range(len(self.offspring))
                with np.errstate(divide='ignore'):
                    maternal_pos = [np.where(np.array(self.candidates) == self.mothers[i])[0][0] for i in ix] # positions of the mothers
                    lik_array[ix, maternal_pos] += np.log(selfing_rate)

        # normalise so rows sum to one.
        lik_array = lik_array - alogsumexp(lik_array, axis=1)[:,np.newaxis]

        return lik_array

    def write(self, path):
        """
        Write a matrix of (unnormalised) likelihoods of paternity to disk.

        ARGUMENTS:

        path Path to write to.

        RETURNS:
        A CSV file indexing offspring ID, mother ID, followed by a matrix of likelihoods
        that each candidate male is the true father of each individual. The final
        column is the likelihood that the paternal alleles are drawn from population
        allele frequencies.
        """
        # append offspring and mother IDs onto the likelihood array.
        # append likelihoods of abset fathers on the back.
        newdata = np.append(self.offspring[:,np.newaxis],
                        np.append(self.mothers[:,np.newaxis],
                                  np.append(self.lik_array,
                                            self.lik_absent[:,np.newaxis],
                                            1),1),1)
        # headers
        cn = ''.join(self.candidates )
        cn = 'offspringID,motherID,' + cn + ',missing_father'
        # write to disk
        np.savetxt(path, newdata, fmt='%s', delimiter=',', comments='', header=cn)
