from faps.genotypeArray import genotypeArray
from faps.alogsumexp import alogsumexp
import numpy as np
from warnings import warn

class paternityArray(object):
    """
    Likelihoods of that any of a set of candidate males is the true father of
    each offspring individual, assuming the mother is known. Call the wrapper
    function `paternity_array`.

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
    missing_parents : float between zero and one
        Input value for the proportion of adults who are missing from the sample.
        This is used to weight the probabilties of paternity for each father
        relative to the probability that a father was not sampled. Defaults to
        zero, but you should definitely change this.
    purge: vector of str, or float between zero and one, optional
        Individuals who can be removed from the paternity array a priori. If
        a float is given, that proportion of individuals is removed from the
        array at random. Alternatively a string or vector of strings containing
        names of specific candidate fathers can be supplied.
    selfing_rate: float between zero and one, optional
        Input value for the prior probability of self-fertilisation.
    clashes: array, optional
        Array of the numbers of opposing homozygous incompatibilities for all
        possible parent-offspring dyads. This should have a row for every offspring
        and a column for every candidate father, and hence be the same shape as
        the `likelihood` array.
    max_clashes: int, optional
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

    def __init__(self, likelihood, lik_absent, offspring, mothers, fathers, candidates, missing_parents=0, mu=None, purge=None, selfing_rate=None, clashes = None, max_clashes = None, covariate=0):
        self.mu         = mu
        self.offspring  = offspring
        self.mothers    = mothers
        self.fathers    = fathers
        self.candidates = candidates
        self.lik_array  = likelihood
        self.lik_absent = lik_absent
        self.clashes    = clashes
        self.covariate  = covariate
        self.mu         = mu
        self.purge      = purge
        self.missing_parents = missing_parents
        self.selfing_rate = selfing_rate
        self.max_clashes = max_clashes

    def add_covariate(self, covariate):
        """
        Include a vector of (log) probabilities associated with covariates that
        provide additional information about paternity beyond that provided by
        genetic information (e.g. geographic distances).

        Parameters
        ----------
        covariate: 1-d array
            Vector of (log) probabilities of paternity based on non-genetic
            information, with one element for every candidate father. If this is a
            function of multiple sources they should be multiplied and included in
            this vector. If a list of offspring arrays have been supplied, this
            should be a list of vectors.

        Returns
        -------
        No output is printed; the covariate is added to the paternityArray as
        the attribute 'covariate'. Any existing information is overwritten. The
        vector is appended with an additional zero to allow for the final column
        of a the prob_array item in a paternityArray that accounts for the
        probability of missing fathers.
        """
        if isinstance(covariate, np.ndarray):
            if len(covariate.shape) > 1:
                raise ValueError("covariate should be a 1-d array, but has shape {}".format(covariate.shape))
            if len(self.candidates) != covariate.shape[0]:
                raise ValueError("Length of vector of covariates ({}) does not match the number of fathers ({})".format(len(self.candidates), covariate.shape[0]))
            if not all(covariate <= 0):
                warn("Not all values in covariate are less or equal to zero. Is it possible probabilities have not been log transformed?")
            covariate = np.append(covariate, 0)
            self.covariate = covariate
            return None
        else:
            raise TypeError("covariate should be a 1-d NumPy array.")


    def prob_array(self):
        """
        Construct an array of log posterior probabilities that each offspring is sired
        by each of the candidate males in the sample, or that the true father is not
        present in the sample. Rows are normalised to some to one.

        Additional arguments can specify the proportion of missing fathers, and the rate
        of self-fertilisation.

        Parameters
        ----------
        None.

        Returns
        -------
        An array with a row for each offspring individual and column for each
        candidate male, with an extra final column for the probability that the offspring
        is drawn from population allele frequencies. Each element is a log
        probability, and as such each row sums to one.

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
        patlik = paternity_array(
            progeny,
            mother,
            adults,
            mu=0.0013,
            missing_parents=0.2
        )
        # View posterior probabilities of paternity (G matrix in the FAPS paper)
        patlik.prob_array()

        # Change some parameters that will affect the G matrix.
        patlik.missing_parents = 0.3
        patlik.selfing_rate = 0
        # Explicitly set some candidates to have probabilities of paternity to zero
        patlik.purge = 'base_0' # Another way to remove the mother
        patlik.purge = ['base_15', 'base_16', 'base_17'] # remove a list of specific candidates
        patlik.purge = 0.2 # Throw two candidates out at random

        # Another way to do the previous steps in a direct call to paternity_array.
        paternity_array(
            progeny,
            mother,
            adults,
            mu=0.0013,
            missing_parents=0.2,
            selfing_rate=0,
            purge = 0.2
        )
        """
        new_array = np.append(self.lik_array, self.lik_absent[:,np.newaxis], 1)

        # set log lik of individuals to be purged to -Inf
        if self.purge is not None:
            nc = len(self.candidates)
            # If a float is given, remove candidates at random.
            if isinstance(self.purge, float):
                if self.purge <= 0 or self.purge >= 1:
                    raise ValueError(" Error: purge must be between zero and one.")
                # Random set of candidate indices to be purged.
                ix = np.random.choice(range(nc), np.round(self.purge*nc).astype('int'), replace=False)
                with(np.errstate(divide='ignore')):
                    new_array[:, ix] = np.log(0)

            # If one or more integers is given, remove candidates at those indices
            elif isinstance(self.purge, (list, np.ndarray, int, str)):
                if isinstance(self.purge, (int, str)):
                    self.purge = [self.purge]
                # If all entries are strings, find the names of the candidates.
                if all([isinstance(x, str) for x in self.purge]):
                    if not all(np.isin(self.purge, self.candidates)):
                        raise ValueError("One or more names in paternityArray.purge are not found in paternityArray.candidates")
                    self.purge = [np.where(x == self.candidates)[0][0] for x in self.purge]
                with(np.errstate(divide='ignore')):
                    new_array[:, self.purge] = np.log(0)
            else:
                raise TypeError("Error: purge should be a float or list of floats between zero and one.")

        # correct for input parameter for proportion of missing fathers.
        if not isinstance(self.missing_parents, (int, float)):
            raise TypeError("missing_parents should be between 0 and 1.")
        # apply correction for the prior on number of missing parents.
        if self.missing_parents < 0 or self.missing_parents >1:
            raise ValueError("missing_parents must be between 0 and 1!")
        # if missing_parents is between zero and one, correct the likelihoods.
        if self.missing_parents >= 0 and self.missing_parents <= 1:
            if self.missing_parents ==0: warn("Missing_parents set to 0. Only continue if you are sure you really have 100% of possible fathers.")
            if self.missing_parents ==1: warn("Missing_parents set to 100%. Are you sure this is what you mean?")
            with(np.errstate(divide='ignore')):
                new_array[:, -1] = new_array[:, -1] + np.log(  self.missing_parents)
                new_array[:,:-1] = new_array[:,:-1] + np.log(1-self.missing_parents)
        # if missing_parents is 0, set the term for unrelated fathers to zero.
        if self.missing_parents == 0:
            with(np.errstate(divide='ignore')): new_array[:,-1] = np.log(0)

        # correct for selfing rate.
        if self.selfing_rate is not None:
            if not isinstance(self.selfing_rate, (int, float)):
                raise TypeError("selfing_rate should be between 0 and 1.")
            if self.selfing_rate < 0 or self.selfing_rate >1:
                raise ValueError("selfing_rate must be between 0 and 1.")
            if self.selfing_rate >=0 and self.selfing_rate <=1:
                if self.selfing_rate == 1: warn("selfing_rate set to 100%. Are you sure that is what you meant?")

                ix = range(len(self.offspring))
                with np.errstate(divide='ignore'):
                    maternal_pos = [np.where(np.array(self.candidates) == self.mothers[i])[0][0] for i in ix] # positions of the mothers
                    new_array[ix, maternal_pos] += np.log(self.selfing_rate)

        # set the likelihood dyads with many incompatibilities to zero
        if self.max_clashes is not None:
            if not isinstance(self.max_clashes, int):
                raise TypeError("paternityArray.max_clashes should be a positive integer")
            if self.max_clashes <= 0:
                raise ValueError("paternityArray.max_clashes should be greater than zero.")
            if self.clashes is None:
                raise TypeError("Unable to adjust for number of incompatible homozygous loci because `paternityArray.clashes` is `None`.")
            elif self.clashes.shape != self.lik_array.shape:
                raise ValueError("Shape of the likelihood array does not match that of the array of clashes.")
            else:
                inc = np.append(self.clashes, np.zeros(self.lik_absent.shape)[:,np.newaxis], 1) # add an extra column so the shapes match
                with np.errstate(divide='ignore'):
                    ix = np.log(inc <= self.max_clashes) # index elements to alter
                new_array = new_array + ix

        # normalise so rows sum to one.
        new_array = new_array - alogsumexp(new_array, axis=1)[:,np.newaxis]

        return new_array

    def subset(self, indices):
        """
        Subset offspring in a paternity array.

        Parameters
        ----------
        indices: List or array of integers
            Positions of individuals to subset.

        Returns
        -------
        A paternityArray object for the individuals indexed by `indices`.

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

        # Pull out data for only the first family
        patlik.subset([0,1,2,3,4])
        """
        # If index is for a single individual, make it a list anyway.
        if isinstance(indices, int):
            individuals = [indices]
        # Subset original data.
        new_array = paternityArray(
            likelihood = self.lik_array[indices],
            lik_absent = self.lik_absent[indices],
            offspring  = self.offspring[indices],
            mothers    = self.mothers[indices],
            fathers    = self.fathers[indices],
            candidates = self.candidates
        )
        # Add additional attributes is these exist
        if self.mu           is not None: new_array.clashes      = self.mu
        if self.clashes      is not None: new_array.clashes      = self.clashes
        if self.covariate    is not None: new_array.covariate    = self.covariate

        # Return substted paternityArray
        return new_array

    def split(self, by, return_dict=True):
        """
        Split up a paternityArray into groups according to some grouping
        factor. For example, divide an array for multiple half-sibling
        arrays by the ID of their mothers.

        Parameters
        ----------
        by: array-like
            Vector containing grouping labels for each individual.
        return_dict: logical
            If True, the output is returned as a dictionary of paternityArray
            objects indexed by entries in `by`. If False, a list is returned.
            Defaults to True.

        Returns
        -------
        A dictionary of paternityArray objects.

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

        # Pull out data for only the first family
        patlik.split(by=progeny.fathers)
        """
        groups = np.unique(by)
        ix = [np.where(by == i)[0] for i in groups]
        # Split into separate arrays.
        if return_dict:
            output = {k:self.subset(i) for k,i in zip(groups, ix)}
        else:
            output = [self.subset(i) for i in ix]
        #return output
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
