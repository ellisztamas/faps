import numpy as np
from warnings import warn
from faps.paternityArray import paternityArray
from faps.alogsumexp import alogsumexp
from faps.relation_matrix import relation_matrix
from faps.draw_fathers import draw_fathers
from faps.lik_partition import lik_partition
from pandas import DataFrame

class sibshipCluster(object):
    """
    Information on  the results of hierarchical clustering of an offspring array
    into full sibling groups.

    This is typcially not called directly, but through an instance of the function
    `paternity_array`.

    Parameters
    ----------
    paternity_array: paternityArray
        Object listing information on paternity of individuals.
    linkage_matrix: array
        Z-matrix from fastcluster.linkage.
    partitions: 2-d array
        Array of possible partition structures from the linkage matrix.
    lik_partitions: 1d-array
        Vector of log likelihoods for each partition structure.

    Returns
    -------
    prob_partitions: array
        log posterior probabilities of each partition structure (`lik_partitions`
        normalised to sum to one).
    mlpartition: list
        maximum-likelihood partition structure.
    noffspring: int
        Number of offspring in the array.
    npartitions: int
        Number of partitions recovered from the dendrogram.
    """
    def __init__(self, paternity_array, linkage_matrix, partitions, lik_partitions, paths, path_likelihoods, path_probs, covariate):
        self.candidates       = paternity_array.candidates
        self.offspring        = paternity_array.offspring
        self.mothers          = paternity_array.mothers
        self.paternity_array  = paternity_array.prob_array()
        self.partitions       = partitions
        self.linkage_matrix   = linkage_matrix
        self.lik_partitions   = lik_partitions
        self.prob_partitions  = self.lik_partitions - alogsumexp(self.lik_partitions)
        self.mlpartition      = self.partitions[np.where(self.lik_partitions == self.lik_partitions.max())[0][0]]
        self.noffspring       = len(self.mlpartition)
        self.npartitions      = len(self.lik_partitions)
        self.paths            = paths
        self.path_likelihoods = path_likelihoods
        self.path_probs       = path_probs
        self.covariate       = covariate

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

    def accuracy(self, progeny, adults):
        """
        Summarise statistics about the accuracy of sibship reconstruction when
        the true genealogy is known (for example from simulated families).

        Parameters
        ----------
        progeny: genotypeArray
            Genotype information on the progeny
        adults: genotypeArray
            Genotype information on the adults

        Returns
        -------
        Vector of statistics:
        0. Binary indiciator for whether the true partition was included in the
            sample of partitions.
        1. Difference in log likelihood for the maximum likelihood partition
            identified and the true partition. Positive values indicate that the
            ML partition had greater support.
        2. Posterior probability of the true number of families.
        3. Mean probabilities that a pair of full sibs are identified as full sibs.
        4. Mean probabilities that a pair of half sibs are identified as half sibs.
        5. Mean probabilities that a pair of half or full sibs are correctly
            assigned as such.
        6. Mean probability of paternity of the true sires for those sires who
            had been sampled (who had non-zero probability in the paternityArray).
        7. Mean probability that the sire had not been sampled for those
            individuals whose sire was truly absent (who had non-zero probability
            in the paternityArray).
        """
        # Was the true partition idenitifed by sibship clustering.
        true_part  = progeny.true_partition()
        nmatches   = np.array([(relation_matrix(self.partitions[x]) == relation_matrix(true_part)).sum()
                            for x in range(self.npartitions)])
        nmatches   = 1.0*nmatches / true_part.shape[0]**2 # divide by matrix size.
        true_found = int(1 in nmatches) # return 1 if the true partition is in self.partitions, otherwise zero

        delta_lik  = round(self.lik_partitions.max() - lik_partition(self.paternity_array, true_part),2) # delta lik
        # Prob correct number of families
        if len(self.nfamilies()) < progeny.nfamilies:
            nfamilies  = 0
        else:
            nfamilies = self.nfamilies()[progeny.nfamilies-1]
        # Pairwise sibship relationships
        full_sibs = self.partition_score(progeny.true_partition(), rtype='fs') # accuracy of full sibship reconstruction
        half_sibs = self.partition_score(progeny.true_partition(), rtype='hs') # accuracy of full sibship reconstruction
        all_sibs  = self.partition_score(progeny.true_partition(), rtype='all')# accuracy of full sibship reconstruction

        # Mean probability of paternity for true sires included in the sample.
        sire_ix = progeny.parent_index('f', adults.names) # positions of the true sires.
        dad_present = np.isfinite(self.paternity_array[range(progeny.size), sire_ix]) # index those sires with non-zero probability of paternity
        if any(dad_present):
            sire_probs = self.posterior_paternity_matrix(sire_ix)
            sire_probs = sire_probs[dad_present]
            sire_probs = alogsumexp(sire_probs) - np.log(len(np.array(sire_probs))) # take mean
            sire_probs = sire_probs.squeeze()
        else:
            sire_probs = np.nan

        # Mean probability that the father is absent
        abs_probs = np.exp(self.posterior_paternity_matrix(-1)).mean()

        output = np.array([true_found,
                           delta_lik,
                           np.round(nfamilies, 3),
                           np.round(full_sibs, 3),
                           np.round(all_sibs,  3),
                           np.round(half_sibs, 3),
                           np.round(np.exp(sire_probs),3),
                           np.round(abs_probs,3)])
        return output

    def nfamilies(self):
        """
        Posterior probability distribution of the number of full sibships in the
        array.

        Returns
        -------
        A vector of (exponentiated) probabilities that the array contains each
        integer value of full sibships from one to the maximum possible.
        """
        pprobs = np.exp(self.prob_partitions) # exponentiate partition likelihoods for simplicity
        # number of families in each partition
        nfams  = np.array([len(np.unique(i)) for i in self.partitions])
        # sum the probabilities of each partition containing each value of family number
        nprobs = np.array([pprobs[np.where(i == nfams)].sum() for i in range(1, self.noffspring+1)])
        nprobs = nprobs / nprobs.sum() # normalise
        return nprobs

    def family_size(self):
        """
        Multinomial posterior distribution of family sizes within the array,
        averaged over all partitions.

        Returns
        -------
        A vector of probabilities of observing a family of size *n*, where *n* is
        all integers from one to the number of offspring in the array.
        """
        pprobs = np.zeros(self.noffspring) # empty vector to store sizes
        # For each partition get the counts of each integer family size.
        for j in range(self.npartitions):
            counts = np.bincount(np.unique(self.partitions[j], return_counts=True)[1], minlength=self.noffspring+1).astype('float')[1:]
            counts = counts / counts.sum() # normalise to sum to one.
            pprobs+= counts * np.exp(self.prob_partitions[j]) # multiply by likelihood of the partition.
        return pprobs

    def mean_nfamilies(self):
        """
        Expected number of families given probabilities of each partition.
        """
        return np.average(np.arange(1, self.noffspring+1), weights=self.nfamilies())

    def full_sib_matrix(self, exp=False):
        """
        Create a matrix of log posterior probabilities that pairs of offspring
        are full siblings. This sums over the probabilities of each partition
        in which two individuals are full siblings, multiplied by the
        probability of that partition.

        By default, this creates a 3-dimensional matrix of log probabilities
        and sums using logsumexp, which preserves values in log space.
        Alternatively values can be exponentiated and summed directly, which
        will be less demanding on memory and the processor for large arrays,
        but probably at some cost to accuracy.

        Parameters
        ----------
        exp: logical
            If True, exponentiate log probabilities and sum these directly.
            Defaults to `False`.

        Returns
        -------
        An n*n array of log probabilities, where n is the number of offspring
        in the `sibshipCluster` object.
        """
        if exp is True:
            sibmat = np.zeros([self.noffspring, self.noffspring])
            for j in range(self.npartitions):
                sibmat+= np.exp(self.prob_partitions[j]) * np.array([self.partitions[j][i] == self.partitions[j] for i in range(self.noffspring)])

        if exp is False:
            sibmat = np.zeros([self.npartitions, self.noffspring, self.noffspring])
            with np.errstate(divide='ignore'):
                for j in range(self.npartitions):
                    sibmat[j] = self.prob_partitions[j] + np.log(np.array([self.partitions[j][i] == self.partitions[j] for i in range(self.noffspring)]))
            sibmat = alogsumexp(sibmat, 0)

        return sibmat

    def partition_score(self, reference, rtype='all'):
        """
        Returns the accuracy score for the `sibshipCluster` relative to a
        reference partition. This is usually the known true partition for a
        simulated array, where the partition is known.

        Accuracies can be calculated for only full-sibling pairs, only
        non-full-sibling pairs, or for all relationships.

        Parameters
        ----------
        reference: list
            Reference partition structure to refer to. This should be a list or
            vector of the same length as the number of offspring.
        rtype str
            Indicate whether to calculate accuracy for full-sibling, half-sibling,
            or all relationships. This is indicated by 'fs', 'hs' and 'all'
            respectively. Note that half-sibling really means 'not a full sibling'.
            The distinction is only meaningful for data sets with multiple
            half-sib families.

        Returns
        -------
        A float between zero and one.
        """

        if len(reference) != self.noffspring:
            raise ValueError("Reference partition should be the same length as the number of offspring.")
            return None

        obs = self.full_sib_matrix()
        rm  = relation_matrix(reference)

        # Matrix of ones and zeroes to reference elements for each relationship type.
        if   rtype == 'all': ix = np.triu(np.ones(rm.shape), 1)
        elif rtype == 'fs':  ix = np.triu(rm, 1)
        elif rtype == 'hs':  ix = np.triu(1-rm, 1)
        else:
            raise ValueError("rtype must be one of 'all', 'fs' or 'hs'.")
            return None

        # Get accuracy scores
        dev = abs(rm - np.exp(obs))
        dev = dev * ix
        dev = dev.sum() / ix.sum()

        return 1- dev

    def posterior_paternity_matrix(self, reference=None):
        """
        Posterior probabilities of paternity accounting for uncertainty in
        sibship structure. This is analogous to paternityArray.prob_array(), 
        except that the latter is calculated on shared alleles between progeny
        and individual candidates only, and therefore does not include
        information from sibships.

        Parameters
        ----------
        reference: int, array-like, optional
            Indices for the candidates to return. If an integer, returns
            probabilties for a single candidate individual. To return
            probabilities for a vector of candidates, supply a list or array of
            integers of the same length as the number of offspring.

        Returns
        -------
        An array of (log) probabilities with a row for each offspring and a
        column for each candidate father. Each element contains the posterior
        probability of paternity for the pair *after* averaging over possible
        sibships structures.
        """
        if reference is None:
            # empty matrix to store probs for each partitions
            probs = np.zeros([self.npartitions, self.noffspring, self.paternity_array.shape[1]])
            # loop over partitions
            for j in range(self.npartitions):
                this_part = self.partitions[j]
                this_array = np.array([self.paternity_array[this_part[i] == this_part].sum(0) for i in range(self.noffspring)])
                this_array = this_array - alogsumexp(this_array,1)[:, np.newaxis] # normalise
                this_array+= self.prob_partitions[j] # multiply by probability of this partition.
                probs[j] = this_array
            probs = alogsumexp(probs, axis=0)
            return probs

        else:
            # If a vector of candidates has been supplied.
            if isinstance(reference, list) or isinstance(reference, np.ndarray):
                if len(reference) != self.noffspring:
                    raise ValueError("If the set of reference candidates is given as a list or numpy vector this must be of the same length as the number of offspring.")
                if any([reference[i] > self.paternity_array.shape[1] for i in range(len(reference))]):
                    raise ValueError("One or more indices in reference are greater than the number of candidates.")
            # If a single candidate has been supplied
            elif isinstance(reference, int):
                if reference > self.paternity_array.shape[1]:
                    raise ValueError("The index for the reference candidate is greater than the number of candidates.")
                else:
                    reference = [reference] * self.noffspring
            else:
                raise TypeError("reference should be given as a list or array of the same length as the number of offspring, or else a single integer.")

            probs = np.zeros([self.npartitions, self.noffspring]) # empty matrix to store probs for each partitions
            for j in range(self.npartitions): # loop over partitions
                this_part = self.partitions[j]
                this_array = np.array([self.paternity_array[this_part[i] == this_part].sum(0) for i in range(self.noffspring)])
                this_array = this_array - alogsumexp(this_array,1)[:, np.newaxis] # normalise
                probs[j]   = np.diag(this_array[:, reference]) # take only diagnical elements
            probs = probs + self.prob_partitions[:, np.newaxis]
            probs = alogsumexp(probs, 0)

            return probs

    def paternity(self, n_candidates = 4):
        """
        For each offspring in a half-sib array, pull out the ID and log
        posterior probability of paternity for the top candidates.
        
        Parameters
        ----------
        n_candidates: int
            Maximum number of top candidates to return. Defaults to the top
            four candidates.

        Returns
        -------
        Dataframe with a row for every offspring, showing the names and log
        posterior probabilities of paternity for the N most-probable candidates.
        """
        probs = self.paternity_array
        sort_index = np.argsort(probs, axis = 1)
        # Add an extra entry to the list of candidates for unsampled fathers.
        candidate_names = np.append(self.candidates, "missing")

        # Data frame with a column of only progeny names for now
        out = DataFrame({
            "progeny" : self.offspring
        })
        # Append the dataframe with columns for the names and posterior probabilities of paternity for the N top candidates.
        for n in range(1,n_candidates + 1):
            out = out.join(
                DataFrame({
                    "candidate_"      + str(n) : candidate_names[ sort_index ][:, -n],
                    "logprob_" + str(n) : np.partition(probs, (-n), axis =1)[:, -n]
                })
            )

        return out

    def sires(self, labels='names'):
        """
        For every candidate drawn as a father in any partition, calculate the
        (log) probability that each sired at least one offspring.

        Probabilities are calculated as the sum of likelihoods for every 
        path in every partition in which a candidate appears.

        Parameters
        ----------
        labels: array, optional
            Labels for candidates. By default, the names of candidates are used.
            If `labels=None`, the index positions in the list of candidates are
            used; this can be useful if you want to use indices to subset
            another dataset and link that to probabilities  of paternity.

        Returns
        -------
        DataFrame giving mother (taken from the keys of the input dictionary),
        fathers (inherited from each sibshipCluster object), probabilties of
        having sired at least one offspring, and the expected number of
        offspring.


        Examples
        --------
        from faps import *
        import numpy as np

        # Generate a population of adults
        allele_freqs = np.random.uniform(0.3,0.5,50)
        adults = make_parents(20, allele_freqs)

        # Ecample with a single family
        # Mate the first adult to the next three.
        mother = adults.subset(0)
        progeny = make_sibships(adults, 0, [1,2,3], 5, 'x')
        patlik = paternity_array(progeny, mother, adults, mu=0.0013)
        sc = sibship_clustering(patlik)

        sc.sires() # returns candidate names by default
        """
        if labels == 'names':
            labels = np.append(self.candidates, np.nan)

        # Get list of the unique set of fathers drawn for any partition.
        sires = np.unique([i for j in [x for y in self.paths.values() for x in y] for i in j])

        # Flatten lists of paths and their likelihoods
        flat_paths    = [x for y in self.paths.values() for x in y]
        flat_pathliks = [x for y in self.path_probs.values() for x in y if np.isfinite(x)]
        # For each sire drawn, find all paths he is involved in and sum their likelihoods.
        output = {}
        for s in sires:
            sx = [s in x for x in flat_paths]
            these_liks = np.array(flat_pathliks)[sx]
            output[s] = alogsumexp(these_liks)

        # Flip into data.frame
        output = DataFrame({
                    'position' : list(output.keys()),
                    'label'    : [labels[k] for k in output.keys()],
                    'log_prob' : list(output.values())
                })
        output['prob'] = np.exp(output['log_prob'])
        # Get the expected number of offspring for each mating event
        noffs = self.posterior_paternity_matrix()[:, output['position']]
        noffs = alogsumexp(noffs, axis=0)
        output['offspring'] = np.exp(noffs)
        # Return DataFrame of information on sires
        return output

    def posterior_mating(self, ndraws=10000, use_covariates=True, covariates_only=False, down_sample = True):
        """
        Simulate plausible mating events from the posterior distribution of all
        possible pairings between the mother and candidate fathers, integrating
        over uncertainty in sibship structure.

        For a single partitition structure, `posterior_mating` draws a sample of
        putative fathers for each full-sib family in propotion to posterior
        probabilities of paternity from genetic information. Paternity 
        information from covariate probabilities can be incorporated as well by
        setting `use_covariates` to `True`. Covariate data are taken from the 
        sibshipCluster object directly (see `sibshipCluster.add_covariate`).
        
        Samples are drawn for every partition structure, excluding partitions 
        with zero posterior probability. Samples for each partition are then 
        subsampled in proportion to the posterior probability of each partition 
        to generate a final sample of plausible fathers for the whole half-sib 
        family. In this way, fathers are drawn in proportion to their
        probability of paternity, and uncertainty in sibship structure is 
        accounted for.

        Samples of fathers can also be drawn in proportion to covariate 
        probabilities only by setting `covariates_only` to `True`. This can be 
        used to compare "real" mating events inferred using genetic (potentially
        including covariate information) to a null distribution of mating based
        on covariates only. For example, if covariates describe a model of 
        dispersal, such a comparison might tell you if there is non-random
        mating for some trait other than distance.

        Parameters
        ----------
        ndraws: int
            Number of Monte Carlo draws for each family.
        use_covariates: logical, optional
            If True, information on prbabilities associated with covariates
            stored in paternityArray objects are incorporated into weights for 
            drawing likely fathers.
        covariates_only: boolean, optional
            If True, candidates are drawn based on covariate probabilities only 
            (i.e. ignoring genetic data)
        down_sample: boolean, optional
            If True, downsample the output so that there are as many mating
            events as there would be in reality, by downsampling mating events
            to the most-likely number of full sibships in the array. If False,
            ndrawds candidates are returned, which are likely to include most 
            of the candidates in the sample at least once.

        Returns
        -------
        A DataFrame giving plausible fathers that could have mated with the
        mother, along with the proportion of pollen coming from each father.

        Examples
        --------
        # Simulate a starting population
        allele_freqs = np.random.uniform(0.3,0.5,50)
        adults = fp.make_parents(100, allele_freqs, family_name='a')
        progeny = fp.make_sibships(adults, 0, [1,2,3], 5, 'x')
        mothers = adults.subset(progeny.mothers)
        patlik  = fp.paternity_array(progeny, mothers, adults, mu = 0.0015, missing_parents=0.01)
        sc = fp.sibship_clustering(patlik)
        # Check posterior_mating returns what it should in ideal case
        me = sc.posterior_mating()

        # Remove one of the fathers and check that a missing dad is sampled.
        patlik.purge = "a_1"
        sc2 = fp.sibship_clustering(patlik)
        me2 = sc2.posterior_mating()

        # Include a nonsense covariate
        cov = np.arange(0,adults.size)
        cov = -cov/cov.sum()
        patlik.add_covariate(cov)
        sc3 = fp.sibship_clustering(patlik, use_covariates=True)
        me3 = sc3.posterior_mating(use_covariates=True)

        # Draw individuals based on the covariate only.
        sc4 = fp.sibship_clustering(patlik, use_covariates=True)
        me4 = sc4.posterior_mating(use_covariates=True, covariates_only=True)
        """
        dad_names = np.append(self.candidates, "missing")

        # only consider partitions that would account for at least one mating event.
        valid_ix = np.around(ndraws * np.exp(self.prob_partitions)) >= 1
        valid_partitions = self.partitions[valid_ix]
        # Get names for those
        unit_names = ['partition_' + str(x) for x in np.where(valid_ix)[0]]

        # draw mating events for each partition.
        unit_events = {}
        for k,v in zip(unit_names, valid_partitions):
            draws = draw_fathers(
                v,
                genetic = self.paternity_array,
                covariate = self.covariate, 
                ndraws=ndraws,
                use_covariates=use_covariates,
                covariates_only = covariates_only
                )
            if len(draws) > 0:
                unit_events[k] = draws
        # Resample mating events for each partition weighted by the probability
        # for that partition to give a total sample of ndraws.
        # In fact, sample size may be a little above or below ndraws, because 
        # rounding probabilities to integers means things don't always add up.
        #
        # First, get an set of integer number of draws for each partition.
        unit_weights = np.around(np.exp(self.prob_partitions[valid_ix]) * ndraws).astype('int')
        unit_weights = {k:v for k,v in zip(unit_names, unit_weights)}
        # Resample unit_events proportional to the prob of each unit.
        total_events = [np.random.choice(a=v, size=unit_weights[k], replace=True) for k,v in unit_events.items()]
        total_events = [item for sublist in total_events for item in sublist]
        total_events = dad_names[total_events]
        # Count up how often each candidate appears and return a DataFrame
        dad, freq = np.unique(total_events, return_counts = True)
        
        # If dads have been drawn based on covariates only, downsample the
        # output so that there are as many mating events as there would be in 
        # reality, by downsampling mating events to the most-likely number of
        # full sibships in the array.
        if covariates_only and down_sample:
            nfamilies = np.argsort(self.nfamilies())[-1] +1
            ix = np.random.choice(range(len(dad)),  nfamilies)
            dad, freq = dad[ix], freq[ix]
        
        output = DataFrame({
            'father'    : dad, 
            'frequency' : freq/sum(freq)
            })

        return output