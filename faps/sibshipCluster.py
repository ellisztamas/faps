import numpy as np
from alogsumexp import alogsumexp
from relation_matrix import relation_matrix
from draw_fathers import draw_fathers
from lik_partition import lik_partition

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
        Z-matrix from scipy.cluster.hierarchy.
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
    def __init__(self, paternity_array, linkage_matrix, partitions, lik_partitions):
        self.paternity_array= paternity_array.prob_array
        self.partitions     = partitions
        self.linkage_matrix = linkage_matrix
        self.lik_partitions = lik_partitions
        self.prob_partitions= self.lik_partitions - alogsumexp(self.lik_partitions)
        self.mlpartition    = self.partitions[np.where(self.lik_partitions == self.lik_partitions.max())[0][0]]
        self.noffspring     = len(self.mlpartition)
        self.npartitions    = len(self.lik_partitions)

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
        nfams  = np.array([len(np.unique(self.partitions[i])) for i in range(self.npartitions)])
        # sum the probabilities of each partition containing each value of family number
        nprobs = np.array([pprobs[np.where(i == nfams)].sum() for i in range(1, len(nfams)+1)])
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

    def full_sib_matrix(self, exp=False):
        """
        Create a matrix of log posterior probabilities that pairs of offspring are
        full siblings. This sums over the probabilities of each partition in which
        two individuals are full siblings, multiplied by the probability of that
        partition.

        By default, this creates a 3-dimensional matrix of log probabilities and
        sums using logsumexp, which preserves values in log space. Alternatively
        values can be exponentiated and summed directly, which will be less
        demanding on memory and the processor for large arrays, but probably at some
        cost to accuracy.

        Parameters
        ----------
        exp: logical
            If True, exponentiate log probabilities and sum these directly.
            Defaults to `False`.

        Returns
        -------
        An n*n array of log probabilities, where n is the number of offspring in the
        `sibshipCluster`.
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
            print "Reference partition should be the same length as the number of offspring."
            return None

        obs = self.full_sib_matrix()
        rm  = relation_matrix(reference)

        # Matrix of ones and zeroes to reference elements for each relationship type.
        if   rtype is 'all': ix = np.triu(np.ones(rm.shape), 1)
        elif rtype is 'fs':  ix = np.triu(rm, 1)
        elif rtype is 'hs':  ix = np.triu(1-rm, 1)
        else:
            print "rtype must be one of 'all', 'fs' or 'hs'."
            return None

        # Get accuracy scores
        dev = abs(rm - np.exp(obs))
        dev = dev * ix
        dev = dev.sum() / ix.sum()

        return 1- dev

    def prob_paternity(self, reference=None):
        """
        Posterior probabilities of paternity for a set of reference fathers accounting
        for uncertainty in sibship structure.

        Parameters
        ----------
        reference: int, array-like, optional
            Indices for the candidates to return. If an integer, returns probabilties
            for a single candidate individual. To return probabilities for a vector
            of candidates, supply a list or array of integers of the same length as
            the number of offspring..

        Returns
        -------
        Array or vector of log posterior probabilities.
        """
        if reference is None:
            probs = np.zeros([self.npartitions, self.noffspring, self.paternity_array.shape[1]]) # empty matrix to store probs for each partitions
            for j in range(self.npartitions): # loop over partitions
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
                    print "If the set of reference candidates is given as a list or numpy vector"
                    print "this must be of the same length as the number of offspring."
                    return None
                if any([reference[i] > self.paternity_array.shape[1] for i in range(len(reference))]):
                    print "One or more indices in reference are greater than the number of candidates."
                    return None
            # If a single candidate has been supplied
            elif isinstance(reference, int):
                if reference > self.paternity_array.shape[1]:
                    print "The index for the reference candidate is greater than the number of candidates."
                    return None
                else:
                    reference = [reference] * self.noffspring
            else:
                print "reference should be given as a list or array of the same length as the number of offspring,"
                print "or else a single integer."
                return None

            probs = np.zeros([self.npartitions, self.noffspring]) # empty matrix to store probs for each partitions
            for j in range(self.npartitions): # loop over partitions
                this_part = self.partitions[j]
                this_array = np.array([self.paternity_array[this_part[i] == this_part].sum(0) for i in range(self.noffspring)])
                this_array = this_array - alogsumexp(this_array,1)[:, np.newaxis] # normalise
                probs[j]   = np.diag(this_array[:, reference]) # take only diagnical elements
            probs = probs + self.prob_partitions[:, np.newaxis]
            probs = alogsumexp(probs, 0)

            return probs

    def prob_mating(self, phenotypes, null_probs=None, ndraws=1000):
        """
        Calulate the probability that the mother mated with a male of each of a set of phenotype
        categories. Sires are drawn for each full-sib family for each partition proportional to
        his probability of paternity for that family. Since two full-sibships cannot share the
        same father, draws for which two families share a father are removed.

        Optionally, a sample of candidates can also be drawn at random, or proportional to some
        other distribution. This can act as a null distribution to determine whether observed
        patterns can be explained by random mating.

        Parameters
        ----------
        phenotypes: array-like
            A vector of phenotypes for each male. Phenotypes should be divided into discrete
            categories. For continuous variables, this can take the form of grouping values
            into bins of a set width.
        ndraws: int
            Number of Monte Carlo draws for each family.
        null_probs: array-like or 'uniform', optional
            Optional argument specifying whether a second set of sires should be drawn corresponding
            to random mating. To draw candidates at random, give 'uniform'. To draw candidates from
            some other distribution, for example a function of distance, supply a vector of
            probabilities for each candidate that sums to one. If no argument is supplied, no extra
            draws are made.

        Returns
        -------
        A tuple containing (1) names of each phenotype, (2) probabilities of mating with each phenotype
        class, and optionally (3) probabilities of mating with each phenotyp class under random mating.
        The term labelled 'absent' indicates the probability that the father was unsampled, and hence
        has no phenotype information.
        """

        # exclude partitions with zero posterior probability
        valid_partitions = self.partitions[np.isfinite(self.lik_partitions)]
        # convert phenotypes to list, and concatenate term for missing fathers.
        if isinstance(phenotypes, np.ndarray): phenotypes = phenotypes.tolist()
        phenotypes = phenotypes + ['absent']

        # list of unique phenotype categories.
        cats = list(set(phenotypes))
        cats = [str(cats[i]) for i in range(len(cats))]

        # an empty matrix to store phenotypes
        count_table = np.zeros([len(cats), len(valid_partitions)])

        if null_probs is None:
            for j in range(len(valid_partitions)): # loop over partitions
                this_part    = valid_partitions[j]
                these_dads   = draw_fathers(this_part, self.paternity_array, null_probs=None, ndraws=ndraws)
                these_phens  = np.array(phenotypes)[these_dads] # subset list of phenotypes for the candidates sampled.
                #print these_phens
                these_counts = np.array([sum(cats[i] == these_phens) for i in range(len(cats))]) # how many sampled fathers match each phenotype
                these_counts = these_counts.astype('float') / len(these_dads)
                #print these_counts
                count_table[:,j] = np.log(these_counts)
            # weight by probability of each partition, and sum over partition structures.
            logprobs = count_table + self.prob_partitions[np.isfinite(self.lik_partitions)]
            logprobs = alogsumexp(logprobs, 1)

            return cats, logprobs
        else:
            null_table = np.copy(count_table) # empty matrix for NULL fathers.
            for j in range(len(valid_partitions)): # loop over partitions
                this_part    = valid_partitions[j]
                these_dads, null_dads = draw_fathers(this_part, self.paternity_array, null_probs, ndraws)
                # # subset list of phenotypes for the candidates sampled.
                these_phens  = np.array(phenotypes)[these_dads]
                null_phens   = np.array(phenotypes)[null_dads]
                # how many sampled fathers match each phenotype.
                these_counts = np.array([sum(cats[i] == these_phens) for i in range(len(cats))])
                these_counts = these_counts.astype('float') / len(these_dads)
                count_table[:,j] = np.log(these_counts)
                # how many null fathers match each phenotype.
                null_counts = np.array([sum(cats[i] == null_phens) for i in range(len(cats))])
                null_counts = null_counts.astype('float') / len(null_dads)
                null_table[:,j] = np.log(null_counts)
            # weight by probability of each partition, and sum over partition structures.
            logprobs = count_table + self.prob_partitions[np.isfinite(self.lik_partitions)]
            lognull  = null_table  + self.prob_partitions[np.isfinite(self.lik_partitions)]
            logprobs = alogsumexp(logprobs, 1)
            lognull  = alogsumexp(lognull, 1)

            return cats, logprobs, lognull

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
            sire_probs = self.prob_paternity(sire_ix)
            sire_probs = sire_probs[dad_present]
            sire_probs = alogsumexp(sire_probs) - np.log(len(np.array(sire_probs))) # take mean
        else:
            sire_probs = np.nan

        # Mean probability of being absent for those sires with zero prob of paternity
        dad_absent = np.isinf(self.paternity_array[range(progeny.size), sire_ix]) # index sires with zero probability of paternity
        if any(dad_absent):
            abs_probs = self.prob_paternity()[dad_absent, -1]
            abs_probs = alogsumexp(abs_probs) - np.log(len(abs_probs))
        else:
            abs_probs = np.nan

        output = np.array([true_found,
                           delta_lik,
                           round(nfamilies, 3),
                           round(full_sibs, 3),
                           round(half_sibs, 3),
                           round(all_sibs,  3),
                           round(sire_probs,3),
                           round(abs_probs,3)])
        return output
