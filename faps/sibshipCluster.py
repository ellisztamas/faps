import numpy as np
from paternityArray import paternityArray
from matingEvents import matingEvents
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

        # Mean probability that the father is absent
        abs_probs = np.exp(self.prob_paternity(-1)).mean()

        output = np.array([true_found,
                           delta_lik,
                           round(nfamilies, 3),
                           round(full_sibs, 3),
                           round(half_sibs, 3),
                           round(all_sibs,  3),
                           round(np.exp(sire_probs),3),
                           round(abs_probs,3)])
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
        
    def mean_nfamilies(self):
        """
        Expected number of families given probabilities of each partition.
        """
        return np.average(np.arange(self.noffspring), weights = self.nfamilies())

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
        Posterior probabilities of paternity for a set of reference fathers
        accounting for uncertainty in sibship structure.

        Parameters
        ----------
        reference: int, array-like, optional
            Indices for the candidates to return. If an integer, returns
            probabilties for a single candidate individual. To return
            probabilities for a vector of candidates, supply a list or array of
            integers of the same length as the number of offspring.

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
            
    def mating_events(self, paternity, unit_draws=1000, total_draws=10000, n_subsamples = None, subsample_size = None, null_probs=None):
        """
        Sample plausible mating events from a sibshipCluster. These can then be used
        to infer mating patterns using and array of phenotypes for each male.

        Indices for possible fathers are drawn based on genetic data, other pertinent
        information (such as distance between parents), or a combination of these
        types. Mating events are drawn for each partition in a `sibshipCluster` object.
        These samples are then resampled in proportion to a weight on each unit.
        To assess uncertainty, the aggregate mating events can be subsampled.

        You can also supply an array of probabilities that a given father is the sire
        based on some appropriare null model, which can then be compared with observed
        mating patterns. For example, to test whether observed patterns could be
        explained by random mating but with spatial clustering, you could supply an 
        array of probabilities derived from an appropriate dispersal function.

        Parameters
        ----------
        paternity: paternityArray
            Probabilities of paternity used to construct the sibshipCluster
            object.
        unit_draws: int
            Number of mating events to sample for each partition.
        total_draws: int
            Total number of mating events to resample for each partition.
        n_subsamples: int, optional
            Number of subsamples to draw from the total mating events.
        subsample_size: int, optional
            Number of mating events in each subsample. Defaults to 0.1*total_draws.
        null_probs: array, optional
            Array of probabilities for paternity if this were not based on marker
            data. This should be the same shape as the prob_array in the paternity
            argument, i.e. have a row for every offspring and a column for every
            candidate.

        Returns
        -------
        A matingEvents object.
        """
        if not isinstance(paternity, paternityArray):
            raise TypeError('paternity should be a paternityArray.')
        elif paternity.prob_array.shape != self.paternity_array.shape:
            raise ValueError('Number of offspring in paternity ({}) does not match the sibshipCluster ({}).'.format(paternity.prob_array.shape, self.paternity_array.shape))

        if not isinstance(unit_draws, int):
            raise ValueError('unit_draws should be an integer.')
        if not isinstance(total_draws, int):
            raise ValueError('total_draws should be an integer.')
        if not isinstance(n_subsamples, int) and n_subsamples is not None:
            raise ValueError('n_subsamples should be an integer.')
        if not isinstance(subsample_size, int) and subsample_size is not None:
            raise ValueError('subsample_size should be an integer.')
        if subsample_size is not None and subsample_size >= total_draws:
            raise ValueError('Number of subsamples must be smaller than total_draws')

        # only consider partitions that would account for at least one mating event.
        valid_ix = np.around(total_draws * np.exp(self.prob_partitions)) >= 1
        valid_partitions = self.partitions[valid_ix]

        if isinstance(null_probs, np.ndarray):
            prob_array = null_probs
        elif null_probs is None:
            prob_array = paternity.prob_array
        else:
            raise TypeError('null_probs should be an array, or None.')
        
        # draw mating events for each partition.
        unit_events = [draw_fathers(i, prob_array, unit_draws) for i in valid_partitions]

        # resample weighted by probability
        unit_weights = np.around(np.exp(self.prob_partitions[valid_ix]) * total_draws).astype('int')
        # resample unit_events proportional to the prob of each unit.
        total_events = [np.random.choice(unit_events[i], unit_weights[i], replace=True)
                        for i in range(len(unit_events)) if len(unit_events[i])>0]
        total_events = [item for sublist in total_events for item in sublist]
        total_events = np.array(total_events)

        # subsample mating events.
        if n_subsamples is None:
            sub_events = "Mating events not subsampled."
        else:
            if subsample_size is None:
                subsample_size = np.around(total_draws*0.1).astype('int')
            sub_events = np.random.choice(total_events, n_subsamples*subsample_size, replace=True)
            sub_events = sub_events.reshape([n_subsamples, subsample_size])

        unit_names = paternity.offspring[valid_ix]

        return matingEvents(unit_names, paternity.candidates, unit_weights/total_draws, unit_events, total_events, sub_events)