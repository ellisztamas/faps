import numpy as np
from faps.paternityArray import paternityArray
from faps.matingEvents import matingEvents
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
    def __init__(self, paternity_array, linkage_matrix, partitions, lik_partitions, paths, path_likelihoods, path_probs):
        self.candidates       = paternity_array.candidates
        self.paternity_array  = paternity_array.prob_array
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
            sire_probs = sire_probs.squeeze()
        else:
            sire_probs = np.nan

        # Mean probability that the father is absent
        abs_probs = np.exp(self.prob_paternity(-1)).mean()

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
            raise ValueError("Reference partition should be the same length as the number of offspring.")
            return None

        obs = self.full_sib_matrix()
        rm  = relation_matrix(reference)

        # Matrix of ones and zeroes to reference elements for each relationship type.
        if   rtype is 'all': ix = np.triu(np.ones(rm.shape), 1)
        elif rtype is 'fs':  ix = np.triu(rm, 1)
        elif rtype is 'hs':  ix = np.triu(1-rm, 1)
        else:
            raise ValueError("rtype must be one of 'all', 'fs' or 'hs'.")
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

    def sires(self, labels='names'):
        """
        For every candidate drawn as a father in any partition, calculate the
        (log) probability that each sired at least one offspring.

        Probabilities are calculated as the sum of probabilties of each
        partition in which a candidate appears.

        Parameters
        ----------
        labels: array, optional
            Labels for candidates. By default, the names of candidates are used.
            If `labels=None`, the index positions in the list of candidates are
            used; this can be useful if you want to use indices to subset
            another dataset and link that to probabilities  of paternity.

        Returns
        -------
        A dictionary with three values for each drawn father:
        'position': Integer position in the list of candidate fathers.
        'label'   : Value of the father in the list of labels. By default,
            the candidates name is returned.
        'log_prob': Log probability that the candidate is the true sire of
            at least one offspring in the maternal family.

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
        if labels is 'names':
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
        # Return DataFrame of information on sires
        return output

    def mating_events(self, paternity_array, unit_draws=1000, total_draws=10000, n_subsamples = 0, subsample_size = None, null_probs=None, use_covariates=False, return_names=True):
        """
        Sample plausible mating events from a sibshipCluster. These can then be
        used to infer mating patterns using and array of phenotypes for each male.

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
        paternity_array: paternityArray
            Object listing information on paternity of individuals. This should be the
            same paternityArray object used to construct the sibshipCluster objects,
            or else results will be meaningless.
        unit_draws: int
            Number of mating events to sample for each partition.
        total_draws: int
            Total number of mating events to resample for each partition.
        n_subsamples: int, optional
            Number of subsamples to draw from the total mating events. Defaults
            to zero.
        subsample_size: int, optional
            Number of mating events in each subsample. Defaults to
            0.1*total_draws.
        null_probs: array, optional
            1-d Array of probabilities for paternity if this were not based on
            marker data. This should have an element for every candidate father.
        use_covariates: logical, optional
            If True, information on prbabilities associated with covariates
            stored in paternityArray objects are incorporated into weights for
            drawing likely fathers.
        return_names: logical, optional
            If True, unit_events and total_events are returned as the names of
            the candidates drawn for each partition. If False, their integer
            positions in the list of candidates is returned.

        Returns
        -------
        A matingEvents object.

        Examples
        --------
        import numpy as np
        from faps import *

        # simulate genotype data
        allele_freqs = np.random.uniform(0.1, 0.5, 50)
        males = make_parents(250, allele_freqs)
        offspring = make_sibships(males, 0, range(1,5), 5)
        mothers = males.subset(offspring.parent_index('m', males.names))

        # Creat paternityArrays and cluster into full sibships.
        patlik = paternity_array(offspring, mothers, males, mu=0.0015)
        patlik.add_covariate(np.log(np.repeat(1.0/males.size, males.size)))
        sc = sibship_clustering(patlik)

        # Basic usage
        sc.mating_events(patlik)

        # Including probabilities from covariates stored in the paternityArray.
        sc.mating_events(patlik, use_covariates=True)

        # Generate a sample in proportion to some other probability function not including genetic information
        null_probs = np.repeat(1.0/males.size, males.size)
        sc.mating_events(patlik, null_probs=null_probs)
        """
        if not isinstance(paternity_array, paternityArray):
            raise TypeError('paternity_array should be a paternityArray.')
        elif paternity_array.prob_array.shape != self.paternity_array.shape:
            raise ValueError('Number of offspring in paternity ({}) does not match the sibshipCluster ({}).'.format(paternity_array.prob_array.shape, self.paternity_array.shape))

        if not isinstance(unit_draws, int):
            raise ValueError('unit_draws should be an integer.')
        if not isinstance(total_draws, int):
            raise ValueError('total_draws should be an integer.')

        # only consider partitions that would account for at least one mating event.
        valid_ix = np.around(total_draws * np.exp(self.prob_partitions)) >= 1
        valid_partitions = self.partitions[valid_ix]
        # Get names for those
        unit_names =  ['partition_' + str(x) for x in np.where(valid_ix)[0]]

        # mating_events draws a sample of fathers from either genetic data,
        # or else from some array of probabilties based on non-genetic data.
        # Draw fathers from genetic data:
        if null_probs is None:
            # draw mating events for each partition.
            unit_events = {}
            for k,v in zip(unit_names, valid_partitions):
                d = draw_fathers(v, paternity_array=paternity_array, ndraws=unit_draws, use_covariates=use_covariates)
                unit_events[k] = d
        # Draw fathers from non-genetic data.
        elif isinstance(null_probs, np.ndarray):
            if len(null_probs.shape) != 1:
                raise ValueError("null_probs should be a 1-d vector, but has shape {}".format("null_probs.shape"))
            elif len(null_probs) != len(paternity_array.candidates):
                raise ValueError("null_probs should have an element for every candidate father, but has length {}.".format(len(null_probs)))
            else:
                unit_events = {}
                for k,v in zip(unit_names, valid_partitions):
                    unit_events[k] = draw_fathers(v, null_probs=null_probs, ndraws=unit_draws)
        else:
            raise TypeError('null_probs is of type {}. If supplied, this should is supplied it should be an array.'.format(type(null_probs)))
        # Resample mating events for each partition weighted by the probability
        # for that partition.
        # First, get an set of integer number of draws for each partition.
        unit_weights = np.around(np.exp(self.prob_partitions[valid_ix]) * total_draws).astype('int')
        unit_weights = {k:v for k,v in zip(unit_names, unit_weights)}
        # Resample unit_events proportional to the prob of each unit.
        total_events = [np.random.choice(a=v, size=unit_weights[k], replace=True) for k,v in unit_events.items()]
        total_events = [item for sublist in total_events for item in sublist]
        total_events = np.array(total_events)

        # Subsample mating events.
        if isinstance(n_subsamples, int):
            if subsample_size is None:
                subsample_size = np.around(total_draws*0.1).astype('int')
            elif not isinstance(subsample_size, int):
                raise TypeError("subsample_size should be an integer or None.")
            if subsample_size >= total_draws:
                raise ValueError('Number of subsamples must be smaller than total_draws')
            sub_events = np.random.choice(total_events, n_subsamples*subsample_size, replace=True)
            sub_events = sub_events.reshape([n_subsamples, subsample_size])
        else:
            raise ValueError('n_subsamples should be an integer.')


        return matingEvents(unit_names=unit_names,
                            candidates=paternity_array.candidates,
                            unit_weights=unit_weights,
                            unit_events=unit_events,
                            total_events=total_events,
                            subsamples=sub_events)
