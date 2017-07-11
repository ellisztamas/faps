import numpy as np

class genotypeArray(object):
    """
    Information about an array of parents or offspring with four components:
    1. A three-dimensional array of genotype information of size, with axes denoting
        number of individuals, number of loci, and the two alleles.
    2. Names of each individual.
    3. Names of the mother of each individual.
    4. Names of the mother of each individual.
    """
    def __init__(self, geno, names, mothers, fathers, markers=None):
        self.geno      = geno
        self.names     = names
        self.mothers   = mothers
        self.fathers   = fathers
        self.markers   = markers
        self.size      = self.geno.shape[0]
        self.nloci     = self.geno.shape[1]
        self.parents   = np.array([str(self.mothers[o]) + '/' + str(self.fathers[o]) for o in range(self.size)])
        self.families  = np.unique(self.parents)
        self.nfamilies = len(self.families)

    def allele_freqs(self):
        """
        Allele frequencies of each locus. The reference allele is whichever is
        labelled as 1.
        """
        diploid = self.geno.sum(2) * 0.5
        return np.array([diploid[:,i][diploid[:,i] >= 0].mean() for i in range(self.nloci)])

    def heterozygosity(self, by='marker'):
        """
        Mean heterozygosity, either averaged across markers for each individual,
        or across individuals for each marker.

        Parameters
        ----------
        by: str
            If 'individual', values are returned averaged across markers for each
            individual. If 'marker' values are returned averaged across
            individuals for each marker. Defaults to 'marker'.

        Returns
        -------
        Vector of floats.
        """
        if by == 'marker' or by == 0:
            return (self.geno.sum(2) == 1).mean(0)
        elif by == 'individual' or by == 1:
            return (self.geno.sum(2) == 1).mean(1)
        else:
            print "Input for `by` not recognised."
            return None

    def missing_data(self, by='marker'):
        """
        Mean genotype drop-out rate, either averaged across markers for each individual,
        or across individuals for each marker.

        Parameters
        ----------
        by: str
            If 'individual', values are returned averaged across markers for each
            individual. If 'marker' values are returned averaged across
            individuals for each marker. Defaults to 'marker'.

        Returns
        -------
        Vector of floats.
        """
        if by == 'marker' or by == 0:
            return (self.geno[:,:,0] == -9).mean(0)
        elif by == 'individual' or by == 1:
            return (self.geno[:,:,0] == -9).mean(1)
        else:
            print "Input for `by` not recognised."
            return None

    def parent_index(self, parent, parent_names):
        """
        Finds the position of parental names in a vector of possible parental names.
        This can be the name of the mother or the father.

        This is essentially a convenient wrapper for np.where().

        ARGUMENTS:
        parent: A string indicating whether the offspring's mother or father is to
            be located. Valid arguments are 'mother', 'father' and 'parents',  or
            equivalently 'm' and 'f' respectively.

        parent_names: 1-d array of parental names.

        RETURNS:
        A list of positions of the parent for each entry in offs_names.
        """
        if parent is 'mother' or parent is 'm':
            return [np.where(parent_names == self.mothers[x])[0][0] for x in range(len(self.mothers))]
        if parent is 'father' or parent is 'f':
            return [np.where(parent_names == self.fathers[x])[0][0] for x in range(len(self.fathers))]
        if parent is 'parents' or parent is 'p':
            return [np.where(parent_names == self.parents[x])[0][0] for x in range(len(self.parents))]
        else:
            print "parent must be 'mother', 'father', or 'parents'."

    def true_partition(self):
        """
        For families of known parentage, usually simulated data, create a full sibship partition
        vector from the identities of known mothers and fathers contained in variables 'mothers'
        and 'fathers' of a genotype array.

        If one or more individuals has at least one missing parent they will be assigned to the
        same full sibship group.

        ARGUMENTS
        offspring: a genotype object of offspring.

        RETURNS
        An array of integers with an entry for each offspring individual. Individuals are labelled
        according to their full sibling group.
        """
        if 'NA' in self.mothers or 'NA' in self.fathers:
            print 'Warning: one or more individuals has at least one parent of unkown identity.'
            print 'All such individuals will be assigned to the same sibship group.'

        # concatenate mother and father names to create a vector of parent pairs.
        #parentage = np.array([str(self.mothers[o]) + '/' + str(self.fathers[o]) for o in range(noffs)])
        possible_families = np.unique(self.parents) # get a list of all unique parent pairs

        partitions = np.zeros(self.size).astype('int') # empty vector of zeros.
        for o in range(self.nfamilies):
            # for every possible family label individuals with an identical integer.
            partitions[self.parents == possible_families[o]] += o

        return partitions

    def subset(self, individuals=None, loci=None):
        """
        Subset the genotype array by individual or number of loci.

        ARGUMENTS
        individuals: an integer or list of integers indexing the individuals to be included.

        loci: a list of loci to be included.

        RETURNS
        A genotype array with only the target individuals included.
        """
        # if no subsetting indices are given, return the whole object.
        if individuals is None and loci is None:
            return self
        if individuals is not None:
            output = genotypeArray(self.geno[individuals], self.names[individuals],
                                    self.mothers[individuals], self.fathers[individuals], markers=self.markers)
        if loci is not None:
            if isinstance(loci, int):
                loci = [loci]
            if isinstance(loci, np.ndarray):
                if np.result_type(loci) == 'bool':
                    loci = np.arange(len(loci))[loci]
                loci = loci.tolist()
            output = genotypeArray(self.geno[:,loci], self.names, self.mothers, self.fathers, markers=self.markers[loci])
        return output

    def drop(self, individuals):
        """
        Remove specific individuals from the genotype array.

        ARGUMENTS
        individuals: an integer or list of integers indexing the individuals to be removed.

        RETURNS
        A genotype array with the target individuals removed.
        """
        if isinstance(individuals, int):
            individuals = [individuals]
        index = range(self.names.shape[0]) # list of indices for each individual.
        # remove the target individual's indices.
        new_index = [i for j, i in enumerate(index) if j not in individuals]
        # create new genotypeArray.
        output = genotypeArray(self.geno[new_index], self.names[new_index],
                        self.mothers[new_index], self.fathers[new_index])
        return output

    def mutations(self, mu):
        """
        Introduce mutations at random to an array of genotype data for multiple individuals.
        For all alleles present draw mutations given error rate mu, then swap zeroes and
        ones in the array.

        ARGUMENTS
        mu: haploid genotyping error rate.

        RETURNS
        A copy of the input genotype data, but with point mutations added.
        """
        ninds    = self.geno.shape[0]
        nloci    = self.geno.shape[1]
        nalleles = self.geno.shape[2]

        new_alleles = np.copy(self.geno).astype(int) # make a copy of the data, and make it an integer
        # for an array of the same shape as newAlleles, draw mutations at each position with probability mu.
        mutate = np.reshape(np.random.binomial(1,mu,ninds*nloci*nalleles),[ninds,nloci,nalleles]) == 1
        new_alleles[mutate] ^= 1 # swap zeroes and ones.

        output = genotypeArray(new_alleles, self.names, self.mothers, self.fathers)

        return output

    def dropouts(self, dr):
        """
        Add allelic dropouts to an array of genotypic data.

        ARGUMENTS
        dr: diploid dropout rate.

        RETURNS
        A copy of the input genotype data, but with dropouts. Alleles which dropout are
        given as -9, in contrast to useful data which can only be 0 or 1 (or 2 for diploid
        genotypes).
        """
        ninds    = self.geno.shape[0]
        nloci    = self.geno.shape[1]

        # pick data points to dropout
        positions = np.reshape(np.random.binomial(1, dr, ninds*nloci),[ninds,nloci])
        positions = np.repeat(positions[:,:,np.newaxis], 2, axis =2)

        # make a copy of the genotype data, just in case
        new_alleles = np.copy(self.geno)
        # insert missing data into parental genotypes
        new_alleles = new_alleles - positions*3 # make all dropped loci < 0
        new_alleles[new_alleles < 0] = -9 # make dropped loci -9, just for tidyness.

        output = genotypeArray(new_alleles, self.names, self.mothers, self.fathers)

        return output

    def split(self, by):
        """
        Split up a gentotypeArray into groups according to some grouping
        factor. For example, divide an array containing genotype data for
        multiple half-sibling arrays by the ID of their mothers.

        Parameters
        ----------
        by: array-like
            Vector containing grouping labels for each individual

        Returns
        -------
        A list of genotypeArray objects.
        """
        groups = np.unique(by)
        ix = [np.where(by == groups[i])[0] for i in range(len(groups))]
        output = [self.subset(ix[i]) for i in range(len(ix))]
        return output
