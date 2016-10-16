import numpy as np

class genotypeArray(object):
    """Information about an array of parents or offspring with four components:
    1. A three-dimensional array of genotype information of size, with axes denoting
        number of individuals, number of loci, and the two alleles.
    2. Names of each individual.
    3. Names of the mother of each individual.
    4. Names of the mother of each individual.
    """
    def __init__(self, geno, names, mothers, fathers):
        self.geno      = geno
        self.names     = names
        self.mothers   = mothers
        self.fathers   = fathers
        self.size      = self.geno.shape[0]
        self.nloci     = self.geno.shape[1]
        self.parents   = np.array([str(self.mothers[o]) + '/' + str(self.fathers[o]) for o in range(self.size)])
        self.nfamilies = len(np.unique(self.parents))
        self.dropouts  = (1.0*sum(self.geno == -9)[:,0]) / self.size
        
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
        A list of positions of the parental for each entry in offs_names.
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
                                    self.mothers[individuals], self.fathers[individuals])
        if loci is not None:
            if isinstance(loci, int):
                loci = [loci]
            if isinstance(loci, np.ndarray):
                loci = loci.tolist()
            output = genotypeArray(self.geno[:,loci], self.names, self.mothers, self.fathers)
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
