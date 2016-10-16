

def lik_partition(paternity_probs, partitions, includes_null=True, ndraws=1000):
    """
    Find the likelihood of a partition by sampling fathers for each sibship proportional to his probability.
    This then removes duplicate sets of fathers, and sets for which more than one family shares a father.
    
    ARGUMENTS:
    paternity_probs: An array of probabilties of paternity supplied by lik_transition() or pr_transition().
    
    partitions: A 1-d array of integers labelling individuals into families. This should have as many
    elements as there are individuals in paternity_probs.
    
    includes_null: Boolean index indicating whether the final column of paternity_probs contains the posterior
    probabilities that each individual is sired by an unsampled father. Defaults to True.
    
    ndraws: The number of paths through fathers to be drawn. Set to 1000 by default. If you find that
    partitions have log likelihoods with differences less than 1 log likelihood unit apart you might want to
    increase this to get a more accurate estimate of the likelihood.
    
    RETURNS
    A float indicating log likelihood for the proposed partition.
    """
    # check the partition is the correct length!
    if paternity_probs.shape[0] is not len(partitions):
        print "Error in lik_partition: paternity_probs and partition are of uneven length."
        return None
    
    siblik = squash_siblings(paternity_probs, partitions) # multiply likelihoods within sibships
    # If all offspring go into a single partition
    if len(siblik.shape) is 1:
        return alogsumexp(siblik) # log likelihood of the model

    # If offspring go into different partitions.
    if len(siblik.shape) >  1:
        siblik = siblik[:, np.isfinite(siblik).sum(0) > 0] # remove columns with no non-zero terms
        prob_array = np.exp(siblik - alogsumexp(siblik, 1)[:, np.newaxis]) # normalise and exponentiate likelihoods
        
        # number of sibships and compatible fathers
        nfamilies = prob_array.shape[0]
        nfathers  = prob_array.shape[1]

        # If there are fewer possible fathers than sibships, then the likelihood must by definition be zero.
        if nfathers < nfamilies:
            with np.errstate(divide='ignore'):
                return np.log(0)

        else:
            # lists to loop over to save repeating them in the list comprehensions
            flist     = range(nfathers)
            slist     = range(nfamilies)

            # generate a sample of possible paths through the matrix of candidate fathers.
            path_samples = np.array([np.random.choice(flist, ndraws, replace=True, p = prob_array[i])
                                     for i in range(nfamilies)]).T
            path_samples = unique_rows(path_samples) # remove duplicate rows.
            
            # find paths which only visit each father once.
            # If unrelated fathers are not included in paternity_probs, remove all duplicates:
            if includes_null is False:
                unique_dads = [all(np.bincount(path_samples[i])      < 2) for i in range(len(path_samples))]
            # If unrelated fathers are included as the last column of paternity_probs, allow for multiple occurences of this.
            if includes_null is True:
                unique_dads = [all(np.bincount(path_samples[i])[:-1] < 2) for i in range(len(path_samples))]
            # remove paths with shared fathers across two or more families.
            paths_to_keep   = np.arange(path_samples.shape[0])[np.array(unique_dads)] # indices for valid paths
            path_samples    = path_samples[paths_to_keep] # keep only valid paths

            # If no possible paths are found, return a likelihood of zero.
            if path_samples.shape[0] == 0:
                with np.errstate(divide='ignore'):
                    return np.log(0)
            # If there are possible paths, multiply over each set of fathers, and sum across sets.
            if path_samples.shape[0] > 0:
                # multiply over viable sets of fathers:
                path_product = [siblik[slist, path_samples[i]].sum() for i in range(path_samples.shape[0])]
                paths_sum    = alogsumexp(np.array(path_product)) # sum across sets
                return paths_sum
