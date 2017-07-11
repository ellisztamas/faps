import numpy as np
from paternityArray import paternityArray

def write_paternity_array(paternity_array, path):
    """
    Write a matrix of (unnormalised) likelihoods of paternity to disk.

    ARGUMENTS:
    paternity_array A paternityArray object.

    path Path to write to.

    RETURNS:
    A CSV file indexing offspring ID, mother ID, followed by a matrix of likelihoods
    that each candidate male is the true father of each individual. The final
    column is the likelihood that the paternal alleles are drawn from population
    allele frequencies.
    """
    if !isinstance(paternityArray, 'paternityArray'):
        print "Data object should be a paternityArray obeject."
        return None
    else:
    # append offspring and mother IDs onto the likelihood array.
    # append likelihoods of abset fathers on the back.
    newdata = np.append(paternity_array.offspring[:,np.newaxis],
                    np.append(paternity_array.mothers[:,np.newaxis],
                              np.append(paternity_array.likelihood,
                                        paternity_array.lik_absent[:,np.newaxis],
                                        1),1),1)
    # headers
    cn = ",".join( paternity_array.candidates )
    cn = 'offspringID,motherID,' + cn + ',missing_father'
    # write to disk
    np.savetxt(path, newdata, fmt='%s', delimiter=',', comments='', header=cn)
