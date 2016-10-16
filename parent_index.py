import numpy as np

def parent_index(offs_names, parent_names):
    """
    Finds the position of parental names in a vector of possible parental names.
    
    This is essentially a convenient wrapper for np.where().
    
    ARGUMENTS:
    offs_names: Either a string denoting the name of the parents to be located,
        or else a list of such strings.
        
    parent_names: 1-d array of parental names.
    
    RETURNS:
    A list of positions of the parental for each entry in offs_names.
    """
    if isinstance(offs_names, str):
        offs_names = [offs_names]
    return [np.where(parent_names == offs_names[x])[0][0] for x in range(len(offs_names))]