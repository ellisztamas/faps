import numpy as np

def unique_rows(a):
    """
    Pull out the unique set of rows of an n*m matrix along axis n.
    
    ARGUMENTS
    a: An n*m matrix
    
    RETURNS
    A subset of one or more rows in a.
    """
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
