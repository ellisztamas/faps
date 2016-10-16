import numpy as np

def alogsumexp(logarray, axis=0):
    """
    Calculate the sum of an array of values which are in log space across a given axis.
    """
    with np.errstate(invalid='ignore', divide='ignore'): # turn off warnings about dividing by zeros.
        shp       = list(logarray.shape)
        shp[axis] = 1
        maxvals   = logarray.max(axis=axis)
        sumarray  = np.log(np.sum(np.exp(logarray - maxvals.reshape(shp)), axis = axis)) + maxvals
        if not isinstance(sumarray, float): 
            sumarray[np.isnan(sumarray)] = np.log(0)
        return sumarray
