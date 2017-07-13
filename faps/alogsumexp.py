import numpy as np

def alogsumexp(logarray, axis=0):
    """
    Calculate the sum of an array of values which are in log space. If an axis is
    specified, the sum across this axis is given.
    
    Parameters
    ----------
    logarray: array
        An array of log values to be summed.
    axis: int, optional
        Axis to sum across.
    
    Returns
    -------
    Either a float corresponding to the log of summed values, or the vector of
    sums across a specific axis.
    """
    with np.errstate(invalid='ignore', divide='ignore'): # turn off warnings about dividing by zeros.
        shp       = list(logarray.shape)
        shp[axis] = 1
        maxvals   = logarray.max(axis=axis)
        sumarray  = np.log(np.sum(np.exp(logarray - maxvals.reshape(shp)), axis = axis)) + maxvals
        if not isinstance(sumarray, float): 
            sumarray[np.isnan(sumarray)] = np.log(0)
        return sumarray
