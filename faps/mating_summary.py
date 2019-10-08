import numpy as np
from pandas import DataFrame
from faps.matingEvents import matingEvents

def mating_summary(mating_event1, mating_event2, phenotype, absent_label=np.nan, quantiles=[2.5, 9.75]):
    """
    Compare patterns of mating in two matingEvents objects, for example based on
    observed genetic data, and a null model. Returns the relative probability
    that any male of a given phenotype sired any offspring.Confidence intervals
    are derived from subsampling mating events in each object.

    At present only categorical phenotypes are supported. Because Numpy cannot
    handle vectors of strings that include possible NaN values well, categories
    should be represented by an integer labels.

    Parameters
    ----------
    mating_event1: matingEvents object
        Information on likley mating events under model 1, for example mating
        imferred from observed genetic data.
    mating_event2: matingEvents object
        Information on likley mating events under model 2, for example mating
        under a null model of equal siring probality, or mating in proportion
        to dispersal only.
    phenotype: array
        1-d numerical vector indicating phenotype category for each candidate.
        Where there is missing data this should be indicated with a suitable
        value, such as -9.
    absent_label: float, int, nan
        Value to use for missing fathers. See documentation for matingEvents.
    percentiles = list
        List of two percentiles between 0 and 100 indicating upper and lower
        bounds for confidence intervals. Defaults to 95% intervals.

    Returns
    -------
    Dataframe listing:
        category: label for each category
        n: frequency of each category in the candidates
        model1: probability of mating with each category under mating model 1.
        model2: probability of mating with each category under mating model 1.
        ratio: ratio of model1 and model2
    Additional columns indicate confidence intervals around each estimate.
    """
    if mating_event1.subsamples.shape[0] != mating_event2.subsamples.shape[0]:
        raise ValueError('Arrays of subsamples are different sizes.')
    if not isintance(mating_event1, matingEvents):
        raise TypeError('mating_event1 and mating_events2 are not matingEvents objects')

    # phenotype vectors, not including NaN.
    dat1 = mating_event1.phenotype(phenotype, absent_label, remove_NaN=True)
    dat2 = mating_event2.phenotype(phenotype, absent_label, remove_NaN=True)
    # List of unique elements
    category = np.unique(np.append(dat1, dat2))
    # count each type
    count1 = [(dat1 == x).sum()      for x in category]
    count2 = [(dat2 == x).sum()      for x in category]
    base   = [(phenotype == x).sum() for x in category]
    # Take the average, included NaNs that were excluded
    count1 = np.array(count1).astype('float') / len(mating_event1.total_events)
    count2 = np.array(count2).astype('float') / len(mating_event2.total_events)
    # get relative probability
    ratio = count1 / count2

    vals = np.unique(phenotype)
    # phenotypes in each subsample
    subcounts1 = phenotype[mating_event1.subsamples.flatten()].values.reshape(mating_event1.subsamples.shape)
    subcounts1 = np.array([(subcounts1 == i).mean(1) for i in vals])

    subcounts2 = phenotype[mating_event2.subsamples.flatten()].values.reshape(mating_event2.subsamples.shape)
    subcounts2 = np.array([(subcounts2 == i).mean(1) for i in vals])

    # ratio among subsamples.
    sub_ratio = subcounts1 / subcounts2
    # confidence intervals.
    SE1 = np.percentile(subcounts1, q=quantiles, axis=1)
    SE2 = np.percentile(subcounts2, q=quantiles, axis=1)
    SEr = np.percentile(sub_ratio,  q=quantiles, axis=1)
    # output the summary statistics
    output = DataFrame({'category':category, 'n':base,
                  'model1':count1, 'lowerCI_1':SE1[0],  'upperCI_1':SE1[1],
                  'model2':count2, 'lowerCI_2':SE2[0],  'upperCI_2':SE2[1],
                  'ratio':ratio,   'lower_ratio':SEr[0],'upperCI_ratio':SEr[1]})
    # rearrange column order and return a dataframe.
    return output[['category','n',
                   'model1', 'lowerCI_1',   'upperCI_1',
                   'model2', 'lowerCI_2',   'upperCI_2',
                   'ratio',  'lower_ratio', 'upperCI_ratio']]
