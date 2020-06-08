Dealing with multiple half-sib families
=======================================

Tom Ellis, March 2018

In the previous sections on `genotype
arrays <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/02_genotype_data.html>`__,
`paternity
arrays <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/03_paternity_arrays.html>`__
and `sibship
clustering <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/04_sibship_clustering.html>`__
we considered only a single half-sibling array. In most real-world
situations, you would probably have multiple half-sibling arrays from
multiple mothers.

FAPS assumes that these families are independent, which seems a
reasonable assumption for most application, so dealing with multiple
families boils down to performing the same operation on these families
through a loop. This guide outlines some tricks to automate this.

This notebook will examine how to:

1. Divide a dataset into multiple families
2. Perform sibship clustering on those families
3. Extract information from objects for multiple families

To illustrate this we will use data from wild-pollinated seed capsules
of the snapdragon *Antirrhinum majus*. Each capsule represents a single
maternal family, which may contain mixtures of full- and half-siblings.
Each maternal family can be treated as independent.

These are the raw data described in Ellis *et al.* (2018), and are
available from the `IST Austria data
repository <https://datarep.app.ist.ac.at/id/eprint/95>`__
(DOI:10.15479/AT:ISTA:95). For the analysis presented in that paper we
did extensive data cleaning and checking, which is given as a `case
study <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/08_data_cleaning_in_Amajus.html>`__
later in this guide. Here, we will skip this process, since it primarily
concerns accuracy of results.

Divide genotype data into families
----------------------------------

There are two ways to divide data into families: by splitting up a
``genotypeArray`` into families, and making a ``paternityArray`` for
each, or create a single ``paternityArray`` and split up that.

Import the data
~~~~~~~~~~~~~~~

Frequently offspring have been genotyped from multiple half-sibling
arrays, and it is convenient to store these data together in a single
file on disk. However, it (usually) only makes sense to look for sibling
relationships *within* known half-sib families, so we need to split
those data up into half-sibling famililes.

First, import the required packages and data for the sample of candidate
fathers and the progeny.

.. code:: ipython3

    import numpy as np
    from faps import *
    import matplotlib.pyplot as plt
    %pylab inline
    
    adults  = read_genotypes('../../data/parents_2012_genotypes.csv', genotype_col=1)
    progeny = read_genotypes('../../data/offspring_2012_genotypes.csv', genotype_col=2, mothers_col=1)


.. parsed-literal::

    Populating the interactive namespace from numpy and matplotlib


For simplicity, let's restrict the progeny to those offspring belonging
to three maternal families.

.. code:: ipython3

    ix = np.isin(progeny.mothers, ['J1246', 'K1809', 'L1872'])
    progeny = progeny.subset(individuals=ix)

We also need to define an array of genotypes for the mothers, and a
genotyping error rate.

.. code:: ipython3

    mothers = adults.subset(individuals=np.unique(progeny.mothers))
    mu= 0.0015

Pull out the numbers of adults and progeny in the dataset, as well as
the number of maternal families.

.. code:: ipython3

    print(adults.size)
    print(progeny.size)
    print(len(np.unique(progeny.mothers)))


.. parsed-literal::

    2124
    76
    3


Most maternal families are between 20 and 30, with some either side.

Split up the ``genotypeArray``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the data import we specified that the ID of the mother of each
offspring individual was given in column 2 of the data file (i.e. column
1 for Python, which starts counting from zero). Currently this
information is contained in ``progeny.mothers``.

To separate a ``genotypeArray`` into separate families you can use
``split``, and the vector of maternal names. This returns a
**dictionary** of ``genotypeArray`` objects for each maternal family.

.. code:: ipython3

    progeny2 = progeny.split(progeny.mothers)
    mothers2 = mothers.split(mothers.names)

If we inspect ``progeny2`` we can see the structure of the dictionary.
Python dictionaries are indexed by a **key**, which in this case is the
maternal family name. Each key refers to some **values**, which in this
case is a ``genotypeArray`` object for each maternal family.

.. code:: ipython3

    progeny2




.. parsed-literal::

    {'J1246': <faps.genotypeArray.genotypeArray at 0x7fe0e58052d0>,
     'K1809': <faps.genotypeArray.genotypeArray at 0x7fe0e5805950>,
     'L1872': <faps.genotypeArray.genotypeArray at 0x7fe0e5805050>}



You can pull attributes about an individual family by indexing the key
like you would for any other python dictionary.

.. code:: ipython3

    progeny2["J1246"].size




.. parsed-literal::

    25



To do this for all families you can iterate with a **dictionary
comprehension**, or loop over the dictionary. Here are three ways to get
the number of offspring in each maternal family:

.. code:: ipython3

    {k: v.size for k,v in progeny2.items()} # the .items() suffix needed to separate keys and values




.. parsed-literal::

    {'J1246': 25, 'K1809': 25, 'L1872': 26}



.. code:: ipython3

    {k : progeny2[k].size for k in progeny2.keys()} # using only the keys.




.. parsed-literal::

    {'J1246': 25, 'K1809': 25, 'L1872': 26}



.. code:: ipython3

    # Using a for loop.
    for k,v in progeny2.items():
        print(k, v.size)


.. parsed-literal::

    J1246 25
    K1809 25
    L1872 26


``paternityArray`` objects with multiple families
-------------------------------------------------

Paternity from a dictionary of ``genotypeArray`` objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The previous section divided up a ``genotypeArray`` containing data for
offspring from multiple mothers and split that up into maternal
families. You can then pass this dictionary of ``genotypeArray`` objects
to ``paternity_array`` directly, just as if they were single objects.
``paternity_array`` detects that these are dictionaries, and returns a
dictionary of ``paternityArray`` objects.

.. code:: ipython3

    %time patlik1 = paternity_array(progeny2, mothers2, adults, mu, missing_parents=0.01)


.. parsed-literal::

    CPU times: user 893 ms, sys: 7.99 ms, total: 901 ms
    Wall time: 899 ms


Split up an existing paternity array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The alternative way to do this is to pass the entire arrays for progeny
and mothers to ``paternity_array``. A word of caution is needed here,
because ``paternity_array`` is quite memory hungry, and for large
datasets there is a very real chance you could exhaust the RAM on your
computer and the machine will grind to a halt. By splitting up the
genotype data first you can deal with small chunks at a time.

.. code:: ipython3

    mothers_full = adults.subset(progeny.mothers)
    
    %time patlik2 = paternity_array(progeny, mothers_full, adults, mu)
    patlik2


.. parsed-literal::

    CPU times: user 863 ms, sys: 84 ms, total: 947 ms
    Wall time: 947 ms




.. parsed-literal::

    <faps.paternityArray.paternityArray at 0x7fe0e58bbb90>



There doesn't seem to be any difference in speed the two methods,
although in other cases I have found that creating a single
``paternityArray`` is slower. Your mileage may vary.

We split up the ``paternity_array`` in the same way as a
``genotype_array``. It returns a list of ``paternityArray`` objects.

.. code:: ipython3

    patlik3 = patlik2.split(progeny.mothers)
    patlik3




.. parsed-literal::

    {'J1246': <faps.paternityArray.paternityArray at 0x7fe0dd4b4e10>,
     'K1809': <faps.paternityArray.paternityArray at 0x7fe0dd4b4890>,
     'L1872': <faps.paternityArray.paternityArray at 0x7fe0dc8c03d0>}



We would hope that ``patlik`` and ``patlik3`` are identical lists of
``paternityArray`` objects. We can inspect family J1246 to check:

.. code:: ipython3

    patlik1['J1246'].offspring




.. parsed-literal::

    array(['J1246_221', 'J1246_222', 'J1246_223', 'J1246_224', 'J1246_225',
           'J1246_226', 'J1246_227', 'J1246_228', 'J1246_229', 'J1246_230',
           'J1246_231', 'J1246_232', 'J1246_233', 'J1246_241', 'J1246_615',
           'J1246_616', 'J1246_617', 'J1246_618', 'J1246_619', 'J1246_620',
           'J1246_621', 'J1246_622', 'J1246_623', 'J1246_624', 'J1246_625'],
          dtype='<U10')



.. code:: ipython3

    patlik3['J1246'].offspring




.. parsed-literal::

    array(['J1246_221', 'J1246_222', 'J1246_223', 'J1246_224', 'J1246_225',
           'J1246_226', 'J1246_227', 'J1246_228', 'J1246_229', 'J1246_230',
           'J1246_231', 'J1246_232', 'J1246_233', 'J1246_241', 'J1246_615',
           'J1246_616', 'J1246_617', 'J1246_618', 'J1246_619', 'J1246_620',
           'J1246_621', 'J1246_622', 'J1246_623', 'J1246_624', 'J1246_625'],
          dtype='<U10')



Clustering multiple families
----------------------------

``sibship_clustering`` is also able to detect when a list of
``paternityArray`` objects is being passed, and treat each
independently. It returns a dictionary of ``sibshipCluster`` objects.

.. code:: ipython3

    %%time
    sc = sibship_clustering(patlik1)
    sc


.. parsed-literal::

    CPU times: user 499 ms, sys: 0 ns, total: 499 ms
    Wall time: 497 ms




.. parsed-literal::

    {'J1246': <faps.sibshipCluster.sibshipCluster at 0x7fe0de42bf90>,
     'K1809': <faps.sibshipCluster.sibshipCluster at 0x7fe0e3917790>,
     'L1872': <faps.sibshipCluster.sibshipCluster at 0x7fe0e57a8550>}



This time there is quite a substantial speed advantage to performing
sibship clustering on each maternal family separately rather than on all
individuals together. This advanatge is modest here, but gets
substantial quickly as you add more families and offspring, because the
number of *pairs* of relationships to consider scales quadratically.

.. code:: ipython3

    %time sibship_clustering(patlik2)


.. parsed-literal::

    CPU times: user 1.48 s, sys: 2.64 ms, total: 1.49 s
    Wall time: 1.48 s




.. parsed-literal::

    <faps.sibshipCluster.sibshipCluster at 0x7fe0e6a5cc10>



You can index any single family to extract information about it in the
same way as was explained in the section on `sibship
clustering <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/04_sibship_clustering.html>`__.
For example, the posterior distribution of full-sibship sizes for the
first maternal family.

.. code:: ipython3

    sc['J1246'].family_size()




.. parsed-literal::

    array([4.57906541e-001, 0.00000000e+000, 1.80697820e-001, 0.00000000e+000,
           0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 2.81585310e-004,
           1.92238646e-001, 7.46085991e-002, 9.22427026e-002, 2.02410615e-003,
           2.11864904e-020, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
           0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
           0.00000000e+000, 1.42303289e-213, 0.00000000e+000, 0.00000000e+000,
           1.00383692e-265])



As with ``genotypeArray`` objects, to extract information about each
``sibshipCluster`` object it is straightforward to set up a list
comprehension. For example, this cell pulls out the number of partition
structures for each maternal family.

.. code:: ipython3

    {k : v.npartitions for k,v in sc.items()}




.. parsed-literal::

    {'J1246': 25, 'K1809': 25, 'L1872': 26}



Paternity for many arrays
-------------------------

Since paternity is likely to be a common aim, there is a handy function
for calling the ``sires`` method on a list of ``sibshipCluster``
objects.

.. code:: ipython3

    summarise_sires(sc)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>mother</th>
          <th>father</th>
          <th>log_prob</th>
          <th>prob</th>
          <th>offspring</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>J1246</td>
          <td>M0551</td>
          <td>-489.404687</td>
          <td>2.846066e-213</td>
          <td>9.925957e-05</td>
        </tr>
        <tr>
          <td>1</td>
          <td>J1246</td>
          <td>M0554</td>
          <td>-4.833608</td>
          <td>7.957758e-03</td>
          <td>9.030774e-04</td>
        </tr>
        <tr>
          <td>2</td>
          <td>J1246</td>
          <td>M0570</td>
          <td>-0.387012</td>
          <td>6.790830e-01</td>
          <td>6.410034e-01</td>
        </tr>
        <tr>
          <td>3</td>
          <td>J1246</td>
          <td>M1077</td>
          <td>-6.021227</td>
          <td>2.426690e-03</td>
          <td>2.602961e-03</td>
        </tr>
        <tr>
          <td>4</td>
          <td>J1246</td>
          <td>M1103</td>
          <td>-6.063947</td>
          <td>2.325205e-03</td>
          <td>2.494105e-03</td>
        </tr>
        <tr>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <td>62</td>
          <td>L1872</td>
          <td>M0878</td>
          <td>-3.022599</td>
          <td>4.867454e-02</td>
          <td>2.358130e-03</td>
        </tr>
        <tr>
          <td>63</td>
          <td>L1872</td>
          <td>M0880</td>
          <td>0.000000</td>
          <td>1.000000e+00</td>
          <td>2.000000e+00</td>
        </tr>
        <tr>
          <td>64</td>
          <td>L1872</td>
          <td>M0819</td>
          <td>0.000000</td>
          <td>1.000000e+00</td>
          <td>9.000000e+00</td>
        </tr>
        <tr>
          <td>65</td>
          <td>L1872</td>
          <td>M0204</td>
          <td>-929.421064</td>
          <td>0.000000e+00</td>
          <td>7.428108e-09</td>
        </tr>
        <tr>
          <td>66</td>
          <td>L1872</td>
          <td>M0124</td>
          <td>-176.623401</td>
          <td>1.965312e-77</td>
          <td>8.497702e-08</td>
        </tr>
      </tbody>
    </table>
    <p>67 rows × 5 columns</p>
    </div>



The output is similar to that of ``sires()`` except that it gives labels
for mother and father separately, replacing the ``label`` column. The
``prob`` and ``offspring`` columns have the same interpretation as for
single ``sibshipCluster`` objects.
