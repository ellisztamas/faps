Dealing with multiple half-sib families
=======================================

Tom Ellis, March 2018, updated June 2020

.. code:: ipython3

    import numpy as np
    import faps as fp
    import matplotlib.pyplot as plt
    
    print("Created using FAPS version {}.".format(fp.__version__))


.. parsed-literal::

    Created using FAPS version 2.6.6.


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

    %pylab inline
    
    adults  = fp.read_genotypes('../../data/parents_2012_genotypes.csv', genotype_col=1)
    progeny = fp.read_genotypes('../../data/offspring_2012_genotypes.csv', genotype_col=2, mothers_col=1)


.. parsed-literal::

    Populating the interactive namespace from numpy and matplotlib


For simplicity, let’s restrict the progeny to those offspring belonging
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
offspring individual was given in column 2 of the data file (i.e. column
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

    {'J1246': <faps.genotypeArray.genotypeArray at 0x7f21789fd400>,
     'K1809': <faps.genotypeArray.genotypeArray at 0x7f21789fd250>,
     'L1872': <faps.genotypeArray.genotypeArray at 0x7f21299dfbb0>}



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

    %time patlik1 = fp.paternity_array(progeny2, mothers2, adults, mu, missing_parents=0.01)


.. parsed-literal::

    CPU times: user 1.04 s, sys: 77.9 ms, total: 1.12 s
    Wall time: 1.12 s


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
    
    %time patlik2 = fp.paternity_array(progeny, mothers_full, adults, mu, missing_parents=0.01)
    patlik2


.. parsed-literal::

    CPU times: user 980 ms, sys: 344 ms, total: 1.32 s
    Wall time: 1.32 s




.. parsed-literal::

    <faps.paternityArray.paternityArray at 0x7f217816c550>



There doesn’t seem to be any difference in speed the two methods,
although in other cases I have found that creating a single
``paternityArray`` is slower. Your mileage may vary.

We split up the ``paternity_array`` in the same way as a
``genotype_array``. It returns a list of ``paternityArray`` objects.

.. code:: ipython3

    patlik3 = patlik2.split(progeny.mothers)
    patlik3




.. parsed-literal::

    {'J1246': <faps.paternityArray.paternityArray at 0x7f2128deaee0>,
     'K1809': <faps.paternityArray.paternityArray at 0x7f2127f45fa0>,
     'L1872': <faps.paternityArray.paternityArray at 0x7f2127fe2460>}



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
    sc = fp.sibship_clustering(patlik1)
    sc


.. parsed-literal::

    CPU times: user 258 ms, sys: 844 µs, total: 258 ms
    Wall time: 257 ms




.. parsed-literal::

    {'J1246': <faps.sibshipCluster.sibshipCluster at 0x7f21789f1ca0>,
     'K1809': <faps.sibshipCluster.sibshipCluster at 0x7f212883d850>,
     'L1872': <faps.sibshipCluster.sibshipCluster at 0x7f2127c456d0>}



This time there is quite a substantial speed advantage to performing
sibship clustering on each maternal family separately rather than on all
individuals together. This advanatge is modest here, but gets
substantial quickly as you add more families and offspring, because the
number of *pairs* of relationships to consider scales quadratically.

.. code:: ipython3

    %time fp.sibship_clustering(patlik2)


.. parsed-literal::

    CPU times: user 637 ms, sys: 12 µs, total: 638 ms
    Wall time: 636 ms




.. parsed-literal::

    <faps.sibshipCluster.sibshipCluster at 0x7f2178a0b280>



You can index any single family to extract information about it in the
same way as was explained in the section on `sibship
clustering <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/04_sibship_clustering.html>`__.
For example, the posterior distribution of full-sibship sizes for the
first maternal family.

.. code:: ipython3

    sc['J1246'].family_size()




.. parsed-literal::

    array([4.98363699e-01, 0.00000000e+00, 2.50818150e-01, 0.00000000e+00,
           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
           0.00000000e+00, 0.00000000e+00, 1.09643257e-03, 2.45353454e-01,
           4.36749302e-03, 7.70743245e-07, 0.00000000e+00, 0.00000000e+00,
           1.58450339e-56])



As with ``genotypeArray`` objects, to extract information about each
``sibshipCluster`` object it is straightforward to set up a list
comprehension. For example, this cell pulls out the number of partition
structures for each maternal family.

.. code:: ipython3

    {k : v.npartitions for k,v in sc.items()}




.. parsed-literal::

    {'J1246': 15, 'K1809': 15, 'L1872': 11}



Paternity for many arrays
-------------------------

Since paternity is likely to be a common aim, there is a handy function
for calling the ``sires`` method for `individual ``sibshipCluster``
objects <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/04_sibship_clustering.html#mating-events>`__
on an entire dictionary of ``sibshipCluster`` objects.

.. code:: ipython3

    fp.summarise_sires(sc)




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
          <th>0</th>
          <td>J1246</td>
          <td>M0551</td>
          <td>-1.319064e-02</td>
          <td>0.986896</td>
          <td>0.973894</td>
        </tr>
        <tr>
          <th>1</th>
          <td>J1246</td>
          <td>M1103</td>
          <td>-1.541488e-06</td>
          <td>0.999998</td>
          <td>0.999881</td>
        </tr>
        <tr>
          <th>2</th>
          <td>J1246</td>
          <td>M0147</td>
          <td>-5.206256e+00</td>
          <td>0.005482</td>
          <td>0.000030</td>
        </tr>
        <tr>
          <th>3</th>
          <td>J1246</td>
          <td>M0025</td>
          <td>0.000000e+00</td>
          <td>1.000000</td>
          <td>3.000000</td>
        </tr>
        <tr>
          <th>4</th>
          <td>J1246</td>
          <td>nan</td>
          <td>0.000000e+00</td>
          <td>1.000000</td>
          <td>20.026195</td>
        </tr>
        <tr>
          <th>5</th>
          <td>K1809</td>
          <td>M0551</td>
          <td>-3.966770e-03</td>
          <td>0.996041</td>
          <td>0.992776</td>
        </tr>
        <tr>
          <th>6</th>
          <td>K1809</td>
          <td>M2047</td>
          <td>-3.327946e+00</td>
          <td>0.035867</td>
          <td>0.016896</td>
        </tr>
        <tr>
          <th>7</th>
          <td>K1809</td>
          <td>M1971</td>
          <td>-9.120326e-01</td>
          <td>0.401707</td>
          <td>0.189237</td>
        </tr>
        <tr>
          <th>8</th>
          <td>K1809</td>
          <td>L0662</td>
          <td>2.220446e-16</td>
          <td>1.000000</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>9</th>
          <td>K1809</td>
          <td>M0392</td>
          <td>-3.327946e+00</td>
          <td>0.035867</td>
          <td>0.016896</td>
        </tr>
        <tr>
          <th>10</th>
          <td>K1809</td>
          <td>nan</td>
          <td>2.220446e-16</td>
          <td>1.000000</td>
          <td>22.784195</td>
        </tr>
        <tr>
          <th>11</th>
          <td>L1872</td>
          <td>L0124</td>
          <td>0.000000e+00</td>
          <td>1.000000</td>
          <td>2.000000</td>
        </tr>
        <tr>
          <th>12</th>
          <td>L1872</td>
          <td>M0107</td>
          <td>-3.686856e+00</td>
          <td>0.025051</td>
          <td>0.000628</td>
        </tr>
        <tr>
          <th>13</th>
          <td>L1872</td>
          <td>M0819</td>
          <td>0.000000e+00</td>
          <td>1.000000</td>
          <td>9.000000</td>
        </tr>
        <tr>
          <th>14</th>
          <td>L1872</td>
          <td>nan</td>
          <td>0.000000e+00</td>
          <td>1.000000</td>
          <td>14.999372</td>
        </tr>
      </tbody>
    </table>
    </div>



The output is similar to that of the ``sires()`` method, except that it
gives labels for mother and father separately, replacing the ``label``
column. The ``prob`` and ``offspring`` columns have the same
interpretation as for single ``sibshipCluster`` objects.
