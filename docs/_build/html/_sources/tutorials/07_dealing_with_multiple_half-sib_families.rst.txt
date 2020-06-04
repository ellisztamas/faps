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

    {'J1246': <faps.genotypeArray.genotypeArray at 0x7ff0cf56eed0>,
     'K1809': <faps.genotypeArray.genotypeArray at 0x7ff0cf5323d0>,
     'L1872': <faps.genotypeArray.genotypeArray at 0x7ff0cf56e450>}



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

    %time patlik1 = paternity_array(progeny2, mothers2, adults, mu)


.. parsed-literal::

    CPU times: user 943 ms, sys: 35.2 ms, total: 979 ms
    Wall time: 980 ms


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

    CPU times: user 884 ms, sys: 59.4 ms, total: 943 ms
    Wall time: 949 ms




.. parsed-literal::

    <faps.paternityArray.paternityArray at 0x7ff0cdf29810>



There doesn't seem to be any difference in speed the two methods,
although in other cases I have found that creating a single
``paternityArray`` is slower. Your mileage may vary.

We split up the ``paternity_array`` in the same way as a
``genotype_array``. It returns a list of ``paternityArray`` objects.

.. code:: ipython3

    patlik3 = patlik2.split(progeny.mothers)
    patlik3




.. parsed-literal::

    {'J1246': <faps.paternityArray.paternityArray at 0x7ff0cdf2c3d0>,
     'K1809': <faps.paternityArray.paternityArray at 0x7ff0ce35be90>,
     'L1872': <faps.paternityArray.paternityArray at 0x7ff0cdcd2550>}



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

    /home/GMI/thomas.ellis/miniconda3/envs/faps/lib/python3.7/site-packages/faps/paternityArray.py:216: UserWarning: Missing_parents set to 0. Only continue if you are sure you really have 100% of possible fathers.
      if self.missing_parents ==0: warn("Missing_parents set to 0. Only continue if you are sure you really have 100% of possible fathers.")


.. parsed-literal::

    CPU times: user 523 ms, sys: 0 ns, total: 523 ms
    Wall time: 521 ms




.. parsed-literal::

    {'J1246': <faps.sibshipCluster.sibshipCluster at 0x7ff0ce647710>,
     'K1809': <faps.sibshipCluster.sibshipCluster at 0x7ff0cd834150>,
     'L1872': <faps.sibshipCluster.sibshipCluster at 0x7ff0cd834b50>}



This time there is quite a substantial speed advantage to performing
sibship clustering on each maternal family separately rather than on all
individuals together. This advanatge is modest here, but gets
substantial quickly as you add more families and offspring, because the
number of *pairs* of relationships to consider scales quadratically.

.. code:: ipython3

    %time sibship_clustering(patlik2)


.. parsed-literal::

    CPU times: user 1.51 s, sys: 0 ns, total: 1.51 s
    Wall time: 1.5 s




.. parsed-literal::

    <faps.sibshipCluster.sibshipCluster at 0x7ff0cdfffb50>



You can index any single family to extract information about it in the
same way as was explained in the section on `sibship
clustering <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/04_sibship_clustering.html>`__.
For example, the posterior distribution of full-sibship sizes for the
first maternal family.

.. code:: ipython3

    sc['J1246'].family_size()




.. parsed-literal::

    array([4.58264645e-001, 0.00000000e+000, 1.80578452e-001, 0.00000000e+000,
           0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 5.35892330e-004,
           1.92140400e-001, 7.43963548e-002, 9.20640706e-002, 2.02018638e-003,
           2.11454618e-020, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
           0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
           0.00000000e+000, 1.42027713e-213, 0.00000000e+000, 0.00000000e+000,
           1.00189294e-265])



As with ``genotypeArray`` objects, to extract information about each
``sibshipCluster`` object it is straightforward to set up a list
comprehension. For example, this cell pulls out the number of partition
structures for each maternal family.

.. code:: ipython3

    {k : v.npartitions for k,v in sc.items()}




.. parsed-literal::

    {'J1246': 25, 'K1809': 25, 'L1872': 26}


