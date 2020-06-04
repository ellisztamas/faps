Genotype data in FAPS
=====================

Tom Ellis, March 2017

In most cases, researchers will have a sample of offspring, maternal and
candidate paternal individuals typed at a set of markers. In this
section we'll look in more detail at how FAPS deals with genotype data
to build a matrix we can use for sibship inference.

This notebook will examine how to:

1. Generate simple ``genotypeArray`` objects and explore what
   information is contained in them.
2. Import external genotype data.
3. Work with genotype data from multiple half sib families.

Checking genotype data is an important step before committing to a full
analysis. A case study of data checking and cleaning using an empirical
dataset is given in
`here <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/08_data_cleaning_in_Amajus.html>`__.
In the `next
section <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/03_paternity_arrays.html>`__
we'll see how to combine genotype information on offspring and a set of
candidate parents to create an array of likelihoods of paternity for
dyads of offspring and candidate fathers.

Note that the first half of this tutorial only deals with the case where
you have a ``genotypeArray`` object for a single maternal family. If you
have multiple families, you can apply what is here to each one, but at
some point you'll have to iterate over those families. See
`below <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/02_genotype_data.html#multiple-families>`__
and the specific
`tutorial <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/07_dealing_with_multiple_half-sib_families.html>`__
on that.

Currently, FAPS ``genotypeArray`` objects assume you are using
biallelic, unlinked SNPs for a diploid. If your system deviates from
these criteria in some way you can also skip this stage by creating your
own array of paternity likelihoods using an appropriate likelihood
function, and importing this directly as a ``paternityArrays``. See the
next section for more on ```paternityArray``
objects <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/03_paternity_arrays.html>`__
and how they should look.

``genotypeArray`` objects
-------------------------

Basic genotype information
~~~~~~~~~~~~~~~~~~~~~~~~~~

Genotype data are stored in a class of objects called a
``genotypeArray``. We'll illustrate how these work with simulated data,
since not all information is available for real-world data sets. We
first generate a vector of population allele frequencies for 10 unlinked
SNP markers, and use these to create a population of five adult
individuals. This is obviously an unrealisticaly small dataset, but
serves for illustration. The optional argument ``family_names`` allows
you to name this generation.

.. code:: ipython3

    import faps as fp
    import numpy as np
    
    allele_freqs = np.random.uniform(0.3,0.5,10)
    mypop = fp.make_parents(5, allele_freqs, family_name='my_population')

The object we just created contains information about the genotypes of
each of the ten parent individuals. Genotypes are stored as
*N*\ x\ *L*\ x2-dimensional arrays, where *N* is the number of
individuals and *L* is the number of loci. We can view the genotype for
the first parent like so (recall that Python starts counting from zero,
not one):

.. code:: ipython3

    mypop.geno[0]




.. parsed-literal::

    array([[1, 1],
           [1, 0],
           [0, 0],
           [0, 0],
           [0, 1],
           [1, 0],
           [0, 1],
           [0, 0],
           [0, 0],
           [0, 0]])



You could subset the array by indexes the genotypes, for example by
taking only the first two individuals and the first five loci:

.. code:: ipython3

    mypop.geno[:2, :5]




.. parsed-literal::

    array([[[1, 1],
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 1]],
    
           [[0, 1],
            [1, 1],
            [0, 1],
            [0, 1],
            [0, 1]]])



For realistic examples with many more loci, this obviously gets unwieldy
pretty soon. It's cleaner to supply a list of individuals to keep or
remove to the ``subset`` and ``drop`` functions. These return return a
new ``genotypeArray`` for the individuals of interest.

.. code:: ipython3

    print(mypop.subset([0,2]).names)
    print(mypop.drop([0,2]).names)


.. parsed-literal::

    ['my_population_0' 'my_population_2']
    ['my_population_1' 'my_population_3' 'my_population_4']


Information on individuals
~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``genotypeArray`` contains other useful information about the
individuals:

.. code:: ipython3

    print(mypop.names) # individual names
    print(mypop.size)  # number of individuals
    print(mypop.nloci) # numbe of loci typed.


.. parsed-literal::

    ['my_population_0' 'my_population_1' 'my_population_2' 'my_population_3'
     'my_population_4']
    5
    10


``make_sibships`` is a convenient way to generate a single half-sibling
array from individuals in ``mypop``. This code mates makes a half-sib
array with individual 0 as the mothers, with individuals 1, 2 and 3
contributing male gametes. Each father has four offspring each.

.. code:: ipython3

    progeny = fp.make_sibships(mypop, 0, [1,2,3], 4, 'myprogeny')

With this generation we can extract a little extra information from the
``genotypeArray`` than we could from the parents about their parents and
family structure.

.. code:: ipython3

    print(progeny.fathers)
    print(progeny.mothers)
    print(progeny.families)
    print(progeny.nfamilies)


.. parsed-literal::

    ['my_population_1' 'my_population_1' 'my_population_1' 'my_population_1'
     'my_population_2' 'my_population_2' 'my_population_2' 'my_population_2'
     'my_population_3' 'my_population_3' 'my_population_3' 'my_population_3']
    ['my_population_0' 'my_population_0' 'my_population_0' 'my_population_0'
     'my_population_0' 'my_population_0' 'my_population_0' 'my_population_0'
     'my_population_0' 'my_population_0' 'my_population_0' 'my_population_0']
    ['my_population_0/my_population_1' 'my_population_0/my_population_2'
     'my_population_0/my_population_3']
    3


Of course with real data we would not normally know the identity of the
father or the number of families, but this is useful for checking
accuracy in simulations. It can also be useful to look up the positions
of the parents in another list of names. This code finds the indices of
the mothers and fathers of the offspring in the names listed in
``mypop``.

.. code:: ipython3

    print(progeny.parent_index('mother', mypop.names))
    print(progeny.parent_index('father', mypop.names))


.. parsed-literal::

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]


Information on markers
~~~~~~~~~~~~~~~~~~~~~~

Pull out marker names with ``marker``. The names here are boring because
they are simulated, but your data can have as exciting names as you'd
like.

.. code:: ipython3

    mypop.markers




.. parsed-literal::

    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])



Check whether the locus names for parents and offspring match. This is
obvious vital for determining who shares alleles with whom, but easy to
overlook! If they don't match, the most likely explanation is that you
have imported genotype data and misspecified where the genotype data
start (the ``genotype_col`` argument).

.. code:: ipython3

    mypop.markers == progeny.markers




.. parsed-literal::

    array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
            True])



FAPS uses population allele frequencies to calculate the likelihood that
paternal alleles are drawn at random. They are are useful to check the
markers are doing what you think they are. Pull out the population
allele frequencies for each locus:

.. code:: ipython3

    mypop.allele_freqs()




.. parsed-literal::

    array([0.6, 0.4, 0.4, 0.4, 0.5, 0.2, 0.7, 0.2, 0.4, 0.2])



We can also check for missing data and heterozygosity for each marker
and individual. By default, data for each marker are returned:

.. code:: ipython3

    print(mypop.missing_data())
    print(mypop.heterozygosity())


.. parsed-literal::

    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0.8 0.4 0.8 0.4 1.  0.4 0.6 0.4 0.4 0.4]


To get summaries for each individual:

.. code:: ipython3

    print(mypop.missing_data(by='individual'))
    print(mypop.heterozygosity(by='individual'))


.. parsed-literal::

    [0. 0. 0. 0. 0.]
    [0.4 0.6 0.6 0.7 0.5]


In this instance there is no missing data, because data are simulated to
be error-free. See the next section on an empircal example where this is
not true.

Importing genotype data
-----------------------

You can import genotype data from a text or CSV (comma-separated text)
file. Both can be easily exported from a spreadsheet program. Rows index
individuals, and columns index each typed locus. More specifically:

1. Offspring names should be given in the first column
2. If the data are offspring, names of the mothers are given in the
   second column.
3. If known for some reason, names of fathers can be given as well.
4. Genotype information should be given *to the right* of columns
   indicating individual or parental names, with locus names in the
   column headers.

SNP genotype data must be biallelic, that is they can only be homozygous
for the first allele, heterozygous, or homozygous for the second allele.
These should be given as 0, 1 and 2 respectively. If genotype data is
missing this should be entered as NA.

The following code imports genotype information on real samples of
offspring from half-sibling array of wild-pollinated snpadragon
seedlings collected in the Spanish Pyrenees. The candidate parents are
as many of the wild adult plants as we could find. You will find the
data files on the `IST Austria data
repository <https://datarep.app.ist.ac.at/id/eprint/95>`__
(DOI:10.15479/AT:ISTA:95). Aside from the path to where the data file is
stored, the two other arguments specify the column containing names of
the mothers, and the first column containing genotype data of the
offspring.

.. code:: ipython3

    offspring = fp.read_genotypes(
        path = '../../data/offspring_2012_genotypes.csv',
        mothers_col=1,
        genotype_col=2)

Again, Python starts counting from zero rather than one, so the first
column is really column zero, and so on. Because these are CSV, there
was no need to specify that data are delimited by commas, but this is
included for illustration.

Offspring are divided into 60 maternal families of different sizes. You
can call the name of the mother of each offspring. You can also call the
names of the fathers, with ``offspring.fathers``, but since these are
unknown this is not informative.

.. code:: ipython3

    np.unique(offspring.mothers)




.. parsed-literal::

    array(['J1246', 'K0451', 'K0632', 'K0635', 'K1768', 'K1809', 'K2036',
           'L0057', 'L0221', 'L0911', 'L0935', 'L1264', 'L1847', 'L1872',
           'L1882', 'L1892', 'M0002', 'M0009', 'M0018', 'M0022', 'M0025',
           'M0028', 'M0034', 'M0042', 'M0043', 'M0045', 'M0047', 'M0054',
           'M0069', 'M0078', 'M0130', 'M0137', 'M0202', 'M0209', 'M0210',
           'M0225', 'M0238', 'M0251', 'M0254', 'M0258', 'M0259', 'M0267',
           'M0283', 'M0310', 'M0323', 'M0329', 'M0333', 'M0344', 'M0345',
           'M0484', 'M0494', 'M0773', 'M0884', 'M1000', 'M1335', 'M1454',
           'M1460', 'M1463', 'M1466', 'M1846'], dtype='<U5')



Offspring names are a combination of maternal family and a unique ID for
ecah offspring.

.. code:: ipython3

    offspring.names




.. parsed-literal::

    array(['J1246_221', 'J1246_222', 'J1246_223', ..., 'M1846_435',
           'M1846_436', 'M1846_437'], dtype='<U10')



You can call summaries of genotype data to help in data cleaning. For
example, this code shows the proportion of loci with missing genotype
data for the first ten offspring individuals.

.. code:: ipython3

    print(offspring.missing_data('individual')[:10])


.. parsed-literal::

    [0.01408451 0.12676056 0.09859155 0.07042254 0.01408451 0.08450704
     0.11267606 0.07042254 0.22535211 0.08450704]


This snippet shows the proportion of missing data points and
heterozygosity for the first ten loci. These can be helpful in
identifying dubious loci.

.. code:: ipython3

    print(offspring.missing_data('marker')[:10])
    print(offspring.heterozygosity()[:10])


.. parsed-literal::

    [0.07616361 0.07616361 0.0606488  0.16643159 0.05500705 0.0909732
     0.09449929 0.05994358 0.05923836 0.07757405]
    [0.36812412 0.39985896 0.45627645 0.33497884 0.42665726 0.4696756
     0.36318759 0.35543018 0.4506347  0.3751763 ]


Multiple families
-----------------

In real data set we generally work with multplie half-sibling arrays at
once. For downstream analyses we need to split up the genotype data into
families to reflect this. This is easy to do with ``split`` and a vector
of labels to group offspring by. This returns a dictionary of
``genotypeArray`` objects labelled by maternal family. These snippet
splits up the data and prints the maternal family names.

.. code:: ipython3

    offs_split = offspring.split(by = offspring.mothers)
    offs_split.keys()




.. parsed-literal::

    dict_keys(['J1246', 'K0451', 'K0632', 'K0635', 'K1768', 'K1809', 'K2036', 'L0057', 'L0221', 'L0911', 'L0935', 'L1264', 'L1847', 'L1872', 'L1882', 'L1892', 'M0002', 'M0009', 'M0018', 'M0022', 'M0025', 'M0028', 'M0034', 'M0042', 'M0043', 'M0045', 'M0047', 'M0054', 'M0069', 'M0078', 'M0130', 'M0137', 'M0202', 'M0209', 'M0210', 'M0225', 'M0238', 'M0251', 'M0254', 'M0258', 'M0259', 'M0267', 'M0283', 'M0310', 'M0323', 'M0329', 'M0333', 'M0344', 'M0345', 'M0484', 'M0494', 'M0773', 'M0884', 'M1000', 'M1335', 'M1454', 'M1460', 'M1463', 'M1466', 'M1846'])



Each entry is an individual ``genotypeArray``. You can pull out
individual families by indexing the dictionary by name. For example,
here are the names of the offspring in family J1246:

.. code:: ipython3

    offs_split["J1246"].names




.. parsed-literal::

    array(['J1246_221', 'J1246_222', 'J1246_223', 'J1246_224', 'J1246_225',
           'J1246_226', 'J1246_227', 'J1246_228', 'J1246_229', 'J1246_230',
           'J1246_231', 'J1246_232', 'J1246_233', 'J1246_241', 'J1246_615',
           'J1246_616', 'J1246_617', 'J1246_618', 'J1246_619', 'J1246_620',
           'J1246_621', 'J1246_622', 'J1246_623', 'J1246_624', 'J1246_625'],
          dtype='<U10')



To perform operations on each ``genotypeArray`` we now have to iterate
over each element. A convenient way to do this is with dictionary
comprehensions by separating out the labels from the ``genotypeArray``
objects using ``items``.

As an example, here's how you call the number of offspring in each
family. It splits up the dictionary into keys for each family, and calls
``size`` on each ``genotypeArray`` (labelled genArray in the
comprehension).

.. code:: ipython3

    {family : genArray.size for family,genArray in offs_split.items()}




.. parsed-literal::

    {'J1246': 25,
     'K0451': 27,
     'K0632': 6,
     'K0635': 25,
     'K1768': 25,
     'K1809': 25,
     'K2036': 19,
     'L0057': 24,
     'L0221': 22,
     'L0911': 24,
     'L0935': 26,
     'L1264': 23,
     'L1847': 33,
     'L1872': 26,
     'L1882': 27,
     'L1892': 9,
     'M0002': 3,
     'M0009': 41,
     'M0018': 10,
     'M0022': 22,
     'M0025': 26,
     'M0028': 33,
     'M0034': 37,
     'M0042': 25,
     'M0043': 11,
     'M0045': 21,
     'M0047': 25,
     'M0054': 27,
     'M0069': 31,
     'M0078': 33,
     'M0130': 20,
     'M0137': 20,
     'M0202': 25,
     'M0209': 33,
     'M0210': 26,
     'M0225': 23,
     'M0238': 24,
     'M0251': 24,
     'M0254': 26,
     'M0258': 23,
     'M0259': 16,
     'M0267': 23,
     'M0283': 25,
     'M0310': 10,
     'M0323': 25,
     'M0329': 26,
     'M0333': 26,
     'M0344': 16,
     'M0345': 24,
     'M0484': 24,
     'M0494': 22,
     'M0773': 25,
     'M0884': 36,
     'M1000': 24,
     'M1335': 28,
     'M1454': 18,
     'M1460': 22,
     'M1463': 26,
     'M1466': 24,
     'M1846': 23}



You can achieve the same thing with a list comprehension, but you lose
information about family ID. It is also more difficult to pass a list on
to downstream functions. This snippet shows the first ten items.

.. code:: ipython3

    [genArray.size for genArray in offs_split.values()][:10]




.. parsed-literal::

    [25, 27, 6, 25, 25, 25, 19, 24, 22, 24]


