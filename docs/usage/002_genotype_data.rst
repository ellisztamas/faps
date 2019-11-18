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
dataset is given in `section
8 <https://github.com/ellisztamas/faps/blob/master/docs/08%20Data%20cleaning%20in%20A.%20majus.ipynb>`__.
In the `next
section <https://github.com/ellisztamas/faps/blob/master/docs/03%20Paternity%20arrays.ipynb>`__
we'll see how to combine genotype information on offspring and a set of
candidate parents to create an array of likelihoods of paternity for
dyads of offspring and candidate fathers. Also relevant is the section
on `simulating data and power
analysis <https://github.com/ellisztamas/faps/blob/master/docs/06%20Simulating%20data.ipynb>`__.

Currently, FAPS ``genotypeArray`` objects assume you are using
biallelic, unlinked SNPs for a diploid. If your system deviates from
these criteria in some way you can also skip this stage by creating your
own array of paternity likelihoods using an appropriate likelihood
function, and importing this directly as a ``paternityArrays``. See the
next section for more on ``paternityArray`` objects and how they should
look.

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

    from faps import *
    import numpy as np
    
    allele_freqs = np.random.uniform(0.3,0.5,10)
    mypop = make_parents(5, allele_freqs, family_name='my_population')

The object we just created contains information about the genotypes of
each of the ten parent individuals. Genotypes are stored as
*N*\ x\ *L*\ x2-dimensional arrays, where *N* is the number of
individuals and *L* is the number of loci. We can view the genotype for
the first parent like so (recall that Python starts counting from zero,
not one):

.. code:: ipython3

    mypop.geno[0]




.. parsed-literal::

    array([[0, 0],
           [0, 1],
           [1, 1],
           [0, 0],
           [0, 1],
           [0, 1],
           [1, 1],
           [0, 1],
           [0, 0],
           [1, 1]])



You could subset the array by indexes the genotypes, for example by
taking only the first two individuals and the first five loci:

.. code:: ipython3

    mypop.geno[:2, :5]




.. parsed-literal::

    array([[[0, 0],
            [0, 1],
            [1, 1],
            [0, 0],
            [0, 1]],
    
           [[0, 0],
            [0, 1],
            [1, 1],
            [0, 0],
            [0, 0]]])



For realistic examples with many more loci, this obviously gets unwieldy
pretty soon. It's cleaner to supply a list of individuals to keep or
remove to the ``subset`` and ``drop`` functions. These return return a
new ``genotypeArray`` for the individuals of interest.

.. code:: ipython3

    print mypop.subset([0,2]).names
    print mypop.drop([0,2]).names


.. parsed-literal::

    ['my_population_0' 'my_population_2']
    ['my_population_1' 'my_population_3' 'my_population_4']


Information on indivuals
~~~~~~~~~~~~~~~~~~~~~~~~

A ``genotypeArray`` contains other useful information about the
individuals:

.. code:: ipython3

    print mypop.names # individual names
    print mypop.size  # number of individuals
    print mypop.nloci # numbe of loci typed.


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

    progeny = make_sibships(mypop, 0, [1,2,3], 4, 'myprogeny')

With this generation we can extract a little extra information from the
``genotypeArray`` than we could from the parents about their parents and
family structure.

.. code:: ipython3

    print progeny.fathers
    print progeny.mothers
    print progeny.families
    print progeny.nfamilies


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

    print progeny.parent_index('mother', mypop.names)
    print progeny.parent_index('father', mypop.names)


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

    array([0.2, 0.4, 0.5, 0.2, 0.3, 0.3, 0.4, 0.5, 0.3, 0.7])



We can also check for missing data and heterozygosity for each marker
and individual. By default, data for each marker are returned:

.. code:: ipython3

    print mypop.missing_data()
    print mypop.heterozygosity()


.. parsed-literal::

    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0.4 0.8 0.2 0.4 0.6 0.6 0.4 0.6 0.6 0.2]


To get summaries for each individual:

.. code:: ipython3

    print mypop.missing_data(by='individual')
    print mypop.heterozygosity(by='individual')


.. parsed-literal::

    [0. 0. 0. 0. 0.]
    [0.4 0.3 0.4 0.7 0.6]


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
offspring and candidate parents. Offspring are a half-sibling array of
wild-pollinated snpadragon seedlings collected in the Spanish Pyrenees.
The candidate parents are as many of the wild adult plants as we could
find. You will find the data files on the `IST Austria data
repository <https://datarep.app.ist.ac.at/id/eprint/95>`__
(DOI:10.15479/AT:ISTA:95).

.. code:: ipython3

    adults   = read_genotypes('../manuscript_faps/data_files/parents_SNPs_2012.csv', genotype_col=1, delimiter=',')
    offspring = read_genotypes('../manuscript_faps/data_files/offspring_SNPs_2012.csv', genotype_col=2, mothers_col=1)

Again, Python starts counting from zero rather than one, so the first
column is really column zero, and so on. Because these are CSV, there
was no need to specify that data are delimited by commas, but this is
included for illustration.

You can call summaries of genotype data to help in data cleaning. For
example, this code shows the proportion of loci with missing genotype
data for the first ten offspring individuals.

.. code:: ipython3

    print offspring.missing_data('individual')


.. parsed-literal::

    [0.01449275 0.11594203 0.08695652 ... 0.05797101 0.07246377 0.07246377]


This snippet shows the proportion of missing data points and
heterozygosity for the first ten loci. These can be helpful in
identifying dubious loci.

.. code:: ipython3

    print offspring.missing_data('marker')[:9]
    print offspring.heterozygosity()[:9]

Multiple families
-----------------

In real data set we generally work with multplie half-sibling arrays at
once. For downstream analyses we need to split up the genotype data into
families to reflect this. To do this we need a list of positions for the
mother of each offspring. We can abbreviate 'mother' to 'm' passed to
``parent_index`` out of clemency to weary fingers.

.. code:: ipython3

    mi = offspring.parent_index('m', offspring.mothers) # index position of the mothers
    np.unique(adults.names[mi]) # names of the mothers




.. parsed-literal::

    array(['L0009', 'L0263', 'L0573', 'L0966', 'L1223', 'L1766', 'L1772',
           'M0015', 'M0018', 'M0043', 'M0084', 'M0110', 'M0130', 'M0155',
           'M0165', 'M0188', 'M0219', 'M0252', 'M0289', 'M0311', 'M0336',
           'M0347', 'M0368', 'M0394', 'M0421', 'M0447', 'M0478', 'M0513',
           'M0534', 'M0570', 'M0595', 'M0630', 'M0657', 'M0680', 'M0705',
           'M0729', 'M0758', 'M0782', 'M0799', 'M0822', 'M0847', 'M0875',
           'M0885', 'M0911', 'M0940', 'M0968', 'M0974', 'M0991', 'M1015',
           'M1039', 'M1065', 'M1088', 'M1116', 'M1137', 'M1175', 'M1200',
           'M1227', 'M1255', 'M1273', 'M1295', 'M1322', 'M1346', 'M1356'],
          dtype='|S5')



We split up the data using ``split``. This returns a list of 63
``genotypeArray`` objects for each of the 63 maternal families in this
dataset.

.. code:: ipython3

    offs2 = offspring.split(mi)
    print len(offs2)


.. parsed-literal::

    63


You can apply any commands to each object in the list just like we did
before. For example, this summarises the number of individuals in family
3, their names, and the names of the mothers. The latter are
(thanksfully) identical.

.. code:: ipython3

    print 'Family size:', offs2[3].size
    print 'Offspring names: ',offs2[3].names
    print 'Mothers names:',offs2[3].mothers


.. parsed-literal::

    Family size: 24
    Offspring names:  ['L0057_745' 'L0057_746' 'L0057_747' 'L0057_748' 'L0057_749' 'L0057_750'
     'L0057_751' 'L0057_752' 'L0057_753' 'L0057_754' 'L0057_755' 'L0057_756'
     'L0057_846' 'L0057_847' 'L0057_848' 'L0057_849' 'L0057_850' 'L0057_851'
     'L0057_852' 'L0057_853' 'L0057_854' 'L0057_855' 'L0057_856' 'L0057_857']
    Mothers names: ['L0057' 'L0057' 'L0057' 'L0057' 'L0057' 'L0057' 'L0057' 'L0057' 'L0057'
     'L0057' 'L0057' 'L0057' 'L0057' 'L0057' 'L0057' 'L0057' 'L0057' 'L0057'
     'L0057' 'L0057' 'L0057' 'L0057' 'L0057' 'L0057']


To perform operations on each ``genotypeArray`` we can use Python's
excellent list comprehension format. These are convenient fast, and
straightforward once you are used to them, but if you aren't familiar
with list comprehensions, it is worth searching for tutorials online.

As an example, here's how you call the number of offspring in each
family.

.. code:: ipython3

    np.array([offs2[i].size for i in range(len(offs2))])




.. parsed-literal::

    array([25, 25, 25, 24, 27,  1, 26,  3, 25, 41, 25, 19, 24, 10, 22, 26, 33,
           37, 22, 25, 11, 21, 25, 27, 24, 31, 33, 20, 33, 25, 33, 26, 23, 24,
           24, 26, 23, 16, 23, 25, 27, 10, 25, 26, 26,  6, 16, 24, 23, 24, 22,
           25, 20, 36, 24, 26, 28, 18, 22, 26, 24,  9, 23])



