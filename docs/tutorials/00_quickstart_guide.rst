Quickstart guide to FAPS
========================

Tom Ellis, May 2020.

If you are impatient to do an analyses as quickly as possible without
reading the rest of the documentation, this page provides a minimal
example. The work flow is as follows:

1. Import marker data on offspring and parents
2. Create a matrix of paternity of each individual offspring
3. Cluster offspring into full sibships.
4. ????
5. Profit.

It goes without saying that to understand what the code is doing and get
the most out of the data, you should read the
`tutorials <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/index.html>`__.

Import the package.

.. code:: ipython3

    import faps as fp
    import numpy as np

Import genotype data. These are CSV files with:

1. A column giving the name of each individual
2. For the offspring, the second column gives the name of the known
   mother.
3. Subsequent columns give genotype data for each marker, with column
   headers giving marker names.

.. code:: ipython3

    adults  = fp.read_genotypes('../data/parents_2012_genotypes.csv', genotype_col=1)
    progeny = fp.read_genotypes('../data/offspring_2012_genotypes.csv', genotype_col=2, mothers_col=1)
    # Mothers are a subset of the adults.
    mothers = adults.subset(individuals=np.unique(progeny.mothers))

In this example, the data are for multiple maternal families, each
containing a mixture of full- and half-siblings. We need to divide the
offspring and mothers into maternal families.

.. code:: ipython3

    progeny = progeny.split(progeny.mothers)
    mothers = mothers.split(mothers.names)

I expect that multiple maternal families will be the most common
scenario, but if you happen to only have a sigle maternal family, you
can skip this.

Calculate paternity of individuals. This is equivalent to the **G**
matrix in `Ellis et al
(2018) <https://doi.org/10.1111/1755-0998.12782>`__.

.. code:: ipython3

    patlik = fp.paternity_array(progeny, mothers, adults, mu = 0.0015)

Cluster offspring in each family into full-sibling families.

.. code:: ipython3

    sibships = fp.sibship_clustering(patlik)

You can pull out `various kinds of
information <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/04_sibship_clustering.html>`__
about the each clustered maternal family. For example, get the
most-likely number of full-sib families in maternal family J1246.

.. code:: ipython3

    sibships["J1246"].mean_nfamilies()




.. parsed-literal::

    5.605375868371062



Or do this for all families with a dict comprehension:

.. code:: ipython3

    {k: v.mean_nfamilies() for k,v in sibships.items()}




.. parsed-literal::

    {'J1246': 5.605375868371062,
     'K0451': 12.679100830502975,
     'K0632': 5.098186791267536,
     'K0635': 6.222576977121563,
     'K1768': 5.95279321064476,
     'K1809': 12.317762689872342,
     'K2036': 4.518681729473807,
     'L0057': 18.53519892725761,
     'L0221': 7.523719666781066,
     'L0911': 21.579949302519644,
     'L0935': 21.584456885870384,
     'L1264': 10.973166572630031,
     'L1847': 12.064523674941354,
     'L1872': 9.048439399512647,
     'L1882': 16.113027728381027,
     'L1892': 7.147054942431994,
     'M0002': 1.047888622290101,
     'M0009': 23.11360020574565,
     'M0018': 7.051482492713087,
     'M0022': 7.450274317790799,
     'M0025': 10.454372677003231,
     'M0028': 4.239820497584428,
     'M0034': 12.435549448178843,
     'M0042': 6.088524327650887,
     'M0043': 4.87419977417076,
     'M0045': 6.000782412960964,
     'M0047': 12.719548559166366,
     'M0054': 18.984647576874096,
     'M0069': 21.02305110499397,
     'M0078': 23.42550345266462,
     'M0130': 17.069045572015895,
     'M0137': 15.029407573170278,
     'M0202': 11.48844273728524,
     'M0209': 8.819699122141314,
     'M0210': 10.999293014192693,
     'M0225': 7.045833239484286,
     'M0238': 10.247537341131476,
     'M0251': 9.39369696108596,
     'M0254': 13.997079852966515,
     'M0258': 9.828694751876757,
     'M0259': 12.199493597014733,
     'M0267': 13.999934870300056,
     'M0283': 12.76441063459917,
     'M0310': 7.9950925640201405,
     'M0323': 10.031892269392502,
     'M0329': 15.65033087966963,
     'M0333': 15.988483638068129,
     'M0344': 9.946009544142706,
     'M0345': 20.309316369318616,
     'M0484': 18.495245747794613,
     'M0494': 8.05463069910333,
     'M0773': 6.824167457325241,
     'M0884': 28.620466685852023,
     'M1000': 7.923972617146549,
     'M1335': 19.898885496992698,
     'M1454': 12.853870585838022,
     'M1460': 7.055349431265118,
     'M1463': 13.841229954609007,
     'M1466': 23.197797611570273,
     'M1846': 12.055278800405954}


