Introduction
============

Tom Ellis, February 2017

FAPS stands for Fractional Analysis of Sibships and Paternity. It is a
Python package for reconstructing genealogical relationships in wild
populations, and making inference about biological processes. The
sections of this document are intended as a user’s guide to introduce
how FAPS works. For full details of the method, see `Ellis et al.
2018 <https://doi.org/10.1111/1755-0998.12782>`__.

Background
----------

The motivation for developing of FAPS was to provide a package to
investigate biological processes in a large population of snapdragons in
the wild. Existing packages to do this relied on computationally intense
Markov-chain algorithms, which limited the scope for subsequent analysis
and for checking assumptions through simulations. As such, most of the
examples in this guide relate to snapdragons. That said, FAPS addresses
general issues in pedigree reconstruction of wild populations, and it is
hoped that FAPS will be useful for other plant and animal systems.

The specific aims of FAPS were to provide a package which would allow us
to:

1. Jointly inferring sibship and paternity relationships, and hence gain
   better estimates of both (Wang 2007), without using a random-walk
   algorithm such as MCMC or simulated annealing.
2. Account for uncertainty about relationships (i.e. to integrate
   fractional paternity assignment with sibship inference).
3. Provide tools for testing biological hypotheses with minimal hassle.
4. Do this efficiently for large populations.
5. Not depend on mating system or genetic marker technology.

Overview of the method
~~~~~~~~~~~~~~~~~~~~~~

FAPS reconstructs relationships for one or more half-sibling arrays,
that is to say a sample of offspring from one or more mothers whose
identity is uncertain. A half-sibling array consistent of seedlings from
the same maternal plant, or a family of lambs from the same ewe, to give
two examples. The paternity of each offspring is unkown, and hence it is
unknown whether pairs of offspring are full or half siblings.

There is also a sample of males, each of whom is a candidate to be the
true sire of each offspring individual. FAPS is used to identify likely
sibling relationships between offspring and their shared fathers based
on typed genetic markers, and to use this information to make meaningful
conclusions about population or matig biology. The procedure can be
summarised as follows:

1. FAPS begins with an matrix with a row for each offspring and a column
   for each candidate male. Each element represents the likelihood that
   the corresponding male is the sire of the offspring individual.
2. Based on this matrix, we apply hierarchical clustering to group
   individuals into possible configurations of full-siblings.
3. Calculate statistics of interest on each configuration, weighted by
   the probability of each candidate male and each configuration.

Like all statistical analyses, FAPS makes a number of assumptions about
your data. It is good to state these explicitly so everyone is aware of
the limitations of the data and method:

1. Individuals come from two generations: a sample of offspring, and a
   sample of reproductive adults.
2. At present FAPS is intended to deal with paternity assignment only,
   i.e. when one parent is known and the second must be inferred. That
   said, it would not take too much work to modify FAPS to accommodate
   bi-parental assignment.
3. Sampling of the adults does not need to be complete, but the more
   true parents are missing the more likely it is that true
   full-siblings are inferred to be half-siblings.
4. Genotype data are biallelic, and genotyping errors can be
   characterised as a haploid genotype being either correct or incorrect
   with some probability. Multi-allelic markers with complex error
   patterns such as microsatellites are not supported at present - see
   the sections on genotype data and paternity arrays for more on this.

Other software
~~~~~~~~~~~~~~

Depending on your biological questions, your data may not fit the
assumptions listed above, and an alternative approach might be more
appropriate. For example:

-  Cervus (Marshall 1998) was the first widely-used program for
   parentage/paternity assignment. It is restricted to categorical
   assignment of parent-offspring pairs and relies on substantial prior
   information about the demography of the population.
-  Colony2 is a dedicated Fortran program for sibship inference (Wang
   2004, Wang and Santure 2009). Analyses can be run with or without a
   sample of parents. Originally designed for microsatellites, it can be
   slow for large SNP datasets, although Wang *et al.* 2012 updated the
   software to support SNP data with a slight cost to accuracy.
-  As argued eloquently by Hadfield (2006), frequently the aim of
   genealogical reconstruction is very often not the pedigree itself,
   but some biological parameter of the population. His MasterBayes
   package for R allows the user to dircetly estimate the posterior
   distribution for the statistic of interest. It is usually slow for
   SNP data, can does not incorporate information about siblings.
-  SNPPIT (Anderson & Garza, 2006) provides very fast and accurate
   assignments of parentage using SNPs.
-  Almudevar (2003), Huisman (2017) and Anderson and Ng (2016) provide
   software for estimating multigenerational pedigrees.

Prerequisites
-------------

Python packages
~~~~~~~~~~~~~~~

It is assumed you have read Ellis *et al.* 2018 for the basic background
to the method. It would also be useful to read Devlin (1988) for an
overview of the motivation and basic methods of fractional assignment.
It is also assumed you have a basic understanding of probability and
likelihood; see Bolker (2006, chapter 6) for an example of a general
introduction.

FAPS uses Python as an interface, but it is hoped that this guide should
allow users who aren’t familiar with Python to adapt the code to their
needs. It would be worthwhile to at least familiarise yourself with
Python’s data types, especially lists and NumPy arrays, and how list
comprehensions work. A general introduction to Python concepts can be
found `here <v>`__. I recommend interacting with FAPS through
`IPython/Jupyter <http://ipython.org/>`__, which allows you to test
small pieces of code and annotate analyses as you go. This document, for
example, is written in IPython.

You will of course need to have Python installed on your machine. If you
do not already have this, instructions can be found
`here <https://wiki.python.org/moin/BeginnersGuide/Download>`__. You
will also need to install the NumPy, fastcluster and Pandas libraries.
These should be installed automatically if you intall FAPS using pip
(see below), but if for some reason they are not, the easiest way to do
this is to install one of the `scientific Python
bundles <http://www.scipy.org/install.html>`__. Some of the simulation
tools also make use of `Jupyter
widgets <http://ipywidgets.readthedocs.io/en/latest/user_install.html>`__,
but these are optional. There are no specific hardware requirements
beyond what is needed to run Python, but it is possible that RAM will be
a limiting factor if you are dealing with large samples (for example
~100 offspring and 10,000 candidate males).

All testing and development of FAPS was done on Linux and Mac machines.
I have not tested it on Windows, nor do I intend to. That said, an
advantage of Python is that it ought to work on any operating system, so
in principle FAPS ought to run as well as on a Unix machine. One
important difference is that Windows uses ‘' instead of’/’ in its file
paths, so you will need to edit accordingly.

Installing FAPS
~~~~~~~~~~~~~~~

The best way to install FAPS is to use Python’s package manager, Pip.
Instructions to do so can be found at that projects `documentation
page <https://pip.pypa.io/en/stable/installing/>`__. Windows users might
also consider
`pip-Win <https://sites.google.com/site/pydatalog/python/pip-for-windows>`__.
To download the stable release run ``pip install faps`` in the command
line. If Python is unable to locate the package, try
``pip install fap --user``. You can download the development version of
FAPS from `the project github
repository <https://github.com/ellisztamas/faps>`__.

Once in Python/IPython you’ll need to import the package, as well as the
NumPy library on which it is based. In the rest of this document, I’ll
assume you’ve run the following lines to do this if this isn’t
explicitly stated.

from faps import \* import numpy as np

The asterisk on the first line is a shortcut to tell Python to import
all the functions and classes in FAPS. This is somewhat lazy, but saves
us having to give the package name every time we call something.

Marker data types
-----------------

The basic unit on which analyses are built is a matrix of likelihoods of
paternities, with a row for each offspring individual and a column for
each candidate father (matrix G in Ellis *et al.* 2018). Each element
represents the likelihood that a single candidate male is the father of
a single offspring individual based on alleles shared between them and
the offspring’s mother. One of the aims of FAPS was to create a method
which did not depend on marker type, mating system, ploidy, or
genotyping technology, with the aim that it should be applicable to as
broad a range of datasets that exist, or may yet exist. As such, the
optimum way to estimate G will vary from case to case.

Although microsatellite data are fairly standard in format, SNP
technologies are moving fast. Since every technology comes with its own
quirks, so FAPS was written with the expectation that *most users will
be using non-standard data in some way*. As such, it is really difficult
to write functions to calculate G that are general, and users are
strongly encouraged to think about the most appropriate way to calculate
G for their data. Once you have done this, all other aspects of the
analysis are independent of marker type. See the sections on `Importing
genotype
data <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/02_genotype_data.html#importing-genotype-data>`__
and `Paternity
arrays <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/03_paternity_arrays.html#importing-a-paternityarray>`__
for more details on how to import data.

FAPS will also work given an appropriate G matrix for a polyploid
species, but you will also need to provide G yourself. See Wang 2016 for
inspiration. This topic is rather involved, and I personally do not feel
comfortable implementing anything in this area myself, but I would be
interested to hear from anyone who is willing to try it.

Literature cited
----------------

-  Anderson, E. C., & Garza, J. C. (2006). The power of
   single-nucleotide polymorphisms for large-scale parentage inference.
   Genetics, 172(4),2567–2582.
-  Anderson, E. C., & Ng, T. C. (2016). Bayesian pedigree inference with
   small numbers of single nucleotide polymorphisms via a factor-graph
   representation. Theoretical Population Biology, 107, 39–51.
   https://doi.org/10.1016/j.tpb.2015.09.005
-  Devlin, B., Roeder, K., & Ellstrand, N. (1988). Fractional paternity
   assignment: Theoretical development and comparison to other methods.
   Theoretical and Applied Genetics, 76(3), 369–380.
-  Ellis, TJ, Field DL, Barton, NH (2018) Efficient inference of
   paternity and sibship inference given known maternity via
   hierarchical clustering. Molecular Ecology Resources
   https://doi.org/10.1111/1755-0998.12782
-  Hadfield, J., Richardson, D., & Burke, T. (2006). Towards unbiased
   parentage assignment: Combining genetic, behavioural and spatial data
   in a Bayesian framework. Molecular Ecology, 15(12), 3715–3730.
   https://doi.org/10.1111/j.1365-294X.2006.03050.x
-  Huisman, J. (2017). Pedigree reconstruction from SNP data: parentage
   assignment, sibship clustering and beyond. Molecular ecology
   resources, 17(5), 1009-1024. https://doi.org/10.1111/1755-0998.12665
-  Marshall, T., Slate, J., Kruuk, L., & Pemberton, J. (1998).
   Statistical confdence for likelihood-based paternity inference in
   natural populations. Molecular Ecology, 7(5), 639–655.
   https://doi.org/10.1046/j.1365-294x.1998.00374.x
-  Wang, J. (2004). Sibship reconstruction from genetic data with
   typingerrors. Genetics, 166(4), 1963–1979.
   https://doi.org/10.1534/genetics.166.4.1963
-  Wang, J. (2007). Parentage and sibship exclusions: Higher statistical
   power with more family members. Heredity, 99(2), 205–217.
   https://doi.org/10.1038/sj.hdy.6800984
-  Wang, J., & Santure, A. W. (2009). Parentage and sibship inference
   from multilocus genotype data under polygamy. Genetics, 181(4),
   1579–1594. https://doi.org/10.1534/genetics.108.100214
-  Wang, J. (2012). Computationally efficient sibship and parentage
   assignment from multilocus marker data. Genetics, 191(1),
   183–194.https://doi.org/10.1534/genetics.111.138149
-  Wang, J., & Scribner, K. T. (2014). Parentage and sibship inference
   from markers in polyploids. Molecular Ecology Resources, 14(3),
   541–553. https://doi.org/10.1111/1755-0998.12210
