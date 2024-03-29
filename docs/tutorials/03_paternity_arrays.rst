Paternity arrays
================

Tom Ellis, March 2017, updated June 2020

.. code:: ipython3

    import faps as fp
    import numpy as np
    print("Created using FAPS version {}.".format(fp.__version__))


.. parsed-literal::

    Created using FAPS version 2.6.4.


Paternity arrays are the what sibship clustering is built on in FAPS.
They contain information about the probability that each candidate male
is the father of each individual offspring - this is what the FAPS paper
refers to as matrix **G**. This information is stored in a
``paternityArray`` object, along with other related information. A
``paternityArray`` can either be imported directly, or created from
genotype data.

This notebook will examine how to:

1. Create a ``paternityArray`` from marker data.
2. Examine what information it contains.
3. Read and write a ``paternityArray`` to disk, or import a custom
   ``paternityArray``.

Once you have made your ``paternityArray``, the `next
step <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/04_sibship_clustering.html>`__
is to cluster the individuals in your array into full sibship groups.

Note that this tutorial only deals with the case where you have a
``paternityArray`` object for a single maternal family. If you have
multiple families, you can apply what is here to each one, but you’ll
have to iterate over those families. See the specific
`tutorial <https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/tutorials/07_dealing_with_multiple_half-sib_families.html>`__
on that.

Creating a ``paternityArray`` from genotype data
------------------------------------------------

To create a ``paternityArray`` from genotype data we need to specficy
``genotypeArray``\ s for the offspring, mothers and candidate males.
Currently only biallelic SNP data are supported.

We will illustrate this with a small simulated example again with four
adults and six offspring typed at 50 loci.

.. code:: ipython3

    np.random.seed(27) # this ensures you get exactly the same answers as I do.
    allele_freqs = np.random.uniform(0.3,0.5, 50)
    mypop        = fp.make_parents(4, allele_freqs, family_name='my_population')
    progeny      = fp.make_sibships(mypop, 0, [1,2], 3, 'myprogeny')

We need to supply a ``genotypeArray`` for the mothers. This needs to
have an entry for for every offspring, i.e. six replicates of the
mother.

.. code:: ipython3

    mum_index = progeny.parent_index('mother', mypop.names) # positions in the mothers in the array of adults
    mothers   = mypop.subset(mum_index) # genotypeArray of the mothers

To create the ``paternityArray`` we also need to supply information on
the genotyping error rate (mu). In this toy example we know the error
rate to be zero. However, in reality this will almost never be true, and
moreover, sibship clustering becomes unstable when errors are zero, so
we will use a small number for the error rate.

.. code:: ipython3

    error_rate = 0.0015
    patlik = fp.paternity_array(
        offspring = progeny,
        mothers = mothers,
        males= mypop,
        mu=error_rate)

``paternityArray`` structure
----------------------------

Basic attributes
~~~~~~~~~~~~~~~~

A ``paternityArray`` inherits information about individuals from found
in a ``genotypeArray``. For example, labels of the candidates, mothers
and offspring.

.. code:: ipython3

    print(patlik.candidates)
    print(patlik.mothers)
    print(patlik.offspring)


.. parsed-literal::

    ['my_population_0' 'my_population_1' 'my_population_2' 'my_population_3']
    ['my_population_0' 'my_population_0' 'my_population_0' 'my_population_0'
     'my_population_0' 'my_population_0']
    ['myprogeny_0' 'myprogeny_1' 'myprogeny_2' 'myprogeny_3' 'myprogeny_4'
     'myprogeny_5']


Representation of matrix **G**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The FAPS paper began with matrix **G** that gives probabilities that
each individual is sired by each candidate father, or that the true
father is absent from the sample. Recall that this matrix had a row for
every offspring and a column for every candidate father, plus and
additional column for the probability that the father was unsampled, and
that these rows sum to one. The relative weight given to these two
sections of **G** is determined by our prior expectation *p* about what
proportion of true fathers were sampled. This section will examine how
that is matrix is constructed.

The most important part of the ``paternityArray`` is the likelihood
array, which represent the log likelihood that each candidate male is
the true father of each offspring individual. In this case it will be a
6x4 dimensional array with a row for each offspring and a column for
each candidate.

.. code:: ipython3

    patlik.lik_array




.. parsed-literal::

    array([[-429.02140753,  -30.49847594, -326.16586129, -271.82010979],
           [-430.40770189,  -29.80532876, -246.13499882, -327.88862788],
           [-456.65242865,  -31.19162313, -434.65442344, -326.85900847],
           [-268.0873111 , -272.91736496,  -29.80532876, -220.69674793],
           [-212.82526887, -356.05728839,  -31.19162313, -244.99556454],
           [-266.70101674, -327.26311646,  -28.4190344 , -190.98628527]])



You can see that the log likelihoods of paternity for the first
individual are much lower than the other candidates. This individual is
the mother, so this makes sense. You can also see that the highest log
likelihoods are in the columns for the real fathers (the 2nd column in
rows one to three, and the third column in rows four to six).

The ``paternityArray`` also includes information that the true sire is
not in the sample of candidate males. In this case this is not helpful,
because we know sampling is complete, but in real examples is seldom the
case. By default this is defined as the likelihood of generating the
offspring genotypes given the known mothers genotype and alleles drawn
from population allele frequencies. Here, values for the six offspring
are higher than the likelihoods for the non-sires, indicating that they
are no more likely to be the true sire than a random unrelated
individual.

.. code:: ipython3

    patlik.lik_absent




.. parsed-literal::

    array([-56.18419755, -56.88945139, -61.84235747, -49.42854881,
           -50.96313387, -48.80522532])



The numbers in the two previous cells are (log) *likelihoods*, either of
paternity, or that the father was missing. These are estimated from the
marker data and are not normalised to probabilities. To join these bits
of information together, we also need to specify our *prior* belief
about the proportion of fathers you think you sampled based on your
domain expertise in the system, which should be a float between 0 and 1.

Let’s assume that we think we missed 10% of the fathers and set that as
an attribute of the ``paternityArray`` object:

.. code:: ipython3

    patlik.missing_parents = 0.1

The function ``prob_array`` creates the **G** matrix by multiplying
``lik_absent`` by 0.1 and ``lik_array`` by 0.9 (i.e. 1-0.1), then
normalising the rows to sum to one. This returns a matrix with an extra
column than ``lik_array`` had.

.. code:: ipython3

    print(patlik.lik_array.shape)
    print(patlik.prob_array().shape)


.. parsed-literal::

    (6, 4)
    (6, 5)


Note that FAPS is doing this on the log scale under the hood. To check
its working, we can check that rows sum to one.

.. code:: ipython3

    np.exp(patlik.prob_array()).sum(axis=1)




.. parsed-literal::

    array([1., 1., 1., 1., 1., 1.])



If we were sure we really had sampled every single father, we could set
the proportion of missing fathers to 0. This will throw a warning urging
you to be cautious about that, but will run. We can see that the last
column has been set to negative infinity, which is log(0).

.. code:: ipython3

    patlik.missing_parents = 0
    patlik.prob_array()


.. parsed-literal::

    /home/thomas.ellis/.local/lib/python3.8/site-packages/faps/paternityArray.py:216: UserWarning: Missing_parents set to 0. Only continue if you are sure you really have 100% of possible fathers.
      if self.missing_parents ==0: warn("Missing_parents set to 0. Only continue if you are sure you really have 100% of possible fathers.")




.. parsed-literal::

    array([[-398.52293159,    0.        , -295.66738534, -241.32163384,
                     -inf],
           [-400.60237313,    0.        , -216.32967006, -298.08329912,
                     -inf],
           [-425.46080552,    0.        , -403.46280032, -295.66738534,
                     -inf],
           [-238.28198233, -243.1120362 ,    0.        , -190.89141917,
                     -inf],
           [-181.63364574, -324.86566527,    0.        , -213.80394141,
                     -inf],
           [-238.28198233, -298.84408206,    0.        , -162.56725087,
                     -inf]])



You can also set the proportion of missing fathers directly when you
create the paternity array.

.. code:: ipython3

    patlik = fp.paternity_array(
        offspring = progeny,
        mothers = mothers,
        males= mypop,
        mu=error_rate,
        missing_parents=0.1)

Modifying a ``paternityArray``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the previous example we saw how to set the proportion of missing
fathers by changing the attributes of the ``paternityArray`` object.
There are a few other attributes that can be set that will modify the
**G** matrix before passing this on to cluster offspring into sibships.

Selfing rate
^^^^^^^^^^^^

Often the mother is included in the sample of candidate males, either
because you are using the same array for multiple families, or
self-fertilisation is a biological possibility. In a lot of cases though
the mother cannot simultaneously be the sperm/pollen donor, and it is
necessary to set the rate of self-fertilisation to zero (the natural
logarithm of zero is negative infinity). This can be done simply by
setting the attribute ``selfing_rate`` to zero:

.. code:: ipython3

    patlik.selfing_rate=0
    patlik.prob_array()




.. parsed-literal::

    array([[           -inf, -7.78044296e-13, -2.95667385e+02,
            -2.41321634e+02, -2.78829462e+01],
           [           -inf, -1.91846539e-13, -2.16329670e+02,
            -2.98083299e+02, -2.92813472e+01],
           [           -inf, -3.55271368e-15, -4.03462800e+02,
            -2.95667385e+02, -3.28479589e+01],
           [           -inf, -2.43112036e+02, -3.33812977e-10,
            -1.90891419e+02, -2.18204446e+01],
           [           -inf, -3.24865665e+02, -2.87805335e-10,
            -2.13803941e+02, -2.19687353e+01],
           [           -inf, -2.98844082e+02, -1.55647939e-10,
            -1.62567251e+02, -2.25834155e+01]])



This has set the prior probability of paternity of the mother (column
zero above) to negative infinity (i.e log(zero)). You can set any
selfing rate between zero and one if you have a good idea of what the
value should be and how much it varies. For example, *Arabidopsis
thaliana* selfs most of the time, so we could set a selfing rate of 95%.

.. code:: ipython3

    patlik.selfing_rate=0.95
    patlik.prob_array()




.. parsed-literal::

    array([[-3.98574225e+02, -7.78044296e-13, -2.95667385e+02,
            -2.41321634e+02, -2.78829462e+01],
           [-4.00653666e+02, -1.91846539e-13, -2.16329670e+02,
            -2.98083299e+02, -2.92813472e+01],
           [-4.25512099e+02, -3.55271368e-15, -4.03462800e+02,
            -2.95667385e+02, -3.28479589e+01],
           [-2.38333276e+02, -2.43112036e+02, -3.33812977e-10,
            -1.90891419e+02, -2.18204446e+01],
           [-1.81684939e+02, -3.24865665e+02, -2.87805335e-10,
            -2.13803941e+02, -2.19687353e+01],
           [-2.38333276e+02, -2.98844082e+02, -1.55647939e-10,
            -1.62567251e+02, -2.25834155e+01]])



However, notice that despite the strong prior favouring the mother, she
still doesn’t have the highest probablity of paternity for any
offspring. That’s because the signal from the genetic markers is so
strong that the true fathers still come out on top.

Removing individual candidates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also set likelihoods for particular individuals to zero
manually. You might want to do this if you wanted to test the effects of
incomplete sampling on your results, or if you had a good reason to
suspect that some candidates could not possibly be the sire (for
example, if the data are multigenerational, and the candidate was born
after the offspring). Let’s remove candidate 3:

.. code:: ipython3

    patlik.purge = 'my_population_3'
    patlik.prob_array()




.. parsed-literal::

    array([[-3.98574225e+02, -7.78044296e-13, -2.95667385e+02,
                       -inf, -2.78829462e+01],
           [-4.00653666e+02, -1.91846539e-13, -2.16329670e+02,
                       -inf, -2.92813472e+01],
           [-4.25512099e+02, -3.55271368e-15, -4.03462800e+02,
                       -inf, -3.28479589e+01],
           [-2.38333276e+02, -2.43112036e+02, -3.33812977e-10,
                       -inf, -2.18204446e+01],
           [-1.81684939e+02, -3.24865665e+02, -2.87805335e-10,
                       -inf, -2.19687353e+01],
           [-2.38333276e+02, -2.98844082e+02, -1.55647939e-10,
                       -inf, -2.25834155e+01]])



This also works using a list of candidates.

.. code:: ipython3

    patlik.purge = ['my_population_0', 'my_population_3']
    patlik.prob_array()




.. parsed-literal::

    array([[           -inf, -7.78044296e-13, -2.95667385e+02,
                       -inf, -2.78829462e+01],
           [           -inf, -1.91846539e-13, -2.16329670e+02,
                       -inf, -2.92813472e+01],
           [           -inf, -3.55271368e-15, -4.03462800e+02,
                       -inf, -3.28479589e+01],
           [           -inf, -2.43112036e+02, -3.33812977e-10,
                       -inf, -2.18204446e+01],
           [           -inf, -3.24865665e+02, -2.87805335e-10,
                       -inf, -2.19687353e+01],
           [           -inf, -2.98844082e+02, -1.55647939e-10,
                       -inf, -2.25834155e+01]])



This has removed the first individual (notice that this is identical to
the previous example, because in this case the first individual is the
mother). Alternatively you can supply a float between zero and one,
which will be interpreted as a proportion of the candidates to be
removed at random, which can be useful for simulations.

.. code:: ipython3

    patlik.purge = 0.4
    patlik.prob_array()




.. parsed-literal::

    array([[-3.70691279e+02,            -inf, -2.67784439e+02,
                       -inf,  0.00000000e+00],
           [-3.71372319e+02,            -inf, -1.87048323e+02,
                       -inf,  0.00000000e+00],
           [-3.92664140e+02,            -inf, -3.70614841e+02,
                       -inf,  0.00000000e+00],
           [-2.38333276e+02,            -inf, -3.33812977e-10,
                       -inf, -2.18204446e+01],
           [-1.81684939e+02,            -inf, -2.87805335e-10,
                       -inf, -2.19687353e+01],
           [-2.38333276e+02,            -inf, -1.55647939e-10,
                       -inf, -2.25834155e+01]])



Reducing the number of candidates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You might want to remove candidates who have an a priori very low
probability of paternity, for example to reduce the memory requirements
of the ``paternityArray``. One simple rule is to exclude any candidates
with more than some arbritray number of loci with opposing homozygous
genotypes relative to the offspring (you want to allow for a small
number, in case there are genotyping errors). This is done with
``max_clashes``.

.. code:: ipython3

    patlik.max_clashes=3

The option ``max_clashes`` refers back to a matrix that counts the
number of such incompatibilities for each offspring-candidate pair. When
you create a ``paternityArray`` from ``genotypeArray`` objects, this
matrix is created automatically ad can be called with:

.. code:: ipython3

    patlik.clashes




.. parsed-literal::

    array([[ 0,  0,  3,  2],
           [ 0,  0,  1,  3],
           [ 0,  0,  6,  3],
           [ 0,  8,  0,  2],
           [ 0, 10,  0,  3],
           [ 0,  9,  0,  2]])



If you import a ``paternityArray`` object, this isn’t automatically
generated, but you can recreate this manually with:

.. code:: ipython3

    fp.incompatibilities(mypop, progeny)




.. parsed-literal::

    array([[ 0,  0,  3,  2],
           [ 0,  0,  1,  3],
           [ 0,  0,  6,  3],
           [ 0,  8,  0,  2],
           [ 0, 10,  0,  3],
           [ 0,  9,  0,  2]])



Notice that this array has a row for each offspring, and a column for
each candidate father. The first column is for the mother, which is why
everything is zero.

Modifying arrays on creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also set the attributes we just described by setting them when
you create the ``paternityArray`` object. For example:

.. code:: ipython3

    patlik = fp.paternity_array(
        offspring = progeny,
        mothers = mothers,
        males= mypop,
        mu=error_rate,
        missing_parents=0.1,
        purge = 'my_population_3',
        selfing_rate = 0
    )

Importing a ``paternityArray``
------------------------------

Frequently you may wish to save an array and reload it. Otherwise, you
may be working with a more exotic system than FAPS currently supports,
such as microsatellite markers or a funky ploidy system. In this case
you can create your own matrix of paternity likelihoods and import this
directly as a ``paternityArray``. Firstly, we can save the array we made
before to disk by supplying a path to save to:

.. code:: ipython3

    patlik.write('../../data/mypatlik.csv')

We can reimport it again using ``read_paternity_array``. This function
is similar to the function for importing a ``genotypeArray``, and the
data need to have a specific structure:

1. Offspring names should be given in the first column
2. Names of the mothers are usually given in the second column.
3. If known for some reason, names of fathers can be given as well.
4. Likelihood information should be given *to the right* of columns
   indicating individual or parental names, with candidates’ names in
   the column headers.
5. The final column should specify a likelihood that the true sire of an
   individual has *not* been sampled. Usually this is given as the
   likelihood of drawing the paternal alleles from population allele
   frequencies.

.. code:: ipython3

    patlik = fp.read_paternity_array(
        path = '../../data/mypatlik.csv',
        mothers_col=1,
        likelihood_col=2)

Of course, you can of course generate your own ``paternityArray`` and
import it in the same way. This is especially useful if your study
system has some specific marker type or genetic system not supported by
FAPS.

One caveat with importing data is that the array of opposing homozygous
loci is not imported automatically. You can either import this as a
separate text file, or you can recreate this as above:

.. code:: ipython3

    fp.incompatibilities(mypop, progeny)




.. parsed-literal::

    array([[ 0,  0,  3,  2],
           [ 0,  0,  1,  3],
           [ 0,  0,  6,  3],
           [ 0,  8,  0,  2],
           [ 0, 10,  0,  3],
           [ 0,  9,  0,  2]])



However, this step is not essential.
