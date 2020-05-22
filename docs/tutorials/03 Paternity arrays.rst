Paternity arrays
================

Tom Ellis, March 2017

Paternity arrays are the what sibship clustering is built on in FAPS.
They contain information about the probability that each candidate male
is the father of each individual offspring. This information is stored
in a ``paternityArray`` object, along with other related information. A
``paternityArray`` can either be imported directly, or created from
genotype data.

This notebook will examine how to:

1. Create a ``paternityArray`` from marker data.
2. Examine what information it contains.
3. Read and write a ``paternityArray`` to disk, or import a custom
   ``paternityArray``.

Once you have made your ``paternityArray``, the [next
step]((https://github.com/ellisztamas/faps/blob/master/docs/04%20Sibship%20clustering.ipynb)
is to cluster the individuals in your array into full sibship groups.

Creating a ``paternityArray`` from genotype data
------------------------------------------------

To create a ``paternityArray`` from genotype data we need to specficy
``genotypeArray``\ s for the offspring, mothers and candidate males.
Currently only biallelic SNP data are supported.

We will illustrate this with a small simulated example again with four
adults and six offspring typed at 50 loci.

.. code:: ipython3

    import faps as fp
    import numpy as np
    
    np.random.seed(27) # this ensures you get exactly the same answers as I do.
    allele_freqs = np.random.uniform(0.3,0.5, 50)
    mypop        = fp.make_parents(4, allele_freqs, family_name='my_population')
    progeny      = fp.make_sibships(mypop, 0, [1,2], 3, 'myprogeny')

We need to supply a ``genotypeArray`` for the mothers. This needs to
have an entry for for every offspring, i.e. six replicates of the
mother.

.. code:: ipython3

    mum_index = progeny.parent_index('mother', mypop.names) # positions in the mothers in the array of adults
    mothers   = mypop.subset(mum_index) # genotypeArray of the mothers

To create the ``paternityArray`` we also need to supply information on
the genotyping error rate (mu), and population allele frequencies. In
toy example we know the error rate to be zero. However, in reality this
will almost never be true, and moreover, sibship clustering becomes
unstable when errors are zero. With that in mind, ``fp.paternity_array``
will throw an error when this is included.

For allele frequencies, we can either take the population allele
frequencies defined above, or estimate them from the data, which will
give slightly different answers. The function ``paternity_array``
creates an object of class ``paternityArray``.

.. code:: ipython3

    error_rate = 0.0015
    
    sample_af = mypop.allele_freqs()
    patlik = fp.paternity_array(
        offspring = progeny,
        mothers = mothers,
        males= mypop,
        mu=error_rate)

Note that if you try running the above cell but with mu=0, it will run,
but throw a warning. The reason is that setting mu to zero tends to make
sibship clustering unstable, so it automatically sets mu to a very small
number.

``paternityArray`` structure
----------------------------

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


The most important part of the ``paternityArray`` is the likelihood
array, which represent the log likelihood that each candidate male is
the true father of each offspring individual. In this case it will be a
6x4 dimensional array with a row for each offspring and a column for
each candidate.

.. code:: ipython3

    patlik.lik_array




.. parsed-literal::

    array([[-211.5074549 , -140.28689445, -195.91484976, -185.21796594],
           [-212.88775824, -139.60346998, -181.34681961, -197.62999358],
           [-217.31266984, -140.97480372, -217.11239559, -196.60499636],
           [-181.50722696, -186.32497429, -139.59977031, -177.72798551],
           [-169.89679708, -203.98172946, -140.97558885, -180.22556077],
           [-180.12692361, -197.01961064, -138.21946697, -169.85455796]])



You can see that the log likelihoods of paternity for the first
individual are much lower than the other candidates. This individual is
the mother, so this makes sense. You can also see that the highest log
likelihoods are in the columns for the real fathers (the 2nd column in
rows one to three, and the third column in rows four to six).

The ``paternityArray`` also includes information that the true sire is
not in the sample of candidate males. In this case this is not helpful,
because we know sampling is complete, but in real examples is seldom the
case. By default this is defined as the liklihood of generating the
offspring genotypes given the known mothers genotype and alleles drawn
from population allele frequencies. Here, values for the six offspring
are higher than the likelihoods for the non-sires, indicating that they
are no more likely to be the true sire than a random unrelated
individual.

.. code:: ipython3

    patlik.lik_absent




.. parsed-literal::

    array([-174.33954085, -178.45865643, -170.2110881 , -183.41884222,
           -177.57516224, -182.36957151])



This becomes more informative when we combine likelihoods about sampled
and unsampled fathers. For fractional analyses we really want to know
the probability that the father was unsampled vs sampled, and how
probable it is that a single candidate is the true sire. To do this,
``patlik.lik_array`` is concatenate with values in
``patlik.lik_absent``, then normalises the array so that values in each
row sum to one. Printing the shape of the array demonstrates that we
have gained a column.

.. code:: ipython3

    patlik.prob_array.shape




.. parsed-literal::

    (6, 5)



If we sum the rows we see that they do indeed add up to one now.
Probabilities are stored as log probabilities, so we have to
exponentiate first.

.. code:: ipython3

    np.exp(patlik.prob_array).sum(axis=1)




.. parsed-literal::

    array([1., 1., 1., 1., 1., 1.])



Modifying a ``paternityArray``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can alter the information in ``patlik.prob_array`` to reflect
different prior beliefs about the dataset. (In contrast, it's seldom a
good idea to manipulate the likelihoods from genetic data contained in
``patlik.lik_array``).

For example, often the mother is included in the sample of candidate
males, either because you are using the same array for multiple
families, or self-fertilisation is a biological possibility. In a lot of
cases though the mother cannot simultaneously be the sperm/pollen donor,
and it is necessary to set the rate of self-fertilisation to zero (the
natural logarithm of zero is negative infinity). Here, only the first
three columns are shown.

.. code:: ipython3

    patlik.adjust_prob_array(selfing_rate=0)[:, :3]




.. parsed-literal::

    array([[           -inf,  0.00000000e+00, -5.56279553e+01],
           [           -inf,  0.00000000e+00, -4.17433496e+01],
           [           -inf, -1.98951966e-13, -7.61375919e+01],
           [           -inf, -4.67252040e+01,  0.00000000e+00],
           [           -inf, -6.30061406e+01,  0.00000000e+00],
           [           -inf, -5.88001437e+01, -2.84217094e-14]])



The likelihoods for the mother have changed to 0 (negative infinity on
the log scale). You can set any selfing rate between zero and one if you
have a good idea of what the value should be and how much it varies.
Otherwise it may be better to estimate the selfing rate from the data,
or else estimate it some other way.

``adjust_prob_array`` always refers back to the original
``patlik.lik_array`` and ``patlik.lik_absent``, which remain unchanged.
Calling ``adjust_prob_array`` will not alter the data stored for
``patlik.prob_array`` unless you assign it yourself.

.. code:: ipython3

    patlik.prob_array = patlik.adjust_prob_array(selfing_rate=0)

You can also set likelihoods for particular individuals to zero
manually. You might want to do this if you wanted to test the effects of
incomplete sampling on your results, or if you had a good reason to
suspect that some candidates could not possibly be the sire (for
example, if the data are multigenerational, and the candidate was born
after the offspring).

.. code:: ipython3

    patlik.adjust_prob_array(purge = 'my_population_0')




.. parsed-literal::

    array([[           -inf,  0.00000000e+00, -5.56279553e+01,
            -4.49310715e+01, -3.40526464e+01],
           [           -inf,  0.00000000e+00, -4.17433496e+01,
            -5.80265236e+01, -3.88551864e+01],
           [           -inf, -1.98951966e-13, -7.61375919e+01,
            -5.56301926e+01, -2.92362844e+01],
           [           -inf, -4.67252040e+01,  0.00000000e+00,
            -3.81282152e+01, -4.38190719e+01],
           [           -inf, -6.30061406e+01,  0.00000000e+00,
            -3.92499719e+01, -3.65995734e+01],
           [           -inf, -5.88001437e+01, -2.84217094e-14,
            -3.16350910e+01, -4.41501045e+01]])



This has removed the first individual (notice that this is identical to
the previous example, because in this case the first individual is the
mother). Alternatively you can supply a float between zero and one,
which will be interpreted as a proportion of the candidates to be
removed at random, which can be useful for simulations.

.. code:: ipython3

    patprob2 = patlik.adjust_prob_array(purge=0.4)
    np.isinf(patprob2).mean(1) # proportion missing along each row.




.. parsed-literal::

    array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])



You can specify the proportion :math:`\theta` of the population of
candidate males which are missing with the option ``missing_parents``.
The likelihoods for non-sampled parents will be weighted by
:math:`\theta`, and likelihoods for sampled candidates by
:math:`1-\theta`.

Of course, rows still need to sum to one. Luckily ``adjust_prob_array``
does that automatically.

.. code:: ipython3

    np.exp(patprob2).sum(1)




.. parsed-literal::

    array([1., 1., 1., 1., 1., 1.])



.. code:: ipython3

    patlik.adjust_prob_array(missing_parents=0.1)




.. parsed-literal::

    array([[-7.12205605e+01,  0.00000000e+00, -5.56279553e+01,
            -4.49310715e+01, -3.62498710e+01],
           [-7.32842883e+01,  0.00000000e+00, -4.17433496e+01,
            -5.80265236e+01, -4.10524110e+01],
           [-7.63378661e+01, -2.84217094e-14, -7.61375919e+01,
            -5.56301926e+01, -3.14335090e+01],
           [-4.19074566e+01, -4.67252040e+01,  0.00000000e+00,
            -3.81282152e+01, -4.60162965e+01],
           [-2.89212082e+01, -6.30061406e+01, -2.84217094e-13,
            -3.92499719e+01, -3.87967980e+01],
           [-4.19074566e+01, -5.88001437e+01, -2.84217094e-14,
            -3.16350910e+01, -4.63473291e+01]])



You might want to remove candidates who have an a priori very low
probability of paternity, for example to reduce the memory requirements
of the ``paternityArray``. One simple rule is to exclude any candidates
with more than some arbritray number of loci with opposing homozygous
genotypes relative to the offspring (you want to allow for a small
number, in case there are genotyping errors). This is done with
``max_clashes``.

.. code:: ipython3

    patlik.adjust_prob_array(max_clashes=3)




.. parsed-literal::

    array([[-7.12205605e+01,  0.00000000e+00, -5.56279553e+01,
            -4.49310715e+01, -3.40526464e+01],
           [-7.32842883e+01,  0.00000000e+00, -4.17433496e+01,
            -5.80265236e+01, -3.88551864e+01],
           [-7.63378661e+01, -1.98951966e-13,            -inf,
            -5.56301926e+01, -2.92362844e+01],
           [-4.19074566e+01,            -inf,  0.00000000e+00,
            -3.81282152e+01, -4.38190719e+01],
           [-2.89212082e+01,            -inf, -2.84217094e-13,
            -3.92499719e+01, -3.65995734e+01],
           [-4.19074566e+01,            -inf, -2.84217094e-14,
            -3.16350910e+01, -4.41501045e+01]])



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



You can recreate this manually with:

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

Importing a ``paternityArray``
------------------------------

Frequently you may wish to save an array and reload it. Otherwise, you
may be working with a more exotic system than FAPS currently supports,
such as microsatellite markers or a funky ploidy system. In this case
you can create your own matrix of paternity likelihoods and import this
directly as a ``paternityArray``. Firstly, we can save the array we made
before to disk by supplying a path to save to:

.. code:: ipython3

    patlik.write('../data/mypatlik.csv')

We can reimport it again using ``read_paternity_array``. This function
is similar to the function for importing a ``genotypeArray``, and the
data need to have a specific structure:

1. Offspring names should be given in the first column
2. Names of the mothers are usually given in the second column.
3. If known for some reason, names of fathers can be given as well.
4. Likelihood information should be given *to the right* of columns
   indicating individual or parental names, with candidates' names in
   the column headers.
5. The final column should specify a likelihood that the true sire of an
   individual has *not* been sampled. Usually this is given as the
   likelihood of drawing the paternal alleles from population allele
   frequencies.

.. code:: ipython3

    patlik = fp.read_paternity_array(
        path = '../data/mypatlik.csv',
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


