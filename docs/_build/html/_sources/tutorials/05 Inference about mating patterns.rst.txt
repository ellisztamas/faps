Inference about mating patterns
===============================

Tom Ellis, August 2017

This section is under construction!

.. code:: ipython3

    import numpy as np
    from faps import *
    
    allele_freqs = np.random.uniform(0.1, 0.5, 50)
    males = make_parents(100, allele_freqs)
    phenotypes = np.random.choice()
    


.. code:: ipython3

    
    offspring = make_sibships(males, 0, range(1,5), 5)
    
    mu = 0.0013
    males = males.dropouts(0.015).mutations(mu)
    offspring= offspring.dropouts(0.025).mutations(mu)
    
    mothers = males.subset(offspring.parent_index('m', males.names))
    
    #mothers = mothers.split(offspring.fathers)
    #offspring = offspring.split(offspring.fathers)
    
    patlik = paternity_array(offspring, mothers, males, allele_freqs, mu)
