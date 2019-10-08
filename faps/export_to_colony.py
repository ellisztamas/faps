import numpy as np
import os

def export_to_colony(offspring, mothers, males, allele_freqs,
                     file_path, project_name, output_name, seed, dropout_rates, error_rates,
                     known_dads, known_mums, update_af = 0, two_sexes = 2, inbreeding = 0,
                     haplodiploid= 0, polygamy = [0,0], infer_clones = 0, size_scaling = 1,
                     sibship_prior   = [0,1.0,1.0], known_allele_fr = 1, nruns = 1,
                     run_length = 2, monitor_method  = 0, monitor_interval= 100000,
                     windows_gui = 0, analysis_method = 1, precision = 3, parents_present = [1,1],
                     n_incompatibilies = 3):
    """
    Export genotypeArray objects to run in Colony2.

    Variable descriptions are mostly copied from the manual for Colony2. See the Colony2
    manual for full descriptions of what they mean.

    The output is rather crude.

    Parameters
    ----------
    offspring, mothers, males: genotypeArray objects
        Genotype information on the the offspring, mothers, and
        candidate males.
    allele_freqs: array
        A vector of population allele frequencies.
    file_path: str
        Folder to store analyses
    project_name: str
        Dataset name.
    output_name: str
        Output file name
    seed: int
        Random seed for random number generator.
    dropout_rates: array-like
        List of dropout rates for each locus.
    error_rates: array-like
        List of error rates for each locus.
    known_dads, known_mums: lists
        Lists of names of the father and mother of each offspring individual.
        There should be a single entry for each offspring. If the parent
        is unknown, enter 'NA'.
    update_af: int
        0/1=Not updating/updating allele frequency
    two_sexes: int
        2/1=Dioecious/Monoecious species
    inbreeding: int
        0/1=No inbreeding/inbreeding
    haplodiploid: int
        0/1=Diploid species/HaploDiploid species
    polygamy: int
        0/1=Polygamy/Monogamy for males and females
    infer_clones: int
        0/1=Clone inference =No/Yes
    size_scaling: int
        0/1=Full sibship size scaling =No/Yes
    sibship_prior: int
        0,1,2,3=No,weak,medium,strong sibship size prior; mean paternal &
        maternal sibship size.
    known_allele_fr: int
        0/1=Unknown/Known population allele frequency
    nruns: int
        Number of runs
    run_length:
        1,2,3,4 =  short, medium, long, very long runs
    monitor_method: int
        0/1=Monitor method by Iterate#/Time in second
    monitor_interval: int
        Monitor interval in Iterate# / in seconds
    windows_gui: int
        0/1 running in Windows GUI version?
    analysis_method: int
        0/1/2 pairwise-likelihood score (PLS), full likelihood (FL), or
        the FL and PLS combined (FPLS) method.
    precision: int
        1/2/3=low/medium/high Precision for Fulllikelihood
    parents_present:int
        prob. of dad/mum included in the candidates

    Returns
    -------
    A .dat file suitable for using in Colony 2.
    """
    # Open a file
    fo = open(os.path.join(file_path, project_name) + '.dat', "wb")
    # write population parameters
    fo.write(project_name+' !Dataset name \n' +
             output_name +' !Output file name \n' +
             str(offspring.size)+' ! Number of offspring in the sample \n' +
             str(offspring.nloci)+' ! Number of loci \n'+
             str(seed)+' ! Seed for random number generator\n' +
             str(update_af)+' ! 0/1=Not updating/updating allele frequency \n' +
             str(two_sexes)+' ! 2/1=Dioecious/Monoecious species \n' +
             str(inbreeding)+' ! 0/1=No inbreeding/inbreeding \n'+
             str(haplodiploid)+' ! 0/1=Diploid species/HaploDiploid species \n' +
             str(polygamy[0])+'  '+str(polygamy[1]) + ' ! 0/1=Polygamy/Monogamy for males & females \n' +
             str(infer_clones)+' ! 0/1=Clone inference =No/Yes \n' +
             str(size_scaling)+' ! 0/1=Full sibship size scaling =No/Yes \n' +
             str(sibship_prior[0])+ ' ' + str(sibship_prior[1])+ ' ' + str(sibship_prior[2])+ ' ' + ' ! 0,1,2,3=No,weak,medium,strong sibship size prior; mean paternal & maternal sibship size \n' +
             str(known_allele_fr) + ' ! 0/1=Unknown/Known population allele frequency \n')

    # Number of alleles per locus
    a = ''
    for i in range(offspring.nloci):
        a += '2 '
    fo.write(a + ' !Number of alleles per locus \n')
    # Allele frequencies
    for l in range(offspring.nloci):
        fo.write(str(9) + '\t' + str(10) + '\n')
        fo.write(('%1.4f'%allele_freqs[l]) + '\t' + ('%1.4f'%(1-allele_freqs[l])) + '\n')
    fo.write('\n')

    # Analysis parameters
    fo.write(str(nruns)+' ! Number of runs \n' +
             str(run_length)+' ! Length of run \n' +
             str(monitor_method) + ' ! 0/1=Monitor method by Iterate#/Time in second \n' +
             str(monitor_interval) + ' ! Monitor interval in Iterate# / in seconds \n' +
             str(windows_gui) + ' ! non-Windows version \n' +
             str(analysis_method) + ' ! Give an indicated value of 0, 1 or 2 to tell Colony to use the pairwise- likelihood score (PLS), full likelihood (FL), or the FL and PLS combined (FPLS) method \n' +
             str(precision) + ' ! 1/2/3=low/medium/high Precision for Fulllikelihood \n' +
             '\n')

    # locus information and error rates
    t, c, d, m = '', '', '',''
    for l in range(offspring.nloci):
        t += 'mk'+str(l+1)+'\t'
        c += str(0) + '\t'
        d += ('%1.4f'%error_rates[l]) + '\t'
        m += ('%1.4f'%error_rates[l]) + '\t'
    fo.write(t + '\n' + # marker name
             c + '\n' + # dominant/codominant marker
             d + '\n' + # heterozygous allelic dropout rate
             m + '\n' + # other error rates
             '\n')

    # Offspring genotypes
    for o in range(offspring.size):
        thisrow = offspring.geno[o].flatten()+9
        t= offspring.names[o]
        for i in range(offspring.nloci*2):
            t = t + '\t' + str(thisrow[i])
        t = t + '\n'
        fo.write(t)

    # prob of including parents and numbers of mothers and fathers.
    if two_sexes is 1:
        fo.write('\n' +
             str(parents_present[0]) + '  ' + str(parents_present[1]) + ' ! prob. of dad/mum included in the candidates \n' +
             str(males.size)+ '  ' + str(males.size) + ' ! numbers of candidate males & females \n'+
             '\n')
    elif two_sexes is 2:
        fo.write('\n' +
             str(parents_present[0]) + '  ' + str(parents_present[1]) + ' ! prob. of dad/mum included in the candidates \n' +
             str(males.size)+ '  ' + str(mothers.size) + ' ! numbers of candidate males & females \n'+
             '\n')

    # Candidate male genotypes
    for m in range(males.size):
        thisrow = males.geno[m].flatten()+9
        t= males.names[m]
        for i in range(males.nloci*2):
            t = t + '\t' + str(thisrow[i])
        t = t + '\n'
        fo.write(t)
    fo.write('\n')

    # Maternal genotypes
    if two_sexes is 1: # Hermaphrodites
        for m in range(males.size):
            thisrow = males.geno[m].flatten()+9
            t= males.names[m]
            for i in range(males.nloci*2):
                t = t + '\t' + str(thisrow[i])
            t = t + '\n'
            fo.write(t)
    elif two_sexes is 2: # separate sexes
        for m in range(mothers.size):
            thisrow = mothers.geno[m].flatten()+9
            t= mothers.names[m]
            for i in range(mothers.nloci*2):
                t = t + '\t' + str(thisrow[i])
            t = t + '\n'
            fo.write(t)
    else:
        raise ValueError("two_sexes should be 1 for hermaphrodites or 2 for separate sexes.")


    fo.write('\n')

    # Number of offspring with a known father in the sample of fathers.
    if all(offspring.fathers == 'NA'):
        fo.write(str(0) + ' ' + str(n_incompatibilies) + ' ! Number of known father-offspring dyads \n\n')
    else:
        check_dads = offspring.fathers[offspring.fathers != 'NA']
        fo.write(str(sum([check_dads[i] in known_dads for i in range(offspring.size)])) + ' ' + str(n_incompatibilies) + ' ! Number of known father-offspring dyads \n')
        # Pairs of dyads of offspring and known fathers
        for o in range(offspring.size):
            if offspring.fathers[o] in known_dads:
                fo.write(offspring.names[o] + ' ' + known_dads[o] + '\n')
        fo.write('\n')

    # Number of offspring with a known mother in the sample of mothers.
    if all(offspring.mothers == 'NA'):
        fo.write(str(0) + ' ' + str(n_incompatibilies) + ' ! Number of known mother-offspring dyads \n\n')
    else:
        check_mums = offspring.mothers[offspring.mothers != 'NA']
        fo.write(str(sum([check_mums[i] in known_mums for i in range(offspring.size)])) + ' ' + str(n_incompatibilies) + ' ! Number of known mother-offspring dyads \n')
        # Pairs of dyads of offspring and known mothers
        for o in range(offspring.size):
            if offspring.mothers[o] in known_mums:
                fo.write(offspring.names[o] + ' ' + known_mums[o] + '\n')
        fo.write('\n')

    # Number of known paternal full sibships
    paternal_families = np.unique(known_dads)
    paternal_families = paternal_families[paternal_families != 'NA']
    fo.write(str(paternal_families.shape[0]) + ' ! Number of know  paternal full sibships \n')
    # individuals in each paternal full sibship
    for m in range(paternal_families.shape[0]):
        this_family =  offspring.names[known_dads == paternal_families[m]]
        f = ''
        for o in range(len(this_family)):
            f += this_family[o] + ' '
        fo.write(str(this_family.shape[0]) + ' ' + f + '\n')

    # number of known maternal full sibships
    fo.write('0' + ' ! Number of known maternal full sibships \n')

    # set number of excluded fathers and mothers to zero.
    fo.write('0' + ' ! Number of offspring with known excluded paternity \n')
    fo.write('0' + ' ! Number of offspring with known excluded maternity \n')

    # set number of excluded sibships to zero.
    fo.write('0' + ' ! Number of offspring with known excluded paternal sibships \n')
    fo.write('0' + ' ! Number of offspring with known excluded maternal sibships  \n')

    # Close opened file
    fo.close()
