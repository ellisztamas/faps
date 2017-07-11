# faps

FAPS stands for Fractional Analysis of Sibships and Paternity. It is a Python package for reconstructing genealogical relationships in wild populations in a way that accounts for uncertainty in the genealogy. It uses a clustering algorithm to sample plausible partitions of offspring into full sibling families, negating the need to apply an iterative search algorithm.

## Background and motivation

Inference of wild pedigrees has most often involved categorical assignment of relationships, usually parent offspring relationships. In categorical asssignment, relationships are inferred to be of a single type, for example the father of offspring A is assigned as adult B. This is intuitive, because we know that each offspring must have a single father.

Unfortunately, categorical assignment leads to flawed genealogical inference. Inference is just that: we do not know the true pedigree and must infer it, which means we can never be completely certain we have the correct answer. Unfortunately, incorrect pedigrees can sometimes be more consistent with genetic data than the true pedigree just by chance. If we use only the most likely answer, but that answer turns out to be incorrect, we will be 100% incorrect when it comes to interpreting the results. It is better to consider a range of possible answers and consider each proportional to their relative probability.

Devlin et al. (1988) and Roeder et al. (1989) first articulated these ideas for as 'fractional paternity'. The idea is to apportion paternity of an offspring individual among candidate fathers in proportion to how compatible they were with the offspring's genotype. This allowed for less biased inference about fertilities in natural populations (e.g. Meagher 1994, Smouse 1999). Unfortunately, fractional methods have been largely eclipsed by categorical methods.

FAPS was designed as a tool for performing joint assignment of sibship and paternity relationships in a fractional framework. The primary motivation for developing of FAPS was to provide a package to investigate biological processes in a large population of snapdragons in the wild. As such, most of the examples in this guide relate to snapdragons. That said, FAPS addresses general issues in pedigree reconstruction of wild populations, and it is hoped that FAPS will be useful for other plant and animal systems.

The specific aims of FAPS were to provide a package which would allow us to:

    1. Jointly inferr sibship and paternity relationships, and hence gain better estimates of both (Wang 2007), without using a random-walk algorithm such as MCMC or simulated annealing.
    2. Account for uncertainty about relationships (i.e. to integrate fractional paternity assignment with sibship inference).
    3. Provide tools for testing biological hypotheses with minimal hassle.
    4. Do this efficiently for large populations.
    5. Not depend on mating system or genetic marker technology.

## Overview of the method

FAPS reconstructs relationships for one or more half-sibling arrays, that is to say a sample of offspring from one or more mothers whose identidy is uncertain. A half-sibling array consistent of seedlings from the same maternal plant, or a family of lambs from the same ewe. The paternity of each offspring is unkown, and hence it is unknown whether pairs of offspring are full or half siblings.

There is also a sample of males, each of whom is a candidate to be the true sire of each offspring individual. FAPS is used to identify likely sibling relationships between offspring and their shared fathers based on typed genetic markers, and to use this information to make meaningful conclusions about population or matig biology. The procedure can be summarised as follows:

    1. FAPS begins with an matrix with a row for each offspring and a column for each candidate male. Each element represents the likelihood that the corresponding male is the sire of the offspring individual.
    2. Based on this matrix, we apply hierarchical clustering to group individuals into possible configurations of full-siblings.
    3. Calculate statistics of interest on each configuration, weighted by the probability of each candidate male and each configuration

Like all statistical analyses, FAPS makes a number of assumptions about your data. It is good to state these explicitly so everyone is aware of the limitations of the data and method:

    1. Individuals come from two generations: a sample of offspring, and a sample of reproductive adults.
    2. At present FAPS is intended to deal with paternity assignment only, i.e. when one parent is known and the second must be inferred. That said, it would not take too much work to modify FAPS to accommodate bi-parental assignment.
    3. Sampling of the adults does not need to be complete, but the more true parents are missing the more likely it is that true full-siblings are inferred to be half-siblings.
    4. Genotype data are biallelic, and genotyping errors can be characterised as a haploid genotype being either correct or incorrect with some probability. Multi-allelic markers with complex error patterns such as microsatellites are not supported at present - see the sections on genotype data and paternity arrays for more on this.

Other software

Depending on your biological questions, your data may not fit the assumptions listed above, and an alternative approach might be more appropriate. For example:

    * Cervus (Marshall 1998, Kalinowski 200?) was the first widely-used program for parentage/paternity assignment. It is restricted to categorical assignment of parent-offspring pairs and relies on substantial prior information about the demography of the population.
    * Colony2 is a dedicated Fortran program for sibship inference (Wang 2004, Wang and Santure 2009). Analyses can be run with or without a sample of parents. Originally designed for microsatellites, it can be slow for large SNP datasets, although Wang et al. 2012 updated the software to support SNP data with a slight cost to accuracy.
    * As argued eloquently by Hadfield (2006), frequently the aim of genealogical reconstruction is very often not the pedigree itself, but some biological parameter of the population. His MasterBayes package for R allows the user to dircetly estimate the posterior distribution for the statistic of interest. It is usually slow for SNP data, can does not incorporate information about siblings.
    * SNPPIT (Anderson & Garza, 2006) provides very fast and accurate assignments of parentage using SNPs.
    * Almudevar (2003), Huisman (2017) and Anderson and Ng (2016) provide software for estimating multigenerational pedigrees.

## Installing FAPS

You can download the development version of FAPS from https://github.com/ellisztamas/faps. The simplest way to start using it is to unzip the package contents to your working directory, and import it from there. For example, if you're working directory is /home/Documents/myproject where you will save your analyses, you need a folder in that directory called faps containing the functions and classes contained in FAPS.

Once in Python/IPython you'll need to import the package, as well as the NumPy library on which it is based. In the rest of this document, I'll assume you've run the following lines to do this if this isn't explicitly stated.
In [1]:

from faps import *
import numpy as np