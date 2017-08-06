# faps

FAPS stands for Fractional Analysis of Sibships and Paternity. It is a Python package for reconstructing genealogical relationships in wild populations in a way that accounts for uncertainty in the genealogy. It uses a clustering algorithm to sample plausible partitions of offspring into full sibling families, negating the need to apply an iterative search algorithm. Simulation tools are provided to assist with planning and verifying results.

## Overview
The basic workflow is:

1. Import data
2. Calculate pairwise likelihoods of paternity between offspring and candidate fathers.
3. Cluster offspring into sibships
4. Infer biological parameters integrating out uncertainty in paternity and sibship structre.

At present only biallelic diploid SNPs are supported. However, the method depends only only being able to calculate likelihoods in stage two, and provided you can do this there is nothing stopping you applying this to other marker types (e.g. microsatellites) or genetic systems (e.g. polyploids). If you are interested in doing this, please contact me (see below).

## Installation
### Dependencies
FAPS is built using the `Numpy` library, with additional tools from `Scipy` and Pandas. These can be accessed together through the bundles of packages available from [Anaconda](https://www.continuum.io/downloads). 

For simulations, it also makes use of `Pandas` dataframes, and [iPython widgets](https://github.com/jupyter-widgets/ipywidgets). iPython widgets can be a little more troublesome to get working, but are only needed for simulations, and can be switched off. See [here](https://github.com/jupyter-widgets/ipywidgets/blob/master/docs/source/user_install.md) for installation instructions.

### The messy way
Download the package from the repository by hitting 'clone or download'. Copy the contents of the folder FAPS to the directory where you are running your analyses, then import the package in your Python script.

### The clean way
The best way to install FAPS is to use Python's package manager, Pip. Instructions to do so can be found at that projects [documentation page](https://pip.pypa.io/en/stable/installing/). You can then either download the package form this repository, and run `pip install .` from the package directory. At some point I will also endeavour to get the package on the PyPi database.

## Using FAPS
A user's guide is provided in this repository. This provides a fairly step-by-step guide to importing data, clustering offspring into sibship groups, and using those clusters to investigate the underlying biological processes. This was written with users in mind who have little experience of working with Python. These documents are written in [iPython](https://ipython.org/), which I highly recommend as an interactive environment for running analyses in Python.

## Issues

Please report any bugs or requests that you have using the GitHub issue tracker! But before you do that, please check the user's guide folder in the docs folder to see if your question is answered there.

This project was part of my PhD work. Since I have now moved on to a different institution, I am not actively working on this any more, but I will still try to respond to questions.

## Authors and license information

Tom Ellis (thomas[dot]ellis[at]ebc[dot]uu[dot]se)

FAPS is available under the MIT license. See LICENSE.txt for more information


## To do

- Correct path names in the analysis files
- Upload code and data files to support FAPS publication to IST repository
- Check file paths in these files.
- Revise user guides
- Upload to Pypi
- Include installation notes for PyPi


