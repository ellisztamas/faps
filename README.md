# faps

FAPS stands for Fractional Analysis of Sibships and Paternity. It is a Python package for reconstructing genealogical relationships in wild populations in a way that accounts for uncertainty in the genealogy. It uses a clustering algorithm to sample plausible partitions of offspring into full sibling families, negating the need to apply an iterative search algorithm. Simulation tools are provided to assist with planning and verifying results.

## Table of contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Using FAPS](#using-faps)
4. [Citing FAPS](#citing-faps)
5. [Authors and license information](#authors-and-license-information)

## Overview
The basic workflow is:

1. Import data
2. Calculate pairwise likelihoods of paternity between offspring and candidate fathers.
3. Cluster offspring into sibships
4. Infer biological parameters integrating out uncertainty in paternity and sibship structre.

At present only biallelic diploid SNPs in half sibling arrays are supported. FAPS can handle multiple half-sibling arrays, but the idendity of one parent must be known. It is assumed that population allele frequencies are known and sampling of candidate fathers is complete, or nearly complete.

There are however a number of extensions which I would be happy to attempt, but that I cannot justify investing time in unless someone has a specific need for it. For example, support for microsatellites, polyploids, bi-parental inference, or sibship inference without access to parental information. If any of these directions would be of use to you, please let me know by email, or better by filing an issue on GitHub directly.

## Installation
The best way to install FAPS is to use Python's package manager, Pip. Instructions to do so can be found at that projects [documentation page](https://pip.pypa.io/en/stable/installing/). Windows users might also consider [pip-Win](https://sites.google.com/site/pydatalog/python/pip-for-windows)

### From PyPi
To download the stable release run `pip install faps` in the command line.
If Python is unable to locate the package, try `pip install fap --user`.

### From GitHub
Download the .zip or tarball file, and unpack it. In a command line move into the folder this creates and run `pip install .`

### The messy way
Download the package from the repository by hitting 'clone or download'. Copy the contents of the folder FAPS to the directory where you are running your analyses, then import the package in your Python script.

### Dependencies
FAPS is built using the Numpy library with additional tools from Scipy and Pandas. If you install with pip, dependencies should be installed automatically.

For simulations, it also makes use of [iPython widgets](https://github.com/jupyter-widgets/ipywidgets). iPython widgets can be a little more troublesome to get working, but are only needed for simulations, and can be switched off. See [here](https://github.com/jupyter-widgets/ipywidgets/blob/master/docs/source/user_install.md) for installation instructions.

## Using FAPS
A user's guide is provided in this repository. This provides a fairly step-by-step guide to importing data, clustering offspring into sibship groups, and using those clusters to investigate the underlying biological processes. This was written with users in mind who have little experience of working with Python. These documents are written in [iPython](https://ipython.org/), which I highly recommend as an interactive environment for running analyses in Python.

Topics covered:

1. [Intorduction](https://github.com/ellisztamas/faps/blob/master/docs/01%20Introduction.ipynb)
2. [Genotype data](https://github.com/ellisztamas/faps/blob/master/docs/02%20Genotype%20data.ipynb)
3. [Paternity arrays](https://github.com/ellisztamas/faps/blob/master/docs/03%20Paternity%20arrays.ipynb)
4. [Sibship clustering](https://github.com/ellisztamas/faps/blob/master/docs/04%20Sibship%20clustering.ipynb)
5. Inference about mating patterns - under construction!
6. [Simulating data](https://github.com/ellisztamas/faps/blob/master/docs/06%20Simulating%20data.ipynb)
7. [Dealing with multiple families](https://github.com/ellisztamas/faps/blob/master/docs/07%20Dealing%20with%20multiple%20half-sib%20families.ipynb)
8. [Case study: data preparation in *Antirrhinum majus*](https://github.com/ellisztamas/faps/blob/master/docs/08%20Data%20cleaning%20in%20A.%20majus.ipynb)

## Citing FAPS
If you use FAPS in any puiblished work please cite:

> Ellis, TJ, Field DL, Barton, NH (2018) Efficient inference of paternity and sibship inference given known maternity via hierarchical clustering. Molecular Ecology Resources 18:988–999. https://doi.org/10.1111/1755-0998.12782

Here is the relevant bibtex reference:

```
@Article{ellis2018efficient,
  Title                    = {Efficient inference of paternity and sibship inference given known maternity via hierarchical clustering},  
  Author                   = {Ellis, Thomas James and Field, David Luke and Barton, Nicholas H},  
  Journal                  = {Molecular ecology resources},  
  Year                     = {2018},  
  Volume                   = {18},  
  pages                    = {988--999},  
  Doi                      = {10.1111/1755-0998.12782},  
  Publisher                = {Wiley Online Library}  
}
```

## Issues

Please report any bugs or requests that you have using the GitHub issue tracker! But before you do that, please check the user's guide folder in the docs folder to see if your question is answered there.

This project was part of my PhD work. Since I have now moved on to a different institution, I am not actively working on this any more, but I will still try to respond to questions.

## Authors and license information

Tom Ellis (thomas[dot]ellis[at]ebc[dot]uu[dot]se)

FAPS is available under the MIT license. See LICENSE.txt for more information
