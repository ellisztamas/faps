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

### With Pip
The easiest way to install FAPS is to use Python's package manager, Pip. Instructions to install Pip so can be found at that projects [documentation page](https://pip.pypa.io/en/stable/installing/). Windows users might also consider [pip-Win](https://sites.google.com/site/pydatalog/python/pip-for-windows)

To download the stable release run `pip install faps` in the command line.
If Python is unable to locate the package, try `pip install faps --user`.

Be aware that when you have multiple versions of Python on your computer, including implementations like Jupyter, Pip might not be installing things to folders where Python is looking. It's useful to check Python is using the correct version of FAPS by opening Python and running
```
import faps as fp
fp.__version__
```
If this isn't showing the version you expected, install FAPS with a conda environment instead.

### With conda
Conda environments let you be explicit about what versions of each package you
would like to use for a given project, and can be cleaner to use when you have
multiple versions of Python on your machine.

If you haven't already, [install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (the lightweight Miniconda is fine). Next install the FAPS Conda environment:

1. copy the contents of [this Conda environment file]https://raw.githubusercontent.com/ellisztamas/faps/master/faps.yml) to a text file named `faps.yml`.
2. In the folder you saved that file to, run:
```
conda env create -f faps.yml
```
3. Acivate the environment. You'll need to do this every time you use FAPS.
```
conda activate faps
```
4. You can now open Python or Jupyter and begin using FAPS. It's best to double check the package version is what you think it should be:
```
import faps as fp
fp.__version__
```
5. Deactivate the environment again when you want to use something else.
```
conda deactivate
```

**Important:** you need to activate the environment every time you use FAPS in Python.

See the [Conda environments documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#) for more details.

#### Remove an environment
conda remove --name myenv --all


### Dependencies
FAPS is built using the Numpy library with additional tools from Scipy and Pandas. If you install with pip, dependencies should be installed automatically.

For simulations, it also makes use of [iPython widgets](https://github.com/jupyter-widgets/ipywidgets). iPython widgets can be a little more troublesome to get working, but are only needed for simulations, and can be switched off. See [here](https://github.com/jupyter-widgets/ipywidgets/blob/master/docs/source/user_install.md) for installation instructions.

## Using FAPS
A user's guide is available [here](https://fractional-analysis-of-paternity-and-sibships.readthedocs.io/en/latest/). This provides a fairly step-by-step guide to importing data, clustering offspring into sibship groups, and using those clusters to investigate the underlying biological processes. This was written with users in mind who have little experience of working with Python.

## Citing FAPS

If you use FAPS in any published work please cite:

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

## Authors and license information

Tom Ellis (thomas.ellis@gmi.oeaw.ac.at)

FAPS is available under the MIT license. See LICENSE.txt for more information
