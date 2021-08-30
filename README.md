# Appendix for Dissertation
## *Stochastic Uncertainty Analysis for Data-Consistent Approaches to Inverse Problems*

Author: Tian Yu Yen

This appendix contains all of the code required to reproduce the example figures contained in my dissertation. 
These are organized in various python Jupyter notebooks.

* **`Square Plate Bayesian Figures.ipynb` and `Square Plate DCI Figures`** contain the figures from Chapter 1 and Chapter 2 which describe the fixed-plate and wobbly-plate examples and compare the Bayesian and data-consistent approaches to solving these two different example problems.

* **`Stocahstic Wobbly Plate.ipynb`** contains an extension of the wobbly-plate problem to include different types of additional stochastic uncertainty (i.e., an extension to stochastic maps) described in Chapter 3.

* **`PDE`** contains the elliptic pde example with uncertain diffusion represented by a KL expansion and demonstrates how the data-consistent approach may be used to handle truncations in the representation of the permeability field. This is also described in Chapter 3.

* **`Review of Density Estimation.ipynb`** contains a review of different density estimation techniques, specifically: kernel density estimation, bayesian mixture models, and Dirichlet process mixture models. These methods are used to estimate a multi-modal distribution (a "tripeak" density) for the examples in Chapter 4.

* **`DCI and Density Estimation.ipynb`** contains the analysis of how these density estimation techniques influence the data-consistent approach to solving the inverse problem. The notebook shows how uncertainty may be measured by constructing confidence intervals (and demonstrates how to construct them). It also contains a demonstration of inverting component distributions from the Dirichlet process mixture models to obtain a decomposition of the data-consistent solution into clusters, as described at the end of Chapter 4.

# Code environment and packages

The following packages are required for running the notebooks. Other than the PDE example (which requires FEniCS), these packages are included with a standard build of [Anaconda](https://www.anaconda.com/).

* python version: 3.8.5
  * numpy: 1.19.2
  * scipy: 1.5.2
  * scikit-learn:  0.23.2
  * pandas: 1.1.3
  * matplotlib: 3.3.2
  * seaborn: 0.11.0

* PDE example with FEniCS
  * [FEniCS](https://fenicsproject.org/): 2019.1.0 (use of docker is recommended but not required)
