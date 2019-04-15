# Optimal reduced space for Variational Data Assimilation
Rossella Arcucci, Laetitia Mottet, Christopher Pain, Yi-Ke Guo

## Todo List :
- Analyse Abstract : Done
- Read the Paper :
- Take Notes of relevant techniques :

## Innovations
- Truncated Singular Value Decomposition (TSVD) is used to increase the scalability of Variational DA.
- Algorithm to compute the optimal truncation parameter
- Proof that it reduces ill-conditioning and removes less significant modes

## Questions
- Difference between DA and VarDA ?
- What is exactly Operational Forecasting ?

## Abstract Analysis

### Abstract

**Data Assimilation** (DA) is an uncertainty quantification technique used to *incorporate observed data into a prediction model* in order to improve numerical forecasted results. **Variational DA** (VarDA) is based on the minimisation of a function which estimates the discrepancy between numerical results and observations. **Operational forecasting** requires real-time data assimilation. This mandates the choice of opportune methods to improve the efficiency of VarDA codes without loosing accuracy. Due to the scale of the forecasting area and the number of state variables used to describe the physical model, DA is a big data problem. In this paper, the Truncated Singular Value Decomposition (TSVD) is used to reduce the space dimension, alleviate the computational cost and reduce the errors. Nevertheless, a consequence is that important information is lost if the truncation parameter is not properly chosen. We provide an algorithm to compute the optimal truncation parameter and we prove that the optimal estimation reduces the ill- conditioning and removes the statistically less significant modes which could add noise to the estimate obtained from DA. In this paper, numerical issues faced in developing VarDA algorithm include the ill-conditioning of the background covariance matrix, the choice of a preconditioning and the choice of the regularisation parameter. We also show how the choice of the regularisation parameter impacts on the efficiency of the VarDA minimisation computed by the L-BFGS (Limited – Broyden Fletcher Goldfarb Shanno). Experimental results are provided for pollutant dispersion within an urban environment.

### Key Ideas of Abstract
What ?
- DA : incorporate observed data into a prediction model
- Variational DA : minimisation of a function which estimates the discrepancy between numerical results and observations

What for ?
- Operational forecasting : real-time data assimilation

Problem ?
- Scale of forecasting area
- Number of States to describe physical model
- => Big Data Problem

Solution ?
- Truncated Singular Value Decomposition (TSVD) :
  - reduce the space dimension
  - alleviate the computational cost
  - reduce the errors
  -
- Choice of optimal truncation parameter (algorithm provided)
  - reduces ill- conditioning
  - removes the statistically less significant modes (which could add noise to the estimate obtained from DA)

Difficulties ?
- Numerical issues :
  - ill-conditioning of the background covariance matrix
  - the choice of a preconditioning and the choice of the regularisation parameter.
- Choice of the regularisation parameter impacts on the efficiency of the VarDA minimisation computed by the L-BFGS (Limited – Broyden Fletcher Goldfarb Shanno).



## Limitations
- Depends on the choice of parameters ?

## References
*Insert*
