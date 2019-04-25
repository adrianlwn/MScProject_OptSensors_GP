# NearOptimal Sensor Placements in Gaussian Processes
2008 - Andreas Krause - Ajit Singh - Carlos Guestrin

## Abstract : 
When monitoring spatial phenomena, which can often be modelled as _Gaussian processes_ (GPs), choosing sensor locations is a fundamental task. There are several common strategies to address this task, for example, geometry or disk models, placing sensors at the points of highest entropy (variance) in the GP model, and A-, D-, or E-optimal design. In this paper, we tackle the combinatorial optimisation problem of maximizing the mutual information between the chosen locations and the locations which are not selected. We prove that the problem of finding the configuration that max- imizes mutual information is NP-complete. To address this issue, we describe a polynomial-time approximation that is within (1 − 1/e) of the optimum by exploiting the submodularity of mutual information. We also show how submodularity can be used to obtain online bounds, and design branch and bound search procedures. We then extend our algorithm to exploit lazy evaluations and local structure in the GP, yielding significant speedups. We also extend our approach to find placements which are robust against node failures and uncertainties in the model. These extensions are again associated with rigorous theoretical approximation guarantees, exploiting the submodularity of the objective function. We demonstrate the advantages of our approach towards optimizing mutual information in a very extensive empirical study on two real-world data sets.

## Introduction

## Gaussian Processes
## Optimizing Sensor Placements
## Approximation Algorithm 
￼￼￼￼￼￼￼￼￼￼￼