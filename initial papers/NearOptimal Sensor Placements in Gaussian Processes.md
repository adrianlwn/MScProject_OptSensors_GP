# NearOptimal Sensor Placements in Gaussian Processes
2008 - Andreas Krause - Ajit Singh - Carlos Guestrin
This document presents the organisation and the relevant information taken from paper.
## I. Abstract :
When monitoring spatial phenomena, which can often be modelled as _Gaussian processes_ (GPs), choosing sensor locations is a fundamental task. There are several common strategies to address this task, for example, geometry or disk models, placing sensors at the points of highest entropy (variance) in the GP model, and A-, D-, or E-optimal design. In this paper, we tackle the combinatorial optimisation problem of maximizing the mutual information between the chosen locations and the locations which are not selected. We prove that the problem of finding the configuration that max- imizes mutual information is NP-complete. To address this issue, we describe a polynomial-time approximation that is within (1 − 1/e) of the optimum by exploiting the submodularity of mutual information. We also show how submodularity can be used to obtain online bounds, and design branch and bound search procedures. We then extend our algorithm to exploit lazy evaluations and local structure in the GP, yielding significant speedups. We also extend our approach to find placements which are robust against node failures and uncertainties in the model. These extensions are again associated with rigorous theoretical approximation guarantees, exploiting the submodularity of the objective function. We demonstrate the advantages of our approach towards optimizing mutual information in a very extensive empirical study on two real-world data sets.
## II. Organisation
1. Introduction
2. Gaussian Processes
	1. Modeling Sensor Data Using the Multivariate Normal Distribution
	2. Modeling Sensor Data Using Gaussian Processes
	3. Nonstationarity
3. Optimizing Sensor Placements
	1. The Entropy Criterion
	2. An Improved Design Criterion: Mutual Information
4. Approximation Algorithm
	1. The Algorithm
	2. An Approximation Bound
	3. Sensor Placement with Non-constant Cost Functions
	4. Online Bounds
	5. Exact Optimization and Tighter Bounds Using Mixed Integer Programming
5. Scaling Up
	1. Lazy Evaluation Using Priority Queues
	2. Local Kernels
6. Robust Sensor Placements
	1. Robustness Against Failures of Nodes
	2. Robustness Against Uncertainty in the Model Parameters
7. Related Work
	1. Objective Functions
	2. Optimization Techniques
	3. Related Work on Extensions
	4. Related Work in Machine Learning
	5. Relationship to Previous Work of the Authors
8. Notes on Optimizing Other Objective Functions
	1. A Note on the Relationship with the Disk Model
	2. A Note on Maximizing the Entropy
	3. A Note on Maximizing the Information Gain
	4. A Note on Using Experimental Design for Sensor Placement
9. Experiments
	1. Data Sets
	2. Comparison of Stationary and Non-stationary Models
	3. Comparison of Data-driven Placements with Geometric Design Criteria
	4. Comparison of the Mutual Information and Entropy Criteria
	5. Comparison of Mutual Information with Classical Experimental Design Criteria
	6. Empirical Analysis of the Greedy Algorithm
	7. Results on Local Kernels
10. Future Work
11. Conclusions

## III. Notes
1. Introduction
Monitoring spatial phenomenon + limited number of sensing devices =\> position of those sensors. 
Modelling of sensor Measurement (what does the sensor measure in term of information in the space)
- Geometric Model
Sensing area ? Fixed sensing Radius (Gonzalez- Banos and Latombe, 2001). Not realistic : correlations between sensor measurement and actual value in the whole environment.
=\> Fundamentally, the notion that a single sensor needs to predict values in a nearby region is too strong
- **Gaussian Process** Model  (Cressie, 1991; Caselton and Zidek, 1984)
Weaker assumptions (more generic. == non-parametric generalization of linear regression). **Learning** model of the phenomenon with _pilot deployment_ or _expert knowledge_
With a GP model, we _asses the quality _of the placement using different criterion
- Highest **Entropy** (variance).  (Cressie, 1991; Shewry and Wynn, 1987)
-  A-, D-, or E-optimal design
- **Mutual information**. (Caselton and Zidek (1984))
Typical sensor _placement technique_ : greedily add sensors where uncertainty about the phenomenon is highest. 
Criterion 1 : Highest **Entropy**: indirect criterion : measure the quality of each sensors measurement, not the prediction quality on the interesting area. Usually characterised by the sensors position that are selected to be as far from each other as possible. Border placement !
Criterion 2 : **Mutual information** : direct criterion : use the posterior of the GP to measure the sensor placement effect. 
Optimisation :  **combinatorial optimization problem** for maximising mutual information.   NP-complete problem.   mutual information is a submodular function =\> first approximation algorithm (in polynomial time) that guaranties a constant-factor approximation ??

2. Gaussian Processes
	1. Modeling Sensor Data Using the Multivariate Normal Distribution
	2. Modeling Sensor Data Using Gaussian Processes
	3. Nonstationarity




￼￼￼￼￼￼￼￼￼￼￼