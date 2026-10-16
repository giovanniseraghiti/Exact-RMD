# Exact ReLU Matrix Decomposition (RMD)

This repository contains MATLAB code for computing **exact ReLU matrix decompositions** of nonnegative and sparse matrices.

## The problem

Given a nonnegative matrix $X \in \mathbb{R}^{m \times n}_{+}$ and a target rank $r$, **RMD**  looks for a low-rank matrix $\Theta \in \mathbb{R}^{m \times n}$ such that $X \approx \max(0, \Theta)$ and $ \mathrm{rank}(\Theta) \leq r,$ where the maximum is taken entrywise. 

In this repository we focus on the **exact** case, that is, we look for $\Theta$ such that $\max(0, \Theta) = X$ holds with equality.

## Formulations of the exact problem

### 1. Feasibility problem

The exact ReLU decomposition can be written as the problem of finding a point in the intersection of two sets:

$$
\text{find } \Theta \in \mathcal{A} \cap \mathcal{B},
$$

with

$$
\mathcal{A} = \lbrace Z \in \mathbb{R}^{m \times n} \mid \max(0, Z) = X \rbrace
\quad \text{and} \quad
\mathcal{B} = \lbrace \Theta \in \mathbb{R}^{m \times n} \mid \mathrm{rank}(\Theta) \leq r \rbrace.
$$

### 2. Factorized (QCQP) formulation

Writing the low-rank matrix explicitly as $\Theta = WH$, with $W \in \mathbb{R}^{m \times r}$ and $H \in \mathbb{R}^{r \times n}$, removes the rank constraint and leads to the following problem:

$$
\begin{aligned}
\text{find} \quad & W \in [-\bar{M}, \bar{M}]^{m \times r}, \quad H \in [-\bar{M}, \bar{M}]^{r \times n} \\
\text{subject to} \quad & (WH)_{ij} = X_{ij}, \qquad \forall\, (i,j) : X_{ij} > 0, \\
& (WH)_{ij} \leq 0, \qquad\ \ \forall\, (i,j) : X_{ij} = 0.
\end{aligned}
$$

## Repository structure

```
Matlab_codes/
├── Script/                 % scripts to run the experiments
└── Algorithms_and_utils/   % solvers and auxiliary functions
```

## References

- L. K. Saul, *A nonlinear matrix decomposition for mining the zeros of sparse data*, SIAM Journal on Mathematics of Data Science, 4(2), 2022.
- G. Seraghiti, A. Awari, A. Vandaele, M. Porcelli, N. Gillis, *Accelerated algorithms for nonlinear matrix decomposition with the ReLU function*, IEEE International Workshop on Machine Learning for Signal Processing (MLSP), 2023.
