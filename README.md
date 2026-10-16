# Exact ReLU Matrix Decomposition (RMD)

This repository contains MATLAB code for computing **exact RMD** of nonnegative and sparse matrices. Moreover, it allows one to reproduce the results presented in our paper [2]; see https://arxiv.org/pdf/2610.22255.

## The problem

Given a nonnegative matrix $X \in \mathbb{R}^{m \times n}_{+}$ and a target rank $r$, **RMD**  looks for a low-rank matrix $\Theta \in \mathbb{R}^{m \times n}$ such that $X \approx \max(0, \Theta)$ and $\text{rank}(\Theta) \leq r$, where the maximum is taken entrywise. 

In this repository we focus on the **exact** case, that is, we look for $\Theta$ such that $\max(0, \Theta) = X$ holds with equality.

## Formulations of the exact problem

### 1. Feasibility problem based on Latent-RMD

Starting from the formulation of Latent-RMD proposed by Saul [1], the exact ReLU decomposition can be written as the problem of finding a point in the intersection of two sets:

$$
\text{find } \Theta \in \mathcal{A} \cap \mathcal{B},
$$

with

$$
\mathcal{A} = \lbrace Z \in \mathbb{R}^{m \times n} \mid \max(0, Z) = X \rbrace
\quad \text{and} \quad
\mathcal{B} = \lbrace \Theta \in \mathbb{R}^{m \times n} \mid \text{rank}(\Theta) \leq r \rbrace.
$$

This feasibility problem can be solved by the Alternating projection Method (APM), which, starting from a feasible point in one of the sets, produces a sequence of iterates by alternately projecting onto the two sets. The algorithm is implemented in Algorithms_and_utils/Naive_RMD.

### 2. Factorized (QCQP) formulation

Writing the low-rank matrix explicitly as $\Theta = WH$, with $W \in \mathbb{R}^{m \times r}$ and $H \in \mathbb{R}^{r \times n}$, removes the rank constraint and leads to the following problem:

$$
\begin{aligned}
\text{find} \quad & W \in [-\bar{M}, \bar{M}]^{m \times r}, \quad H \in [-\bar{M}, \bar{M}]^{r \times n} \\
\text{subject to} \quad & (WH)_{ij} = X_{ij}, \qquad \forall\, (i,j) : X_{ij} > 0, \\
& (WH)_{ij} \leq 0, \qquad\ \ \forall\, (i,j) : X_{ij} = 0.
\end{aligned}
$$

This formulation can be tackled by a global solver such as Gurobi. In particular, we used YALMIP with Gurobi subroutine. Therefore, to run our codes, the YALMIP package is required.

## Repository structure

```
Exact_RMD/
├── Script/                 % scripts to run the experiments
└── Algorithms_and_utils/   % solvers and auxiliary functions
```

The numerical experiments folder contains codes to replicate the experiments contained in the paper [2]:

- Proposition_2_1: Compute the RMD of the example shown in Proposition 2.1 from [2], highlighting the ill-posedness of RMD.
- Table_1: compares the Naive-RMD algorithm with the Gurobi solver on randomly generated matrices for different values of rank and dimensions.
- Figure_2: computes an exact rank-3 factorization of the identity matrix and plot the configuration in points and hyperplanes associated to its factors.
- Section_4_3_2: Computes a rank-3 approximation of the identity matrix using Naive-RMD with random initialization.
- Figure_3: Computes a rank-3 approximation of the identity matrix using Naive-RMD initalized with a noisy exact solution.

## References

1 L. K. Saul, *A nonlinear matrix decomposition for mining the zeros of sparse data*, SIAM Journal on Mathematics of Data Science, 4(2), 2022.\\
2 G. Seraghiti, A. Vandaele, N. Gillis, *The ReLU Matrix Decomposition: NP-Hardness, Ill-Posedness, and ReLU-Rank*, https://arxiv.org/pdf/2610.22255 , 2026.
