# Increasing Quantum Advantage with Tailored QAOA for 1-in-3 SAT

This documentation is a work-in-progress during the Hackathon.

## Project Overview 

This Hackathon is work for a future publication based in part on the results gathered here. Our project is about benchmarking an exciting tailored Quantum Alternating Operator Ansatz (`QAOA`)[[1]](#1) approach for solving `1-in-3 SAT`. 

Given the promising recent results obtained with the Quantum Approximate Optimization Algorithm (`QAOA1`)[[2]](#2) by [Boulebnane and Montanaro](https://arxiv.org/abs/2208.06909)[[3]](#3) showing quantum advantage, we wish to understand what computational benefit tailored QAOA can bring on Boolean Satisfiability problems. 

`QAOA` can show massive reductions in the search space considered by quantum devices in place of `QAOA1` by tailoring the mixing and phase-separating operators to smaller subspace[[1]](#1)[[4]](#4). This includes **superpolynomial** reductions in the resulting search space and such a reduction will be seen here as well. (current focus for day one, generate instances and benchmark space reduction)


We plan to generate a collection of `1-in-3 SAT` instances of increasing sizes in the phase transition (see below for more details), following the example set by their paper. While they considered `k-SAT` problems, we will see that our approach has a more meaningful impact for `1-in-3 SAT` style problems. 

Primarily, we will train our `QAOA` and `QAOA1` on instances of size `12` within the band of angles prescribed by `B&M` and utilize this to solve instances of increasing size up to `20`. We will implement `CPU` and `CUDA` `GPU` functions in `Julia` to accomplish this task. 

We also use a brute-force solver to find the entire solution space and generate a collection of `1-in-3 SAT` instances in `JSON` that holds essential information about the instances to be able to benchmark our approach. 

# Quantum Approximate Optimization Algorithm for 1-in-3 SAT

Here we describe `QAOA1` at a high level. Let $ \sigma^{0} = \sigma^{z} + Id $ and $ \sigma^{1} = \sigma^{z} + Id $

```math
    U_{p}(\alpha, \beta) = U_{\text{mixer}}(\beta_{p}) U_{\text{cost}}(\alpha_{p}) \,  \ldots \, U_{\text{mixer}}(\beta_1) \, U_{\text{cost}}(\alpha_1)
```

The cost of a clause is given by satisfying one and unsatisfying the others:
```math
    H_{\text{clause}} = \sum_{ (e_{1}, ..., e_{k} ; v_{1}, ..., v_{k} ) = \text{clause} } \sum_{i=1}^{k} \sigma_{v_{i}}^{e_{i}} \prod_{k\neq i} \sigma_{v_{k}}^{1 - 2 \, e_{k}} 
```

Then the phase-separating operator, given a specific $ \alpha $, is just:
```math
    U_{\text{cost}} = \prod_{\text{clause} \in \text{cost}} e^{i \, \alpha \, H_{\text{clause}} }
```

For the mixing operator, we have the typical `X` rotations per qubit for $ \beta $:
```math
    U_{\text{mixer}}(\beta) = \prod_{j=1}^{n} e^{-i \, \beta \, \sigma_{j}^{x}}.
```

Notice we place the negative sign for the exponent of the mixing operator and the positive sign for the exponent of the phase-separating operator. 

# Tailored Quantum Alternating Operator Ansatz for 1-in-3 SAT

Here we describe our approach at a high level. 



# 1-in-3 SAT 

Given a collection of `m` clauses, each with $3$ literals, we wish to find an assignment to all `n` variables such that one and exactly one of the literals in each clause is satisfied. So $1$ literal is satisfied and $2$ are unsatisfied for each clause in the collection. 


## Random 1-in-3 SAT instances around the Satisfiability Threshold

The [transition](https://www.researchgate.net/publication/2400280_The_phase_transition_in_1-in-k_SAT_and_NAE_3-SAT) [[5]](#5) from likely to be satisfiable to likely to be unsatisfiable for random `1-in-k` SAT problems occurs with $n/{k \choose 2}$ clauses. For random  `1-in-3` SAT, this occurs with `n/3` clauses. 

# Benchmarking



## References

<a id="1">[1]</a> 
Hadfield, S., Wang, Z., O’gorman, B., Rieffel, E. G., Venturelli, D., & Biswas, R. (2019). 
From the quantum approximate optimization algorithm to a quantum alternating operator ansatz. Algorithms, 12(2), 34.

<a id="1">[2]</a> 
Farhi, E., Goldstone, J., & Gutmann, S. (2014). 
A quantum approximate optimization algorithm. arXiv preprint arXiv:1411.4028.

<a id="1">[3]</a> 
Leipold, H., Spedalieri, F. M., & Rieffel, E. (2022). Tailored Quantum Alternating Operator Ansätzes for Circuit Fault Diagnostics. Algorithms, 15(10), 356.

<a id="1">[4]</a> 
Boulebnane, S., & Montanaro, A. (2022). 
Solving boolean satisfiability problems with the quantum approximate optimization algorithm. arXiv preprint arXiv:2208.06909.

<a id="3">[5]</a> 
Achlioptas, D., Chtcherba, A., Istrate, G., & Moore, C. (2001, January). 
The phase transition in 1-in-k SAT and NAE 3-SAT. In Proceedings of the twelfth annual ACM-SIAM symposium on Discrete algorithms (pp. 721-722).