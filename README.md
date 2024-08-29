This project aims to discover general optimization techniques. I'm currently working on Simulated Annealing, and will later move to other algorithms. A distant goal is to apply these techniques to QUBO problems.

The QUBO problem deals with finding a bit vector that optimizes the map $f_Q$, where Q is an upper matrix of weights. If we define $f_Q:$ {0,1}$^n \to \mathbb R$ for an upper triangular matrix Q such that $$f_Q(\mathbf x) = \mathbf x^T Q \mathbf x = \sum_{i = 0}^n\sum_{j    = 0}^n Q_{ij}\mathbf x_i \mathbf x_j$$
