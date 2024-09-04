This project aims to discover general optimization techniques. I'm currently working on Simulated Annealing, and will later move to other algorithms. A distant goal is to apply these techniques to QUBO problems.
# Simulated Annealing Versions 
## v1 
This version uses primitive simulated annealing to determine the minimum value of a hexic polynomial. `initial_temp` was set to 1000 and `number_iterations = 10000000`. 
## v2 
The problem aims to rearrange the pixels of a randomly generated image to minimize a given potential energy function. Depending on the annealing schedule, the output is either an amorphous (fast) image or crystalline (slow).
### v2.1 
 It utilizes a k- dimensional tree to find the neighbors and calculate the potential energy using $$\frac{| x_i - x_j | - 125}{125 * r_{ij}}$$. 
This version used `initial_temp = 1000` and `number_iterations = 100000`. It was unsuccessful due to oversights in creating the potential energy functions and being computationally inefficient. 

### v2.2 
 It uses NumPy methods to calculate potential energy and anneal it. This version used `initial_temp = 1000` and `number_iterations = 100000`. It was unsuccessful due to oversights in creating the potential energy functions and being computationally inefficient.

### v2.3 
It uses for loops with time complexity O(4n^2) to determine the potential energy of neighbors of each element. This version used `initial_temp = 1000` and `number_iterations = 200000`. It is the first successful version and takes around 6 hours to anneal a 100x100 pixel image. Further versions utilize the same potential energy function with different implementations to optimize computation time.

### v2.4
This version uses NumPy to create a `height*width X height*width` matrix which contains the pairwise norm differences of the RGB values of each pixel. Multiplying by the required mask followed by summation provides a method of determining potential energy using vectorized NumPy functions. 
`number_iterations = 200000`, `initial_temp = 1000`.

# The QUBO Problem

The QUBO problem deals with finding a bit vector that optimizes the map $f_Q$, where Q is an upper matrix of weights. If we define $f_Q:$ {0,1}$^n \to \mathbb R$ for an upper triangular matrix Q such that $$f_Q(\mathbf x) = \mathbf x^T Q \mathbf x = \sum_{i = 0}^n\sum_{j    = 0}^n Q_{ij}\mathbf x_i \mathbf x_j$$
