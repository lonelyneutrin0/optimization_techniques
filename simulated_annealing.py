"""
Start with some initial element e = e_0  
Run a for loop
Have a temperature function T(r) that steadily decreases, which tells the
program how frequently to make unfavorable transitions to avoid getting stuck 
in a local minimum. 
Select a random neighbor of e_0 and pass it through the acceptance probability 
function. If the probability is greater than a random number between 0 and 1, assign e = e_new.
"""
import random
import numpy as np
import math
def temperature(iterator):
    temp =  10-0.000001*iterator
    
    return temp; 
def objective(x): 
    return x*x - 6
def probability_acceptance(i, u, t): 
    difference = objective(i) - objective(u)
    return math.exp(difference/(t))
initial_value = -5; 
intermediate_value = initial_value; 
iterator = 0;
while(iterator < 10000000): 
    random_closeby_value = intermediate_value + np.random.uniform(-0.01, 0.01)
    
    if(probability_acceptance(intermediate_value, random_closeby_value, temperature(iterator)) > random.random()): 
        
        intermediate_value = random_closeby_value    
    iterator+=1 
print(intermediate_value, objective(intermediate_value))