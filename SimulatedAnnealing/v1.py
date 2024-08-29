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
import matplotlib.pyplot as plt

temperatures = np.linspace(1000, 1, 10000000)
def objective(x): 
    return (x-1)*(x-2)*(x-3)*(x-5)*(x-6)*(x-9)
def probability_acceptance(i, u, t): 
    difference = objective(i) - objective(u)
    return math.exp(difference/(t))
values = []
initial_value = -2; 
intermediate_value = initial_value; 
for i in temperatures :
    random_closeby_value = intermediate_value + np.random.uniform(-0.01, 0.01)
    if(probability_acceptance(intermediate_value, random_closeby_value, i) > random.random()):         
        intermediate_value = random_closeby_value    
        values.append(objective(intermediate_value))
print(intermediate_value, objective(intermediate_value))
x_values = np.linspace(1,len(values), len(values))
plt.plot(x_values, values)
plt.show()