import PIL #imports PIL library needed for pillow
from PIL import Image # imports Image class from pillow
import numpy as np
import math
import random
#Generates the image matrix
im=Image.open("SimulatedAnnealing/image.png")
image_matrix = np.array(im)



# #Objective Function 
# objective = lambda A, x : ((np.dot(x.T, np.dot(A,x))).sum(axis=1))
# # Temperature Function
# def temperature(parm): 
#     return 100000-(0.01*parm)
# # Potential Energy Function
# def probability_acceptance(i,u,v,t): 
#     difference = objective(i, v) - objective(u, v)
#     return math.exp(difference/t)




