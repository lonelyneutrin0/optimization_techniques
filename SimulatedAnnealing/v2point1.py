# Aim to improve using noniterative methods 
import numpy as np 
import math 
import matplotlib.pyplot as plt 
import random
from PIL import Image
import PIL 
import scipy.spatial as spat
import time

im = Image.open("SimulatedAnnealing/image.png")
image_matrix = np.array(im)
height, width, rgb = image_matrix.shape
start_temp = 1000 
temperatures = np.linspace(start_temp, 1, 100000) # use temperature.size for number of iterations 
x = temperatures[::1]
energies = []
final_matrix = image_matrix
norm_differences = np.linalg.norm(image_matrix[:, :, np.newaxis, :] - image_matrix.reshape(-1, 3), axis=3)
norm_differences = norm_differences - np.full(norm_differences.shape, 125)

distance_matrix = np.empty((height, width, 2))
for i in range(height): 
    for j in range(width): 
        if(i == j): 
            distance_matrix[i,j] = float('inf') 
        distance_matrix[i,j] = [i,j]

distance_matrix = np.multiply(np.linalg.norm(distance_matrix[:, :, np.newaxis, :] - distance_matrix.reshape(-1 ,2), axis = 3), np.full((height, width, height*width), 125))
distance_matrix[distance_matrix == 0] = float('inf')

def energy(image_matrix): 
    prior_time = time.perf_counter()
    norm_differences = np.linalg.norm(image_matrix[:, :, np.newaxis, :] - image_matrix.reshape(-1, 3), axis=3)
    norm_differences = norm_differences - np.full(norm_differences.shape, 125)
    energy_matrix = np.divide(norm_differences, distance_matrix)
    post_time = time.perf_counter()
    return np.sum(energy_matrix)

def probability_acceptance(old_energy, new_energy, temp): 
    if temp == 0: 
        return 0
    return math.exp((old_energy - new_energy) / temp)

for i in temperatures: 
    prior_time = time.perf_counter()
    temp_matrix = np.copy(final_matrix)
    ran_row1 = random.randint(0, image_matrix.shape[0] - 1)
    ran_col1 = random.randint(0, image_matrix.shape[1] - 1)
    ran_row2 = random.randint(0, image_matrix.shape[0] - 1)
    ran_col2 = random.randint(0, image_matrix.shape[1] - 1)
    
    temp_pixel = temp_matrix[ran_row1, ran_col1].copy()
    temp_matrix[ran_row1, ran_col1] = temp_matrix[ran_row2, ran_col2]
    temp_matrix[ran_row2, ran_col2] = temp_pixel
  
    if probability_acceptance(energy(final_matrix), energy(temp_matrix), i) > random.random(): 
        final_matrix = temp_matrix
    if(i == start_temp):    
        post_time = time.perf_counter()
        print((post_time - prior_time)*temperatures.size/60)

final_image = Image.fromarray(np.uint8(final_matrix))
final_image.show()
