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
temperatures = np.linspace(start_temp, 1, 200000) # use temperature.size for number of iterations 
x = temperatures[::1]

final_matrix = image_matrix
def get_4_connected_neighbors(matrix, i, j):
    matrix = np.array(matrix)  # Ensure matrix is a NumPy array
    rows, cols, se = matrix.shape
    
    # Define potential neighbors relative positions (delta_i, delta_j)
    shifts = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    
    # Compute the new indices
    neighbors = np.array([(i + di, j + dj) for di, dj in shifts])
    
    # Filter out neighbors that are out of bounds
    valid_neighbors = []
    for ni, nj in neighbors:
        if 0 <= ni < rows and 0 <= nj < cols:
            valid_neighbors.append((ni, nj))
    
    return np.asarray(valid_neighbors)

def probability_acceptance(old_energy, new_energy, temp): 
    if temp == 0: 
        return 0
    return math.exp((old_energy - new_energy) / temp)

def energy(image_matrix): 
    energy = 0
    for i in range(height): 
        for j in range(i): 
            for neighbor in get_4_connected_neighbors(image_matrix, i ,j): 
                energy += np.linalg.norm(image_matrix[i, j] - image_matrix[neighbor])
    return energy

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
