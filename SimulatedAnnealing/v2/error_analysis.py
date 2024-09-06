import numpy as np
import time
import matplotlib.pyplot as plt
import random
# Define a sample image matrix (3x3) with RGB values
image_matrix = np.arange(0, 300, 1).reshape((10, 10, 3))
height, width, channels = image_matrix.shape
# Define the first energy function
def energy_first(image):
    
    left = np.roll(image, 1, axis=1)
    right = np.roll(image, -1, axis=1)
    down = np.roll(image, -1, axis=0)
    up = np.roll(image, 1, axis=0)
    
    diff_l = np.linalg.norm(image-left, axis=-1)
    diff_l[:, 0] = 0
    
    diff_r = np.linalg.norm(image-right, axis=-1)
    diff_r[:, -1] = 0    
   
    diff_u = np.linalg.norm(image- up, axis=-1)
    diff_u[0, :] = 0
    
    diff_d = np.linalg.norm(image- down, axis=-1)
    diff_d[-1, :] = 0
    return np.sum(diff_u+diff_d+diff_r+diff_l)
# Define the second energy function
def get_4_connected_neighbors(matrix, i, j):
  
    matrix = np.array(matrix) 
    rows, cols, se = matrix.shape
    shifts = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    neighbors = np.array([(i + di, j + dj) for di, dj in shifts])
    valid_neighbors = []
    for ni, nj in neighbors:
        if 0 <= ni < rows and 0 <= nj < cols:
            valid_neighbors.append((ni, nj))
   
    return np.asarray(valid_neighbors)


def energy_second(image_matrix): 
    energy = 0
    counter = 0
    for i in range(height): 
        for j in range(i): 
            for neighbor in get_4_connected_neighbors(image_matrix, i ,j): 
                energy += np.linalg.norm(image_matrix[i, j] - image_matrix[neighbor])
                
    print(counter)
    return energy
err = 0
print(energy_first(image_matrix), (energy_second(image_matrix)))
# temp_matrix = np.copy(image_matrix)
# ran_row1 = random.randint(0, image_matrix.shape[0] - 1)
# ran_col1 = random.randint(0, image_matrix.shape[1] - 1)
# ran_row2 = random.randint(0, image_matrix.shape[0] - 1)
# ran_col2 = random.randint(0, image_matrix.shape[1] - 1)

# temp_pixel = temp_matrix[ran_row1, ran_col1].copy()
# temp_matrix[ran_row1, ran_col1] = temp_matrix[ran_row2, ran_col2]
# temp_matrix[ran_row2, ran_col2] = temp_pixel

# print(energy_first(temp_matrix), energy_second(temp_matrix))