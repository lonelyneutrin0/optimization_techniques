import numpy as np
import time
import matplotlib.pyplot as plt
# Define a sample image matrix (3x3) with RGB values
image_matrix = 256*np.random.rand(100, 100,3)

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
    rows, cols, _ = matrix.shape
    shifts = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    neighbors = np.array([(i + di, j + dj) for di, dj in shifts])
    valid_neighbors = []
    for ni, nj in neighbors:
        if 0 <= ni < rows and 0 <= nj < cols:
            valid_neighbors.append((ni, nj))
    return np.asarray(valid_neighbors)

def energy_second(image_matrix): 
    energy = 0
    rows, cols, _ = image_matrix.shape
    norm_matrix = np.empty((rows, cols))
    for i in range(rows): 
        for j in range(cols): 
            diff = 0
            for neighbor in get_4_connected_neighbors(image_matrix, i, j): 
                ni, nj = neighbor
                diff += np.linalg.norm(image_matrix[i, j] - image_matrix[ni, nj])
            energy+=diff 
    return energy
err = 0
for i in range(100): 
    image_matrix = 256*np.random.rand(100,100,3)
    err += energy_first(image_matrix) - energy_second(image_matrix)
print(err)