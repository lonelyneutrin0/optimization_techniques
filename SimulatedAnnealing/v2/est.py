import numpy as np

# Define a sample image matrix (3x3) with RGB values
image_matrix = 256*np.random.rand(100,100, 3)

# Define the first energy function
def energy_first(image):
    left = image.copy()
    left = np.roll(left, 1, axis=1)
    left[:, 0] = image[:, 0]
    diff_l = np.linalg.norm(image- left, axis=-1)
    
    right = image.copy()
    right = np.roll(right, -1, axis=1)
    right[:, -1] = image[:, -1]
    diff_r = np.linalg.norm(image- right, axis=-1)
    
    up = image.copy()
    up = np.roll(up, -1, axis=0)
    up[0, :] = image[0, :]
    diff_u = np.linalg.norm(image- up, axis=-1)
    
    down = image.copy() 
    down = np.roll(down, 1, axis=0)
    down[-1, :] = image[-1, :]
    diff_d = np.linalg.norm(image- down, axis=-1)
    
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
 
    for i in range(rows): 
        for j in range(cols): 
            for neighbor in get_4_connected_neighbors(image_matrix, i, j): 
                ni, nj = neighbor
                energy += np.linalg.norm(image_matrix[i, j] - image_matrix[ni, nj])
 
    return energy

# Compute the energies
while 1>0:
    image_matrix = 256*np.random.rand(100,100, 3)
    energy1 = energy_first(image_matrix)
    energy2 = energy_second(image_matrix)
    
    print((energy1-energy2)/energy1)