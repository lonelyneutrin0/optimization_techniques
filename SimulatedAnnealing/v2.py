import PIL
from PIL import Image
import numpy as np
import math
import random   
from scipy.spatial import KDTree
import time
import matplotlib.pyplot as plt

# Load and convert the image to a NumPy array
im = Image.open("SimulatedAnnealing/image.png")
image_matrix = np.array(im)
final_matrix = np.copy(image_matrix)  # Make sure to copy the matrix
start_temp = 1000
temperatures = np.linspace(start_temp, 1, 100000)
energies = []
x = np.linspace(1, temperatures.size, temperatures.size)
def compute_potential_energy(image):
    height, width, _ = image.shape
    
    # Flatten the image to 2D array of shape (height*width, 3)
    pixels = image.reshape(-1, 3)
    pixel_positions = np.indices((height, width)).reshape(2, -1).T
    
    # Create KD-tree using pixel positions
    kdtree = KDTree(pixel_positions)
    
    # Initialize total energy
    total_energy = 0.0
   
    # Iterate over each pixel and compute its potential energy with neighbors
    for i, pos in enumerate(pixel_positions):
        
        pixel_rgb = pixels[i]
        distances, indices = kdtree.query([pos], k=5)  
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx != i:  # Skip self
                neighbor_pos = pixel_positions[idx]
                neighbor_rgb = pixels[idx]
     
                spatial_distance = np.linalg.norm(pos - neighbor_pos)
                
              
                rgb_diff = np.linalg.norm(pixel_rgb - neighbor_rgb)
                
            
                if spatial_distance > 0:
                    total_energy += (rgb_diff-125)/125*spatial_distance
        
    return total_energy

def probability_acceptance(old_energy, new_energy, temp): 
    if temp == 0: 
        return 0
    return math.exp((old_energy - new_energy) / temp)

for temp in temperatures: 
    first = False
    prior_time = time.perf_counter()
    temp_matrix = np.copy(final_matrix)
    ran_row1 = random.randint(0, image_matrix.shape[0] - 1)
    ran_col1 = random.randint(0, image_matrix.shape[1] - 1)
    ran_row2 = random.randint(0, image_matrix.shape[0] - 1)
    ran_col2 = random.randint(0, image_matrix.shape[1] - 1)
    
    temp_pixel = temp_matrix[ran_row1, ran_col1].copy()
    temp_matrix[ran_row1, ran_col1] = temp_matrix[ran_row2, ran_col2]
    temp_matrix[ran_row2, ran_col2] = temp_pixel
    
    old_energy = compute_potential_energy(final_matrix)
    new_energy = compute_potential_energy(temp_matrix)
    energies.append(new_energy)
    
    if probability_acceptance(old_energy, new_energy, temp) > random.random(): 
        final_matrix = temp_matrix
        
    if(temp == start_temp):    
        post_time = time.perf_counter()
        print((post_time - prior_time)*temperatures.size/60)

plt.plot(x, energies)
plt.show()
final_image = Image.fromarray(np.uint8(final_matrix))
final_image.show()