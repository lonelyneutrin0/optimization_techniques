import numpy as np 
from PIL import Image
import math
import time
import random
im = Image.open("SimulatedAnnealing/image.png")
image = np.array(im)
height, width, rgb = image.shape
# Parameters
alpha= 1
beta = 0.1
gamma = 0.01
delta = 0.05
# Distance Array
index_array = np.stack(np.indices((5,5)), axis=-1)
distance_matrix = np.linalg.norm(index_array[:, :, np.newaxis] - index_array.reshape(-1, 2), axis=-1).reshape(5,5,5,5)
attractive_distance = np.exp(-beta*distance_matrix)
repulsive_distance = np.exp(-delta*distance_matrix)
#Partition into elements 
grid = image.reshape(height//5, 5, width//5, 5, rgb).swapaxes(1,2)
#Energy Function 
def energy(matrix): 
    norm_diff = np.linalg.norm(matrix[:, :, np.newaxis] - matrix.reshape(-1,3), axis=-1).reshape(5,5,5,5)**2
    energy = np.sum(alpha*attractive_distance*norm_diff - gamma*repulsive_distance*norm_diff, axis=(-1,-2))
    return energy
energy_grid = np.empty((height//5, width//5, 5,5))
print(energy_grid.shape)
for i in range(height//5): 
    for j in range(width//5): 
        energy_grid[i,j] = energy(grid[i,j])

start_temp = 1000 
iterations = 1000000
intermediate_grid = grid
intermediate_energy_grid = energy_grid
temps = np.linspace(start_temp, 1, iterations)
def prob(e1, e2, t): 
    try: 
        res = math.exp((e1-e2)/t)
    except OverflowError: 
        res = np.inf
    return res

for i in temps: 
    initial_energy = np.sum(intermediate_energy_grid)
    temp_grid = intermediate_grid.copy()
    temp_energy_grid = intermediate_energy_grid.copy()
    if(i == start_temp): 
        prior_time = time.perf_counter()
    #Swapping the pixels
    ran_row1 = random.randint(0, temp_grid.shape[0] - 1)
    ran_col1 = random.randint(0, temp_grid.shape[1] - 1)
    ran_subrow1 = random.randint(0,temp_grid.shape[2] - 1)
    ran_subcol1 = random.randint(0,temp_grid.shape[3] - 1)
    ran_row2 = random.randint(0, temp_grid.shape[0] - 1)
    ran_col2 = random.randint(0, temp_grid.shape[1] - 1)
    ran_subrow2 = random.randint(0,temp_grid.shape[2] - 1)
    ran_subcol2 = random.randint(0,temp_grid.shape[3] - 1)
    
    temp_pixel = temp_grid[ran_row1, ran_col1, ran_subrow1, ran_subcol1].copy()
    temp_grid[ran_row1, ran_col1, ran_subrow1, ran_subcol1] = temp_grid[ran_row2, ran_col2, ran_subrow2, ran_subcol2]
    temp_grid[ran_row2, ran_col2, ran_subrow2, ran_subcol2] = temp_pixel
    
    # Recalculating the subgrids of the changed pixels
    temp_energy_grid[ran_row1, ran_col1] = energy(temp_grid[ran_row1, ran_col1])
    temp_energy_grid[ran_row2, ran_col2] = energy(temp_grid[ran_row2, ran_col2])
    temp_energy = np.sum(temp_energy_grid)
    # print(initial_energy- temp_energy)
    #Probability Function
    if prob(initial_energy, temp_energy, i) > random.random(): 
        # Reassigned the temporary variables to be permanent
        intermediate_grid = temp_grid
        intermediate_energy_grid = temp_energy_grid
    if(start_temp == i):
        print((time.perf_counter()-prior_time)*temps.size/60)
final_image = Image.fromarray(intermediate_grid.reshape(height, width, rgb))
final_image.show()