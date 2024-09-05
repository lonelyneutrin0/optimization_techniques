import numpy as np 
from PIL import Image
import time
import random
import math
import matplotlib.pyplot as plt 
im = Image.open("SimulatedAnnealing/image.png")
image_matrix = np.array(im)
final_matrix = image_matrix
height, width, rgb = image_matrix.shape 
energies = []
start_temp = 1000
temperatures = np.linspace(start_temp, 1, 1000000)
x = temperatures[::-1]
def probability_acceptance(old_energy, new_energy, temp): 
    if temp == 0: 
        return 0
    return math.exp((old_energy - new_energy) / temp)

def energy(image):
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
# print(energy(image=image_matrix))
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
    final_energy = energy(final_matrix)
    temp_energy = energy(temp_matrix)
    # print(final_energy)
    energies.append(temp_energy)
    if probability_acceptance(final_energy, temp_energy, i) > random.random(): 
        final_matrix = temp_matrix
    if(i == start_temp):    
        post_time = time.perf_counter()
        print((post_time - prior_time)*temperatures.size/60)

final_image = Image.fromarray(np.uint8(final_matrix))
final_image.show()
plt.plot(x, energies)
plt.show()