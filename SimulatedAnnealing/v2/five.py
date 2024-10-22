import numpy as np 
from PIL import Image
import time
import random
import math
import matplotlib.pyplot as plt 
im = Image.open("SimulatedAnnealing/image.png")
image_matrix = np.array(im)
height, width, rgb = image_matrix.shape 
start_temp = 10000
temperatures = np.linspace(start_temp, 1, 1000000)
x = temperatures[::-1]
energies = []
final_matrix = image_matrix
def probability_acceptance(old_energy, new_energy, temp): 
    if temp == 0:
        return 0
    return math.exp((old_energy - new_energy) / temp)

def energy(image):
    
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
    return np.sum(diff_d)
    # return np.sum(diff_u+diff_d+diff_r+diff_l)
# print(energy(image=image_matrix))
for i in temperatures: 
    if(i == start_temp): 
        prior_time = time.perf_counter()
    temp_matrix = np.copy(final_matrix)
    ran_row1 = random.randint(0, image_matrix.shape[0] - 1)
    ran_col1 = random.randint(0, image_matrix.shape[1] - 1)
    ran_row2 = random.randint(0, image_matrix.shape[0] - 1)
    ran_col2 = random.randint(0, image_matrix.shape[1] - 1)
    
    temp_pixel = temp_matrix[ran_row1, ran_col1].copy()
    temp_matrix[ran_row1, ran_col1] = temp_matrix[ran_row2, ran_col2]
    temp_matrix[ran_row2, ran_col2] = temp_pixel
    temp_energy = energy(temp_matrix)
    energies.append(temp_energy)
    if probability_acceptance(energy(final_matrix), temp_energy, i) > random.random(): 
        final_matrix = temp_matrix
    if(i == start_temp):    
        post_time = time.perf_counter()
        print((post_time - prior_time)*temperatures.size/60)

plt.plot(x, energies)
plt.show()
final_image = Image.fromarray(np.uint8(final_matrix))
final_image.save("SimulatedAnnealing/finalimage.png")
final_image.show()