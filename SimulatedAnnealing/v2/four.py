import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from PIL import Image
import time
import random
import math

im = Image.open("SimulatedAnnealing/image.png")
image_matrix = np.array(im)
final_matrix = image_matrix
height, width, rgb = image_matrix.shape


start_temp = 1000
temperatures = np.linspace(start_temp, 1, 200000)

mask = np.eye(height * width, k=1) + np.eye(height * width, k=-1)
def probability_acceptance(old_energy, new_energy, temp): 
    if temp == 0: 
        return 0
    return math.exp((old_energy - new_energy) / temp)
def energy(image):
    prior = time.perf_counter()
    image = image.reshape(height * width, rgb)
    diffs = image[:, np.newaxis, :] - image[np.newaxis, :, :]
    norm_diffs = np.linalg.norm(diffs, axis=-1)
    norm_diffs = 0.5 * (norm_diffs + norm_diffs.T)
    print(time.perf_counter()-prior)
    return np.sum(norm_diffs * mask)
print(energy(image_matrix))
# for i in temperatures: 
#     prior_time = time.perf_counter()
#     temp_matrix = np.copy(final_matrix)
#     ran_row1 = random.randint(0, image_matrix.shape[0] - 1)
#     ran_col1 = random.randint(0, image_matrix.shape[1] - 1)
#     ran_row2 = random.randint(0, image_matrix.shape[0] - 1)
#     ran_col2 = random.randint(0, image_matrix.shape[1] - 1)
    
#     temp_pixel = temp_matrix[ran_row1, ran_col1].copy()
#     temp_matrix[ran_row1, ran_col1] = temp_matrix[ran_row2, ran_col2]
#     temp_matrix[ran_row2, ran_col2] = temp_pixel
  
#     if probability_acceptance(energy(final_matrix), energy(temp_matrix), i) > random.random(): 
#         final_matrix = temp_matrix
#     if(i == start_temp):    
#         post_time = time.perf_counter()
#         print((post_time - prior_time)*temperatures.size/60)

# final_image = Image.fromarray(np.uint8(final_matrix))
# final_image.show()