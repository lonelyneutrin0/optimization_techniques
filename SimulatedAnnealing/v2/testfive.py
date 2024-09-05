import numpy as np
import time
norms = np.asarray([[1,2,3], [4,5,6], [7,8,9]])
def energy(image_matrix):
    prior = time.perf_counter()
    norms = np.linalg.norm(image_matrix, axis=2) # Creates a norm matrix 
    
    left = norms.copy()
    left[:, -1] = 0
    left = np.roll(left, 1, axis=-1)
        
    right = norms.copy()
    right[:, 0] = 0
    right = np.roll(right, -1, axis=-1)
        
    up = norms.copy() 
    up[-1, :] = 0
    up = np.roll(up, 1, axis=0)
        
    down = norms.copy()
    down[0, :] = 0 
    down = np.roll(down, -1, axis=0)
    
    diff_l = np.sqrt(norms*norms + left*left - 2*norms*left)
    diff_r = np.sqrt(norms*norms + right*right - 2*norms*right)
    diff_u = np.sqrt(norms*norms + up*up - 2*norms*up)
    diff_d = np.sqrt(norms*norms + down*down - 2*norms*down)

    return np.sum(diff_l + diff_r + diff_u + diff_d)/2
