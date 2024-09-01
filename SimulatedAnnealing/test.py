import numpy as np
from scipy.spatial import KDTree

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
        distances, indices = kdtree.query([pos], k=5)  # k=9 to include the pixel itself and 8 neighbors
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx != i:  # Skip self
                neighbor_pos = pixel_positions[idx]
                neighbor_rgb = pixels[idx]
                
                # Compute spatial distance
                spatial_distance = np.linalg.norm(pos - neighbor_pos)
                
                # Compute RGB difference norm
                rgb_diff = np.linalg.norm(pixel_rgb - neighbor_rgb)
                
                # Compute potential energy contribution
                total_energy += (spatial_distance**(-6) - spatial_distance**(-12)) * (256-rgb_diff)
    
    return total_energy

# Example usage
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
total_energy = compute_potential_energy(image)
print(f'Total Potential Energy: {total_energy}')