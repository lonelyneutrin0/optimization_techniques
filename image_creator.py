import numpy as np
from PIL import Image

# numpy.random.randint returns an array of random integers
# from low (inclusive) to high (exclusive). i.e. low <= value < high

pixel_data = np.random.randint(
    low=0, 
    high=256,
    size=(20,20, 3),
    dtype=np.uint8
)

image = Image.fromarray(pixel_data)
image.save("SimulatedAnnealing/image.png")