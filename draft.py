import numpy as np


x = np.array([[9,2,3],
             [4,5,6],
             [1,10,4]])

y = np.array([[12,2,3],
             [4,5,6],
             [1,10,4]])

x_max = np.amax(x, axis=0)
x_min = np.amin(x, axis=0)


normalized_x = (x - x_min)/(x_max - x_min)


normalized_y = (y - x_min)/(x_max - x_min)
normalized_y = np.clip(normalized_y, 0,1)

