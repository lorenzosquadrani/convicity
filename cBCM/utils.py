import matplotlib.pyplot as plt
import numpy as np

def view_weights(weights, grid_shape=None, figsize=(10, 10)):

    weights -= weights.min()
    weights *= 1 / weights.max()

    num_images = weights.shape[0]
    L = int(np.sqrt(weights.shape[1]))
    W = int(weights.shape[1]/L)
    if grid_shape is None:
        rows = int(num_images**(1 / 2))
        cols = rows
    else:
        rows, cols = grid_shape

    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i in range(rows):
        for j in range(cols):
            ax[i, j].imshow(weights[2 * i + j].reshape(L, W))
            ax[i, j].axis('off')
                          
    return fig, ax
