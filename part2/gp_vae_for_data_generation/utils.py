import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def display_mnist_chars(images_tensor):
    matplotlib.use('Qt5Agg')

    num_rows = int(images_tensor.shape[1])       # time dimension - same digit rotated
    num_columns = int(images_tensor.shape[0])    # different original samples

    fig, ax = plt.subplots(num_columns, num_rows)
    for column in range(num_columns):
        for image in range(num_rows):
            pixels = np.array(images_tensor[column, image, :]).reshape((28, 28))
            ax[column, image].imshow(pixels, cmap='gray')

    plt.show()




