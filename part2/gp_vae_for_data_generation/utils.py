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


def apply_noise(dataset, bit_flip_ratio):
    '''
    Generatates a masked dataset based on an original dataset and a bit_flip ratio
    '''
    masked_dataset = dataset.copy()
    mask_set = np.random.random(size=masked_dataset.shape) < bit_flip_ratio
    masked_dataset[mask_set & (masked_dataset == 1)] = 1 - masked_dataset[mask_set & (masked_dataset == 1)]
    return masked_dataset, mask_set


