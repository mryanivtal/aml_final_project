import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def display_mnist_chars(images_tensor):
    matplotlib.use('Qt5Agg')

    num_rows = int(images_tensor.shape[1])       # time dimension - same digit rotated
    num_columns = int(images_tensor.shape[0])    # different original samples

    fig, ax = plt.subplots(num_columns, num_rows+1)
    for column in range(num_columns):
        for image in range(num_rows):
            pixels = np.array(images_tensor[column, image, :]).reshape((28, 28))
            ax[column, image].imshow(pixels, cmap='gray')

    plt.show()


def apply_random_noise(dataset, bit_flip_ratio):
    '''
    Generatates a masked dataset based on an original dataset and a bit_flip ratio
    '''
    masked_dataset = dataset.copy()
    mask_set = np.random.random(size=masked_dataset.shape) < bit_flip_ratio
    masked_dataset[mask_set & (masked_dataset == 1)] = 1 - masked_dataset[mask_set & (masked_dataset == 1)]
    return masked_dataset, mask_set


def apply_mnar_noise(dataset, white_flip_ratio=0.8, black_flip_ratio=0.2):
    '''
    Generatates a masked dataset based on an original dataset and a bit_flip ratio
    '''
    masked_dataset = dataset.copy()
    mask_set = np.zeros(masked_dataset.shape)

    mask_set[dataset == 1] = np.random.random(size=mask_set[dataset == 1].shape) < white_flip_ratio
    mask_set[dataset == 0] = np.random.random(size=mask_set[dataset == 0].shape) < black_flip_ratio
    masked_dataset[(mask_set == 1) & (masked_dataset == 1)] = 1 - masked_dataset[(mask_set == 1) & (masked_dataset == 1)]

    return masked_dataset, mask_set


def save_visual_sample_triplets_to_file(samples_to_show, filename, img_shape, base_data_full, x_val_imputed_no_gt, x_val_imputed):
    idx = np.random.randint(0, len(x_val_imputed_no_gt), samples_to_show)
    fig, axes = plt.subplots(nrows=samples_to_show * 3,
                             ncols=base_data_full.shape[1],
                             figsize=(2 * base_data_full.shape[1], 2 * 3 * samples_to_show))
    zip_seqs = list(zip(base_data_full[idx], x_val_imputed_no_gt[idx], x_val_imputed[idx]))
    seqs = [item for z in zip_seqs for item in z]
    for axs, seq in zip(axes, seqs):
        for ax, img in zip(axs, seq):
            ax.imshow(img.reshape(img_shape), cmap='gray')
            ax.axis('off')
    fig.suptitle('Series reconstruction triplets:\n base data, imputed, imputed within mask limits', size=12)
    fig.savefig(Path(f'{filename}_{samples_to_show}'))
