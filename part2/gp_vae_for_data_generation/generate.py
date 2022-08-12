"""

Script to generate data based on existing GP-VAE model.

"""
from pathlib import Path

import threading
import utils
import sys
import os
import time
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from absl import app
from absl import flags

sys.path.append("..")
from lib.models import *


FLAGS = flags.FLAGS

# HMNIST config
flags.DEFINE_integer('latent_dim', 256, 'Dimensionality of the latent space')
flags.DEFINE_list('encoder_sizes', [256, 256], 'Layer sizes of the encoder')
flags.DEFINE_list('decoder_sizes', [256, 256, 256], 'Layer sizes of the decoder')
flags.DEFINE_integer('window_size', 3, 'Window size for the inference CNN: Ignored if model_type is not gp-vae')
flags.DEFINE_float('sigma', 1.0, 'Sigma value for the GP prior: Ignored if model_type is not gp-vae')
flags.DEFINE_float('length_scale', 2.0, 'Length scale value for the GP prior: Ignored if model_type is not gp-vae')
flags.DEFINE_float('beta', 0.8, 'Factor to weigh the KL term (similar to beta-VAE)')
flags.DEFINE_integer('num_epochs', 20, 'Number of training epochs')

# SPRITES config GP-VAE
# flags.DEFINE_integer('latent_dim', 256, 'Dimensionality of the latent space')
# flags.DEFINE_list('encoder_sizes', [32, 256, 256], 'Layer sizes of the encoder')
# flags.DEFINE_list('decoder_sizes', [256, 256, 256], 'Layer sizes of the decoder')
# flags.DEFINE_integer('window_size', 3, 'Window size for the inference CNN: Ignored if model_type is not gp-vae')
# flags.DEFINE_float('sigma', 1.0, 'Sigma value for the GP prior: Ignored if model_type is not gp-vae')
# flags.DEFINE_float('length_scale', 2.0, 'Length scale value for the GP prior: Ignored if model_type is not gp-vae')
# flags.DEFINE_float('beta', 0.1, 'Factor to weigh the KL term (similar to beta-VAE)')
# flags.DEFINE_integer('num_epochs', 20, 'Number of training epochs')

# Physionet config
# flags.DEFINE_integer('latent_dim', 35, 'Dimensionality of the latent space')
# flags.DEFINE_list('encoder_sizes', [128, 128], 'Layer sizes of the encoder')
# flags.DEFINE_list('decoder_sizes', [256, 256], 'Layer sizes of the decoder')
# flags.DEFINE_integer('window_size', 24, 'Window size for the inference CNN: Ignored if model_type is not gp-vae')
# flags.DEFINE_float('sigma', 1.005, 'Sigma value for the GP prior: Ignored if model_type is not gp-vae')
# flags.DEFINE_float('length_scale', 7.0, 'Length scale value for the GP prior: Ignored if model_type is not gp-vae')
# flags.DEFINE_float('beta', 0.2, 'Factor to weigh the KL term (similar to beta-VAE)')
# flags.DEFINE_integer('num_epochs', 40, 'Number of training epochs')

# Flags with common default values for all three datasets
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_float('gradient_clip', 1e4, 'Maximum global gradient norm for the gradient clipping during training')
flags.DEFINE_integer('num_steps', 0, 'Number of training steps: If non-zero it overwrites num_epochs')
flags.DEFINE_integer('print_interval', 0, 'Interval for printing the loss and saving the model during training')
flags.DEFINE_string('exp_name', "debug", 'Name of the experiment')
flags.DEFINE_string('base_dir', "models", 'Directory where the models should be stored')
flags.DEFINE_string('data_dir', "", 'Directory from where the data should be read in')
flags.DEFINE_enum('data_type', 'hmnist', ['hmnist', 'physionet', 'sprites'], 'Type of data to be trained on')
flags.DEFINE_integer('seed', 1337, 'Seed for the random number generator')
flags.DEFINE_enum('model_type', 'gp-vae', ['vae', 'hi-vae', 'gp-vae'], 'Type of model to be trained')
flags.DEFINE_integer('cnn_kernel_size', 3, 'Kernel size for the CNN preprocessor')
flags.DEFINE_list('cnn_sizes', [256], 'Number of filters for the layers of the CNN preprocessor')
flags.DEFINE_boolean('testing', False, 'Use the actual test set for testing')
flags.DEFINE_boolean('banded_covar', False, 'Use a banded covariance matrix instead of a diagonal one for the output1 of the inference network: Ignored if model_type is not gp-vae')
flags.DEFINE_integer('batch_size', 64, 'Batch size for training')

flags.DEFINE_integer('M', 1, 'Number of samples for ELBO estimation')
flags.DEFINE_integer('K', 1, 'Number of importance sampling weights')

flags.DEFINE_enum('kernel', 'cauchy', ['rbf', 'diffusion', 'matern', 'cauchy'], 'Kernel to be used for the GP prior: Ignored if model_type is not (m)gp-vae')
flags.DEFINE_integer('kernel_scales', 1, 'Number of different length scales sigma for the GP prior: Ignored if model_type is not gp-vae')


def main(argv):
    del argv  # unused
    np.random.seed(FLAGS.seed)
    tf.compat.v1.set_random_seed(FLAGS.seed)

    print("Testing: ", FLAGS.testing, f"\t Seed: {FLAGS.seed}")

    FLAGS.encoder_sizes = [int(size) for size in FLAGS.encoder_sizes]
    FLAGS.decoder_sizes = [int(size) for size in FLAGS.decoder_sizes]

    if 0 in FLAGS.encoder_sizes:
        FLAGS.encoder_sizes.remove(0)
    if 0 in FLAGS.decoder_sizes:
        FLAGS.decoder_sizes.remove(0)


    ###################################
    # Define data specific parameters #
    ###################################

    if FLAGS.data_type == "hmnist":
        # TODO:Yaniv: update paths if necessary
        BASE_PATH = Path(FLAGS.base_dir)
        MODEL_PATH = BASE_PATH / Path('model')
        BASE_DATA_PATH = BASE_PATH / Path('base_data.npy')
        GENERATED_DATA_PATH = BASE_PATH / Path('generated_data')

        # Create output folders on local (Colab) VM
        Path(GENERATED_DATA_PATH).mkdir(parents=True, exist_ok=True)
        print(f'GENERATED_DATA_PATH = {GENERATED_DATA_PATH}, exists = {Path.exists(GENERATED_DATA_PATH)}')

        data_dim = 784
        time_length = 10
        num_classes = 10
        decoder = BernoulliDecoder
        img_shape = (28, 28, 1)
        val_split = 50000
    elif FLAGS.data_type == "physionet":
        if FLAGS.data_dir == "":
            FLAGS.data_dir = "data/physionet/physionet.npz"
        data_dim = 35
        time_length = 48
        num_classes = 2
        decoder = GaussianDecoder
    elif FLAGS.data_type == "sprites":
        if FLAGS.data_dir == "":
            FLAGS.data_dir = "data/sprites/sprites.npz"
        data_dim = 12288
        time_length = 8
        decoder = GaussianDecoder
        img_shape = (64, 64, 3)
        val_split = 8000
    else:
        raise ValueError("Data type must be one of ['hmnist', 'physionet', 'sprites']")


    #############
    # Load and prepare data #
    #############

    #load base dataset from disk
    base_data_full = np.load(BASE_DATA_PATH)

    # Generate masked data
    base_data_masked, mask_set = utils.apply_mnar_noise(base_data_full, white_flip_ratio=0.8, black_flip_ratio=0.2)

    # Display rotated set - base and masked
    # utils.display_mnist_chars(base_data_full[0:5, :, :])
    # utils.display_mnist_chars(base_data_masked[0:5, :, :])

    # Build Conv2D preprocessor for image data
    if FLAGS.data_type in ['hmnist', 'sprites']:
        print("Using CNN preprocessor")
        image_preprocessor = ImagePreprocessor(img_shape, FLAGS.cnn_sizes, FLAGS.cnn_kernel_size)
    elif FLAGS.data_type == 'physionet':
        image_preprocessor = None
    else:
        raise ValueError("Data type must be one of ['hmnist', 'physionet', 'sprites']")


    ###############
    # Build model #
    ###############

    if FLAGS.model_type == "vae":
        model = VAE(latent_dim=FLAGS.latent_dim, data_dim=data_dim, time_length=time_length,
                    encoder_sizes=FLAGS.encoder_sizes, encoder=DiagonalEncoder,
                    decoder_sizes=FLAGS.decoder_sizes, decoder=decoder,
                    image_preprocessor=image_preprocessor, window_size=FLAGS.window_size,
                    beta=FLAGS.beta, M=FLAGS.M, K=FLAGS.K)
    elif FLAGS.model_type == "hi-vae":
        model = HI_VAE(latent_dim=FLAGS.latent_dim, data_dim=data_dim, time_length=time_length,
                       encoder_sizes=FLAGS.encoder_sizes, encoder=DiagonalEncoder,
                       decoder_sizes=FLAGS.decoder_sizes, decoder=decoder,
                       image_preprocessor=image_preprocessor, window_size=FLAGS.window_size,
                       beta=FLAGS.beta, M=FLAGS.M, K=FLAGS.K)
    elif FLAGS.model_type == "gp-vae":
        encoder = BandedJointEncoder if FLAGS.banded_covar else JointEncoder
        model = GP_VAE(latent_dim=FLAGS.latent_dim, data_dim=data_dim, time_length=time_length,
                       encoder_sizes=FLAGS.encoder_sizes, encoder=encoder,
                       decoder_sizes=FLAGS.decoder_sizes, decoder=decoder,
                       kernel=FLAGS.kernel, sigma=FLAGS.sigma,
                       length_scale=FLAGS.length_scale, kernel_scales = FLAGS.kernel_scales,
                       image_preprocessor=image_preprocessor, window_size=FLAGS.window_size,
                       beta=FLAGS.beta, M=FLAGS.M, K=FLAGS.K, data_type=FLAGS.data_type)
    else:
        raise ValueError("Model type must be one of ['vae', 'hi-vae', 'gp-vae']")

    # Load last saved checkpoint
    print("GPU support: ", tf.test.is_gpu_available())

    _ = tf.compat.v1.train.get_or_create_global_step()
    _ = model.get_trainable_vars()

    print("Encoder: ", model.encoder.net.summary())
    print("Decoder: ", model.decoder.net.summary())

    # Load model checkpoint (Trained model parameters)
    latest_checkpoint = tf.train.latest_checkpoint(MODEL_PATH)
    model.load_weights(latest_checkpoint)

    ##############
    # Imputation / Generation #
    ##############

    print("Generating...")

    # Split data on batches
    x_val_miss_batches = np.array_split(base_data_masked, FLAGS.batch_size, axis=0)

    # Save imputed values
    z_mean = [model.encode(x_batch).mean().numpy() for x_batch in x_val_miss_batches]
    np.save(os.path.join(GENERATED_DATA_PATH, "z_mean"), np.vstack(z_mean))
    x_val_imputed = np.vstack([model.decode(z_batch).mean().numpy() for z_batch in z_mean])
    np.save(os.path.join(GENERATED_DATA_PATH, "imputed_no_gt"), x_val_imputed)

    # reset non-masked pixels to original value
    x_val_imputed[mask_set == 0] = base_data_masked[mask_set == 0]
    np.save(os.path.join(GENERATED_DATA_PATH, "imputed"), x_val_imputed)

    # Visualize reconstructions
    if FLAGS.data_type in ["hmnist", "sprites"]:
        img_index = 0
        if FLAGS.data_type == "hmnist":
            img_shape = (28, 28)
            cmap = "gray"
        elif FLAGS.data_type == "sprites":
            img_shape = (64, 64, 3)
            cmap = None

        fig, axes = plt.subplots(nrows=3, ncols=base_data_masked.shape[1], figsize=(2*base_data_masked.shape[1], 6))

        x_hat = model.decode(model.encode(base_data_masked[img_index: img_index+1]).mean()).mean().numpy()
        seqs = [base_data_masked[img_index:img_index+1], x_hat>0.5]

        for axs, seq in zip(axes, seqs):
            for ax, img in zip(axs, seq[0]):
                ax.imshow(img.reshape(img_shape), cmap=cmap)
                ax.axis('off')

        suptitle = FLAGS.model_type + f" reconstruction"
        fig.suptitle(suptitle, size=18)
        fig.savefig(os.path.join(GENERATED_DATA_PATH, FLAGS.data_type + "_reconstruction.pdf"))

    print("Generation finished.")


if __name__ == '__main__':
    app.run(main)
