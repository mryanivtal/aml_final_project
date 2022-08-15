"""

Script to generate data based on existing GP-VAE model.

"""
import pickle
from pathlib import Path
import utils
import sys
import os
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

# Flags with common default values for all three datasets
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_float('gradient_clip', 1e4, 'Maximum global gradient norm for the gradient clipping during training')
flags.DEFINE_integer('num_steps', 0, 'Number of training steps: If non-zero it overwrites num_epochs')
flags.DEFINE_integer('print_interval', 0, 'Interval for printing the loss and saving the model during training')
flags.DEFINE_string('base_dir', "models", 'Directory where the models should be stored')
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

flags.DEFINE_float('white_flip_ratio', 0.6, 'white pixel flip rate')
flags.DEFINE_float('black_flip_ratio', 0.75, 'Black pixel flip rate')


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
    FLAGS.data_type == "hmnist"

    BASE_PATH = Path(FLAGS.base_dir)
    MODEL_PATH = BASE_PATH / Path('model/final_model_weights.pickle')
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

    #############
    # Load and prepare data #
    #############
    #load base dataset from disk
    base_data_full = np.load(BASE_DATA_PATH)

    # Generate masked data
    base_data_masked, mask_set = utils.apply_mnar_noise(base_data_full, white_flip_ratio=FLAGS.white_flip_ratio, black_flip_ratio=FLAGS.black_flip_ratio)

    # Display rotated set - base and masked
    # utils.display_mnist_chars(base_data_full[0:5, :, :])
    # utils.display_mnist_chars(base_data_masked[0:5, :, :])

    ###############
    # Build model #
    ###############
    # Build Conv2D preprocessor for image data
    image_preprocessor = ImagePreprocessor(img_shape, FLAGS.cnn_sizes, FLAGS.cnn_kernel_size)

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
                       length_scale=FLAGS.length_scale, kernel_scales=FLAGS.kernel_scales,
                       image_preprocessor=image_preprocessor, window_size=FLAGS.window_size,
                       beta=FLAGS.beta, M=FLAGS.M, K=FLAGS.K, data_type=FLAGS.data_type)
    else:
        raise ValueError("Model type must be one of ['vae', 'hi-vae', 'gp-vae']")

    _ = tf.compat.v1.train.get_or_create_global_step()
    trainable_vars = model.get_trainable_vars()
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    # Load model weights (Trained model parameters)
    with open(MODEL_PATH, mode='rb') as file:
        model_weights = pickle.load(file)

    model.set_weights(model_weights)

    print("Encoder: ", model.encoder.net.summary())
    print("Decoder: ", model.decoder.net.summary())

    ##############
    # Imputation / Generation #
    ##############

    print("Generating...")
    # Split data on batches
    x_val_miss_batches = np.array_split(base_data_masked, FLAGS.batch_size, axis=0)

    # Save imputed values
    z_mean = [model.encode(x_batch).mean().numpy() for x_batch in x_val_miss_batches]
    np.save(os.path.join(GENERATED_DATA_PATH, "z_mean"), np.vstack(z_mean))
    x_val_imputed_no_gt = np.vstack([model.decode(z_batch).mean().numpy() for z_batch in z_mean])
    np.save(os.path.join(GENERATED_DATA_PATH, "imputed_no_gt"), x_val_imputed_no_gt)

    # reset non-masked pixels to original value
    x_val_imputed = x_val_imputed_no_gt.copy()
    x_val_imputed[mask_set == 0] = base_data_masked[mask_set == 0]
    np.save(os.path.join(GENERATED_DATA_PATH, "imputed"), x_val_imputed)

    filename = str(GENERATED_DATA_PATH / Path(FLAGS.data_type))
    utils.save_visual_sample_triplets_to_file(30, filename, (28, 28), base_data_full, x_val_imputed_no_gt, x_val_imputed)

    print("Generation finished.")


if __name__ == '__main__':
    app.run(main)
