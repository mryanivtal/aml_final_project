"""

Script to train the proposed GP-VAE model.

"""
import threading
import pickle
import utils
import sys
import os
import time
from datetime import datetime
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
from pathlib import Path
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
flags.DEFINE_string('exp_name', "debug", 'Name of the experiment')
flags.DEFINE_string('base_dir', "output1", 'Directory where the models should be stored')
flags.DEFINE_enum('data_type', 'hmnist', ['hmnist', 'physionet', 'sprites'], 'Type of data to be trained on')
flags.DEFINE_integer('seed', 1337, 'Seed for the random number generator')
flags.DEFINE_enum('model_type', 'gp-vae', ['vae', 'hi-vae', 'gp-vae'], 'Type of model to be trained')
flags.DEFINE_integer('cnn_kernel_size', 3, 'Kernel size for the CNN preprocessor')
flags.DEFINE_list('cnn_sizes', [256], 'Number of filters for the layers of the CNN preprocessor')
flags.DEFINE_boolean('banded_covar', False, 'Use a banded covariance matrix instead of a diagonal one for the output1 of the inference network: Ignored if model_type is not gp-vae')
flags.DEFINE_integer('batch_size', 64, 'Batch size for training')

# TODO: Additions
flags.DEFINE_string('data_dir', "", 'Directory from where the data should be read in')
flags.DEFINE_integer('train_class_number', 6000, "max number of class exmaples in training set")
flags.DEFINE_float('white_flip_ratio', 0.6, 'Learning rate for training')
flags.DEFINE_float('black_flip_ratio', 0.75, 'Learning rate for training')
flags.DEFINE_boolean('evaluate_results', False, 'run generation/evaluation as part of train')
# --------- end of addition

flags.DEFINE_integer('M', 1, 'Number of samples for ELBO estimation')
flags.DEFINE_integer('K', 1, 'Number of importance sampling weights')

flags.DEFINE_enum('kernel', 'cauchy', ['rbf', 'diffusion', 'matern', 'cauchy'], 'Kernel to be used for the GP prior: Ignored if model_type is not (m)gp-vae')
flags.DEFINE_integer('kernel_scales', 1, 'Number of different length scales sigma for the GP prior: Ignored if model_type is not gp-vae')


def main(argv):
    del argv  # unused
    np.random.seed(FLAGS.seed)
    tf.compat.v1.set_random_seed(FLAGS.seed)

    print("Seed: {FLAGS.seed}")
    FLAGS.encoder_sizes = [int(size) for size in FLAGS.encoder_sizes]
    FLAGS.decoder_sizes = [int(size) for size in FLAGS.decoder_sizes]

    if 0 in FLAGS.encoder_sizes:
        FLAGS.encoder_sizes.remove(0)
    if 0 in FLAGS.decoder_sizes:
        FLAGS.decoder_sizes.remove(0)

    # Generate folder names
    outdir = os.path.join(FLAGS.base_dir, 'model')
    if not os.path.exists(outdir): os.mkdir(outdir)
    checkpoint_prefix = os.path.join(outdir, "ckpt")
    print("Outputs will be saved to: ", outdir)

    ###################################
    # Define data specific parameters #
    ###################################
    data_dim = 784
    time_length = 10
    num_classes = 10
    decoder = BernoulliDecoder
    img_shape = (28, 28, 1)

    #############
    # Load data #
    #############
    # load data
    print('Loading data...', end='')
    data = np.load(FLAGS.data_dir)
    x_train_full = data['xtrain'].astype('float32')
    x_val_full = data['xtest'].astype('float32')
    y_train = data['ytrain'].astype('uint8')
    y_val = data['ytest'].astype('uint8')
    print('Done!')

    # Reduce train set to required num samples
    if FLAGS.train_class_number:
        indexes = np.empty(0, dtype=int)
        for value in np.unique(y_train):
            itemindex = np.where(y_train == value)
            indexes = np.append(indexes, np.random.choice(itemindex[0], FLAGS.train_class_number, replace=False), axis=0)
        print(len(indexes))
        x_train_full = x_train_full[indexes]
        y_train = y_train[indexes]

    # Save X_train_full and y_train data for generation and classification later - into outputs folder
    data_output_path = Path(FLAGS.base_dir) / Path('train_data.npz')
    np.savez_compressed(data_output_path, x_train=x_train_full, y_train=y_train)
    print(f'Saved train dataset to: {data_output_path}')

    # Mask train and test data using statistical mask
    x_train_miss, m_train_miss = utils.apply_mnar_noise(x_train_full, white_flip_ratio=FLAGS.white_flip_ratio, black_flip_ratio=FLAGS.black_flip_ratio)
    x_val_miss, m_val_miss = utils.apply_mnar_noise(x_val_full, white_flip_ratio=FLAGS.white_flip_ratio, black_flip_ratio=FLAGS.black_flip_ratio)           # Also use random masks in validation set

    # Create tensors from data
    tf_x_train_miss = tf.data.Dataset.from_tensor_slices((x_train_miss, m_train_miss))\
                                     .shuffle(len(x_train_miss)).batch(FLAGS.batch_size).repeat()
    tf_x_val_miss = tf.data.Dataset.from_tensor_slices((x_val_miss, m_val_miss)).batch(FLAGS.batch_size).repeat()
    tf_x_val_miss = tf.compat.v1.data.make_one_shot_iterator(tf_x_val_miss)

    # # Displaysome samples - doesnt work in Jupyter!
    # utils.display_mnist_chars(x_train_full[0:5, :, :])
    # utils.display_mnist_chars(x_train_miss[0:5, :, :])
    # utils.display_mnist_chars(m_train_miss[0:5, :, :])


    ###############
    # Build model #
    ###############
    # Build Conv2D preprocessor for image data
    print("Using CNN preprocessor")
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
                       length_scale=FLAGS.length_scale, kernel_scales = FLAGS.kernel_scales,
                       image_preprocessor=image_preprocessor, window_size=FLAGS.window_size,
                       beta=FLAGS.beta, M=FLAGS.M, K=FLAGS.K, data_type=FLAGS.data_type)
    else:
        raise ValueError("Model type must be one of ['vae', 'hi-vae', 'gp-vae']")


    ########################
    # Training preparation #
    ########################
    print("GPU support: ", tf.test.is_gpu_available())

    print("Training...")
    _ = tf.compat.v1.train.get_or_create_global_step()
    trainable_vars = model.get_trainable_vars()
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    print("Encoder: ", model.encoder.net.summary())
    print("Decoder: ", model.decoder.net.summary())

    if model.preprocessor is not None:
        print("Preprocessor: ", model.preprocessor.net.summary())
        saver = tf.compat.v1.train.Checkpoint(optimizer=optimizer, encoder=model.encoder.net,
                                              decoder=model.decoder.net, preprocessor=model.preprocessor.net,
                                              optimizer_step=tf.compat.v1.train.get_or_create_global_step())

    else:
        saver = tf.compat.v1.train.Checkpoint(optimizer=optimizer, encoder=model.encoder.net, decoder=model.decoder.net,
                                              optimizer_step=tf.compat.v1.train.get_or_create_global_step())

    summary_writer = tf.contrib.summary.create_file_writer(outdir, flush_millis=10000)

    if FLAGS.num_steps == 0:
        num_steps = FLAGS.num_epochs * len(x_train_miss) // FLAGS.batch_size
    else:
        num_steps = FLAGS.num_steps

    if FLAGS.print_interval == 0:
        FLAGS.print_interval = num_steps // FLAGS.num_epochs


    ############
    # Training #
    ############
    losses_train = []
    losses_val = []

    t0 = time.time()
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        for i, (x_seq, m_seq) in enumerate(tf_x_train_miss.take(num_steps)):

            # # Display x_seq, m_seq and combinations to understand data
            # utils.display_mnist_chars(x_seq[0:10])
            # utils.display_mnist_chars(m_seq[0:10])
            # utils.display_mnist_chars(x_seq[0:10] + m_seq[0:10])

            try:
                with tf.GradientTape() as tape:
                    tape.watch(trainable_vars)
                    loss = model.compute_loss(x_seq, m_mask=m_seq)
                    losses_train.append(loss.numpy())
                grads = tape.gradient(loss, trainable_vars)
                grads = [np.nan_to_num(grad) for grad in grads]
                grads, global_norm = tf.clip_by_global_norm(grads, FLAGS.gradient_clip)
                optimizer.apply_gradients(zip(grads, trainable_vars),
                                          global_step=tf.compat.v1.train.get_or_create_global_step())

                # Print intermediate results
                if i % FLAGS.print_interval == 0:
                    print("================================================")
                    print("Learning rate: {} | Global gradient norm: {:.2f}".format(optimizer._lr, global_norm))
                    print("Step {}) Time = {:2f}".format(i, time.time() - t0))
                    loss, nll, kl = model.compute_loss(x_seq, m_mask=m_seq, return_parts=True)
                    print("Train loss = {:.3f} | NLL = {:.3f} | KL = {:.3f}".format(loss, nll, kl))

                    saver.save(checkpoint_prefix)

                    tf.contrib.summary.scalar("loss_train", loss)
                    tf.contrib.summary.scalar("kl_train", kl)
                    tf.contrib.summary.scalar("nll_train", nll)

                    # Validation loss
                    x_val_batch, m_val_batch = tf_x_val_miss.get_next()
                    val_loss, val_nll, val_kl = model.compute_loss(x_val_batch, m_mask=m_val_batch, return_parts=True)
                    losses_val.append(val_loss.numpy())
                    print("Validation loss = {:.3f} | NLL = {:.3f} | KL = {:.3f}".format(val_loss, val_nll, val_kl))

                    tf.contrib.summary.scalar("loss_val", val_loss)
                    tf.contrib.summary.scalar("kl_val", val_kl)
                    tf.contrib.summary.scalar("nll_val", val_nll)

                    # Draw reconstructed images
                    x_hat = model.decode(model.encode(x_seq).sample()).mean()
                    tf.contrib.summary.image("input_train", tf.reshape(x_seq, [-1]+list(img_shape)))
                    tf.contrib.summary.image("reconstruction_train", tf.reshape(x_hat, [-1]+list(img_shape)))

                    t0 = time.time()

            except KeyboardInterrupt:
                saver.save(checkpoint_prefix)
                if FLAGS.debug:
                    import ipdb
                    ipdb.set_trace()
                break

    weights_file_path = Path(outdir) / Path('final_model_weights.pickle')
    model_weights = model.get_weights()
    with open(weights_file_path, mode='wb') as file:
        pickle.dump(model_weights, file)



    if FLAGS.evaluate_results:
        ##############
        # Evaluation #
        ##############
        print("Evaluation...")

        # Split data on batches
        x_val_miss_batches = np.array_split(x_val_miss, FLAGS.batch_size, axis=0)
        x_val_full_batches = np.array_split(x_val_full, FLAGS.batch_size, axis=0)
        m_val_batches = np.array_split(m_val_miss, FLAGS.batch_size, axis=0)

        get_val_batches = lambda: zip(x_val_miss_batches, x_val_full_batches, m_val_batches)

        # Compute NLL and MSE on missing values
        n_missings = m_val_miss.sum()
        nll_miss = np.sum([model.compute_nll(x, y=y, m_mask=m).numpy()
                           for x, y, m in get_val_batches()]) / n_missings
        mse_miss = np.sum([model.compute_mse(x, y=y, m_mask=m, binary=FLAGS.data_type=="hmnist").numpy()
                           for x, y, m in get_val_batches()]) / n_missings
        print("NLL miss: {:.4f}".format(nll_miss))
        print("MSE miss: {:.4f}".format(mse_miss))

        # Save imputed values
        z_mean = [model.encode(x_batch).mean().numpy() for x_batch in x_val_miss_batches]
        np.save(os.path.join(outdir, "z_mean"), np.vstack(z_mean))
        x_val_imputed = np.vstack([model.decode(z_batch).mean().numpy() for z_batch in z_mean])
        np.save(os.path.join(outdir, "imputed_no_gt"), x_val_imputed)

        # impute gt observed values
        x_val_imputed[m_val_miss == 0] = x_val_miss[m_val_miss == 0]
        np.save(os.path.join(outdir, "imputed"), x_val_imputed)

        # Visualize reconstructions
        img_index = 0
        if FLAGS.data_type == "hmnist":
            img_shape = (28, 28)
            cmap = "gray"
        elif FLAGS.data_type == "sprites":
            img_shape = (64, 64, 3)
            cmap = None

        fig, axes = plt.subplots(nrows=3, ncols=x_val_miss.shape[1], figsize=(2*x_val_miss.shape[1], 6))

        x_hat = model.decode(model.encode(x_val_miss[img_index: img_index+1]).mean()).mean().numpy()
        seqs = [x_val_miss[img_index:img_index+1], x_hat, x_val_full[img_index:img_index+1]]

        for axs, seq in zip(axes, seqs):
            for ax, img in zip(axs, seq[0]):
                ax.imshow(img.reshape(img_shape), cmap=cmap)
                ax.axis('off')

        # results table
        suptitle = FLAGS.model_type + f" reconstruction, NLL missing = {mse_miss}"
        fig.suptitle(suptitle, size=18)
        fig.savefig(os.path.join(outdir, FLAGS.data_type + "_reconstruction.pdf"))

        # ==== TODO: Yaniv
        # ==== AURC prediction metrics calculation
        # ==== Decide whether to update the labels or remove this section entirely.
        #
        # # AUROC evaluation using Logistic Regression
        # x_val_imputed = np.round(x_val_imputed)
        # x_val_imputed = x_val_imputed.reshape([-1, time_length * data_dim])
        #
        # cls_model = LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-10, max_iter=10000)
        # val_split = len(x_val_imputed) // 2
        #
        # cls_model.fit(x_val_imputed[:val_split], y_val[:val_split])
        # probs = cls_model.predict_proba(x_val_imputed[val_split:])
        #
        # auprc = average_precision_score(np.eye(num_classes)[y_val[val_split:]], probs)
        # auroc = roc_auc_score(np.eye(num_classes)[y_val[val_split:]], probs)
        # print("AUROC: {:.4f}".format(auroc))
        # print("AUPRC: {:.4f}".format(auprc))
        #
        # results_all = [FLAGS.seed, FLAGS.model_type, FLAGS.data_type, FLAGS.kernel, FLAGS.beta, FLAGS.latent_dim,
        #                FLAGS.num_epochs, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.window_size,
        #                FLAGS.kernel_scales, FLAGS.sigma, FLAGS.length_scale,
        #                len(FLAGS.encoder_sizes), FLAGS.encoder_sizes[0] if len(FLAGS.encoder_sizes) > 0 else 0,
        #                len(FLAGS.decoder_sizes), FLAGS.decoder_sizes[0] if len(FLAGS.decoder_sizes) > 0 else 0,
        #                FLAGS.cnn_kernel_size, FLAGS.cnn_sizes,
        #                nll_miss, mse_miss, losses_train[-1], losses_val[-1], auprc, auroc, FLAGS.data_dir]
        #
        # with open(os.path.join(outdir, "results.tsv"), "w") as outfile:
        #     outfile.write("seed\tmodel\tdata\tkernel\tbeta\tz_size\tnum_epochs"
        #                   "\tbatch_size\tlearning_rate\twindow_size\tkernel_scales\t"
        #                   "sigma\tlength_scale\tencoder_depth\tencoder_width\t"
        #                   "decoder_depth\tdecoder_width\tcnn_kernel_size\t"
        #                   "cnn_sizes\tNLL\tMSE\tlast_train_loss\tlast_val_loss\tAUPRC\tAUROC\tdata_dir\n")
        #     outfile.write("\t".join(map(str, results_all)))
        #
        # with open(os.path.join(outdir, "training_curve.tsv"), "w") as outfile:
        #     outfile.write("\t".join(map(str, losses_train)))
        #     outfile.write("\n")
        #     outfile.write("\t".join(map(str, losses_val)))

    print("Training finished.")


if __name__ == '__main__':
    app.run(main)
