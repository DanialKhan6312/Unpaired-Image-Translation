import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import datetime
from Layers import *
from Models import *
from Preprocess import *
tfds.disable_progress_bar()
autotune = tf.data.experimental.AUTOTUNE

dataset, _ = tfds.load("cycle_gan/cezanne2photo", with_info=True, as_supervised=True)
train_art, train_pic = dataset["trainA"], dataset["trainB"]
test_art, test_pic = dataset["testA"], dataset["testB"]

# Define the standard image size.
orig_img_size = (286, 286)
# Size of the random crops to be used during training.
input_img_size = (128, 128, 3)
# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

buffer_size = 256
batch_size = 11

test_pic = (
    test_pic.map(preprocess_test_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size)
)
test_art = (
    test_art.map(preprocess_test_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size)
)
_, ax = plt.subplots(4, 2, figsize=(10, 16))
for i, samples in enumerate(zip(train_pic.take(4), train_art.take(4))):
    picture = (((samples[0][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    artwork = (((samples[1][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    ax[i, 0].imshow(picture)
    ax[i, 1].imshow(artwork)
plt.show()
# Create cycle gan model
# Get the generators
gen_G = get_resnet_generator(name="generator_G")
gen_F = get_resnet_generator(name="generator_F")

# Get the discriminators
disc_X = get_discriminator(name="discriminator_X")
disc_Y = get_discriminator(name="discriminator_Y")
adv_loss_fn = keras.losses.MeanSquaredError()
cycle_gan_model = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)

# Compile the model
cycle_gan_model.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.3),
    gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.3),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,
)
# Callbacks
plotter = GANMonitor()
checkpoint_filepath = "./model_checkpoints/cyclegan_checkpoints.{epoch:03d}"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath
)
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

cycle_gan_model.fit(
    tf.data.Dataset.zip((train_pic, train_art)),
    epochs=150,
    callbacks=[tensorboard_callback,plotter],
)