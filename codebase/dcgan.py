import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_datasets as tfd
import numpy as np

import matplotlib.pyplot as plt


def make_generator():
    model = tfk.Sequential()
    model.add(tfk.layers.Dense(7*7*BATCH_SIZE, use_bias=False, input_shape=(100,)))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.LeakyReLU())

    model.add(tfk.layers.Reshape((7, 7, BATCH_SIZE)))
    assert model.output_shape == (None, 7, 7, BATCH_SIZE)

    model.add(tfk.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.LeakyReLU())

    model.add(tfk.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.LeakyReLU())

    model.add(tfk.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

###
generator = make_generator()
z = tf.random.normal([1, 100])
x_fake = generator(z, training=False)

plt.imshow(x_fake[0, :, :, 0], cmap='gray')

################################################################################
def make_discriminator():
    model = tfk.Sequential()
    model.add(tfk.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(tfk.layers.LeakyReLU())
    model.add(tfk.layers.Dropout(0.3))

    model.add(tfk.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tfk.layers.LeakyReLU())
    model.add(tfk.layers.Dropout(0.3))

    model.add(tfk.layers.Flatten())
    model.add(tfk.layers.Dense(1))

    return model

###
discriminator = make_discriminator()
p = discriminator(x_fake)
print(p)
################################################################################
cross_entropy = tfk.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(output_real, output_fake):
    loss_real = cross_entropy(tf.ones_like(output_real), output_real)
    loss_fake = cross_entropy(tf.zeros_like(output_fake), output_fake)
    return loss_real + loss_fake

def generator_loss(output_fake):
    return cross_entropy(tf.ones_like(output_fake), output_fake)
