import os
import time
import argparse
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# =========================
# ARGUMENTOS
# =========================
parser = argparse.ArgumentParser()

parser.add_argument("--img_size", type=int, default=28) # porque es 28 x 28
parser.add_argument("--channels", type=int, default=1) # porque es blanco y negro
parser.add_argument("--latent_dim", type=int, default=100) # Cuantas dimensiones tiene el espacio latente - Vector de entrada de GENERADORA
parser.add_argument("--batch_size", type=int, default=128) # Cuantas imagenes entran en lote
parser.add_argument("--epoch", type=int, default=30) # Cuantas epocas de entrenamiento
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--b1", type=float, default=0.5)
parser.add_argument("--b2", type=float, default=0.999)

args = parser.parse_args(args=[])

image_dim = args.img_size * args.img_size * args.channels

# =========================
# DATA
# =========================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = (x_train.reshape(-1, 28, 28, 1).astype("float32") - 127.5) /  127.5

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_train)
    .shuffle(60000)
    .batch(args.batch_size)
)

# =========================
# GENERATOR
# =========================
def build_generator():
    inputs = layers.Input(shape=(args.latent_dim,))
    x = layers.Dense(128, kernel_initializer="he_uniform")(inputs)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(256, kernel_initializer="he_uniform")(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=1e-3)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(512, kernel_initializer="he_uniform")(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=1e-3)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(1024, kernel_initializer="he_uniform")(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=1e-3)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(image_dim, activation="tanh")(x)
    outputs = layers.Reshape(
        (args.img_size, args.img_size, args.channels)
    )(x)

    return tf.keras.Model(inputs, outputs, name="Generator")

# =========================
# DISCRIMINATOR
# =========================
def build_discriminator():
    inputs = layers.Input(shape=(args.img_size, args.img_size, args.channels))
    x = layers.Flatten()(inputs)

    x = layers.Dense(512, kernel_initializer="he_uniform")(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(256, kernel_initializer="he_uniform")(x)
    x = layers.LeakyReLU(0.2)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="Discriminator")

generator = build_generator()
discriminator = build_discriminator()

# =========================
# LOSSES
# =========================
adversarial_loss = tf.keras.losses.BinaryCrossentropy()

def generator_loss(fake_output):
    return adversarial_loss(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = adversarial_loss(tf.ones_like(real_output), real_output)
    fake_loss = adversarial_loss(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# =========================
# OPTIMIZERS
# =========================
generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=args.lr,
    beta_1=args.b1,
    beta_2=args.b2
)

discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=args.lr,
    beta_1=args.b1,
    beta_2=args.b2
)

# =========================
# TRAIN STEP
# =========================
@tf.function
def train_step(images):

    noise = tf.random.normal([tf.shape(images)[0], args.latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_gen, generator.trainable_variables)
    )

    discriminator_optimizer.apply_gradients(
        zip(gradients_of_disc, discriminator.trainable_variables)
    )
    return gen_loss, disc_loss

# =========================
# TRAIN
# =========================
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        generator_losses = []
        discriminator_losses = []

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)

            generator_losses.append(gen_loss.numpy())
            discriminator_losses.append(disc_loss.numpy())

        print(
            f"Generador     -> "
            f"Media: {np.mean(generator_losses):.4f} | "
            f"Desv: {np.std(generator_losses):.4f}"
        )

        print(
            f"Discriminador -> "
            f"Media: {np.mean(discriminator_losses):.4f} | "
            f"Desv: {np.std(discriminator_losses):.4f}"
        )

        print(f"Epoch {epoch+1} - Time: {time.time()-start:.2f}s\n")

train(train_dataset, args.epoch)

noise = tf.random.normal([25, args.latent_dim])
generated_images = generator(noise, training=False)
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    img = generated_images[i,:,:,0]

    # si usás tanh
    img = (img + 1) / 2.0

    plt.imshow(img, cmap="gray")
    plt.axis("off")
plt.show()