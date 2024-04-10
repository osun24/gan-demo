import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import glob
import matplotlib.pyplot as plt

# Count number of pngs to get current trial
model_num = glob.glob("*.h5")
trial = len(model_num) + 1

def make_generator_model(input_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def plot_training_history(history):
    plt.plot(history['gen_loss'], label='gen_loss')
    plt.plot(history['disc_loss'], label='disc_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'training_history_{trial}.png')
    plt.show(False)
    plt.clf()

def generate_and_save_images(model, epoch, test_input):
    # 'training' is set to False so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    # Save the generated images
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.show(False)
    plt.clf()

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels)= mnist.load_data()
train_images = tf.concat([train_images, test_images], axis=0)
train_images = tf.reshape(train_images, (train_images.shape[0], 28, 28, 1))
train_images = tf.cast(train_images, tf.float32)
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Instantiate the generator and the discriminator
generator = make_generator_model(input_dim=100)
discriminator = make_discriminator_model()

# Define the training loop
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    # Train discriminator
    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=False)  # Freeze generator

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Train generator
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)  # Unfreeze generator

        fake_output = discriminator(generated_images, training=False)  # Freeze discriminator

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss, disc_loss

# Train the GAN for 50 epochs
EPOCHS = 50
gen_loss = 0
disc_loss = 0

# Seed for visualization
seed = tf.random.normal([16, 100])

history = {'gen_loss': [], 'disc_loss': []}
for epoch in range(EPOCHS):
    for image_batch in train_dataset:
        gen_loss, disc_loss = train_step(image_batch)
    print(f'Epoch {epoch+1}, gen loss={gen_loss}, disc loss={disc_loss}')
    history['gen_loss'].append(gen_loss)
    history['disc_loss'].append(disc_loss)

    # Plot loss and preview generated images every 5 epochs
    if epoch % 5 == 0:
        plot_training_history(history)
        generate_and_save_images(generator, epoch, seed)

        # Save the generator model
        generator.save(f'mnist_generator_model_{trial}.h5')