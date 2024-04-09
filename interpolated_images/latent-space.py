import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob

def load_generator_model(model_path):
    return tf.keras.models.load_model(model_path)

def interpolate_points(p1, p2, n_steps=10):
    # Linear interpolation
    ratios = np.linspace(0, 1, num=n_steps)
    vectors = (1 - ratios[:, None]) * p1 + ratios[:, None] * p2
    return vectors

def generate_images(model, points):
    # Reshape the points in case they come with an extra dimension
    points = tf.reshape(points, (points.shape[0], -1))
    return model(points, training=False)

def show_images(images, n_rows, n_cols, filename):
    fig = plt.figure(figsize=(2*n_cols, 2*n_rows))
    for i in range(images.shape[0]):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    plt.clf()

# Parameters
latent_dim = 100
n_steps = 25

# Load the trained generator model
generator = load_generator_model('mnist_generator_model_all.h5')

# Generate a bunch of interpolated images to see interesting changes
for i in range(5):
    # Count number of existing latent space images
    latent_space_images = glob.glob('interpolated_images_*.png')
    trials = len(latent_space_images) + 1

    # Generate two random points in the latent space
    point1 = tf.random.normal([latent_dim])
    point2 = tf.random.normal([latent_dim])

    # Interpolate between the two points
    interpolated_points = interpolate_points(point1, point2, n_steps=n_steps)

    # Generate images from the interpolated points
    interpolated_images = generate_images(generator, interpolated_points)

    # Define how many rows and cols we want to display
    rows = cols = int(np.sqrt(n_steps))

    # Show images
    show_images(interpolated_images, rows, cols, f'interpolated_images_{trials}.png')
