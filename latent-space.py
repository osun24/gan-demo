import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_generator_model(model_path):
    return tf.keras.models.load_model(model_path)

def interpolate_points(p1, p2, n_steps=10):
    # Linear interpolation
    ratios = np.linspace(0, 1, num=n_steps)
    vectors = (1 - ratios[:, None]) * p1 + ratios[:, None] * p2
    return vectors

def plot_3d_interpolated_points(points, filename):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Assume points are in shape (n_steps, latent_dim) and we plot first three dimensions
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', marker='o')

    # Connect points with lines to show interpolation
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color='red')

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    
    plt.savefig(filename)
    plt.show()

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
    plt.show()

# Parameters
latent_dim = 100
n_steps = 49

# Load the trained generator model
generator = load_generator_model('mnist_generator_model_all.h5')

# Generate two random points in the latent space
point1 = tf.random.normal([latent_dim])
point2 = tf.random.normal([latent_dim])

# Interpolate between the two points
interpolated_points = interpolate_points(point1, point2, n_steps=n_steps)

# Plot the interpolated points in 3D
plot_3d_interpolated_points(interpolated_points, 'latent_space_3d.png')

# Generate images from the interpolated points
interpolated_images = generate_images(generator, interpolated_points)

# Define how many rows and cols we want to display
rows = cols = int(np.sqrt(n_steps))

# Show images
show_images(interpolated_images, rows, cols, 'interpolated_images.png')
