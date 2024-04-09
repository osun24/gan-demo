import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

def plot_latent_space(generator_model_path, n, dim):
    # Load the generator model
    generator = load_model(generator_model_path)

    # Generate n*n points within the latent space
    z_input = np.random.normal(size=(n * n, dim))
    x_generated = generator.predict(z_input)

    # Use PCA to reduce dimension to 3
    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x_generated.reshape(n*n, -1))

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2])
    plt.show()

plot_latent_space('mnist_generator_model_all.h5', 10, 100) # Adjust '10' for the grid size, '100' for the latent dimension size