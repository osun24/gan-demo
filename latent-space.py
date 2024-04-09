import numpy as np
import matplotlib.pyplot as plt

def plot_latent_space(generator, n, dim):
    # Generate n*n points within the latent space
    z_input = np.random.normal(size=(n * n, dim))
    x_generated = generator.predict(z_input)
    plt.figure(figsize=(10, 10))
    for i in range(n*n):
        plt.subplot(n, n, i + 1)
        plt.imshow(x_generated[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.show()

plot_latent_space(generator, 10, 100) # Adjust '10' for the grid size, '100' for the latent dimension size