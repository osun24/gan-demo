from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the generator model
loaded_generator = load_model('generator_model.h5')

# Generate random noise
noise = tf.random.normal([1, 100])

# Use the generator to create a new image
generated_image = loaded_generator(noise, training=False)

# Postprocess the image
generated_image = (generated_image * 127.5) + 127.5  # Rescale to [0, 255]
generated_image = tf.cast(generated_image, tf.uint8)  # Cast to uint8

# Display the image
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()