import tensorflow as tf
from tensorflow.keras import layers
from models import make_generator_model, make_discriminator_model

# Instantiate the generator and the discriminator
generator = make_generator_model(input_dim=100)
discriminator = make_discriminator_model()

# Save model schematics
tf.keras.utils.plot_model(generator, to_file=f'generator.png', show_shapes=True)
tf.keras.utils.plot_model(discriminator, to_file=f'discriminator.png', show_shapes=True)