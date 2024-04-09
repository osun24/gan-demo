import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from tkinter import font as tkFont

class MNISTGallery:
    def __init__(self, root):
        self.root = root

        # Load the model from h5 file
        self.load_model("mnist_generator_model_3.h5")

        # Set up the UI components: canvas, buttons, and labels
        self.canvas = tk.Canvas(root, width=280, height=280)
        self.canvas.pack()

        frame_buttons = tk.Frame(root)
        frame_buttons.pack(side=tk.BOTTOM, fill=tk.X)

        btn_generate = tk.Button(frame_buttons, text="Generate", command=self.generate_image)
        btn_generate.pack(side=tk.LEFT)

        self.update_canvas(self.generate_image())

    # Load the trained model
    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    # Update the canvas with the new image
    def update_canvas(self, image):
        photo = ImageTk.PhotoImage(image=Image.fromarray(image).resize((280, 280)))
        self.canvas.create_image(140, 140, image=photo)
        self.canvas.image = photo  # Keep a reference!

    def generate_image(self):
        # Generate noise as input
        noise = np.random.normal(0, 1, (1, 100))
        # Generate image
        generated_image = self.model.predict(noise)
        # Rescale image from [-1,1] to [0,255]
        generated_image = ((generated_image + 1) / 2) * 255
        # Convert to uint8
        generated_image = generated_image.astype(np.uint8)
        # Reshape image
        generated_image = generated_image.reshape(28, 28)
        self.update_canvas(generated_image)
        return generated_image

# Create the main window
root = tk.Tk()
root.title("MNIST Gallery")

# Initialize and run the gallery application
app = MNISTGallery(root)
root.mainloop()
