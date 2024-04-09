import imageio
import glob

filenames = glob.glob('*.png')
filenames.sort()

images = []

for filename in filenames:
    images.append(imageio.v2.imread(filename))
    
# set duration
duration = 600
imageio.mimsave('training_history.gif', images, duration=duration)