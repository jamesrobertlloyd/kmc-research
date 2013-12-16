import os
import numpy as np

images = np.zeros((0,28*28))
labels = np.zeros((0,1))

for directory in os.listdir('./temp') :
    if os.path.isdir(directory):
        image_filename = os.path.join(directory, 'images.csv')
        label_filename = os.path.join(directory, 'labels.csv')
        if os.path.isfile(image_filename):
            some_images = np.genfromtxt(image_filename, delimiter=',')
            some_labels = np.genfromtxt(label_filename, delimiter=',')
            if some_labels.ndim == 1: some_labels = some_labels[:, np.newaxis]
            images = np.vstack((images, some_images))
            labels = np.vstack((labels, some_labels))


np.savetxt('images.csv', images, delimiter=',')
np.savetxt('labels.csv', labels, delimiter=',')
