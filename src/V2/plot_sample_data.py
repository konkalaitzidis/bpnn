import matplotlib.pyplot as plt
import numpy as np


def plot_sample_frames(images, labels):

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
    axes = axes.flatten()

    # Generate a list of 5 random integers between 0 and the length of the images variable
    indices = np.random.randint(0, len(images), 5)

    # Loop over the indices and plot each frame with its corresponding label
    for i, index in enumerate(indices):
        axes[i].imshow(images[index])
        axes[i].set_title("Label: " + str(labels[index]))

    plt.tight_layout()
    plt.show()