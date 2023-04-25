import numpy as np
import h5py
from tensorflow.keras.utils import Sequence

class NWBDataGenerator(Sequence):

    def __init__(self, images, labels, idx, batch_size=32):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.frame_order = np.random.permutation(idx)
    def __len__(self):
        return len(self.frame_order) // self.batch_size
    def __getitem__(self, batch_index):
        N = self.batch_size
        indices = self.frame_order[batch_index * N:(batch_index+1)*N]
        return self.images[indices, :, :], self.labels[indices, :]