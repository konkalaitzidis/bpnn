import numpy as np
import h5py
from tensorflow.keras.utils import Sequence
from pixel_values_normalization import normalize_video

class NWBDataGenerator(Sequence):

    
    # list of video urls
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
    
        # find min frame from the batch
        min_frame = np.min(self.images[indices, :, :], axis=0)
        # remove the background
        self.images[indices, :, :] = self.images[indices, :, :] - min_frame
        
        # normalize the values in the batch
        self.images[indices, :, :] = normalize_video(self.images[indices, :, :])
        
        return self.images[indices, :, :], self.labels[indices, :]




    
#############
    
# class NWBDataGenerator(Sequence):

#     def __init__(self, images, labels, idx, batch_size=32):
#         self.images = images
#         self.labels = labels
#         self.batch_size = batch_size
#         self.frame_order = np.random.permutation(idx)
#     def __len__(self):
#         return len(self.frame_order) // self.batch_size
#     def __getitem__(self, batch_index):
#         N = self.batch_size
#         indices = self.frame_order[batch_index * N:(batch_index+1)*N]
#         return self.images[indices, :, :], self.labels[indices, :]