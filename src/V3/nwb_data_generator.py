import numpy as np
import h5py
from tensorflow.keras.utils import Sequence

class NWBDataGenerator(Sequence):

    
    # list of video urls
    def __init__(self, images, labels, idx, batch_size=32):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.frame_order = np.random.permutation(idx)
        self.min_frame = np.min(self.images, axis = 0)
        #self.images = self.images - self.min_frame
        
        self.max_frame = np.max(self.images, axis = 0)
        self.max_pixel_value = np.max(self.max_frame - self.min_frame)
        self.min_pixel_value = 0
        
    

    def get_labels(self):
        return self.labels[self.frame_order[:self.batch_size*(len(self.frame_order)//self.batch_size)]]
        
        
    def __len__(self):
        return len(self.frame_order) // self.batch_size
    
    def __getitem__(self, batch_index):
        N = self.batch_size
        indices = self.frame_order[batch_index * N:(batch_index+1)*N]
    
        # find min frame from the batch
        # remove the background
        batch_images = self.images[indices, :, :] - self.min_frame
        batch_images = (batch_images - self.min_pixel_value) / (self.max_pixel_value - self.min_pixel_value)
        
        # normalize the values in the batch

        
        return batch_images, self.labels[indices, :]


    
    
    
class NWBDataGeneratorTime(Sequence):

    
    # list of video urls
    def __init__(self, images, labels, idx, batch_size=32):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.frame_order = np.random.permutation(idx)
        self.frame_order[self.frame_order >= self.images.shape[0]-1] -= 1
        self.min_frame = np.min(self.images, axis = 0)
        #self.images = self.images - self.min_frame
        
        self.max_frame = np.max(self.images, axis = 0)
        self.max_pixel_value = np.max(self.max_frame - self.min_frame)
        self.min_pixel_value = 0
        
    

    def get_labels(self):
        return self.labels[self.frame_order[:self.batch_size*(len(self.frame_order)//self.batch_size)]]
        
        
    def __len__(self):
        return len(self.frame_order) // self.batch_size
    
    def __getitem__(self, batch_index):
        N = self.batch_size
        indices = self.frame_order[batch_index * N:(batch_index+1)*N]
    
        # find min frame from the batch
        # remove the background
        batch_images = np.empty((N, self.images.shape[1], self.images.shape[2], 3), dtype=self.images.dtype)
        for i, ind in enumerate(indices):
            for t in range(3):
                batch_images[i, :, :, t] = self.images[ind+t-1, :, :] - self.min_frame
        
            
            
        batch_images = (batch_images - self.min_pixel_value) / (self.max_pixel_value - self.min_pixel_value)
        
        # normalize the values in the batch

        
        return batch_images, self.labels[indices, :]


    
    
    
    
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