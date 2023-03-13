#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:20:52 2023

@author: konstantinoskalaitzidis
"""

import numpy as np
import cv2
import os
from pynwb import NWBHDF5IO # python interface for working with Neurodata Without Borders format files
import pandas as pd
import h5py

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Replace 'path/to/file.nwb' with the path to your .nwb file
# io = NWBHDF5IO('/Users/konstantinoskalaitzidis/Developer/dmc/thesis_data/20211016_163921_animal1learnday1.nwb', 'r')
# calcium_video = io.read()

# Open the HDF5 file
with h5py.File('/Users/konstantinoskalaitzidis/Developer/dmc/thesis_data/20211016_163921_animal1learnday1.nwb', 'r') as f:
    # Print the keys of the file
    print(list(f.keys()))
    # dataset = f['identifier'][()]
    # print(dataset)


#raw_calcium_video_path = '/Users/konstantinoskalaitzidis/Developer/dmc/thesis_data/20211016_163921_animal1learnday1.nwb'