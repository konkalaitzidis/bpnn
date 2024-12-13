import time
import numpy as np
import h5py
import fsspec
import os
import csv
from fsspec.implementations.cached import CachingFileSystem

# load calcium video to video_data variable
def load_video_data(video_paths, fov_info, video_name_list, video_data_list):
    
    
    for path in video_paths:
    
        # Open the NWB file
        with h5py.File(path, 'r') as f:

            # extract video name from path and add it to a list
            video_name = os.path.basename(path)
            video_name_list.append(video_name)

            # Access the root group of the file
            root_group = f['/']

            # Access any other groups or datasets in the file as needed
            analysis_group = root_group['analysis']
            recording_group_name = list(analysis_group.keys())[0]

            print(video_name, recording_group_name)

            fov_info = csv.reader(fov_info)
            # Loop through each row in the CSV file
            for row in fov_info:
                if row[0] in video_name:

                    left = int(row[1])
                    top = int(row[2])
                    right = int(row[3])
                    bottom = int(row[4])
                
                # load the full video from the path
                video_data = np.array(root_group["analysis/"+str(recording_group_name)+"/data"])

                # crop the video
                video_data = video_data[:, top:bottom, left:right]   
                print(video_name+" is cropped")

                # save video in a list
                video_data_list.append(video_data)
                
                # print the shape
                print(video_data.shape)
    
    #concatenate the videos
    images = np.vstack(video_data_list)
    print("Concatenated video shape:", images.shape)

    return images



def load_one_video(video_path, video_name_list):
    for path in video_path:
    
        # Open the NWB file
        with h5py.File(path, 'r') as f:

            # extract video name from path and add it to a list
            video_name = os.path.basename(path)
            video_name_list.append(video_name)

            # Access the root group of the file
            root_group = f['/']

            # Access any other groups or datasets in the file as needed
            analysis_group = root_group['analysis']
            recording_group_name = list(analysis_group.keys())[0]

            print(video_name, recording_group_name)
                
            # load the full video from the path
            images = np.array(root_group["analysis/"+str(recording_group_name)+"/data"])

            # print the shape
            print(images.shape)

    return images