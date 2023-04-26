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

            # read csv file with crop coordinates
            with open('/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V3/aligned_videos_animal3.csv', 'r') as csv_file:

                fov_info = csv.reader(csv_file)
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

    
#     start_time = time.time()

#     fs = CachingFileSystem(
#         fs=fsspec.filesystem("http"),
#         cache_storage="nwb-cache",  # Local folder for the cache
#     )
            
#     with fs.open(s3_calcium_url, "rb") as f:
#         with h5py.File(f) as file:
#             analysis_group = file["analysis"]
#             recording_group_name = list(analysis_group.keys())[0]
#             # recording_group = analysis_group[recording_group_name]
#             # print("working..")
#             # video_data = np.array(recording_group["data"][:])
#             print(recording_group_name)

#     with fs.open(s3_calcium_url, "rb") as f:
#         with h5py.File(f) as file:
#             video_data = np.array(file["analysis/recording_20211026_142935-PP-BP-MC/data"])
            
#             video_data = video_data[:, 1:357, 35:433]

#     video_data = np.stack(video_data, ... )
            
            
#     # with fs.open(s3_calcium_url, "rb") as f:
#     #         with h5py.File(f) as file:
#     #             video_data = np.array(file["analysis/recording_20211028_181307-PP-BP-MC/data"])

#     end_time = time.time()
#     execution_time = end_time - start_time
#     hours, remainder = divmod(execution_time, 3600)
#     minutes, seconds = divmod(remainder, 60)
 
#     print(f"Execution time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
    
#     return video_data