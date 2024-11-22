import time
import numpy as np
import h5py
import fsspec
import os
import csv
from fsspec.implementations.cached import CachingFileSystem
from dataclasses import dataclass
from typing import List

@dataclass
class VideoDataParams:
    video_paths: List[str]
    fov_info: str
    video_name_list: List[str]
    video_data_list: List[np.ndarray]

def read_fov_info(fov_info_path):
    fov_info_dict = {}
    with open(fov_info_path, 'r') as fov_file:
        fov_info = csv.reader(fov_file)
        for row in fov_info:
            fov_info_dict[row[0]] = (int(row[1]), int(row[2]), int(row[3]), int(row[4]))
    return fov_info_dict

def process_video_file(path, fov_info_dict=None):
    with h5py.File(path, 'r') as f:
        video_name = os.path.basename(path)
        root_group = f['/']
        analysis_group = root_group['analysis']
        recording_group_name = list(analysis_group.keys())[0]
        print(video_name, recording_group_name)
        
        try:
            video_data = np.array(root_group["analysis/"+str(recording_group_name)+"/data"])
        except KeyError:
            print(f"Dataset path 'analysis/{recording_group_name}/data' does not exist in {video_name}")
            return None, video_name
        
        if fov_info_dict and video_name in fov_info_dict:
            left, top, right, bottom = fov_info_dict[video_name]
            video_height, video_width = video_data.shape[1:3]
            left = max(0, min(left, video_width))
            right = max(0, min(right, video_width))
            top = max(0, min(top, video_height))
            bottom = max(0, min(bottom, video_height))
            video_data = video_data[:, top:bottom, left:right]
            print(video_name + " is cropped")
        
        print(video_data.shape)
        return video_data, video_name

# load calcium video to video_data variable
def load_video_data(params: VideoDataParams):
    fov_info_dict = read_fov_info(params.fov_info)
    for path in params.video_paths:
        video_data, video_name = process_video_file(path, fov_info_dict)
        if video_data is not None:
            params.video_name_list.append(video_name)
            params.video_data_list.append(video_data)
    
    images = np.vstack(params.video_data_list)
    print("Concatenated video shape:", images.shape)
    return images

def load_one_video(video_path, video_name_list):
    video_data_list = []
    for path in video_path:
        video_data, video_name = process_video_file(path)
        if video_data is not None:
            video_name_list.append(video_name)
            video_data_list.append(video_data)
    
    if video_data_list:
        images = np.vstack(video_data_list)
        print("Concatenated video shape:", images.shape)
        return images
    else:
        print("No valid video data found.")
    return None
