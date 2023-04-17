import time
import numpy as np
import h5py
import fsspec
from fsspec.implementations.cached import CachingFileSystem

# load calcium video to video_data variable
def load_video_data(s3_calcium_url):
    start_time = time.time()

    fs = CachingFileSystem(
        fs=fsspec.filesystem("http"),
        cache_storage="nwb-cache",  # Local folder for the cache
    )
            
    with fs.open(s3_calcium_url, "rb") as f:
        with h5py.File(f) as file:
            analysis_group = file["analysis"]
            recording_group_name = list(analysis_group.keys())[0]
            recording_group = analysis_group[recording_group_name]
            print("working..")
            video_data = np.array(recording_group["data"][:])
            return video_data


    # with fs.open(s3_calcium_url, "rb") as f:
    #         with h5py.File(f) as file:
    #             video_data = np.array(file["analysis/recording_20211028_181307-PP-BP-MC/data"])

    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
 
    print(f"Execution time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
    
    #return video_data