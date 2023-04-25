import numpy as np
import h5py
from tensorflow.keras.utils import Sequence

class NWBDataGenerator(Sequence):

    
    # list of video urls
    
    
    def __init__(self, url_lists, labels, idx, batch_size=32):
        
        
        
        video_data_list = load_video_data(url_lists)
        
        # now I have processed three videos
        # I would like to process the labels first and THEN align ALL. 
        
        # now that the video has been processed, lets load the labels. 
        bonsai_data, df_behavior = loading_labels()

        # now that we've loaded our other files lets align them with the video
        df_new_annotations, df_unique_states = align_files(bonsai_data, df_behavior)
        
        # lets check our unique annotations
        df_new_annotations_unique = df_new_annotations.unique()

        labels = df_new_annotations

        # lets preprocess our data
        images, labels, num_classes = model_preprocessing(images, labels, df_new_annotations_unique)        
        
        
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
    
    
    
    
    # load calcium video to video_data variable
    def load_video_data(url_lists):
        
        
        start_time = time.time()

        fs = CachingFileSystem(
            fs=fsspec.filesystem("http"),
            cache_storage="nwb-cache",  # Local folder for the cache
        )

                
        # find the file paths
        # with fs.open(s3_calcium_url, "rb") as f:
        #     with h5py.File(f) as file:
        #         analysis_group = file["analysis"]
        #         recording_group_name = list(analysis_group.keys())[0]
        #         # recording_group = analysis_group[recording_group_name]
        #         # print("working..")
        #         # video_data = np.array(recording_group["data"][:])
        #         print(recording_group_name)

        
        # load multiple videos here
        video_data_list = []
        for i in enumerate(url_lists):
            
            with fs.open(url_lists[i], "rb") as f:
                with h5py.File(f) as file:
                    video_data = np.array(file["analysis/recording_20211026_142935-PP-BP-MC/data"])
                    video_data_list.append(video_data)


        # video_data_list
        
        

        video_data_list = process_video(video_data_list)               

        end_time = time.time()
        execution_time = end_time - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"Execution time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")

        return video_data_list
    
    def process_video(video_data_list):
        
        
        for i in enumerate(video_data_list)
            min_frame = np.min(video_data_list[i], axis=0)
            video_data_list[i] = video_data_list[i] - min_frame

            video_data_list[i] = normalize_video(video_data_list[i])
        
        return video_data_list
    
    def loading_labels():
        
        #loading_files 
        #list with bonsai data paths
        #list with behavior data path
        
        
        
        bonsai_data = pd.read_csv('/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/data/tmaze_2021-10-26T14_29_27.csv', header=None)
        df_behavior = pd.read_hdf('/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/data/20211026_142935_animal3learnday9.h5', 'per_frame')
        
        return bonsai_data, df_behavior