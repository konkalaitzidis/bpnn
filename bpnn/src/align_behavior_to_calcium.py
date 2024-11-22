import pandas as pd
import h5py


#====== old labels ======#
def align_files_old_labels(bonsai_paths, behavior_paths, merge_labels):
    """
    Align bonsai and behavior data files with old labels.

    Parameters:
    bonsai_paths (list): List of paths to bonsai CSV files.
    for bonsai_path, behavior_path in zip(bonsai_paths, behavior_paths):
        # for path in bonsai_paths:
        bonsai_data = pd.read_csv(bonsai_path, header=None)
    Returns:
    tuple: DataFrame of new annotations and DataFrame of new annotations check.
    """

    bonsai_data_list = []
    behavior_data_list = []
    df_new_annotations_list = []

    for bonsai_data_path, behavior_data_path in zip(bonsai_paths, behavior_paths):
        # for path in bonsai_paths:
        bonsai_data = pd.read_csv(bonsai_data_path, header=None)
        behavior_data = pd.read_hdf(behavior_data_path, 'per_frame')
        # Rename the columns of the DataFrame
        bonsai_data = bonsai_data.rename(columns={
            0: 'Time', 1: 'Trial_Number',
            2: 'Reward', 3: 'Frame_Number', 4: 'Central_Zone'})
        bonsai_data_list.append(bonsai_data)
        behavior_data_list.append(behavior_data)
        
    for i, (bonsai_data, behavior_data) in enumerate(zip(bonsai_data_list, behavior_data_list)):
        df_aligned = behavior_data.loc[bonsai_data.groupby('Calcium_frame').first()[1:].Frame_Number].reset_index()
        df_new_annotations = df_aligned[['state_id', 'state_name']]
        df_unique_states = df_new_annotations[['state_id', 'state_name']].drop_duplicates(subset='state_id').set_index('state_id')['state_name'].sort_index()
    
        # state id mapping for main corridor, left corridor, right corridor for animal3learnday8
        if merge_labels:
            if i == 0:
                state_id_map = {
                    1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0,
                    9: 1, 11: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1,
                    10: 2, 12: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25:2
                }
            elif i == 1:
                state_id_map = {
                    1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0,
                    9: 1, 11: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1,
                    10: 2, 12: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25:2
                }
            else:
                state_id_map = {
                    1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0,
                    9: 1, 11: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1,
                    10: 2, 12: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25:2
                }
            df_new_annotations.loc[:, 'state_id'] = df_new_annotations.loc[:, 'state_id'].replace(state_id_map)
            names_of_labels = 'Main Corr', 'Left Corr', 'Right Corr'
        
        df_new_annotations_check = df_new_annotations[['state_id', 'state_name']]
        df_new_annotations_list.append(df_new_annotations)
            
    df_new_annotations = pd.concat(df_new_annotations_list, axis=0)
    
    return df_new_annotations, df_new_annotations_check

#====== new labels ======#

def align_files_new_labels(bonsai_paths, h5_path, multiple_videos):
    """
    Align bonsai and behavior data files with new labels.

    Parameters:
    bonsai_paths (list): List of paths to bonsai CSV files.
    h5_path (str): Path to the HDF5 file containing behavior data.
    for bonsai_path in bonsai_paths:

    Returns:
    tuple: DataFrame of new annotations and DataFrame of new annotations check.
    """

    bonsai_data_list = []
    behavior_data_list = []
    df_new_annotations_list = []
    

    for bonsai_data_path in bonsai_paths:
        # Read the CSV file into a pandas DataFrame
        bonsai_data = pd.read_csv(bonsai_data_path, header=None)
        # Rename the columns of the DataFrame
        bonsai_data = bonsai_data.rename(columns={
            0: 'Time', 1: 'Trial_Number',
            2: 'Reward', 3: 'Frame_Number', 4: 'Central_Zone',
            5: 'L_Zone', 6: 'R_Zone', 7: 'Calcium_frame'})

        # Append the DataFrame to the list of CSV data
        bonsai_data_list.append(bonsai_data)

    with (h5py.File(h5_path, 'r')) as f:
    # print(f['animal3learnday8'].keys())
    
        if multiple_videos == True: 
            behavior_data_8 = pd.read_hdf(h5_path, 'animal3learnday8')
            behavior_data_8 = behavior_data_8.rename(columns={'state_id': 'state_name'})
        if multiple_videos: 
            behavior_data_8.loc[behavior_data_8['state_name'] == 'grooming', 'state_id'] = 0
            behavior_data_8.loc[behavior_data_8['state_name'] == 'immobile', 'state_id'] = 1
            behavior_data_8.loc[behavior_data_8['state_name'] == 'still', 'state_id'] = 2
            behavior_data_8.loc[behavior_data_8['state_name'] == 'moving', 'state_id'] = 3
            behavior_data_8.loc[behavior_data_8['state_name'] == 'rightTurn', 'state_id'] = 4
            behavior_data_8.loc[behavior_data_8['state_name'] == 'leftTurn', 'state_id'] = 5
            behavior_data_8 = behavior_data_8[['state_id', 'state_name']]
            behavior_data_list.append(behavior_data_8)
            # set the value of state_id to 0 when state_name is 'rightTurn'

            behavior_data_9 = pd.read_hdf(h5_path, 'animal3learnday9')
            behavior_data_9 = behavior_data_9.rename(columns={'state_id': 'state_name'})
            behavior_data_9['state_id'] = None
            behavior_data_9.loc[behavior_data_9['state_name'] == 'grooming', 'state_id'] = 0
            behavior_data_9.loc[behavior_data_9['state_name'] == 'immobile', 'state_id'] = 1
            behavior_data_9.loc[behavior_data_9['state_name'] == 'still', 'state_id'] = 2
            behavior_data_9.loc[behavior_data_9['state_name'] == 'moving', 'state_id'] = 3
            behavior_data_9.loc[behavior_data_9['state_name'] == 'rightTurn', 'state_id'] = 4
            behavior_data_9.loc[behavior_data_9['state_name'] == 'leftTurn', 'state_id'] = 5
            behavior_data_9 = behavior_data_9[['state_id', 'state_name']]
            behavior_data_list.append(behavior_data_9)

            behavior_data_10 = pd.read_hdf(h5_path, 'animal3learnday10')
            behavior_data_10 = behavior_data_10.rename(columns={'state_id': 'state_name'})
            behavior_data_10['state_id'] = None
            behavior_data_10.loc[behavior_data_10['state_name'] == 'grooming', 'state_id'] = 0
            behavior_data_10.loc[behavior_data_10['state_name'] == 'immobile', 'state_id'] = 1
            behavior_data_10.loc[behavior_data_10['state_name'] == 'still', 'state_id'] = 2
            behavior_data_10.loc[behavior_data_10['state_name'] == 'moving', 'state_id'] = 3
            behavior_data_10.loc[behavior_data_10['state_name'] == 'rightTurn', 'state_id'] = 4
            behavior_data_10.loc[behavior_data_10['state_name'] == 'leftTurn', 'state_id'] = 5
            behavior_data_10 = behavior_data_10[['state_id', 'state_name']]
            behavior_data_list.append(behavior_data_10)

            behavior_data_11 = pd.read_hdf(h5_path, 'animal2learnday11')
            behavior_data_11 = behavior_data_11.rename(columns={'state_id': 'state_name'})
            behavior_data_11['state_id'] = None
            behavior_data_11.loc[behavior_data_11['state_name'] == 'grooming', 'state_id'] = 0
            behavior_data_11.loc[behavior_data_11['state_name'] == 'immobile', 'state_id'] = 1
            behavior_data_11.loc[behavior_data_11['state_name'] == 'still', 'state_id'] = 2
            behavior_data_11.loc[behavior_data_11['state_name'] == 'moving', 'state_id'] = 3
            behavior_data_11.loc[behavior_data_11['state_name'] == 'rightTurn', 'state_id'] = 4
            behavior_data_11.loc[behavior_data_11['state_name'] == 'leftTurn', 'state_id'] = 5
            behavior_data_11 = behavior_data_11[['state_id', 'state_name']]
            behavior_data_list.append(behavior_data_11)
        else:
            behavior_data_11 = pd.read_hdf(h5_path, 'animal3learnday11')
            behavior_data_11 = behavior_data_11.rename(columns={'state_id': 'state_name'})

    for i, (bonsai_path, path2) in enumerate(zip(bonsai_data_list, behavior_data_list)):
            behavior_data_11.loc[behavior_data_11['state_name'] == 'grooming', 'state_id'] = 0
            behavior_data_11.loc[behavior_data_11['state_name'] == 'immobile', 'state_id'] = 1
            behavior_data_11.loc[behavior_data_11['state_name'] == 'still', 'state_id'] = 2
            behavior_data_11.loc[behavior_data_11['state_name'] == 'moving', 'state_id'] = 3
            behavior_data_11.loc[behavior_data_11['state_name'] == 'rightTurn', 'state_id'] = 4
            behavior_data_11.loc[behavior_data_11['state_name'] == 'leftTurn', 'state_id'] = 5
            behavior_data_11 = behavior_data_11[['state_id', 'state_name']]
            behavior_data_list.append(behavior_data_11)
    

    for i, (bonsai_data, behavior_data) in enumerate(zip(bonsai_data_list, behavior_data_list)):
        # print(bonsai_data)
        # print(behavior_data)
        df_aligned = behavior_data.loc[bonsai_data.groupby('Calcium_frame').first()[1:].Frame_Number].reset_index()
        
        # df_aligned
        df_new_annotations = df_aligned[['state_id', 'state_name']]
        df_unique_states = df_new_annotations[['state_id', 'state_name']].drop_duplicates(subset='state_id').set_index('state_id')['state_name'].sort_index()
        # df_new_annotations = df_new_annotations.loc[:, 'state_id']
        df_new_annotations_list.append(df_new_annotations)
    df_new_annotations = pd.concat(df_new_annotations_list, axis=0)
    df_new_annotations_check = df_new_annotations[['state_id', 'state_name']]
    df_new_annotations_series = df_new_annotations['state_id']
    return df_new_annotations_series, df_new_annotations_check #df_new_annotations_unique


            