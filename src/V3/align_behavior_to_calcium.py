import pandas as pd


# def align_files(bonsai_data, df_behavior):
    
#     # Adding column names
#     bonsai_data = bonsai_data.rename(columns={
#         0: 'Time', 1: 'Trial_Number',
#         2: 'Reward', 3: 'Frame_Number', 4: 'Central_Zone',
#         5: 'L_Zone', 6: 'R_Zone', 7: 'Calcium_frame'})
        
#     df_aligned = df_behavior.loc[bonsai_data.groupby('Calcium_frame').first()[1:].Frame_Number].reset_index()
        
#     df_new_annotations = df_aligned[['state_id', 'state_name']]
    
#     df_unique_states = df_new_annotations[['state_id', 'state_name']].drop_duplicates(subset='state_id').set_index('state_id')['state_name'].sort_index()

#     # state id mapping for main corridor, left corridor, right corridor
#     state_id_map = {
#         1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0,
#         9: 1, 11: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1,
#         10: 2, 12: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2,
#     }
    
    
# #     # state_id map for forward, left turn, right turn, 
# #     state_id_map = {
# #         #forward
# #         0: 0, 3: 0, 4: 0, 5: 0, 6: 0, 13: 0, 14: 0, 15: 0, 19: 0, 20: 0, 21: 0, 
        
# #         #left
# #         1: 1, 7: 1, 9: 1, 11: 1, 16: 1, 18: 1, 22: 1, 
        
# #         #right
# #         2: 2, 8: 2, 10: 2, 12: 2, 17: 2, 23: 2, 24: 2
# #     }
    

#     df_new_annotations.loc[:, 'state_id'] = df_new_annotations.loc[:, 'state_id'].replace(state_id_map)
#     df_new_annotations = df_new_annotations.loc[:, 'state_id']    
    
#     return df_new_annotations, df_unique_states


def align_files(bonsai_paths, behavior_paths, num_of_videos):
    
    bonsai_data_list = []
    behavior_data_list = []
    df_new_annotations_list = []


    for path1, path2 in zip(bonsai_paths, behavior_paths):
        # for path in bonsai_paths:
            # print(path)
             # Read the CSV file into a pandas DataFrame
        bonsai_data = pd.read_csv(path1, header=None)

        # Rename the columns of the DataFrame
        bonsai_data = bonsai_data.rename(columns={
            0: 'Time', 1: 'Trial_Number',
            2: 'Reward', 3: 'Frame_Number', 4: 'Central_Zone',
            5: 'L_Zone', 6: 'R_Zone', 7: 'Calcium_frame'})

        
        # Append the DataFrame to the list of CSV data
        bonsai_data_list.append(bonsai_data)
    

        # print(path)
         # Read the CSV file into a pandas DataFrame
        behavior_data = pd.read_hdf(path2, 'per_frame')

        # Append the DataFrame to the list of CSV data
        behavior_data_list.append(behavior_data)
    

    del path1, path2
    
    
    
    for i, (path1, path2) in enumerate(zip(bonsai_data_list, behavior_data_list)):
            df_aligned = path2.loc[path1.groupby('Calcium_frame').first()[1:].Frame_Number].reset_index()
            # df_aligned
            df_new_annotations = df_aligned[['state_id', 'state_name']]
            df_unique_states = df_new_annotations[['state_id', 'state_name']].drop_duplicates(subset='state_id').set_index('state_id')['state_name'].sort_index()
            
            # state id mapping for main corridor, left corridor, right corridor for animal3learnday8
            # print(path2)
            # break
            
            if i == 0:
                state_id_map = {
                    1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0,
                    9: 1, 11: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1,
                    10: 2, 12: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2
                }
                
            elif i == 1:
                state_id_map = {
                    1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0,
                    9: 1, 11: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1,
                    10: 2, 12: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2
                }
                
            else:
                state_id_map = {
                    1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0,
                    9: 1, 11: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1,
                    10: 2, 12: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2
                }
                
            df_new_annotations.loc[:, 'state_id'] = df_new_annotations.loc[:, 'state_id'].replace(state_id_map)
            
            df_new_annotations = df_new_annotations.loc[:, 'state_id']
            
            df_new_annotations_list.append(df_new_annotations)
            
    
    df_new_annotations = pd.concat(df_new_annotations_list, axis=0)
    df_new_annotations = df_new_annotations.reset_index()
    df_new_annotations_unique = df_new_annotations['state_id'].unique()
    return df_new_annotations, df_new_annotations_unique

                
            
