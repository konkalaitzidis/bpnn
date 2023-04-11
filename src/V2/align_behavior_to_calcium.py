import pandas as pd


def align_files(bonsai_data, df_behavior):

    # Adding column names
    bonsai_data = bonsai_data.rename(columns={
        0: 'Time', 1: 'Trial_Number',
        2: 'Reward', 3: 'Frame_Number', 4: 'Central_Zone',
        5: 'L_Zone', 6: 'R_Zone', 7: 'Calcium_frame'})
    
    df_aligned = df_behavior.loc[bonsai_data.groupby('Calcium_frame').first()[1:].Frame_Number].reset_index()
        
    df_new_annotations = df_aligned[['state_id', 'state_name']]
    df_unique_states = df_new_annotations[['state_id', 'state_name']].drop_duplicates(subset='state_id').set_index('state_id')['state_name'].sort_index()

    # # state id mapping for main corridor, left corridor, right corridor
    # state_id_map = {
    #     1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0,
    #     9: 1, 11: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1,
    #     10: 2, 12: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2,
    # }
    
    # state_id map for forward, left turn, right turn, 
    state_id_map = {
        #forward
        0: 0, 3: 0, 4: 0, 5: 0, 6: 0, 13: 0, 14: 0, 15: 0, 19: 0, 20: 0, 21: 0, 
        
        #left
        1: 1, 7: 1, 9: 1, 11: 1, 16: 1, 18: 1, 22: 1, 
        
        #right
        2: 2, 8: 2, 10: 2, 12: 2, 17: 2, 23: 2, 24: 2
    }
    
    
    

    df_new_annotations.loc[:, 'state_id'] = df_new_annotations.loc[:, 'state_id'].replace(state_id_map)
    # if df_new_annotations.loc[:, 'state_id'] == 0:
    #     forward = 0
    # elif df_new_annotations.loc[:, 'state_id'] == 1:
    #     left = 1
    # else right = 2

    df_new_annotations = df_new_annotations.loc[:, 'state_id']    
    
    return df_new_annotations, df_unique_states