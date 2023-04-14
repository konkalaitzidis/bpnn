import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime # for saving date and time information
import csv
import os



def save_training_info(model=None, history=None, video_name=None, comment=None, experiment_ID=None, save_dir=None, f1_score=None):
    # Set the model name
    model_name = model.name if model else "No Model"
    
    # Get the current date and time
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Handle None values for other arguments
    video_name = video_name if video_name else "No Video Name"
    comment = comment if comment else "No Comment"
    experiment_ID = experiment_ID if experiment_ID else "No Experiment ID"
    save_dir = save_dir if save_dir else "No Save Directory"
    
    
    # dir_name = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V2/output"
    # Save the history object to a CSV file
    with open(os.path.join(save_dir, str(model_name)+'-_history.csv'), 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header row if file is empty
        if f.tell() == 0:
            writer.writerow(['Experiment ID', 'Model', 'Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'F1-Score', 'Date/Time', 'Video Name','Comment'])

        # Write data for each epoch
        for i, (tl, ta, vl, va) in enumerate(zip(history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy'])):
            writer.writerow([experiment_ID, model_name, i+1, tl, ta, vl, va, f1_score, date_time, video_name, comment])
        writer.writerow(['', '', '', '', '', '', '', '', '' , '', ''])
        
        print("Training info saved in csv file.")