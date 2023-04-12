import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime # for saving date and time information
import csv
import os



def save_training_info(model, history, video_name, comment, experiment_ID):
    # Set the model name
    model_name = model.name
    
    # Get the current date and time
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    
    
    dir_name = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V2/output"
    # Save the history object to a CSV file
    with open(os.path.join(dir_name, str(model_name)+'-training_history.csv'), 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header row if file is empty
        if f.tell() == 0:
            writer.writerow(['Experiment ID', 'Model', 'Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Date/Time', 'Video Name','Comment'])

        # Write data for each epoch
        for i, (tl, ta, vl, va) in enumerate(zip(history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy'])):
            writer.writerow([experiment_ID, model_name, i+1, tl, ta, vl, va, date_time, video_name, comment])
        writer.writerow(['', '', '', '', '', '', '', '', '' , ''])
        
        print("Training info saved in csv file.")