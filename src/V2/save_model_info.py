import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime # for saving date and time information
import csv



def save_training_info(model, history, video_name):
    # Set the model name
    model_name = model.name
    
    # Get the current date and time
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")

    # Save the history object to a CSV file
    with open(str(model_name)+'-training_history.csv', 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header row if file is empty
        if f.tell() == 0:
            writer.writerow(['Model', 'Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Date/Time', 'Video Name','Comment'])

        # Write data for each epoch
        for i, (tl, ta, vl, va) in enumerate(zip(history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy'])):
            writer.writerow([model_name, i+1, tl, ta, vl, va, date_time, video_name, 'without frame subtraction'])
        writer.writerow(['', '', '', '', '', '', '', '' , ''])
        
        print("Training info saved in csv file.")