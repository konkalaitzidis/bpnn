# import tensorflow.compat.v1 as tf
# import time # for time-related functions
# from tensorflow.keras.optimizers.legacy import Adam
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras.callbacks import EarlyStopping
# import pickle
# import csv # CSV module is used for working with CSV (Comma Separated Values) files in Python.
# import os

import tensorflow.compat.v1 as tf
import time # for time-related functions
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import csv # CSV module is used for working with CSV (Comma Separated Values) files in Python.
# from f1_score import f1_score
import os

def base_model_execution(base_params, save_dir, model_version):    
    
    model = base_params['model']
    tf = base_params['tf']
    train_images = base_params['train_images']
    train_labels = base_params['train_labels']
    epochs = base_params['epochs']
    batch_size = base_params['batch_size']
    validation_data = base_params['validation_data']
    val_images = base_params['val_images']
    val_labels = base_params['val_labels']
    
    start_time = time.time()
    
    # f1_score = f1_score()
    
    print("Compiling model...")
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(), metrics=['accuracy'])
    
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
    
    
    print("Running model. Go grab a coffee or smth.")
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks=[early_stopping]) 
    
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Execution time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")

    
    # dir_name = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V2/output/pickles"
    model_save_name = str(model_version)+'.h5'
    print("Now saving model data to pickles. Please wait...")
    model.save(f"{save_dir}/{model_save_name}")
    
    # Save the history object to a pickle file
    with open(os.path.join(save_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
    
    with open(os.path.join(save_dir, 'train_images.pkl'), 'wb') as f:
        pickle.dump(train_images, f)

    with open(os.path.join(save_dir, 'val_images.pkl'), 'wb') as f:
        pickle.dump(val_images, f)

    with open(os.path.join(save_dir, 'train_labels.pkl'), 'wb') as f:
        pickle.dump(train_labels, f)
        
    with open(os.path.join(save_dir, 'val_labels.pkl'), 'wb') as f:
        pickle.dump(val_labels, f)
        
    print("Done!")

    
    return history