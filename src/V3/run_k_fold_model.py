import csv  # CSV module is used for working with CSV (Comma Separated Values) files in Python.
# from f1_score import f1_score
import os
import pickle
import time  # for time-related functions
import numpy as np

import tensorflow.compat.v1 as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import KFold
# Train and evaluate the model for each fold
from keras.models import load_model

# import NWBDataGenerator and use it in your code
from nwb_data_generator import NWBDataGenerator
from model_architecture import construct_model

import gc



def run_k_fold(images, 
               labels, 
               num_folds, 
               shuffle, 
               input_shape, 
               num_classes, 
               name, 
               epochs,
               train_loss_all,
               val_loss_all,
               train_acc_all,
               val_acc_all):
    
    
    # Define the EarlyStopping callback to stop training when the validation loss stops decreasing
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')

    # Define the KFold cross-validator
    kf = KFold(n_splits=num_folds, shuffle=shuffle)
    
    
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(images), 1):
    
    
        # loop control
        print(f'Fold {fold}/{num_folds}\n')

        # if fold == 4:
        #     break

        print("Splitting data with NWBDataGenerator\n")

        # Define the data generators for the training and validation sets
        train_generator = NWBDataGenerator(images, labels, train_idx)
        val_generator = NWBDataGenerator(images, labels, test_idx)
        
        # construct model
        print("Creating Model\n")
        model = construct_model(input_shape, num_classes, name)

        # reset the weights. 
        
        
        # Training model
        print("Training model. Go grab a coffee or take a walk.")
        
        # history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_labels), callbacks=[early_stopping]) # callbacks=[early_stopping])
        
        history = model.fit_generator(train_generator, 
                                      epochs=epochs, 
                                      validation_data=val_generator, 
                                      callbacks=[early_stopping])
        
        # all_histories.append(history)  # save the history object to the list

        
        
         # Append the training and validation loss and accuracy values to the corresponding lists
        train_loss_all.append(np.array(history.history['loss']))
        val_loss_all.append(np.array(history.history['val_loss']))
        train_acc_all.append(np.array(history.history['accuracy']))
        val_acc_all.append(np.array(history.history['val_accuracy']))

        # Evaluate the model on the validation set
        print("\nEvaluating model.")
        accuracy_score = model.evaluate_generator(val_generator, verbose=0)
        
        # accuracy_score = model.evaluate(val_images, val_labels, verbose=0)
        print(f'Validation accuracy: {accuracy_score[1]:.4f}\n')
        # accuracy_score_list.append(accuracy_score)


        # performance management, delete variables that are no longer needed
        # del train_images, train_labels, val_images, val_labels, model, history, accuracy_score
        del train_generator, val_generator, model, history, accuracy_score

        # Run the garbage collector to free up memory
        gc.collect()
        
        return train_loss_all, val_loss_all, train_acc_all, val_acc_all
        
    
    print("\nDone!\n")
    
    
    
#     # dir_name = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V2/output/pickles"
#     model_save_name = str(model_version)+'.h5'
#     print("Now saving model data to pickles. Please wait...")
#     model.save(f"{save_dir}/{model_save_name}")
    
#     dir_name_pickles = "src/V2/output/pickels"
#     # Save the history object to a pickle file
#     with open(os.path.join(save_dir, 'history.pkl'), 'wb') as f:
#         pickle.dump(history.history, f)
    
#     with open(os.path.join(save_dir, 'train_images.pkl'), 'wb') as f:
#         pickle.dump(train_images, f)

#     with open(os.path.join(save_dir, 'val_images.pkl'), 'wb') as f:
#         pickle.dump(val_images, f)

#     with open(os.path.join(save_dir, 'train_labels.pkl'), 'wb') as f:
#         pickle.dump(train_labels, f)
        
#     with open(os.path.join(save_dir, 'val_labels.pkl'), 'wb') as f:
#         pickle.dump(val_labels, f)
        
#     print("Done!")

    
#     return history