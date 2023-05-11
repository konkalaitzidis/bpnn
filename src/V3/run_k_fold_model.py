import csv  # CSV module is used for working with CSV (Comma Separated Values) files in Python.
# from f1_score import f1_score
import os
import pickle
import time  # for time-related functions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


import tensorflow.compat.v1 as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import KFold
# Train and evaluate the model for each fold
from keras.models import load_model

# import NWBDataGenerator and use it in your code
from nwb_data_generator import NWBDataGeneratorTime
from model_architecture import construct_model
from sklearn.metrics import confusion_matrix


from plots import plot_cm_k_fold

import gc



def run_k_fold(params,
               train_loss_all,
               val_loss_all,
               train_acc_all,
               val_acc_all,
               average_score_list,
               experiment_ID):
    
    
    images = params['images']
    labels = params['labels']
    num_folds = params['number_of_folds']
    shuffle = params['shuffle_data']
    input_shape = params['input_shape']
    num_classes = params['number_of_classes']
    name = params['model_name']
    epochs = params['epochs']
    no_of_behaviors = params['behaviours'],
    df_new_annotations = params['df_new_annotations']
    
    
    # Define the EarlyStopping callback to stop training when the validation loss stops decreasing
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='min')

    # Define the KFold cross-validator
    kf = KFold(n_splits=num_folds, shuffle=shuffle)
    
    conf_matrices = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(images), 1):
    
    
        # loop control
        print(f'\n\n\nFold {fold}/{num_folds}\n')

        # if fold == 4:
        #     break
        
        
        # labels = pd.Series(labels)
        train_labels_fold = df_new_annotations[train_idx]
        test_labels_fold = df_new_annotations[test_idx]

        train_class_counts = train_labels_fold.value_counts()
        test_class_counts = test_labels_fold.value_counts()

        plt.figure(figsize=(8, 6))
        plt.bar(train_class_counts.index, train_class_counts.values, color='b', alpha=0.5, label='Train')
        plt.bar(test_class_counts.index, test_class_counts.values, color='r', alpha=0.5, label='Test')
        plt.xlabel('Class Label')
        plt.ylabel('Number of Instances')
        plt.title(f'Distribution of Class Labels in Fold {fold}/{num_folds}')
        plt.xticks(np.arange(len(train_class_counts.index)), train_class_counts.index)
        plt.legend()
        plt.show()

        
        
#         print("Splitting data with NWBDataGenerator\n")

#         # Define the data generators for the training and validation sets
#         train_generator = NWBDataGeneratorTime(images, labels, train_idx)
#         val_generator = NWBDataGeneratorTime(images, labels, test_idx)
        
#         print("Length of training and validation sets:", len(train_idx), len(test_idx))
        


#         # construct model
#         print("Creating Model\n")
#         model = construct_model(input_shape, num_classes, name)

#         # reset the weights.    
        
#         # Training model
#         print("Training model. Go grab a coffee or take a walk.")
        
#         # history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_labels), callbacks=[early_stopping]) # callbacks=[early_stopping])
        
#         history = model.fit(train_generator, 
#                             epochs=epochs,
#                             validation_data=val_generator,
#                             callbacks=[early_stopping])

        
        
        
#         # all_histories.append(history)  # save the history object to the list

        
#          # Append the training and validation loss and accuracy values to the corresponding lists
#         train_loss_all.append(np.array(history.history['loss']))
#         val_loss_all.append(np.array(history.history['val_loss']))
#         train_acc_all.append(np.array(history.history['accuracy']))
#         val_acc_all.append(np.array(history.history['val_accuracy']))

        
#         # Evaluate the model on the validation set
        
        
#         print("\nEvaluating model...")
#         accuracy_score = model.evaluate(val_generator, verbose=0)
#         print(f'Validation loss: {accuracy_score[0]:.4f}')
#         print(f'Validation accuracy: {accuracy_score[1]:.4f}\n')
#         average_score_list.append(accuracy_score[1])
        
        
        
#         # accuracy_score_list.append(accuracy_score)
        
        
        
#         # get the true labels and predicted labels for the validation set
#         print("Generating predictions on validation data")
#         true_labels = val_generator.get_labels()
#         predictions = model.predict(val_generator)
        
#         y_true = np.array([np.argmax(x) for x in true_labels])
#         y_pred = np.array([np.argmax(x) for x in predictions])
        
#         # print(np.mean(y_true == y_pred))
        
#         # calculate the confusion matrix for this fold
#         cm = confusion_matrix(y_true, y_pred)
#         conf_matrices.append(cm)
        
        
        # model_cm_dir = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V3/output/cm"
        
        # plot_cm_k_fold(val_generator, no_of_behaviors, experiment_ID, model_cm_dir, model)
        
        # del train_images, train_labels, val_images, val_labels, model, history, accuracy_score
        # del train_generator, val_generator, model, history, accuracy_score
        gc.collect()
        
    
    print("\nDone!\n")
    
    # no_of_behaviors = ['Main Corr', 'Left Corr', 'Right Corr']
    
#     model_cm_dir = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V3/output/cm"
    
#     plot_cm_k_fold(conf_matrices, no_of_behaviors, num_classes, experiment_ID, model_cm_dir)
        
    return train_loss_all, val_loss_all, train_acc_all, val_acc_all, average_score_list, conf_matrices
 
    

    
    
    
    

    # plot_confusion_matrix(experiment_ID, no_of_behaviors, train_generator, val_generator, train_labels, val_labels, train_images, val_images, model_cm_dir, model_path, model_version)

    
    
    
    
    
    
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