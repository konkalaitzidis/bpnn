import csv 
import os
import pickle
import time  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow.compat.v1 as tf
import gc

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from keras.models import load_model
from nwb_data_generator import NWBDataGeneratorTime, NWBDataGenerator
from model_architecture import construct_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, f1_score
from bpnn.src.plots import plot_cm_k_fold
from bpnn.src.class_balance import check_class_imbalance_k_fold
from matplotlib.backends.backend_pdf import PdfPages


def run_k_fold(params,
               train_loss_all,
               val_loss_all,
               train_acc_all,
               val_acc_all,
               average_score_list,
               experiment_ID):
    
    # Initializing parameters
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
    df_new_annotations_unique = params['unique_annotations']
    df_new_annotations_check = params['check_annotations']
    # df_new_annotations_names = params['label_names']
    output_dir = params['output_directory']
    df_new_annotations_names = params['label_names']
 

    
    # Define the EarlyStopping callback to stop training when the validation loss stops decreasing
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='min')

    # Define the KFold cross-validator
    kf = KFold(n_splits=num_folds, shuffle=shuffle)
    
    conf_matrices = []
    f1_score_val_list = []
    model_balance_dir = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V3/"+str(output_dir)+"/balance"

    
    with PdfPages(model_balance_dir+'/class_distributions.pdf') as pdf:
        for fold, (train_idx, test_idx) in enumerate(kf.split(images), 1):


            # loop control
            print(f'\n\n\nFold {fold}/{num_folds}\n')

            # save balance output
            save_dir = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V3/"+str(output_dir)+"/balance"     
            
            
            train_labels_fold = df_new_annotations[train_idx]
            test_labels_fold = df_new_annotations[test_idx]
            
            train_labels_names = df_new_annotations_names
            # train_labels_names = df_new_annotations_check['state_name'].unique()
            # test_labels_names = df_new_annotations_check['state_name']
                        
            train_class_counts = train_labels_fold.value_counts()
            test_class_counts = test_labels_fold.value_counts()

            no_of_labels = no_of_behaviors #len(train_labels_names)
            
            
            fig = check_class_imbalance_k_fold(train_class_counts, 
                                               test_class_counts, 
                                               fold, 
                                               num_folds, 
                                               experiment_ID, 
                                               save_dir, 
                                               df_new_annotations_unique, 
                                               df_new_annotations_check, 
                                               train_labels_names, 
                                               no_of_labels)
            

            # Save the plot in the pdf
            pdf.savefig(fig, bbox_inches='tight')
            plt.show()

            print("Splitting data with NWBDataGenerator\n")

            # Define the data generators for the training and validation sets
            train_generator = NWBDataGeneratorTime(images, labels, train_idx)
            val_generator = NWBDataGeneratorTime(images, labels, test_idx)

            print("Length of training and validation sets:", len(train_idx), len(test_idx))



            # construct model
            print("Creating Model\n")
            model = construct_model(input_shape, num_classes, name)


            
            # Training model
            print("Training model. Go grab a coffee or take a walk.")

            # history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_labels), callbacks=[early_stopping]) # callbacks=[early_stopping])

            history = model.fit(train_generator, 
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
            print("\nEvaluating model...")
            accuracy_score = model.evaluate(val_generator, verbose=0)
            print(f'Validation loss: {accuracy_score[0]:.4f}')
            print(f'Validation accuracy: {accuracy_score[1]:.4f}\n')
            average_score_list.append(accuracy_score[1])

            # get the true labels and predicted labels for the validation set
            print("Generating predictions on validation data")
            true_labels = val_generator.get_labels()
            predictions = model.predict(val_generator)

            y_true = np.array([np.argmax(x) for x in true_labels])
            y_pred = np.array([np.argmax(x) for x in predictions])

            # calculate the confusion matrix for this fold
            cm = confusion_matrix(y_true, y_pred)
            conf_matrices.append(cm)


            # find the f1 score
            f1_score_val = f1_score(y_true, y_pred, average='micro')
            print("F1 score is: {:.3f}" .format(f1_score_val))
            f1_score_val_list.append(f1_score_val)
            

            # del train_images, train_labels, val_images, val_labels, model, history, accuracy_score
            del train_generator, val_generator, model, history, accuracy_score
            gc.collect()
        
    
    print("\nDone!\n")
    
    # Plot Confusion Matrices for all Folds

    # specify the file path where you want to save the PDF
    model_cm_dir = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V3/"+str(output_dir)+"/cm"
    
    plot_cm_k_fold(model_cm_dir, fold, cm, conf_matrices, train_labels_names)
    
    #plot the mean confusion matrix
    mean_cm = np.mean(conf_matrices, axis=0)
    # train_labels_names = ['moving', 'rightTurn', 'immobile', 'grooming', 'still', 'leftTurn']
    plt.figure(figsize=(8, 6))
    sns.heatmap(mean_cm, annot=True, cmap='Blues', fmt='g')
    plt.title('BPNNt - Mean Confusion Matrix - K-Fold Cross Validation')
    plt.xlabel('Predicted Labels')
    plt.xticks(np.arange(len(train_labels_names)), train_labels_names, rotation=90, fontsize=6)
    plt.ylabel('True Labels')
    plt.yticks(np.arange(len(train_labels_names)), train_labels_names, rotation=0, fontsize=6)
    plt.savefig("mean_CM"+str(experiment_ID)+'.svg', bbox_inches='tight', dpi=300)
    plt.show()
    
    
    
        
    return train_loss_all, val_loss_all, train_acc_all, val_acc_all, average_score_list, conf_matrices, f1_score_val_list, train_labels_names
 
    

    
    
    
    
    
    
    
#==== run k-fold without time element ====#
    
def run_k_fold_basic(params,
               train_loss_all,
               val_loss_all,
               train_acc_all,
               val_acc_all,
               average_score_list,
               experiment_ID):
    
    # Initializing parameters
    images = params['images']
    labels = params['labels']
    num_folds = params['number_of_folds']
    shuffle = params['shuffle_data']
    input_shape = params['input_shape']
    num_classes = params['number_of_classes']
    name = params['model_name']
    epochs = params['epochs']
    no_of_behaviors = params['behaviours']
    df_new_annotations = params['df_new_annotations']
    df_new_annotations_unique = params['unique_annotations']
    df_new_annotations_check = params['check_annotations']
    df_new_annotations_names = params['label_names']
    output_dir = params['output_directory']

    
    # Define the EarlyStopping callback to stop training when the validation loss stops decreasing
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='min')

    # Define the KFold cross-validator
    kf = KFold(n_splits=num_folds, shuffle=shuffle)
    
    
    conf_matrices = []
    f1_score_val_list = []
    model_balance_dir = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V3/"+str(output_dir)+"/balance"
    
    
    with PdfPages(model_balance_dir+'/class_distributions.pdf') as pdf:
        for fold, (train_idx, test_idx) in enumerate(kf.split(images), 1):

            # loop control
            print(f'\n\n\nFold {fold}/{num_folds}\n')
            
            # save balance output
            save_dir = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V3/"+str(output_dir)+"/balance"           
            
            train_labels_fold = df_new_annotations[train_idx]
            test_labels_fold = df_new_annotations[test_idx]
            
            train_labels_names = df_new_annotations_names
            # train_labels_names = df_new_annotations_check['state_name'].unique()
            # test_labels_names = df_new_annotations_check['state_name']
                        
            train_class_counts = train_labels_fold.value_counts()
            test_class_counts = test_labels_fold.value_counts()

            no_of_labels = no_of_behaviors #no_of_behaviors
            
            
            fig = check_class_imbalance_k_fold(train_class_counts, 
                                               test_class_counts, 
                                               fold, 
                                               num_folds, 
                                               experiment_ID, 
                                               save_dir, 
                                               df_new_annotations_unique, 
                                               df_new_annotations_check, 
                                               train_labels_names, 
                                               no_of_labels)
            

            # Save the plot in the pdf
            pdf.savefig(fig, bbox_inches='tight')
            plt.show()
            
            print("Splitting data with NWBDataGenerator\n")

            # Define the data generators for the training and validation sets
            train_generator = NWBDataGenerator(images, labels, train_idx)
            val_generator = NWBDataGenerator(images, labels, test_idx)

            print("Length of training and validation sets:", len(train_idx), len(test_idx))



            # construct model
            print("Creating Model\n")
            model = construct_model(input_shape, num_classes, name)

            

            # Training model
            print("Training model. Go grab a coffee or take a walk.")

            # history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_labels), callbacks=[early_stopping]) # callbacks=[early_stopping])

            history = model.fit(train_generator, 
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
            print("\nEvaluating model...")
            accuracy_score = model.evaluate(val_generator, verbose=0)
            print(f'Validation loss: {accuracy_score[0]:.4f}')
            print(f'Validation accuracy: {accuracy_score[1]:.4f}\n')
            average_score_list.append(accuracy_score[1])

            # get the true labels and predicted labels for the validation set
            print("Generating predictions on validation data")
            true_labels = val_generator.get_labels()
            predictions = model.predict(val_generator)

            y_true = np.array([np.argmax(x) for x in true_labels])
            y_pred = np.array([np.argmax(x) for x in predictions])

            # calculate the confusion matrix for this fold
            cm = confusion_matrix(y_true, y_pred)
            conf_matrices.append(cm)
            
            # find the f1 score
            f1_score_val = f1_score(y_true, y_pred, average='micro')
            print("F1 score is: {:.3f}" .format(f1_score_val))
            f1_score_val_list.append(f1_score_val)
            
            
            # del train_images, train_labels, val_images, val_labels, model, history, accuracy_score
            del train_generator, val_generator, model, history, accuracy_score
            gc.collect()
    
    print("\nDone!\n")
    
            
    # Plot Confusion Matrices for all Folds
    
    # specify the file path where you want to save the PDF
    model_cm_dir = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V3/"+str(output_dir)+"/cm"
    

    # create a PdfPages object to save the figures
    
    plot_cm_k_fold(model_cm_dir, fold, cm, conf_matrices, train_labels_names)
    
    
    #plot the mean confusion matrix
    mean_cm = np.mean(conf_matrices, axis=0)
    train_labels_names = ['moving', 'rightTurn', 'immobile', 'grooming', 'still', 'leftTurn']
    plt.figure(figsize=(8, 6))
    sns.heatmap(mean_cm, annot=True, cmap='Blues', fmt='g')
    plt.title('Mean Confusion Matrix - K-Fold Cross Validation')
    plt.xlabel('Predicted Labels')
    plt.xticks(np.arange(len(train_labels_names)), train_labels_names, rotation=90, fontsize=6)
    plt.ylabel('True Labels')
    plt.yticks(np.arange(len(train_labels_names)), train_labels_names, rotation=0, fontsize=6)
    plt.savefig("mean_CM"+str(experiment_ID)+'.svg', bbox_inches='tight', dpi=300)
    plt.show()
    
    
    
    # dir_name = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V3/output/pickles"
  
    return train_loss_all, val_loss_all, train_acc_all, val_acc_all, average_score_list, conf_matrices, f1_score_val_list, train_labels_names

